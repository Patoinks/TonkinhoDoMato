import { Chess, type Color, type Move } from 'chess.js';
import { encode, softmax } from './encoding';
import { pieceSenses, positionRisk, type PieceValuer } from './pieceSafety';
import { kingSenses, kingBalance } from './kingSafety';
import { forcedMate, rankLast } from './mateSearch';
import { exchangeAfterMoves } from './exchangeSearch';

type Tensor = { data: Float32Array; shape: number[] };
type Entry = { key: number[]; target: number; game: number; trust: number; own_turn: boolean | null };
type Skill = { key: number[]; actions: string[]; value: number; trust: number; games: number[] };
export interface TemporalManifest {
  model_kind?: string;
  config?: { [key: string]: number | string | boolean | undefined; objective?: string };
  kd_values?: Record<string, number>;
  king_safety_model?: boolean;
  knowledge?: { entries: Entry[]; skills: Skill[]; calibration: { count: number[]; square_error: number[] } };
}
type State = { memory: Float64Array; context: Float64Array; root: string; history: string[]; self: Color;
  plan: string[]; prefix: string[] };
const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));
const uci = (m: Move) => m.from + m.to + (m.promotion ?? '');
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let sum = 0; for (let i = 0; i < a.length; i++) sum += a[i] * b[i]; return sum;
};
const normalize = (a: number[]) => { const n = Math.max(1e-8, Math.sqrt(dot(a, a))); return a.map(v => v / n); };
const moveUci = (b: Chess, s: string) => b.move({ from: s.slice(0, 2), to: s.slice(2, 4), promotion: s[4] });
const square = (s: string) => (Number(s[1]) - 1) * 8 + s.charCodeAt(0) - 97;

// Preserve Python's deterministic tie ordering, including promotions and castling.
export function pythonMoves(b: Chess): Move[] {
  const inCheck = b.isCheck();
  const key = (m: Move): number[] => {
    const from = square(m.from), to = square(m.to);
    if (inCheck && m.piece === 'k') return [-1, -from, -to, 0];
    if (m.isKingsideCastle() || m.isQueensideCastle()) return [1, 0, -to, 0];
    if (m.piece !== 'p') return [0, -from, -to, 0];
    const promotion = m.promotion ? 'qrbn'.indexOf(m.promotion) : 0;
    if (m.isEnPassant()) return [5, -from, -to, 0];
    if (m.isCapture()) return [2, -from, -to, promotion];
    return [m.isBigPawn() ? 4 : 3, -to, 0, promotion];
  };
  return b.moves({ verbose: true }).sort((a, b) => {
    const x = key(a), y = key(b);
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return x[i] - y[i];
    return 0;
  });
}

export function encodedFen(b: Chess, last?: Move): string {
  const fields = b.fen({ forceEnpassantSquare: true }).split(' ');
  if (last) fields[3] = last.isBigPawn() ? last.from[0] + String((Number(last.from[1]) + Number(last.to[1])) / 2) : '-';
  return fields.join(' ');
}

/** Same draw-claim convention used by the Python training actor. */
export function terminal(b: Chess, self: Color, objective = 'survival', weight = .5): number | null {
  const draw = objective === 'victory' ? .5 : objective === 'shaped' ? 1 - weight / 2 : 1;
  if (b.isCheckmate()) return b.turn() === self ? 0 : 1;
  if (b.isDraw()) return draw;
  const half = Number(b.fen().split(' ')[4]);
  const history = b.history({ verbose: true });
  const position = (fen: string) => fen.split(' ').slice(0, 4).join(' ');
  const keys = history.map(m => position(m.before)); keys.push(position(b.fen()));
  const repeated = half >= 7 && new Set(keys).size < keys.length;
  if (half >= 99 || repeated) {
    for (const move of b.moves({ verbose: true })) {
      b.move(move);
      const draw = !b.isCheckmate() && (b.isDrawByFiftyMoves() || (repeated && b.isThreefoldRepetition()));
      b.undo();
      if (draw) return objective === 'victory' ? .5 : objective === 'shaped' ? 1 - weight / 2 : 1;
    }
  }
  return null;
}

const positionKey = (fen: string) => fen.split(' ').slice(0, 4).join(' ');
/** How many times the current position has occurred (python-chess is_repetition counting). */
export function repetitions(b: Chess) {
  const current = positionKey(b.fen());
  return b.history({ verbose: true }).filter(m => positionKey(m.before) === current).length + 1;
}
/** python-chess can_claim_threefold_repetition: now, or by a move that repeats a position a third time. */
export function canClaimThreefold(b: Chess) {
  if (repetitions(b) >= 3) return true;
  for (const move of b.moves({ verbose: true })) {
    b.move(move);
    try { if (repetitions(b) >= 3) return true; } finally { b.undo(); }
  }
  return false;
}

export class TemporalNetwork {
  private states = new WeakMap<Chess, State>();
  private cfg: Record<string, number>;
  private manifest: TemporalManifest;
  private tensors: Map<string, Tensor>;
  private exchangeEvaluations = new WeakMap<Chess, Record<string, number> | null>();
  constructor(manifest: TemporalManifest, tensors: Map<string, Tensor>) {
    this.manifest = manifest; this.tensors = tensors;
    this.cfg = Object.fromEntries(Object.entries(manifest.config ?? {}).filter(([, v]) => typeof v === 'number')) as Record<string, number>;
    if (!this.cfg || !tensors.has('net.memory_cell.weight_ih')) throw new Error('Missing temporal model state');
  }
  private affine(x: Float64Array, weightName: string, biasName: string): Float64Array {
    const weight = this.tensors.get(weightName), bias = this.tensors.get(biasName);
    if (!weight || !bias || weight.shape[1] !== x.length) throw new Error(`Invalid tensor ${weightName}`);
    const [rows, cols] = weight.shape;
    return Float64Array.from({ length: rows }, (_, r) => {
      let v = bias.data[r]; for (let c = 0; c < cols; c++) v += weight.data[r * cols + c] * x[c]; return v;
    });
  }
  protected linear(x: Float64Array, name: string, activation = false) {
    const y = this.affine(x, `${name}.weight`, `${name}.bias`);
    return activation ? y.map(v => v * sigmoid(v)) : y;
  }
  protected encoder(x: Float64Array, name = 'net.encoder') {
    let latent = this.linear(this.linear(x, `${name}.0`, true), `${name}.2`, true);
    for (let i = 4; this.tensors.has(`${name}.${i}.fc1.weight`); i++) {
      const residual = this.linear(this.linear(latent, `${name}.${i}.fc1`, true), `${name}.${i}.fc2`);
      latent = latent.map((v, j) => v + residual[j]);
    }
    return latent;
  }
  protected step(fen: string, self: Color, previous: Float64Array) {
    const latent = this.encoder(encode(fen, self));
    const ih = this.affine(latent, 'net.memory_cell.weight_ih', 'net.memory_cell.bias_ih');
    const hh = this.affine(previous, 'net.memory_cell.weight_hh', 'net.memory_cell.bias_hh');
    const n = previous.length;
    const memory = previous.map((old, i) => {
      const reset = sigmoid(ih[i] + hh[i]), update = sigmoid(ih[n + i] + hh[n + i]);
      const next = Math.tanh(ih[2 * n + i] + reset * hh[2 * n + i]);
      return (1 - update) * next + update * old;
    });
    const joined = new Float64Array(latent.length + n); joined.set(latent); joined.set(memory, latent.length);
    return { memory, context: this.linear(joined, 'net.fusion.0', true) };
  }
  protected value(context: Float64Array) {
    const heads = Array.from({ length: this.cfg.ensemble_size }, (_, i) =>
      sigmoid(this.linear(this.linear(context, `net.survival_heads.${i}.0`, true), `net.survival_heads.${i}.2`)[0]));
    const mean = heads.reduce((a, b) => a + b, 0) / heads.length;
    const deviation = Math.sqrt(heads.reduce((a, b) => a + (b - mean) ** 2, 0) / heads.length);
    const predictLogit = this.linear(this.linear(context, 'net.predict_head.0', true), 'net.predict_head.2')[0];
    return { mean, deviation, predictLogit };
  }
  observe(board: Chess, self: Color) {
    const moves = board.history({ verbose: true }), history = moves.map(uci);
    const root = moves[0]?.before ?? board.fen({ forceEnpassantSquare: true });
    let state = this.states.get(board);
    if (state && state.self === self && state.root === root && state.history.length === history.length
      && state.history.every((m, i) => m === history[i])) return state;
    const incremental = state && state.self === self && state.root === root
      && state.history.length <= history.length && state.history.every((m, i) => m === history[i]);
    if (!incremental) {
      state = { ...this.step(root, self, new Float64Array(this.cfg.memory_size)), root, history: [], self, plan: [], prefix: [] };
    }
    state = state!;
    const replay = new Chess(root);
    for (let i = 0; i < moves.length; i++) {
      const last = moveUci(replay, history[i]);
      if (i >= state.history.length) Object.assign(state, this.step(encodedFen(replay, last), self, state.memory));
    }
    state.history = history; this.states.set(board, state); return state;
  }
  snapshot(board: Chess, self: Color) {
    const s = this.observe(board, self); return { memory: Array.from(s.memory), value: this.value(s.context).mean };
  }
  kingSafety(board: Chess, owner: Color, last?: Move) {
    if (!this.manifest.king_safety_model) return 1 - kingSenses(board)[owner].risk;
    const x = encode(encodedFen(board, last), owner);
    return sigmoid(this.linear(this.linear(this.linear(x, 'king_safety.body.0', true),
      'king_safety.body.2', true), 'king_safety.body.4')[0]);
  }
  private kingBalance(board: Chess, self: Color, last?: Move) {
    return this.manifest.king_safety_model
      ? this.kingSafety(board, self, last) - this.kingSafety(board, self === 'w' ? 'b' : 'w', last)
      : kingBalance(kingSenses(board), self);
  }
  private ending(board: Chess, self: Color) {
    return terminal(board, self, this.manifest.config?.objective, this.cfg.victory_weight ?? .5);
  }
  private continuation(value: number) {
    const objective = this.manifest.config?.objective;
    const survival = 1 - this.cfg.survival_gamma + this.cfg.survival_gamma * value;
    const weight = this.cfg.victory_weight ?? .5;
    return objective === 'victory' ? value : objective === 'shaped' ? (1 - weight) * survival + weight * value : survival;
  }
  private mateEvaluations = new WeakMap<Chess, { forced_mate: number; allows_mate: Record<string, number> }>();
  mateDiagnostics(board: Chess) { return this.mateEvaluations.get(board); }
  private exchangeValues(): number[] | null {
    if (!(this.cfg.exchange_depth > 0) || !this.manifest.kd_values) return null;
    const values = [0, ...Array.from({ length: 5 }, (_, i) => this.manifest.kd_values![String(i + 1)])];
    return values.every(Number.isFinite) ? values : null;
  }
  /** Python _exchange_nets: immediate capture/promotion minus the opponent's forced gain. */
  exchangeNets(board: Chess, moves = pythonMoves(board)): Record<string, number> | null {
    const values = this.exchangeValues();
    if (!values) return null;
    try {
      const losses = exchangeAfterMoves(board, moves, values, this.cfg.exchange_depth,
        this.cfg.exchange_node_limit ?? 200000);
      return Object.fromEntries(moves.map((move, i) => {
        const victim = move.isEnPassant() ? 1 : move.captured ? ' pnbrqk'.indexOf(move.captured) : 0;
        const gain = victim ? values[victim] : 0;
        const promotion = move.promotion ? values[' pnbrqk'.indexOf(move.promotion)] - values[1] : 0;
        return [uci(move), gain + promotion - losses[i]];
      }));
    } catch (error) {
      if (error instanceof RangeError && error.message.includes('node_limit')) return null;
      throw error;
    }
  }
  exchangeDiagnostics(board: Chess) { return this.exchangeEvaluations.get(board); }
  private key(fen: string, self: Color) {
    const x = encode(fen, self), pooled: number[] = [];
    for (let p = 0; p < 12; p++) for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
      const k = p * 64 + r * 16 + c * 2;
      pooled.push((x[k] + x[k + 1] + x[k + 8] + x[k + 9]) / 4);
    }
    const latent = normalize(Array.from(this.encoder(x, 'knowledge_encoder'))).map(v => v * .5);
    return normalize([...normalize(pooled), ...Array.from(x.slice(768)).map(v => v * .5), ...latent]);
  }
  private remembered(key: number[]) {
    const entries = this.manifest.knowledge?.entries ?? [];
    const nearest = entries.map((e, i) => ({ e, i, sim: e.own_turn === true ? -1 : dot(e.key, key) }))
      .sort((a, b) => b.sim - a.sim || b.i - a.i).slice(0, this.cfg.memory_neighbors)
      .filter(item => item.sim >= this.cfg.memory_similarity);
    if (!nearest.length) return { strength: 0, value: 0 };
    const weights = nearest.map(({ e, sim }) => Math.max(.01, sim - this.cfg.memory_similarity + .01) * e.trust);
    const sum = weights.reduce((a, b) => a + b, 0);
    const value = nearest.reduce((a, { e }, i) => a + e.target * weights[i], 0) / sum;
    const trust = nearest.reduce((a, { e }, i) => a + e.trust * weights[i], 0) / sum;
    const support = new Set(nearest.map(({ e }) => e.game)).size;
    return { value, strength: this.cfg.memory_weight * trust * nearest[0].sim * support / (support + 2) };
  }
  private candidates(board: Chess, self: Color) {
    const state = this.observe(board, self);
    return pythonMoves(board).map(move => {
      board.move(move);
      try {
        const fen = encodedFen(board, move), step = this.step(fen, self, state.memory);
        const value = this.value(step.context);
        return { move, fen, ...step, ...value, uci: uci(move), san: move.san, utility: value.mean, logits: [] as number[], wdl: [] as number[] };
      } finally { board.undo(); }
    });
  }
  sensePieces(board: Chess, self: Color, context?: Float64Array) {
    return pieceSenses(board, this.pieceValuer(board, self, context), this.cfg.piece_exchange_tolerance, this.cfg.piece_exchange_depth);
  }
  /** Python CognitiveAgent.piece_valuer: learned values in pawn units, blended with kd_values. */
  pieceValuer(board: Chess, self: Color, context?: Float64Array): PieceValuer {
    const current = context ?? this.observe(board, self).context;
    // Python _piece_index: with symmetric_piece_values an enemy piece is read as its owner sees it.
    const symmetric = Boolean(this.manifest.config?.symmetric_piece_values);
    const rawValue: PieceValuer = (piece, location) => {
      const enemy = piece.color !== self;
      const side = enemy && !symmetric ? 1 : 0;
      const relative = square(location) ^ (self === 'w' ? 0 : 56);
      const parts = [['net.type_embed.weight', 'pnbrqk'.indexOf(piece.type)],
        ['net.square_embed.weight', enemy && symmetric ? relative ^ 56 : relative], ['net.side_embed.weight', side]] as const;
      const embeddings = parts.map(([name, index]) => {
        const tensor = this.tensors.get(name)!; return tensor.data.slice(index * tensor.shape[1], (index + 1) * tensor.shape[1]);
      });
      const input = new Float64Array(current.length + embeddings.reduce((n, e) => n + e.length, 0));
      input.set(current); let offset = current.length;
      for (const embedding of embeddings) { input.set(embedding, offset); offset += embedding.length; }
      return this.linear(this.linear(input, 'net.piece_head.0', true), 'net.piece_head.2')[0];
    };
    const pieces = board.board().flat().filter(p => p !== null), raw = pieces.map(p => rawValue(p, p.square));
    const pawns = pieces.flatMap((p, i) => p.type === 'p' ? [Math.abs(raw[i])] : []).sort((a, b) => a - b);
    const unit = pawns.length ? Math.max(.001, (pawns[Math.floor(pawns.length / 2)] + pawns[Math.floor((pawns.length - 1) / 2)]) / 2) : .001;
    const key = (piece: { color: Color; type: string }, sq: string) => piece.color + piece.type + sq;
    const normalize = (v: number, piece: { color: Color; type: string }) => {
      const sign = piece.color === self ? 1 : -1;
      const learned = Math.max(-12, Math.min(12, v * sign / unit));
      const kd = this.manifest.kd_values?.[String('pnbrqk'.indexOf(piece.type) + 1)], weight = this.cfg.kd_weight ?? 0;
      return weight > 0 && kd !== undefined && piece.type !== 'k' ? weight * sign * kd + (1 - weight) * learned : learned;
    };
    const values = new Map(pieces.map((p, i) => [key(p, p.square), normalize(raw[i], p)]));
    const value: PieceValuer = (piece, sq) => {
      if (piece.type === 'k') return 10000;
      const id = key(piece, sq);
      if (!values.has(id)) values.set(id, normalize(rawValue(piece, sq), piece));
      return values.get(id)!;
    };
    return value;
  }
  /** Python _avoid_repetition_when_ahead: ahead by `margin` pawns, a repetition draw is nearly worthless. */
  private avoidRepetitionWhenAhead(board: Chess, self: Color, choices: { move: Move; uci: string; utility: number }[],
                                   endings: Map<string, number | null>) {
    const margin = Number(this.manifest.config?.avoid_repetition_margin ?? 0);
    if (!(margin > 0)) return;
    const objective = this.manifest.config?.objective ?? 'survival', weight = this.cfg.victory_weight ?? .5;
    const drawValue = objective === 'victory' ? .5 : objective === 'shaped' ? 1 - weight / 2 : 1;
    const repeated = choices.filter(c => {
      if (endings.get(c.uci) !== drawValue) return false;
      board.move(c.move);
      try { return canClaimThreefold(board) || repetitions(board) >= 5; } finally { board.undo(); }
    });
    if (!repeated.length) return;
    const value = this.pieceValuer(board, self);
    let balance = 0;
    for (const piece of board.board().flat()) if (piece && piece.type !== 'k') balance += value(piece, piece.square);
    if (balance >= margin) for (const c of repeated) c.utility = Math.min(c.utility, .05);
  }
  predict(board: Chess, self: Color) {
    const choices = this.candidates(board, self), probabilities = softmax(choices.map(c => c.predictLogit));
    return choices.map((c, i) => ({ uci: c.uci, san: c.san, p: probabilities[i] })).sort((a, b) => b.p - a.p);
  }
  rank(board: Chess, self: Color) {
    const choices = this.candidates(board, self), cfg = this.cfg, knowledge = this.manifest.knowledge;
    if (!knowledge) return choices.sort((a, b) => b.utility - a.utility);
    const kingBonuses = new Map<string, number>();
    const kingPressure = new Set<string>();
    const nets = cfg.piece_safety_weight ? this.exchangeNets(board, choices.map(c => c.move)) : null;
    this.exchangeEvaluations.set(board, nets);
    for (const c of choices) {
      const recall = this.remembered(this.key(c.fen, self));
      c.utility = (1 - recall.strength) * c.mean + recall.strength * recall.value;
      if (cfg.piece_safety_weight) {
        if (nets) c.utility += cfg.piece_safety_weight * Math.tanh(nets[c.uci] / 3);
        else {
          board.move(c.move);
          try {
            const senses = this.sensePieces(board, self, c.context);
            c.utility += cfg.piece_safety_weight * (positionRisk(senses, self === 'w' ? 'b' : 'w') - positionRisk(senses, self));
          } finally { board.undo(); }
        }
      }
      if (cfg.king_safety_weight) {
        board.move(c.move);
        try {
          const bonus = cfg.king_safety_weight * this.kingBalance(board, self, c.move);
          kingBonuses.set(c.uci, bonus); c.utility += bonus;
          if (1 - this.kingSafety(board, 'w', c.move) > .1 || 1 - this.kingSafety(board, 'b', c.move) > .1) kingPressure.add(c.uci);
        } finally { board.undo(); }
      }
    }
    const diagnostics = { forced_mate: 0, allows_mate: {} as Record<string, number> };
    this.mateEvaluations.set(board, diagnostics);
    const limit = cfg.mate_node_limit ?? 20000;
    if (cfg.mate_defense_moves > 0) for (const c of choices) {
      board.move(c.move);
      try {
        if (this.ending(board, self) === null) {
          const found = forcedMate(board, self === 'w' ? 'b' : 'w', cfg.mate_defense_moves, limit).found;
          if (found) diagnostics.allows_mate[c.uci] = found;
        }
      } finally { board.undo(); }
    }
    rankLast(choices, diagnostics.allows_mate);
    const order = choices.map((c, i) => ({ c, i })).sort((a, b) => b.c.utility - a.c.utility || b.i - a.i);
    const top = order[0].c, gap = order.length > 1 ? top.utility - order[1].c.utility : 1;
    const calibration = knowledge.calibration, bin = Math.min(calibration.count.length - 1, Math.floor(top.mean * calibration.count.length));
    let uncertainty = Math.max(top.deviation, Math.sqrt((calibration.square_error[bin] + .25) / (calibration.count[bin] + 1)));
    if (calibration.count[bin] < cfg.calibration_min_samples) uncertainty = Math.max(uncertainty, cfg.planning_uncertainty);
    if (top.mean < .35 || board.isCheck() || uncertainty >= cfg.planning_uncertainty || gap <= cfg.planning_gap || kingPressure.has(top.uci)) {
      const roots = order.slice(0, cfg.planning_candidates), revised = new Set<number>(); let used = 0;
      for (let ordinal = 0; ordinal < roots.length; ordinal++) {
        const { c, i } = roots[ordinal]; board.move(c.move);
        try {
          const ending = this.ending(board, self);
          if (ending !== null) { c.utility += ending - c.mean - (kingBonuses.get(c.uci) ?? 0); revised.add(i); continue; }
          const available = Math.floor((cfg.planning_nodes - used) / (roots.length - ordinal));
          if (available < 1) break;
          let replies = pythonMoves(board);
          if (replies.length > available) replies = replies.sort((a, b) =>
            Number(!/[+#]/.test(a.san)) - Number(!/[+#]/.test(b.san)) ||
            Number(!a.isCapture() && !a.isEnPassant()) - Number(!b.isCapture() && !b.isEnPassant()) || uci(a).localeCompare(uci(b))).slice(0, available);
          const previews = replies.map(reply => {
            board.move(reply);
            try {
              const step = this.step(encodedFen(board, reply), self, c.memory), v = this.value(step.context);
              const ending = this.ending(board, self);
              const bonus = ending === null && cfg.king_safety_weight ? cfg.king_safety_weight * this.kingBalance(board, self, reply) : 0;
              return { value: ending ?? v.mean + bonus, logit: v.predictLogit };
            } finally { board.undo(); }
          });
          used += replies.length;
          const p = softmax(previews.map(v => v.logit));
          const selected = p.map((q, j) => ({ q, j })).sort((a, b) => b.q - a.q || b.j - a.j)
            .slice(0, Math.max(1, cfg.planning_replies - 1)).map(v => v.j);
          const worst = previews.reduce((best, v, j) => v.value < previews[best].value ? j : best, 0);
          if (!selected.includes(worst)) selected.push(worst);
          const sum = selected.reduce((a, j) => a + p[j] + 1e-8, 0);
          const expected = selected.reduce((a, j) => a + previews[j].value * (p[j] + 1e-8), 0) / sum;
          const continuation = (1 - cfg.robust_weight) * expected + cfg.robust_weight * Math.min(...selected.map(j => previews[j].value));
          c.utility += this.continuation(continuation) - c.mean - (kingBonuses.get(c.uci) ?? 0); revised.add(i);
        } finally { board.undo(); }
      }
      if (revised.size) choices.forEach((c, i) => { if (!revised.has(i)) c.utility = -1e9; });
    }
    const endings = new Map<string, number | null>();
    for (const c of choices) {
      if (cfg.piece_safety_weight || cfg.king_safety_weight || ['victory', 'shaped'].includes(this.manifest.config?.objective ?? '')) c.utility = Math.min(c.utility, .999999);
      board.move(c.move); const ending = this.ending(board, self); board.undo();
      if (ending !== null) c.utility = ending;
      endings.set(c.uci, ending);
    }
    this.avoidRepetitionWhenAhead(board, self, choices, endings);
    rankLast(choices, diagnostics.allows_mate);
    const state = this.observe(board, self), history = state.history;
    const expected = [...state.prefix, ...state.plan], consumed = history.length - state.prefix.length;
    if (!state.plan.length || consumed < 0 || consumed >= state.plan.length || !history.every((m, i) => m === expected[i])) {
      const key = this.key(encodedFen(board, board.history({ verbose: true }).at(-1)), self);
      const mirror = (s: string) => self === 'w' ? s : s[0] + (9 - Number(s[1])) + s[2] + (9 - Number(s[3])) + s.slice(4);
      const candidates = knowledge.skills.map(s => ({ s, sim: dot(s.key, key) }))
        .filter(({ s, sim }) => s.games.length >= cfg.skill_min_support && sim >= cfg.memory_similarity && s.trust >= .5)
        .sort((a, b) => b.sim * b.s.value * b.s.trust - a.sim * a.s.value * a.s.trust);
      state.plan = []; state.prefix = [...history];
      for (const { s } of candidates) {
        const actions = s.actions.map(mirror); let pushed = 0, legal = true;
        try {
          for (const action of actions) {
            if (terminal(board, board.turn()) !== null || !pythonMoves(board).some(m => uci(m) === action)) { legal = false; break; }
            moveUci(board, action); pushed++;
          }
        } finally { while (pushed-- > 0) board.undo(); }
        if (legal) { state.plan = actions; break; }
      }
    }
    if (state.plan.length) {
      const suggested = state.plan[history.length - state.prefix.length];
      const choice = choices.find(c => c.uci === suggested);
      if (choice && choice.utility >= Math.max(...choices.map(c => c.utility)) - cfg.skill_bonus) choice.utility += cfg.skill_bonus;
      else state.plan = [];
    }
    let winning = choices.filter(c => {
      if (!['victory', 'shaped'].includes(this.manifest.config?.objective ?? '')) return false;
      board.move(c.move);
      try { return terminal(board, self, 'victory') === 1; } finally { board.undo(); }
    });
    if (!winning.length && cfg.mate_search_moves > 0) {
      const found = forcedMate(board, self, cfg.mate_search_moves, limit).found;
      if (found) winning = choices.filter(c => {
        board.move(c.move);
        try {
          const ending = terminal(board, self, 'victory');
          return ending === 1 || (ending === null && found > 1 && forcedMate(board, self, found - 1, limit).found > 0);
        } finally { board.undo(); }
      });
      if (winning.length) diagnostics.forced_mate = found;
    }
    const chosen = winning[0];
    return choices.sort((a, b) => Number(b === chosen) - Number(a === chosen) || b.utility - a.utility);
  }
}
