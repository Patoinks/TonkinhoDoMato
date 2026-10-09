import { Chess, type Color, type Move } from 'chess.js';
import { TemporalNetwork, encodedFen } from './temporal';
import type { Manifest } from './network';
import type { PieceValuer } from './pieceSafety';

export interface G16Manifest extends Manifest {
  model_name: string;
  claudio_config: Record<string, number | string | boolean>;
  options: { finish_pieces: number; finish_moves: number; finish_nodes: number };
  book: Record<string, Record<string, [number, number]>>;
  native: { wasm_sha256: string; loader_sha256: string };
  piece_value_mode?: 'contextual';
}
export interface G16Rules { call(operation: string, args: unknown[]): unknown }
type Plan = { moves: number; move: number; nodes: number; proven: number; aborted: boolean };
type Tensor = { data: Float32Array; shape: number[] };
const key = (fen: string) => fen.split(' ').slice(0, 4).join(' ');
const index = (s: string) => (Number(s[1]) - 1) * 8 + s.charCodeAt(0) - 97;
const codeOf = (m: Move) => index(m.from) | index(m.to) << 6 | (m.promotion ? ' pnbrqk'.indexOf(m.promotion) << 12 : 0);
const uciOf = (code: number) => {
  const square = (n: number) => String.fromCharCode(97 + (n & 7)) + (1 + (n >> 3));
  return square(code & 63) + square(code >> 6 & 63) + ((code >> 12) ? ' pnbrqk'[code >> 12] : '');
};
const avg = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
const softmax32 = (a: number[]) => { const maximum = Math.max(...a), e = a.map(v => Math.fround(Math.exp(v - maximum))), total = e.reduce((s, v) => s + v, 0); return e.map(v => Math.fround(v / total)); };
const order = (a: number[]) => a.map((_, i) => i).sort((i, j) => a[j] - a[i] || j - i);

/** Same pinned C++ searches and G16 v2 decision order. Every budget is the checkpoint's. */
export class G16Network extends TemporalNetwork {
  private rules: G16Rules;
  private settings: Record<string, number | string | boolean>;
  private gManifest: G16Manifest;
  private weights: Map<string, Tensor>;
  private counters = { decisions: 0, plannerHits: 0, policyReorders: 0, bookHits: 0 };
  private rootMaterial: number[] | null = null;
  constructor(manifest: G16Manifest, tensors: Map<string, Tensor>, rules: G16Rules) {
    super(manifest, tensors); this.gManifest = manifest; this.settings = manifest.claudio_config;
    this.weights = tensors; this.rules = rules;
  }
  private n(name: string) { return Number(this.settings[name]); }
  private call<T>(name: string, ...args: unknown[]): T { return this.rules.call(name, args) as T; }
  protected override linear(x: Float64Array, name: string, activation = false) {
    return super.linear(x, name, activation).map(Math.fround);
  }
  private setup(board: Chess) {
    this.rootMaterial = null;
    const history = board.history({ verbose: true });
    const root = history[0]?.before ?? board.fen({ forceEnpassantSquare: true });
    this.call('init', root, history.map(codeOf));
    const keys = [key(root), ...history.map(m => key(encodedFen(new Chess(m.after), m)))];
    let since = 0;
    history.forEach((m, i) => { if (m.isCapture() || m.isEnPassant() || m.piece === 'p') since = i + 1; });
    return { keys, since };
  }
  private phaseValues(board: Chess) {
    if (this.gManifest.piece_value_mode === 'contextual') return [0, ...Array.from({ length: 5 }, (_, i) => this.gManifest.kd_values![String(i + 1)])];
    const pieces = board.board().flat().filter(Boolean).length;
    const fields = board.fen().split(' '), plies = 2 * (Number(fields[5]) - 1) + Number(fields[1] === 'b');
    const rows = String(this.settings.phase_values).split(';').map(row => [0, ...row.split(',').map(Number)]);
    return rows.length === 3 ? rows[plies < this.n('phase_opening') ? 0 : pieces <= this.n('phase_ending') ? 2 : 1]
      : [0, ...Array.from({ length: 5 }, (_, i) => this.gManifest.kd_values![String(i + 1)])];
  }
  override pieceValuer(board: Chess, self: Color, context?: Float64Array): PieceValuer {
    if (this.gManifest.piece_value_mode === 'contextual') return super.pieceValuer(board, self, context);
    const values = this.phaseValues(board);
    return piece => (piece.color === self ? 1 : -1) * (piece.type === 'k' ? 10000 : values[' pnbrqk'.indexOf(piece.type)]);
  }
  terminal(board: Chess, self: Color): number | null {
    this.setup(board);
    const end = this.call<number>('terminal');
    if (end === 0 || this.call<boolean>('dead', board.fen())) return .5;
    return end < 0 ? null : (end === 1) === (self === 'w') ? 1 : 0;
  }
  diagnostics() { return { ...this.counters }; }
  private contextualMaterial(board: Chess, self: Color, context: Float64Array) {
    const value = this.pieceValuer(board, self, context), material = new Array<number>(128).fill(0);
    for (const piece of board.board().flat()) if (piece && piece.type !== 'k') {
      material[index(piece.square)] = (piece.color === self ? 1 : -1) * value(piece, piece.square);
    }
    for (const color of ['b','w'] as const) for (const type of ['n','b','r','q'] as const) for (let file=0;file<8;file++) {
      const location = String.fromCharCode(97+file) + (color === 'w' ? '8' : '1');
      material[64 + Number(color==='w')*32 + (' nbrq'.indexOf(type)-1)*8 + file]
        = (color === self ? 1 : -1) * value({type,color}, location as Move['to']);
    }
    if (!material.every(Number.isFinite)) throw new Error('Non-finite contextual piece values');
    this.rootMaterial = material;
    this.call('context', material);
  }
  private candidatesV2(board: Chess, self: Color) {
    const state = this.observe(board, self), codes = this.call<number[]>('legal');
    const legal = new Map(board.moves({ verbose: true }).map(move => [codeOf(move), move]));
    return codes.map(code => {
      const move = legal.get(code); if (!move) throw new Error('G16 native/browser legal moves disagree');
      board.move(move); const fen = encodedFen(board, move); board.undo();
      const step = this.step(fen, self, state.memory), v = this.value(step.context);
      return { code, move, uci: uciOf(code), san: move.san, fen, ...step, ...v, utility: v.mean,
        logits: [Math.log(v.mean), -Infinity, Math.log(1-v.mean)], wdl: [v.mean, 0, 1-v.mean] };
    });
  }
  override predict(board: Chess, self: Color) {
    this.setup(board);
    const choices = this.candidatesV2(board, self), p = softmax32(choices.map(c => c.predictLogit));
    return choices.map((c, i) => ({ uci: c.uci, san: c.san, p: p[i] })).sort((a, b) => b.p - a.p);
  }
  private endingValue(ending: number, self: Color) {
    return ending === 0 ? .5 : ending === -1 ? null : (ending === 1) === (self === 'w') ? 10 : -10;
  }
  private ends(codes: number[], previous: string[], dead: boolean, self: Color) {
    const [ends, repeats] = this.call<[number[], boolean[]]>('endings', codes, previous, dead);
    return ends.map((end, i) => this.endingValue(end, self) ?? (repeats[i] ? .5 : null));
  }
  private gain(move: Move, values: number[]) {
    if (this.rootMaterial) {
      const victim = move.isEnPassant() ? index(move.to) + (move.color === 'w' ? -8 : 8) : index(move.to);
      const capture = move.captured || move.isEnPassant() ? this.rootMaterial[victim] : 0;
      const promotion = move.promotion ? this.rootMaterial[64+Number(move.color==='w')*32+(' nbrq'.indexOf(move.promotion)-1)*8+(index(move.to)&7)] - this.rootMaterial[index(move.from)] : 0;
      return capture + promotion;
    }
    return (move.isEnPassant() ? values[1] : move.captured ? values[' pnbrqk'.indexOf(move.captured)] : 0)
      + (move.promotion ? values[' pnbrqk'.indexOf(move.promotion)] - values[1] : 0);
  }
  override rank(board: Chess, self: Color) {
    if (board.turn() !== self) throw new Error('G16 must be on turn');
    const history = this.setup(board), white = self === 'w', values = this.phaseValues(board);
    const state = this.observe(board, self), pieces = board.board().flat().filter(Boolean);
    if (this.gManifest.piece_value_mode === 'contextual') this.contextualMaterial(board, self, state.context);
    const previous = this.settings.repeat_as_draw && history.keys.length - history.since >= 4 ? history.keys.slice(history.since) : [];
    const dead = pieces.filter(p => p!.type !== 'p' && p!.type !== 'k').length <= 2;
    const codes = this.call<number[]>('legal');
    if (!codes.length) return [];
    this.counters.decisions++;
    const chooseOnly = (code: number) => {
      const move = board.moves({ verbose: true }).find(m => codeOf(m) === code);
      if (!move) throw new Error('G16 returned an illegal move');
      return [{ code, move, uci: uciOf(code), san: move.san, fen: board.fen(), memory: state.memory,
        context: state.context, mean: 1, deviation: 0, predictLogit: 0, utility: 10, logits: [0,-Infinity,-Infinity], wdl: [1,0,0] }];
    };
    const cfg = this.gManifest.options;
    const enemy = pieces.filter(p => p!.color !== self && p!.type !== 'k');
    const lead = pieces.reduce((s, p) => s + (p!.type === 'k' ? 0 : (this.rootMaterial ? this.rootMaterial[index(p!.square)] : values[' pnbrqk'.indexOf(p!.type)]) * (p!.color === self ? 1 : -1)), 0);
    if (pieces.length <= cfg.finish_pieces && !enemy.length && lead > 0) {
      let plan = this.call<Plan>('plan', this.call<string>('fen'), history.keys.slice(history.since, -1), Math.min(4, cfg.finish_moves), 40000);
      if (!plan.moves && (this.value(state.context).mean >= this.n('ahead_value') || pieces.some(p => p!.color === self && (p!.type === 'q' || p!.type === 'r')))) {
        plan = this.call<Plan>('plan', this.call<string>('fen'), history.keys.slice(history.since, -1), cfg.finish_moves, cfg.finish_nodes);
      }
      if (plan.moves && plan.move) { this.counters.plannerHits++; return chooseOnly(plan.move); }
    }
    // Evidence-gated fast opening: full native safety and mate checks remain mandatory.
    const ply = 2 * (Number(board.fen().split(' ')[5]) - 1) + Number(board.turn() === 'b');
    const seen = this.gManifest.book[key(board.fen())];
    if (seen && ply < this.n('book_plies') && !board.isCheck()) {
      const interval = (n: number, good: number) => {
        const z = 1.96, p = good / n, centre = (p + z*z/(2*n))/(1 + z*z/n), width = z*Math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n);
        return [centre-width, centre+width];
      };
      const supported = Object.entries(seen).filter(([, [n]]) => n >= Math.max(16, this.n('book_min')))
        .sort((a, b) => interval(...b[1])[0] - interval(...a[1])[0]);
      if (supported.length) {
        const [uci, row] = supported[0], low = interval(...row)[0];
        const move = board.moves({ verbose: true }).find(m => m.from + m.to + (m.promotion ?? '') === uci);
        if (move && low >= .5 && supported.slice(1).every(([, row]) => low > interval(...row)[1])
          && !this.call<number[]>('mate', white, this.n('mate_attack'), this.n('mate_nodes'), this.n('pawn_reach'))[0]) {
          const code = codeOf(move), loss = this.call<[number, number[]][]>('deep', [code], values, this.n('tactic_depth'), this.n('exchange_depth'), this.n('tactic_nodes'))[0][0];
          this.call('push', code);
          const unsafe = this.call<number>('strict') !== -1 || history.keys.slice(history.since).includes(key(this.call<string>('fen')))
            || this.call<number[]>('mate', !white, this.n('mate_defence'), this.n('mate_nodes'), this.n('pawn_reach'))[0];
          this.call('pop');
          if (!unsafe && Number.isFinite(loss) && this.gain(move, values) - loss >= -1e-9) return chooseOnly(code);
        }
      }
    }
    if (codes.length === 1) return chooseOnly(codes[0]);
    const endgame = this.n('endgame_pieces') > 0 && pieces.length <= this.n('endgame_pieces');
    let depth = endgame ? Math.max(this.n('tactic_depth'), this.n('endgame_depth')) : this.n('tactic_depth');
    let attack = this.n('mate_attack'), mateNodes = this.n('mate_nodes');
    if (endgame && this.n('endgame_mate') > attack) { attack = this.n('endgame_mate'); mateNodes = Math.max(mateNodes, this.n('endgame_nodes')); }
    else if (attack > 2 && this.n('mate_gate') > 0 && this.kingSafety(board, white ? 'b' : 'w') >= this.n('mate_gate')) attack = 2;
    if (attack > 0) {
      const found = this.call<number[]>('mate', white, attack, mateNodes, this.n('pawn_reach'))[0];
      if (found) for (const code of codes) {
        this.call('push', code); const end = this.call<number>('strict');
        const wins = end === (white ? 1 : 2) || (end === -1 && found > 1 && this.call<number[]>('mate', white, found - 1, mateNodes, this.n('pawn_reach'))[0]);
        this.call('pop'); if (wins) return chooseOnly(code);
      }
    }
    const choices = this.candidatesV2(board, self), plain = choices.map(c => c.mean), scores = [...plain], known = codes.map(() => false);
    if (this.n('king_weight')) choices.forEach((c, i) => {
      board.move(c.move);
      scores[i] += this.n('king_weight') * (this.kingSafety(board, self, c.move) - this.kingSafety(board, white ? 'b' : 'w', c.move)); board.undo();
    });
    const gains = choices.map(c => this.gain(c.move, values));
    let nets: number[] | null = null;
    if (this.n('safety_weight')) {
      try { const losses = this.call<number[]>('exchange', codes, values, this.n('exchange_depth'), this.n('exchange_nodes')); nets = gains.map((g, i) => g - losses[i]); }
      catch { /* Same original fallback on an exhausted exchange budget. */ }
    }
    const weight = this.n('safety_weight');
    if (nets) nets.forEach((net, i) => { scores[i] += weight * Math.tanh(net / 3); });
    const recount = (i: number, loss: number) => { if (nets && !Number.isNaN(loss)) {
      const next = gains[i] - loss; scores[i] += weight * (Math.tanh(next / 3) - Math.tanh(nets[i] / 3)); nets[i] = next;
    } };
    if (nets && this.n('scan_depth') > 0) this.call<[number, number[]][]>('deep', codes, values, this.n('scan_depth'), this.n('exchange_depth'), this.n('tactic_nodes')).forEach(([loss], i) => recount(i, loss));
    if (nets && this.n('simplify_weight')) {
      const share = Math.min(1, Math.max(0, (lead - this.n('simplify_margin')) / 3));
      gains.forEach((gain, i) => { if (gain > 0 && nets![i] >= -1e-9) scores[i] += this.n('simplify_weight') * share; });
    }
    // Distinguish an actual terminal draw from a conservative history-return score.
    const [endings, repeated] = this.call<[number[], boolean[]]>('endings', codes, previous, dead);
    endings.forEach((ending, i) => { const exact = this.endingValue(ending, self);
      if (exact !== null) { scores[i] = exact; known[i] = true; }
      else if (repeated[i] && !choices[i].move.isCapture() && !choices[i].move.isEnPassant() && choices[i].move.piece !== 'p') { scores[i] = .5; known[i] = true; }
      else { known[i] = false; /* no change was made above for a zeroing nonterminal */ }
    });
    if (seen && ply < this.n('book_plies')) {
      this.counters.bookHits++; const rows = Object.values(seen), total = rows.reduce((s, r) => s+r[0], 0), mean = rows.reduce((s,r) => s+r[1],0)/Math.max(1,total);
      choices.forEach((c, i) => { const row = seen[c.uci]; if (row && row[0] >= this.n('book_min') && !known[i]) {
        scores[i] += this.n('book_weight') * ((row[1] + this.n('book_min')*mean)/(row[0] + this.n('book_min')) - mean);
      } });
    }
    const replyTop = order(scores).filter(i => !known[i]).slice(0, this.n('reply_top'));
    if (replyTop.length >= 2) {
      const deeper = replyTop.map(i => {
        this.call('push', codes[i]); const replies = this.call<number[]>('legal'), ends = this.ends(replies, previous, dead, self);
        const fens: string[] = [];
        for (const code of replies) { this.call('push', code); fens.push(this.call<string>('fen')); this.call('pop'); }
        this.call('pop');
        const values = fens.map(fen => this.value(this.step(fen, self, choices[i].memory).context));
        const answers = values.map((v, j) => ends[j] ?? v.mean), chance = softmax32(values.map(v => v.predictLogit));
        return this.settings.reply_mode === 'min' || Math.min(...answers) <= -5 ? Math.min(...answers) : answers.reduce((s, v, j) => s + v*chance[j], 0);
      });
      const shifts = deeper.flatMap((v, k) => v > -5 ? [plain[replyTop[k]] - v] : []), shift = shifts.length ? avg(shifts) : 0;
      replyTop.forEach((i, k) => { scores[i] += this.n('reply_weight') * (deeper[k] + shift - plain[i]); });
    }
    if (this.n('critical_depth') > depth) {
      const ranked = order(scores).filter(i => !known[i]);
      if (ranked.length >= 2 && (scores[ranked[0]] - scores[ranked[1]] < this.n('critical_gap') || (this.n('critical_spread') > 0 && choices[ranked[0]].deviation * Math.sqrt(5/4) > this.n('critical_spread')))) depth = this.n('critical_depth');
    }
    if (nets && depth > 0 && this.n('tactic_top') > 1) {
      let top = order(scores).filter(i => !known[i]).slice(0, this.n('tactic_top'));
      if (top.length) {
        const temperature = this.weights.get('policy.temperature')!.data[0], maximum = Math.max(...top.map(i => scores[i]));
        const ranks = top.map(i => {
          const base = Math.fround(scores[i] - maximum), f = new Float64Array([Math.fround(Math.tanh(gains[i]/3)), Math.fround(Math.tanh(nets![i]/3)), Math.fround(Math.max(-10, Math.min(0,base/temperature)))]);
          const residual = 2*Math.tanh(this.linear(this.linear(choices[i].context, 'policy.rank.0', true), 'policy.rank.2')[0]
            + this.linear(this.linear(f, 'policy.features.0', true), 'policy.features.2')[0]);
          return Math.fround(base/temperature + residual);
        });
        const reordered = ranks.map((_, k) => k).sort((i, j) => ranks[j] - ranks[i] || i-j).map(i => top[i]);
        if (reordered.some((v, i) => v !== top[i])) this.counters.policyReorders++; top = reordered;
        const floor = Math.max(-Infinity, ...scores.filter((_, i) => !top.includes(i))), base = top.map(i => scores[i] - weight*Math.tanh(nets![i]/3));
        const [, best, losses, status] = this.call<[number, number, number[], number[]]>('root', top.map(i => codes[i]), base, top.map(i => gains[i]), weight, floor, values, depth, this.n('exchange_depth'), this.n('tactic_nodes'));
        top.forEach((i, k) => { if (status[k] === 1) { nets![i] = gains[i]-losses[k]; scores[i] = base[k] + weight*Math.tanh(nets![i]/3); known[i] = true; }
          else if (status[k] === 0 || status[k] === 2) scores[i] = Math.min(scores[i], best-1e-9); });
      }
    }
    const checked = new Set<number>(); let rounds = 0, chosen = 0;
    const argmax = () => scores.reduce((best, v, i) => v > scores[best] ? i : best, 0);
    while (true) {
      const best = argmax(); chosen = best;
      if (checked.has(best) || scores[best] >= 10) break;
      checked.add(best); const before = scores[best]; this.call('push', codes[best]);
      const mate = this.call<number>('strict') === -1 && this.n('mate_defence') > 0 && this.call<number[]>('mate', !white, this.n('mate_defence'), this.n('mate_nodes'), this.n('pawn_reach'))[0];
      this.call('pop');
      if (mate) { scores[best] = -20 + .001*mate; if (checked.size >= codes.length) { chosen = argmax(); break; } continue; }
      if (!known[best] && nets && depth > 0 && rounds < this.n('tactic_rounds')) {
        rounds++; recount(best, this.call<[number, number[]][]>('deep', [codes[best]], values, depth, this.n('exchange_depth'), this.n('tactic_nodes'))[0][0]);
      }
      if (scores[best] >= before-1e-12 || argmax() === best) break;
    }
    choices.forEach((c,i) => { c.utility = scores[i]; });
    return choices.sort((a, b) => Number(b.code === codes[chosen]) - Number(a.code === codes[chosen]) || b.utility-a.utility);
  }
}

export async function loadG16Rules(manifest: G16Manifest): Promise<G16Rules> {
  const [loader, wasm] = await Promise.all([fetch('/tonkai/g16-rules.mjs', { cache: 'no-cache' }), fetch('/tonkai/g16-rules.wasm', { cache: 'no-cache' })]);
  if (!loader.ok || !wasm.ok) throw new Error('Não foi possível carregar o motor G16 v2.');
  const source = await loader.arrayBuffer(), binary = await wasm.arrayBuffer();
  const digest = async (b: ArrayBuffer) => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', b))).map(v => v.toString(16).padStart(2,'0')).join('');
  if (await digest(source) !== manifest.native.loader_sha256 || await digest(binary) !== manifest.native.wasm_sha256) throw new Error('G16 v2: checksum do motor inválido.');
  const blob = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
  try {
    const { default: create } = await import(/* @vite-ignore */ blob);
    return await create({ wasmBinary: binary, locateFile: (path: string) => new URL(`/tonkai/${path}`, window.location.href).href });
  } finally { URL.revokeObjectURL(blob); }
}
