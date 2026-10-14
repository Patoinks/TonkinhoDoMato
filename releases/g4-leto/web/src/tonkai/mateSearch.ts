import { Chess, type Color } from 'chess.js';

// Isolated adapter to chess.js's move generator: public verbose moves compute
// SAN/FEN for every child, which is unnecessary in this rules-only search.
type RawMove = { from: number; to: number; piece: string; flags: number; promotion?: string };
type SearchBoard = { _moves(): RawMove[]; _makeMove(m: RawMove): void; _undoMove(): unknown };
const square = (s: number) => (7 - (s >> 4)) * 8 + (s & 7);
function moves(board: SearchBoard) {
  const key = (m: RawMove) => {
    if (m.flags & 96) return [1, 0, square(m.to), 0];
    if (m.flags & 8) return [3, square(m.from), square(m.to), 0];
    if (m.piece !== 'p') return [0, square(m.from), square(m.to), 0];
    return [2, square(m.from), m.flags & 2 ? 0 : m.flags & 4 ? 2 : 1,
      square(m.to) * 4 + (m.promotion ? 'qrbn'.indexOf(m.promotion) : 0)];
  };
  // fast.cpp Pos::pseudo scans low squares first, unlike pythonMoves.
  return board._moves().sort((a, b) => {
    const x = key(a), y = key(b);
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return x[i] - y[i];
    return 0;
  });
}

/** Faithful MateSearch port; draw claims are checked by the policy, not here. */
export function forcedMate(board: Chess, attacker: Color, maxMoves: number, nodeLimit: number) {
  const work = new Chess(board.fen({ forceEnpassantSquare: true }));
  const raw = work as unknown as SearchBoard;
  let nodes = 0, aborted = false, proven = 0;
  const visit = () => { if (++nodes > nodeLimit) aborted = true; return !aborted; };
  function child(move: RawMove, run: () => boolean) {
    raw._makeMove(move);
    try { return run(); } finally { raw._undoMove(); }
  }
  function attack(n: number): boolean {
    if (!visit()) return false;
    const checking: RawMove[] = [], quiet: RawMove[] = [];
    for (const move of moves(raw)) {
      (child(move, () => work.isCheck()) ? checking : quiet).push(move);
    }
    for (const move of checking) {
      if (child(move, () => raw._moves().length === 0 || (n > 1 && defend(n - 1)))) return true;
      if (aborted) return false;
    }
    if (n > 1) for (const move of quiet) {
      if (child(move, () => defend(n - 1))) return true;
      if (aborted) return false;
    }
    return false;
  }
  function defend(n: number): boolean {
    if (!visit()) return false;
    const replies = moves(raw);
    if (!replies.length) return false;
    for (const reply of replies) if (!child(reply, () => attack(n)) || aborted) return false;
    return true;
  }
  for (let n = 1; n <= maxMoves; n++) {
    const mate = work.turn() === attacker ? attack(n) : defend(n);
    if (aborted) break;
    if (mate) return { found: n, proven, nodes };
    proven = n;
  }
  return { found: 0, proven, nodes };
}

export function rankLast(choices: { uci: string; utility: number }[], allows: Record<string, number>) {
  const others = choices.filter(c => !(c.uci in allows) && c.utility > -1e8);
  const floor = others.length ? Math.min(...others.map(c => c.utility)) - 1 : 0;
  for (const c of choices) if (c.uci in allows && c.utility > -1e8) c.utility = floor + .001 * allows[c.uci];
}
