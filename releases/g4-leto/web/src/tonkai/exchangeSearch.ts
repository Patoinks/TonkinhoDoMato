import { Chess, type Move } from 'chess.js';

// chess.js's internal legal moves avoid constructing SAN for every search node.
type RawMove = { from: number; to: number; piece: string; captured?: string; flags: number; promotion?: string };
type SearchBoard = { _moves(): RawMove[]; _makeMove(move: RawMove): void; _undoMove(): unknown };
const pieceIndex = (piece?: string) => piece ? ' pnbrqk'.indexOf(piece) : 0;
const square = (name: string) => (8 - Number(name[1])) * 16 + name.charCodeAt(0) - 97;

function validate(values: number[], depth: number, nodeLimit: number) {
  if (values.length !== 6 || !Number.isInteger(depth) || depth < 0 || !Number.isInteger(nodeLimit) || nodeLimit < 0)
    throw new RangeError('exchange search requires six values and non-negative integer limits');
}

function moveGain(move: RawMove, values: number[]) {
  const victim = move.flags & 8 ? 1 : pieceIndex(move.captured);
  const capture = victim >= 1 && victim <= 5 ? values[victim] : 0;
  return capture + (move.promotion ? values[pieceIndex(move.promotion)] - values[1] : 0);
}

function search(board: Chess, raw: SearchBoard, values: number[], depth: number, nodeLimit: number) {
  let nodes = 0;
  function visit(remaining: number, alpha: number, beta: number): number {
    if (++nodes > nodeLimit) throw new RangeError('exchange search exceeded node_limit');
    const check = board.isCheck();
    let moves = check || remaining > 0 ? raw._moves() : [];
    if (check && !moves.length) return -1000;
    if (remaining === 0) return 0;
    if (!check) moves = moves.filter(move => Boolean(move.flags & (2 | 8)));
    // C++ stable_sort: largest victim first, then cheapest attacker.
    moves.sort((a, b) => moveGain(b, values) - moveGain(a, values)
      || values[Math.min(pieceIndex(a.piece), 5)] - values[Math.min(pieceIndex(b.piece), 5)]);

    let best = check ? -Infinity : 0;
    if (best > alpha) alpha = best;
    if (alpha >= beta) return best;
    for (const move of moves) {
      const gain = moveGain(move, values);
      raw._makeMove(move);
      let score: number;
      try { score = gain - visit(remaining - 1, gain - beta, gain - alpha); }
      finally { raw._undoMove(); }
      if (score > best) best = score;
      if (best > alpha) alpha = best;
      if (alpha >= beta) break;
    }
    return best;
  }
  return visit(depth, -Infinity, Infinity);
}

/** Exact material the side to move can force within `depth` half-moves. */
export function exchangeGain(board: Chess, values: number[], depth: number, nodeLimit: number): number {
  validate(values, depth, nodeLimit);
  const work = new Chess(board.fen({ forceEnpassantSquare: true }));
  return search(work, work as unknown as SearchBoard, values, depth, nodeLimit);
}

/** Opponent's exact subsequent exchange gain after each legal move. */
export function exchangeAfterMoves(board: Chess, moves: Move[], values: number[], depth: number, nodeLimit: number): number[] {
  validate(values, depth, nodeLimit);
  const work = new Chess(board.fen({ forceEnpassantSquare: true }));
  const raw = work as unknown as SearchBoard;
  const legal = raw._moves();
  return moves.map(move => {
    const from = square(move.from), to = square(move.to);
    const found = legal.find(candidate => candidate.from === from && candidate.to === to
      && candidate.promotion === move.promotion);
    if (!found) throw new RangeError(`illegal move in exchangeAfterMoves: ${move.from}${move.to}${move.promotion ?? ''}`);
    raw._makeMove(found);
    try { return search(work, raw, values, depth, nodeLimit); }
    finally { raw._undoMove(); }
  });
}
