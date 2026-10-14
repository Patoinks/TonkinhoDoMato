import { Chess, type Color, type Move, type PieceSymbol, type Square } from 'chess.js';
import { pythonMoves } from './temporal';

type Piece = { type: PieceSymbol; color: Color };
export type PieceValuer = (piece: Piece, square: string) => number;
export type PieceSense = { square: string; color: Color; piece: string; value: number | null; attacked: boolean;
  attackers: number; defenders: number; required_defenders: number; defender_squares: string[];
  exchange_balance: number | null; priority: number; deficit: number; risk: number;
  branches: { move: string; balance: number; defenders: string[] }[] };
const action = (m: Move) => m.from + m.to + (m.promotion ?? '');
const captured = (b: Chess, m: Move) => m.isEnPassant() ? m.to[0] + (Number(m.to[1]) + (b.turn() === 'w' ? -1 : 1)) : m.to;
const captures = (b: Chess, square?: string) => pythonMoves(b).filter(m => (m.isCapture() || m.isEnPassant()) && (!square || m.to === square));
const pieces = (b: Chess) => b.board().flat().filter(p => p !== null);

export function exchangeGain(board: Chess, first: Move, value: PieceValuer, depth: number): number {
  const work = new Chess(board.fen({ forceEnpassantSquare: true }));
  const values = new Map(pieces(work).map(p => [p.square as string, value(p, p.square)]));
  function follow(move: Move, remaining: number): number {
    const target = captured(work, move), mover = work.get(move.from)!;
    let gain = values.get(target) ?? 0;
    const moving = values.get(move.from)!;
    const promoted = move.promotion ? value({ type: move.promotion, color: mover.color }, move.to) : moving;
    if (move.promotion) gain += promoted - moving;
    values.delete(target); values.delete(move.from); values.set(move.to, promoted);
    work.move(move);
    if (remaining <= 1) return gain;
    const replies = captures(work, move.to).sort((a, b) => values.get(a.from)! - values.get(b.from)! || action(a).localeCompare(action(b)));
    return replies.length ? gain - Math.max(0, follow(replies[0], remaining - 1)) : gain;
  }
  return follow(first, depth);
}

export function pieceSenses(board: Chess, value: PieceValuer, tolerance: number, depth: number): PieceSense[] {
  const byColor = new Map<Color, { work: Chess; attacks: Map<string, Move[]> }>();
  for (const color of ['w', 'b'] as const) {
    const fields = board.fen({ forceEnpassantSquare: true }).split(' ');
    fields[1] = color === 'w' ? 'b' : 'w';
    if (fields[1] !== board.turn()) fields[3] = '-';
    const work = new Chess(fields.join(' ')), attacks = new Map<string, Move[]>();
    for (const move of captures(work)) {
      const square = captured(work, move), victim = work.get(square as Square);
      if (victim && victim.color === color && victim.type !== 'k') attacks.set(square, [...(attacks.get(square) ?? []), move]);
    }
    byColor.set(color, { work, attacks });
  }
  return pieces(board).map(piece => {
    const { work, attacks } = byColor.get(piece.color)!;
    const moves = attacks.get(piece.square) ?? [];
    const row: PieceSense = { square: piece.square, color: piece.color,
      piece: piece.color === 'w' ? piece.type.toUpperCase() : piece.type,
      value: piece.type === 'k' ? null : value(piece, piece.square), attacked: false, attackers: 0, defenders: 0,
      required_defenders: 0, defender_squares: [], exchange_balance: null, priority: 0, deficit: 0, risk: 0, branches: [] };
    if (piece.type === 'k') {
      row.attacked = board.isAttacked(piece.square, piece.color === 'w' ? 'b' : 'w');
      row.priority = row.attacked ? 1 : 0; return row;
    }
    if (!moves.length) return row;
    row.attackers = new Set(moves.map(m => m.from)).size;
    for (const move of moves) {
      const gain = exchangeGain(work, move, value, depth);
      work.move(move);
      const defenders = [...new Set(captures(work, move.to).map(m => m.from))].sort();
      work.undo();
      row.branches.push({ move: action(move), balance: -gain, defenders });
    }
    const worst = [...row.branches].sort((a, b) => a.balance - b.balance || a.defenders.length - b.defenders.length)[0];
    const balance = worst.balance;
    Object.assign(row, { attacked: true, exchange_balance: balance, priority: 1,
      defenders: worst.defenders.length, defender_squares: worst.defenders });
    if (balance < -tolerance) row.risk = .5 + .5 * Math.min(1, (-balance - tolerance) / 4);
    else if (Math.abs(balance) <= tolerance) {
      row.priority = row.defenders === 0 ? 2 : 3;
      row.required_defenders = row.attackers; row.deficit = Math.max(0, row.attackers - row.defenders);
      row.risk = !row.defenders ? .4 : .3 * row.deficit / row.attackers;
    }
    return row;
  }).sort((a, b) => (Number(a.square[1]) - Number(b.square[1])) || a.square[0].localeCompare(b.square[0]));
}

export function positionRisk(rows: PieceSense[], self: Color): number {
  const risks = rows.filter(r => r.color === self).map(r => r.risk);
  return Math.min(1, Math.max(0, ...risks) + .05 * risks.reduce((a, b) => a + b, 0));
}
