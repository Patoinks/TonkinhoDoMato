import { type Color } from 'chess.js';
import { LetoPosition, distance, fileOf, other, pieceIndex, rankOf } from './letoPosition';
import { PROMOTION_DECAY } from './pawnomicsConstants';

const KING_VALUE = 1000;
const RANK_WORTH = [0, 0, 0, .1, .25, .5, .9, 1.6];
const worth = (board: LetoPosition, square: number, values: number[]) => {
  const type = board.cells[square]!.type;
  return type === 'k' ? KING_VALUE : values[pieceIndex(type)];
};

/** Static exchange on a square, including newly uncovered rays and ascending-square ties. */
export function lostOn(board: LetoPosition, square: number, values: number[]) {
  let side = other(board.cells[square]!.color), standing = worth(board, square, values);
  const next = board.copy(), gains: number[] = [];
  for (;;) {
    let attacker: number | undefined, value = Infinity;
    for (const s of next.attackers(side, square)) {
      const candidate = worth(next, s, values);
      if (attacker === undefined || candidate < value) { attacker = s; value = candidate; }
    }
    if (attacker === undefined || (value === KING_VALUE && next.attackers(other(side), square).length)) break;
    gains.push(standing - (gains.at(-1) ?? 0)); standing = value;
    next.cells[square] = next.cells[attacker]; next.cells[attacker] = null; side = other(side);
  }
  for (let i = gains.length - 1; i > 0; i--) gains[i - 1] = -Math.max(-gains[i - 1], gains[i]);
  return gains.length ? Math.max(0, gains[0]) : 0;
}

function twoMoveReach(board: LetoPosition, side: Color) {
  const reach = new Set<number>();
  for (const type of ['n','b','r','q'] as const) for (const s of board.pieces(type, side)) {
    const first = board.attacks(s).filter(t => board.cells[t]?.color !== side);
    const next = board.copy(); next.cells[s] = null;
    for (const t of first) {
      reach.add(t);
      for (const u of next.attacksFrom(type, t)) if (board.cells[u]?.color !== side) reach.add(u);
    }
  }
  return reach;
}

function structure(board: LetoPosition, square: number, color: Color) {
  const file = fileOf(square), rank = rankOf(square), front = (s: number) => color === 'w' ? rankOf(s) > rank : rankOf(s) < rank;
  if (board.pieces('p', other(color)).some(s => Math.abs(fileOf(s) - file) <= 1 && front(s))) return 1;
  let factor = 1;
  if (board.attackers(color, square).some(s => board.cells[s]!.type === 'p')) factor *= 1.3;
  if (board.pieces('p', color).some(s => Math.abs(fileOf(s) - file) === 1 && Math.abs(rankOf(s) - rank) <= 1)) factor *= 1.2;
  const behind = (s: number) => fileOf(s) === file && s !== square && !front(s);
  if (board.attackers(color, square).some(s => behind(s) && board.cells[s]!.type === 'r')) factor *= 1.2;
  if (board.attackers(other(color), square).some(s => behind(s) && board.cells[s]!.type === 'r')) factor *= .8;
  return factor;
}

function pawn(board: LetoPosition, square: number, color: Color, first: boolean, values: number[], enemyMaterial: number, reach2?: Set<number>) {
  const rank = rankOf(square), step = color === 'w' ? 8 : -8;
  const ahead = Array.from({ length: color === 'w' ? 7 - rank : rank }, (_, i) => square + step * (i + 1));
  if (ahead.some(s => board.cells[s]?.type === 'p' && board.cells[s]?.color !== color)) return null;
  const relative = color === 'w' ? rank + 1 : 8 - rank, free = board.copy(), king = board.king(other(color));
  free.cells[square] = null;
  let tempo = 0, reached = relative;
  for (const [i, s] of ahead.entries()) {
    let moves = i + 1 - Number(relative === 2 && i >= 1) + tempo, before = first ? moves - 1 : moves;
    const occupant = board.cells[s];
    if (occupant) {
      if (occupant.color === color) {
        if (occupant.type === 'p') break;
        tempo++; moves++; before++;
      } else if (occupant.type === 'k') break;
      else {
        const chased = lostOn(board, s, values) > 0 || board.attackers(color, s).some(t => worth(board, t, values) < worth(board, s, values));
        if (!chased) break;
        tempo += 2; moves += 2; before += 2;
      }
    }
    if (board.attackers(other(color), s).some(t => board.cells[t]!.type === 'p')) break;
    const there = free.copy(); there.cells[s] = { type: 'p', color };
    if (lostOn(there, s, values) > 0) break;
    const defended = there.attackers(color, s).length > 0;
    if (king !== undefined && !defended && distance(king, s) - 1 <= before) break;
    if (!defended && before >= 1 && !occupant && board.attackers(other(color), s).some(t => !['p','k'].includes(board.cells[t]!.type))) break;
    if (reach2 && !defended && before >= 2 && !occupant && reach2.has(s)) break;
    reached = relative + i + 1;
    if (reached === 8) {
      const catchFactor = Math.max(.2, 1 - enemyMaterial / 15);
      // A path has at most seven squares and two extra tempi per blocker: exponent <= 20.
      // V8 pow differs from Python by one ulp for some exponents; use the generated constants.
      const value = (values[5] - values[1]) * catchFactor * PROMOTION_DECAY[moves - 1];
      return value * (reach2 ? structure(board, square, color) : 1);
    }
  }
  const value = RANK_WORTH[Math.min(reached, 7)];
  // Iteration and arithmetic follow pawnomics.py; only v2 applies structure bonuses.
  return value * (reach2 && value ? structure(board, square, color) : 1);
}

export function pawnomics(board: LetoPosition, color: Color, first: boolean, values: number[], v2 = false) {
  const reach2 = v2 ? twoMoveReach(board, other(color)) : undefined;
  let enemyMaterial = 0, total = 0;
  for (const type of ['n','b','r','q'] as const) enemyMaterial += values[pieceIndex(type)] * board.pieces(type, other(color)).length;
  for (const square of board.pieces('p', color)) {
    const value = pawn(board, square, color, first, values, enemyMaterial, reach2);
    if (value) total += value;
  }
  return total;
}

export function pawnBalance(board: LetoPosition, me: Color, values: number[], v2 = false) {
  return pawnomics(board, me, board.turn === me, values, v2) - pawnomics(board, other(me), board.turn !== me, values, v2);
}

export function hasRunners(board: LetoPosition) {
  for (const color of ['w','b'] as const) for (const square of board.pieces('p', color)) {
    if (!board.pieces('p', other(color)).some(s => fileOf(s) === fileOf(square) && (color === 'w' ? s > square : s < square))) return true;
  }
  return false;
}
