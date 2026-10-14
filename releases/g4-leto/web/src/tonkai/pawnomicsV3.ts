import { type Chess, type Color } from 'chess.js';
import { PawnBoard, pawnRank, sideOf } from './pawnBoard';
import { PROMOTION_DECAY } from './pawnomicsConstants';
import { probePawnProof } from './pawnProofs';

export const PAWNOMICS_VERSION = 3;
const cache = new Map<string, PawnAssessment>();
export type PawnAssessment = { version: number; balance: number; white: number; black: number; exact: boolean;
  raceNodes: number; raceDepth: number; raceComplete: boolean; routeNodes: number };
type Counters = { routeNodes: number };
export function clearPawnCache() { cache.clear(); }

function provenValue(position: PawnBoard, values: number[]) {
  const proof = probePawnProof(position);
  return proof ? proof.winning ? proof.owner * (values[5] - values[1]) * PROMOTION_DECAY[Math.min(Math.floor(proof.resetDistance / 2), 31)] : 0 : null;
}

/** Legal exchanges: pinned attackers cannot leave their king exposed. */
export function legalPawnLoss(board: PawnBoard, square: number, values: number[]) {
  const target = board.cells[square];
  if (!target || Math.abs(target) === 6) return 0;
  const memo = new Map<string, number>();
  let remaining = 128;
  const capture = (position: PawnBoard, side: number, targetSquare = square): number => {
    const piece = position.cells[targetSquare];
    if (!piece || remaining-- <= 0) return 0;
    const key = position.key() + ':' + side + ':' + targetSquare;
    const previous = memo.get(key); if (previous !== undefined) return previous;
    let best = 0;
    const moves = position.capturesTo(targetSquare, side);
    if (Math.abs(piece) === 1 && position.turn === side && position.ep - side * 8 === targetSquare) {
      moves.push(...position.moves(side).filter(move => move.enPassant && move.to - side * 8 === targetSquare));
    }
    for (const move of moves) {
      const next = position.make(move, side), promotion = move.promotion ? values[move.promotion] - values[1] : 0;
      const gain = values[Math.abs(piece)] + promotion - capture(next, -side, move.to);
      best = Math.max(best, gain);
    }
    memo.set(key, best); return best;
  };
  return capture(board, -sideOf(target));
}

/** Shortest king route on the actual board, avoiding occupied and attacked squares. */
function kingCanCatch(before: PawnBoard, after: PawnBoard, pawn: number, enemy: number, preparation: number) {
  const king = before.king(enemy);
  if (king < 0 || Math.max(Math.abs((king & 7) - (pawn & 7)), Math.abs(pawnRank(king) - pawnRank(pawn))) > preparation + 1) return false;
  const queue: { board: PawnBoard; king: number; depth: number }[] = [{ board: before, king, depth: 0 }];
  const seen = new Set<number>([king]);
  for (let i = 0; i < queue.length; i++) {
    const state = queue[i];
    if (state.king === pawn) return true; // The defender occupies the arrival square first.
    const arrival = after.copy();
    arrival.cells[king] = 0; arrival.cells[state.king] = enemy * 6;
    if (arrival.movesFrom(state.king, enemy, true).some(move => move.to === pawn)) return true;
    if (state.depth >= preparation) continue;
    for (const move of state.board.movesFrom(state.king, enemy)) {
      if (move.to === pawn) return true;
      // Capturing another piece changes the route's occupancy; do not merge that state.
      if (!state.board.cells[move.to] && !seen.has(move.to)) {
        seen.add(move.to); queue.push({ board: state.board.make(move, enemy), king: move.to, depth: state.depth + 1 });
      }
    }
  }
  return false;
}

function canBlock(board: PawnBoard, square: number, enemy: number, values: number[]) {
  if (board.cells[square]) return false;
  for (let from = 0; from < 64; from++) {
    const type = Math.abs(board.cells[from]);
    if (board.cells[from] * enemy <= 0 || type === 1 || type === 6 || !board.attacks(from, square)) continue;
    const next = board.make({ from, to: square }, enemy);
    if (!next.checked(enemy) && legalPawnLoss(next, square, values) <= 0) return true;
  }
  return false;
}

/** A defender can use an earlier tempo to prepare a profitable capture on arrival. */
function canPrepareCapture(board: PawnBoard, square: number, enemy: number, values: number[]) {
  for (let from = 0; from < 64; from++) {
    const type = Math.abs(board.cells[from]);
    if (board.cells[from] * enemy <= 0 || type === 1 || type === 6) continue;
    for (const move of board.movesFrom(from, enemy)) {
      if (move.to === square) continue; // Immediate captures were already checked by legal SEE.
      const next = board.make(move, enemy);
      if (next.attacks(move.to, square) && legalPawnLoss(next, square, values) > 1e-9) return true;
    }
  }
  return false;
}

/** Follow legal pushes AND captures, updating blockers, x-rays and en passant. */
function routeValue(root: PawnBoard, start: number, values: number[], counters: Counters, budget = 32, interception = true) {
  const side = sideOf(root.cells[start]), first = root.turn === side;
  if (!first && legalPawnLoss(root, start, values) > 1e-9) return 0;
  const distance = side === 1 ? 7 - pawnRank(start) : pawnRank(start);
  if (!distance) return 0;
  const queenGain = Math.max(0, values[5] - values[1]);
  // Unsearched future counterplay grows smoothly with the defender's actual material.
  // There is no phase cutoff or fixed rank-value table; immediate safe promotions are certain.
  let defenderMaterial = 0;
  for (const piece of root.cells) if (piece * side < 0 && Math.abs(piece) !== 1 && Math.abs(piece) !== 6) defenderMaterial += values[Math.abs(piece)];
  const conversionConfidence = queenGain / Math.max(queenGain + defenderMaterial, 1e-9);
  let remaining = budget;
  const walk = (board: PawnBoard, square: number, elapsed: number): number => {
    if (remaining-- <= 0) return 0;
    counters.routeNodes++;
    let best = 0;
    for (const move of board.movesFrom(square, side)) {
      if (move.promotion && move.promotion !== 5) continue;
      // A future interception must itself be a legal, sustainable move.
      if (interception && elapsed > 0 && !board.cells[move.to] && canBlock(board, move.to, -side, values)) continue;
      const next = board.make(move, side), count = elapsed + 1;
      if (legalPawnLoss(next, move.to, values) > 1e-9) continue;
      const before = count - Number(first);
      if (interception && before > 0 && canPrepareCapture(next, move.to, -side, values)) continue;
      // Approach before arrival, then capture after it. A future queen cannot attack retroactively.
      if (interception && kingCanCatch(board, next, move.to, -side, before)) continue;
      if (move.promotion) {
        const confidence = first && count === 1 ? 1 : conversionConfidence;
        best = Math.max(best, confidence * queenGain * PROMOTION_DECAY[Math.min(count - 1, 31)]);
      } else {
        const advance = Math.abs(pawnRank(move.to) - pawnRank(start)), fraction = advance / distance;
        const left = side === 1 ? 7 - pawnRank(move.to) : pawnRank(move.to);
        // Progress is measured against the actual promotion distance, not a rank-value table.
        // Partial progress is still a pawn; reserve the queen premium for a complete safe route.
        const blockedFile = Array.from(next.cells).some((piece, target) => piece === -side
          && (target & 7) === (move.to & 7) && (pawnRank(target) - pawnRank(move.to)) * side > 0);
        const progress = blockedFile ? 0 : values[1] * fraction * fraction * PROMOTION_DECAY[Math.min(count + left - 1, 31)];
        best = Math.max(best, progress, walk(next, move.to, count));
      }
    }
    return best;
  };
  return walk(root, start, 0);
}

function projection(board: PawnBoard, values: number[], counters: Counters, routeBudget = 32, interception = true) {
  let white = 0, black = 0;
  for (let square = 0; square < 64; square++) if (Math.abs(board.cells[square]) === 1) {
    const value = routeValue(board, square, values, counters, routeBudget, interception);
    if (board.cells[square] > 0) white += value; else black += value;
  }
  return { white, black, balance: white - black };
}

/** Joint king/pawn search: one defender cannot answer both promotion threats at once. */
function race(board: PawnBoard, values: number[], counters: Counters) {
  let remaining = 384, nodes = 0, complete = true;
  const visited = new Map<string, number>();
  const discount = (ply: number) => PROMOTION_DECAY[Math.min(Math.floor(ply / 2), 31)];
  const leaf = (position: PawnBoard, accrued = 0, ply = 0) => accrued + discount(ply) * projection(position, values, counters, 8, true).balance;
  const search = (position: PawnBoard, depth: number, alpha: number, beta: number, accrued = 0, ply = 0): number => {
    if (remaining-- <= 0) { complete = false; return leaf(position, accrued, ply); }
    nodes++;
    const materialPieces = Array.from(position.cells).filter(piece => piece && Math.abs(piece) !== 6);
    if (!materialPieces.length || materialPieces.length === 1 && [2, 3].includes(Math.abs(materialPieces[0]))) return 0;
    const exact = provenValue(position, values);
    if (exact !== null) return exact * discount(ply);
    if (position.halfmove >= 100) return 0;
    const key = position.key(), times = visited.get(key) ?? 0;
    if (times >= 2) return 0;
    const moves = position.moves();
    if (!moves.length) return position.checked(position.turn) ? -position.turn * values[5] : 0;
    if (!depth) return leaf(position, accrued, ply);
    visited.set(key, times + 1);
    moves.sort((a, b) => Number(!!b.promotion) - Number(!!a.promotion)
      || Number(!!position.cells[b.to] || b.enPassant) - Number(!!position.cells[a.to] || a.enPassant)
      || Number(Math.abs(position.cells[b.from]) === 1) - Number(Math.abs(position.cells[a.from]) === 1));
    let best = position.turn === 1 ? -Infinity : Infinity;
    for (const move of moves) {
      const next = position.make(move);
      // A promotion is credited when it happens, so losing a tempo is not a free tie.
      const gain = (next.material(values) - position.material(values)) * discount(ply);
      const value = search(next, depth - 1, alpha, beta, accrued + gain, ply + 1);
      if (position.turn === 1) { best = Math.max(best, value); alpha = Math.max(alpha, best); }
      else { best = Math.min(best, value); beta = Math.min(beta, best); }
      if (alpha >= beta) break;
      if (remaining <= 0) { complete = false; break; }
    }
    if (times) visited.set(key, times); else visited.delete(key);
    return best;
  };
  let value = leaf(board), completedDepth = 0;
  // Never use the biased prefix of a search interrupted by its node budget.
  for (const depth of [2, 4, 6]) {
    complete = true;
    const next = search(board, depth, -Infinity, Infinity);
    if (!complete) break;
    value = next; completedDepth = depth;
    if (remaining <= 0) break;
  }
  return { balance: value, nodes, depth: completedDepth, complete: completedDepth === 6 };
}

export function assessPawns(board: Chess, values: number[], options: { race?: boolean } = {}): PawnAssessment {
  const key = board.fen() + ':' + values.join(',') + ':' + (options.race !== false);
  const cached = cache.get(key); if (cached) return cached;
  const position = PawnBoard.from(board), counters = { routeNodes: 0 };
  const proven = provenValue(position, values);
  if (proven !== null) {
    const result = { version: PAWNOMICS_VERSION, balance: proven, white: Math.max(0, proven), black: Math.max(0, -proven),
      exact: true, raceNodes: 0, raceDepth: 0, raceComplete: true, routeNodes: 0 };
    if (cache.size >= 512) cache.delete(cache.keys().next().value!);
    cache.set(key, result); return result;
  }
  const estimate = projection(position, values, counters);
  const pieces = Array.from(position.cells).filter(Boolean);
  const purePawns = pieces.every(piece => Math.abs(piece) === 1 || Math.abs(piece) === 6);
  const close = pieces.length <= 6 && Array.from(position.cells).some((piece, square) => Math.abs(piece) === 1
    && (piece > 0 ? pawnRank(square) : 7 - pawnRank(square)) >= 3);
  const joint = options.race !== false && purePawns && close ? race(position, values, counters) : null;
  const result = { version: PAWNOMICS_VERSION, ...estimate, exact: false, balance: joint?.balance ?? estimate.balance,
    raceNodes: joint?.nodes ?? 0, raceDepth: joint?.depth ?? 0, raceComplete: joint?.complete ?? true, routeNodes: counters.routeNodes };
  if (cache.size >= 512) cache.delete(cache.keys().next().value!);
  cache.set(key, result); return result;
}

export function pawnBalanceV3(board: Chess, me: Color, values: number[]) {
  return assessPawns(board, values).balance * (me === 'w' ? 1 : -1);
}
