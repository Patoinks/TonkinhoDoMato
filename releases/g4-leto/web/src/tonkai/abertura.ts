import { type Chess, type Color, type Move } from 'chess.js';
import { LetoPosition, fileOf, other, rankOf, squareIndex } from './letoPosition';

export type OpeningRepertoire = Record<string, string>;
const minor = (type: string) => type === 'n' || type === 'b';
const homeRank = (color: Color) => color === 'w' ? 0 : 7;
const centre = new Set([27,28,35,36]);
const castle = (move: Move) => move.isKingsideCastle() || move.isQueensideCastle();
const capture = (move: Move) => move.isCapture() || move.isEnPassant();

/** FEN uses legal en passant, as abertura._key(version=2) does; clocks are omitted. */
export const repertoireKey = (board: Chess) => board.fen().split(' ').slice(0, 4).join(' ');

export function weightAt(ply: number, plies: number, fade = 'legacy') {
  if (ply >= plies) return 0;
  if (fade === 'linear') return Math.max(0, (plies - ply) / Math.max(1, plies));
  if (fade === 'hold') return 1;
  if (fade !== 'legacy') throw new Error('opening_fade must be legacy, linear or hold');
  const full = Math.floor(plies * 2 / 3);
  return ply < full ? 1 : (plies - ply) / Math.max(1, plies - full);
}

function principlesV1(board: LetoPosition, move: Move, ply: number) {
  const from = squareIndex(move.from), to = squareIndex(move.to), piece = board.cells[from];
  if (!piece) return 0;
  const me = board.turn, kind = piece.type, canCastle = board.canCastle(me);
  if (castle(move)) return 1.5;
  if (kind === 'k') return canCastle ? -2 : -.5;
  if (capture(move)) return 0;
  let units = 0;
  const undeveloped = ['n','b'].reduce((n, t) => n + board.pieces(t as 'n' | 'b', me).filter(s => rankOf(s) === homeRank(me)).length, 0);
  if (minor(kind)) {
    if (rankOf(from) === homeRank(me)) {
      units += .8;
      if (kind === 'n') {
        if ([0,7].includes(fileOf(to))) units -= 1.6;
        else if ([3,4].includes(fileOf(to))) units -= .4;
      }
    } else {
      units -= undeveloped ? .6 : .2;
      if (rankOf(to) === homeRank(me)) units -= .6;
    }
    if (kind === 'b') {
      const front = to + (me === 'w' ? -8 : 8), p = board.cells[front];
      if (front >= 0 && front < 64 && [3,4].includes(fileOf(to)) && p?.type === 'p' && p.color === me) units -= .7;
    }
  } else if (kind === 'q') units -= ply < 16 ? .8 : .3;
  else if (kind === 'r') {
    if (canCastle) units -= 1;
    else if (![2,3,4].includes(fileOf(to))) units -= .4;
  } else if (kind === 'p') {
    const file = fileOf(from), start = rankOf(from) === (me === 'w' ? 1 : 6);
    if ([3,4].includes(file)) units += start ? (Math.abs(to - from) === 16 ? .9 : .4) : .1;
    else if (file === 2) units += start ? .3 : 0;
    else if ([0,7].includes(file)) units -= .6;
    else if (file === 5) units -= .2;
    else if ([1,6].includes(file)) units -= start && Math.abs(to - from) === 8 ? .1 : .4;
  }
  return units;
}

function pawnStructure(board: LetoPosition, color: Color) {
  const counts = new Array<number>(8).fill(0);
  for (const s of board.pieces('p', color)) counts[fileOf(s)]++;
  return [counts.reduce((sum, n) => sum + Math.max(0, n - 1), 0),
    counts.reduce((sum, n, file) => sum + (!counts[file - 1] && !counts[file + 1] ? n : 0), 0)];
}
function connectedRooks(board: LetoPosition, color: Color) {
  const rooks = board.pieces('r', color);
  return rooks.some((a, i) => rooks.slice(i + 1).some(b => (rankOf(a) === rankOf(b) || fileOf(a) === fileOf(b)) && board.attacks(a).includes(b)));
}
type Context = { structure: number[]; connected: boolean; flank: number[] | null };
function context(board: LetoPosition): Context {
  const me = board.turn, king = board.king(me);
  let flank: number[] | null = null;
  if (king !== undefined && rankOf(king) === homeRank(me) && !board.canCastle(me)) {
    flank = fileOf(king) >= 6 ? [5,6,7] : fileOf(king) <= 2 ? [0,1,2] : null;
  }
  return { structure: pawnStructure(board, me), connected: connectedRooks(board, me), flank };
}

function principlesV2(board: LetoPosition, after: LetoPosition, move: Move, ply: number, ctx: Context) {
  const from = squareIndex(move.from), to = squareIndex(move.to), piece = board.cells[from];
  if (!piece) return 0;
  const me = piece.color, kind = piece.type;
  let units = principlesV1(board, move, ply);
  if (castle(move) || kind === 'k') return units;
  const { structure, connected, flank } = ctx, captures = capture(move);
  if (kind === 'p') {
    const [doubled, isolated] = pawnStructure(after, me), scale = captures ? .5 : 1;
    units -= scale * (.55 * (doubled - structure[0]) + .3 * (isolated - structure[1]));
    const fromFile = fileOf(from), toFile = fileOf(to);
    if (flank?.includes(fromFile)) {
      const relative = me === 'w' ? rankOf(to) : 7 - rankOf(to);
      if (toFile !== fromFile) units -= .35;
      else if (relative >= 3) units -= .65;
      else if (relative === 2) units -= .1;
      if (flank[0] === 5 && fromFile === 5) units -= .55;
    }
  }
  if (minor(kind) || kind === 'p') {
    const beforeControl = board.attacks(from).filter(s => centre.has(s)).length, afterControl = after.attacks(to).filter(s => centre.has(s)).length;
    units += .18 * (afterControl - beforeControl);
    if (kind === 'p') units += .25 * (Number(centre.has(to)) - Number(centre.has(from)));
  }
  if (kind !== 'p' && kind !== 'b') {
    const behind = to + (me === 'w' ? -8 : 8), pawn = board.cells[behind], rank = me === 'w' ? rankOf(behind) : 7 - rankOf(behind);
    if (behind >= 0 && behind < 64 && [3,4].includes(fileOf(to)) && pawn?.color === me && pawn.type === 'p' && [1,2].includes(rank)) units -= .7;
  }
  if (kind === 'b' && flank) {
    const home = flank[0] === 5 ? (me === 'w' ? 14 : 54) : (me === 'w' ? 9 : 49);
    if (from === home) {
      const target = board.cells[to];
      if (captures && target && minor(target.type) && board.attackers(other(me), to).length) units -= .65;
      else if (!captures) units -= .25;
    }
  }
  if (!connected && connectedRooks(after, me)) units += .55;
  return units;
}

function undevelopedMinors(board: LetoPosition, color: Color) {
  const offset = color === 'w' ? 0 : 56;
  return [1,6,2,5].map(s => s + offset).filter(s => board.cells[s]?.color === color && board.cells[s]?.type === ([1,6].includes(s - offset) ? 'n' : 'b'));
}
function castleObstacles(board: LetoPosition, color: Color) {
  const home = color === 'w' ? 0 : 56;
  const routes = board.castleRooks(color).map(rook => (rook === home + 7 ? [5,6] : [1,2,3]).filter(f => board.cells[home + f]).length);
  return routes.length ? Math.min(...routes) : null;
}
function bishopActivity(board: LetoPosition, color: Color) {
  const pawnAttacks = new Set(board.pieces('p', other(color)).flatMap(s => board.attacks(s)));
  return board.pieces('b', color).reduce((sum, s) => sum + board.attacks(s).filter(t => board.cells[t]?.color !== color && !pawnAttacks.has(t)).length, 0);
}
function contextV3(board: LetoPosition) {
  const me = board.turn, pawns = board.pieces('p', me);
  return { undeveloped: undevelopedMinors(board, me).length, obstacles: castleObstacles(board, me), activity: bishopActivity(board, me),
    openCentre: !pawns.some(s => fileOf(s) === 3) || !pawns.some(s => fileOf(s) === 4) };
}

function principlesV3(board: LetoPosition, after: LetoPosition, move: Move, ply: number, ctx: Context, v3: ReturnType<typeof contextV3>, scale: number) {
  const units = principlesV2(board, after, move, ply, ctx);
  if (!scale) return units;
  const from = squareIndex(move.from), to = squareIndex(move.to), piece = board.cells[from];
  if (!piece) return units;
  const me = piece.color, kind = piece.type, { undeveloped, obstacles, activity, openCentre } = v3;
  if (castle(move)) return units + scale * (.6 + (ply >= 12 ? .4 : 0) + (openCentre ? .3 : 0));
  if (kind === 'k') return units;
  const captures = capture(move), developing = minor(kind) && undevelopedMinors(board, me).includes(from);
  let extra = developing ? (captures ? .55 : .2) : 0;
  if (minor(kind) || kind === 'p') extra += .08 * Math.max(-4, Math.min(4, bishopActivity(after, me) - activity));
  if (obstacles !== null) {
    const remaining = castleObstacles(after, me);
    if (remaining !== null && remaining < obstacles) extra += .45 * (obstacles - remaining);
  }
  const forced = captures || after.isCheck() || board.isCheck() || board.attackers(other(me), from).length > 0;
  if (!forced) {
    if (minor(kind) && !developing) {
      if (undeveloped) extra -= .25 * Math.min(2, undeveloped);
      else if (['n','b'].some(t => board.pieces(t as 'n' | 'b', me).some(s => rankOf(s) === homeRank(me)))) extra += .4;
      if ([0,7].includes(fileOf(to)) && !after.attacks(to).some(s => after.cells[s]?.color !== me && ['r','q'].includes(after.cells[s]?.type ?? ''))) extra -= .45;
    } else if (kind === 'q') {
      if (undeveloped) extra -= .25 * undeveloped;
      else if (ply >= 16) extra += .3;
    } else if (kind === 'r' && obstacles !== null) extra -= undeveloped ? .45 : .15;
    else if (kind === 'p') {
      const file = fileOf(from), rank = me === 'w' ? rankOf(to) : 7 - rankOf(to);
      if (undeveloped && [0,1,6,7].includes(file) && rank >= 3) extra -= .25 * Math.min(2, undeveloped);
      if (obstacles !== null) {
        if (file === 5) extra -= .65;
        else if (file === 6 && rank >= 3) extra -= .55;
      }
    }
  }
  return units + scale * extra;
}

export function principleScores(board: LetoPosition, moves: Move[], ply: number, version = 2, scale = 1) {
  if (version === 1) return moves.map(move => principlesV1(board, move, ply));
  if (version !== 2 && version !== 3) throw new Error('opening principles version must be 1, 2 or 3');
  const ctx = context(board), v3 = contextV3(board);
  return moves.map(move => version === 2 ? principlesV2(board, board.after(move), move, ply, ctx)
    : principlesV3(board, board.after(move), move, ply, ctx, v3, scale));
}
