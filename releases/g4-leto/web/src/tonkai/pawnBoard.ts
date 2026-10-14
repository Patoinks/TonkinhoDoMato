import { type Chess, type Color } from 'chess.js';

export type PawnMove = { from: number; to: number; promotion?: number; enPassant?: boolean };
export const sideOf = (piece: number) => piece > 0 ? 1 : -1;
export const pawnFile = (square: number) => square & 7;
export const pawnRank = (square: number) => square >> 3;
const types = ' pnbrqk';

/** Small legal board used by Pawnomics; kings remain part of every legality check. */
export class PawnBoard {
  cells: Int8Array;
  turn: number;
  ep: number;
  halfmove: number;
  constructor(cells: Int8Array, turn: number, ep = -1, halfmove = 0) {
    this.cells = cells; this.turn = turn; this.ep = ep; this.halfmove = halfmove;
  }
  static from(board: Chess) {
    return PawnBoard.fromFen(board.fen());
  }
  static fromFen(fen: string) {
    const cells = new Int8Array(64);
    const fields = fen.split(' ');
    for (const [row, text] of fields[0].split('/').entries()) {
      let file = 0;
      for (const character of text) {
        if (/\d/.test(character)) file += Number(character);
        else { cells[(7 - row) * 8 + file++] = types.indexOf(character.toLowerCase()) * (character === character.toUpperCase() ? 1 : -1); }
      }
    }
    const ep = fields[3];
    return new PawnBoard(cells, fields[1] === 'w' ? 1 : -1,
      ep === '-' ? -1 : (Number(ep[1]) - 1) * 8 + ep.charCodeAt(0) - 97, Number(fields[4]));
  }
  copy() { return new PawnBoard(this.cells.slice(), this.turn, this.ep, this.halfmove); }
  king(side: number) { return this.cells.indexOf(side * 6); }
  key() { return Array.from(this.cells).join(',') + ':' + this.turn + ':' + this.ep; }
  attacks(from: number, to: number) {
    const piece = this.cells[from], type = Math.abs(piece);
    if (!piece || from === to) return false;
    const df = pawnFile(to) - pawnFile(from), dr = pawnRank(to) - pawnRank(from), af = Math.abs(df), ar = Math.abs(dr);
    if (type === 1) return af === 1 && dr === sideOf(piece);
    if (type === 2) return af * ar === 2;
    if (type === 6) return Math.max(af, ar) === 1;
    if (!(type === 3 && af === ar || type === 4 && (!df || !dr) || type === 5 && (af === ar || !df || !dr))) return false;
    const step = Math.sign(dr) * 8 + Math.sign(df);
    for (let square = from + step; square !== to; square += step) if (this.cells[square]) return false;
    return true;
  }
  attacked(square: number, side: number) {
    for (let from = 0; from < 64; from++) if (this.cells[from] * side > 0 && this.attacks(from, square)) return true;
    return false;
  }
  checked(side: number) { const king = this.king(side); return king >= 0 && this.attacked(king, -side); }
  make(move: PawnMove, side = this.turn) {
    const next = this.copy(), piece = next.cells[move.from], captured = next.cells[move.to];
    next.cells[move.from] = 0;
    if (move.enPassant) next.cells[move.to - side * 8] = 0;
    next.cells[move.to] = move.promotion ? side * move.promotion : piece;
    next.turn = -side;
    next.ep = Math.abs(piece) === 1 && Math.abs(move.to - move.from) === 16 ? (move.to + move.from) / 2 : -1;
    next.halfmove = Math.abs(piece) === 1 || captured || move.enPassant ? 0 : this.halfmove + 1;
    return next;
  }
  private candidates(from: number, capturesOnly: boolean, side: number): PawnMove[] {
    const piece = this.cells[from], type = Math.abs(piece), moves: PawnMove[] = [];
    const add = (to: number, enPassant = false) => {
      if (this.cells[to] * side > 0 || Math.abs(this.cells[to]) === 6) return;
      if (type === 1 && (pawnRank(to) === 0 || pawnRank(to) === 7)) {
        for (const promotion of [5, 4, 3, 2]) moves.push({ from, to, promotion, enPassant });
      } else moves.push({ from, to, enPassant });
    };
    if (type === 1) {
      const to = from + side * 8;
      if (!capturesOnly && to >= 0 && to < 64 && !this.cells[to]) {
        add(to);
        const twice = to + side * 8;
        if (pawnRank(from) === (side === 1 ? 1 : 6) && !this.cells[twice]) add(twice);
      }
      for (const df of [-1, 1]) {
        const target = from + side * 8 + df;
        if (target < 0 || target >= 64 || Math.abs(pawnFile(target) - pawnFile(from)) !== 1) continue;
        if (this.cells[target] * side < 0) add(target);
        else if (side === this.turn && target === this.ep && this.cells[target - side * 8] === -side) add(target, true);
      }
    } else {
      for (let to = 0; to < 64; to++) if ((!capturesOnly || this.cells[to] * side < 0) && this.attacks(from, to)) add(to);
    }
    return moves;
  }
  movesFrom(from: number, side = this.turn, capturesOnly = false) {
    if (this.cells[from] * side <= 0) return [];
    return this.candidates(from, capturesOnly, side).filter(move => !this.make(move, side).checked(side));
  }
  moves(side = this.turn) {
    const moves: PawnMove[] = [];
    for (let from = 0; from < 64; from++) if (this.cells[from] * side > 0) moves.push(...this.movesFrom(from, side));
    return moves;
  }
  capturesTo(square: number, side: number) {
    if (!this.cells[square] || this.cells[square] * side > 0 || Math.abs(this.cells[square]) === 6) return [];
    const moves: PawnMove[] = [];
    for (let from = 0; from < 64; from++) if (this.cells[from] * side > 0 && this.attacks(from, square)) {
      if (Math.abs(this.cells[from]) === 1 && (pawnRank(square) === 0 || pawnRank(square) === 7)) {
        for (const promotion of [5, 4, 3, 2]) moves.push({ from, to: square, promotion });
      } else moves.push({ from, to: square });
    }
    return moves.filter(move => !this.make(move, side).checked(side));
  }
  material(values: number[]) {
    let result = 0;
    for (const piece of this.cells) if (piece && Math.abs(piece) !== 6) result += sideOf(piece) * values[Math.abs(piece)];
    return result;
  }
  color(side: number): Color { return side === 1 ? 'w' : 'b'; }
}
