import { type Chess, type Color, type Move, type PieceSymbol } from 'chess.js';

export type LetoPiece = { type: PieceSymbol; color: Color };
export const other = (color: Color): Color => color === 'w' ? 'b' : 'w';
export const fileOf = (square: number) => square & 7;
export const rankOf = (square: number) => square >> 3;
export const squareIndex = (square: string) => (Number(square[1]) - 1) * 8 + square.charCodeAt(0) - 97;
export const pieceIndex = (type: PieceSymbol) => ' pnbrqk'.indexOf(type);
export const distance = (a: number, b: number) => Math.max(Math.abs(fileOf(a) - fileOf(b)), Math.abs(rankOf(a) - rankOf(b)));

/** python-chess attack semantics: pinned pieces attack, and rays include the first blocker. */
export class LetoPosition {
  cells: (LetoPiece | null)[];
  turn: Color;
  rights: string;
  constructor(cells: (LetoPiece | null)[], turn: Color, rights: string) {
    this.cells = cells; this.turn = turn; this.rights = rights;
  }
  static from(board: Chess) {
    const cells: (LetoPiece | null)[] = new Array(64).fill(null);
    for (const piece of board.board().flat()) if (piece) cells[squareIndex(piece.square)] = { type: piece.type, color: piece.color };
    return new LetoPosition(cells, board.turn(), board.fen().split(' ')[2]);
  }
  copy() { return new LetoPosition([...this.cells], this.turn, this.rights); }
  pieces(type: PieceSymbol, color: Color) {
    const squares: number[] = [];
    for (let s = 0; s < 64; s++) if (this.cells[s]?.type === type && this.cells[s]?.color === color) squares.push(s);
    return squares;
  }
  king(color: Color) { return this.pieces('k', color).at(-1); }
  attacksFrom(type: PieceSymbol, square: number, color?: Color) {
    const attacks: number[] = [], file = fileOf(square), rank = rankOf(square);
    const add = (df: number, dr: number) => {
      const f = file + df, r = rank + dr;
      if (f >= 0 && f < 8 && r >= 0 && r < 8) attacks.push(r * 8 + f);
    };
    if (type === 'p') { add(-1, color === 'w' ? 1 : -1); add(1, color === 'w' ? 1 : -1); }
    else if (type === 'n') for (const [df, dr] of [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]) add(df, dr);
    else if (type === 'k') {
      for (let df = -1; df <= 1; df++) for (let dr = -1; dr <= 1; dr++) if (df || dr) add(df, dr);
    }
    else {
      const directions = [...(type === 'b' || type === 'q' ? [[-1,-1],[-1,1],[1,-1],[1,1]] : []),
        ...(type === 'r' || type === 'q' ? [[-1,0],[1,0],[0,-1],[0,1]] : [])];
      for (const [df, dr] of directions) {
        for (let f = file + df, r = rank + dr; f >= 0 && f < 8 && r >= 0 && r < 8; f += df, r += dr) {
          const s = r * 8 + f; attacks.push(s); if (this.cells[s]) break;
        }
      }
    }
    return attacks;
  }
  attacks(square: number) {
    const piece = this.cells[square];
    return piece ? this.attacksFrom(piece.type, square, piece.color) : [];
  }
  attackers(color: Color, square: number) {
    const squares: number[] = [];
    for (let s = 0; s < 64; s++) if (this.cells[s]?.color === color && this.attacks(s).includes(square)) squares.push(s);
    return squares;
  }
  isCheck() { const king = this.king(this.turn); return king !== undefined && this.attackers(other(this.turn), king).length > 0; }
  canCastle(color: Color) { return this.castleRooks(color).length > 0; }
  castleRooks(color: Color) {
    const home = color === 'w' ? 0 : 56;
    if (this.king(color) !== home + 4) return [];
    return [[home + 7, color === 'w' ? 'K' : 'k'], [home, color === 'w' ? 'Q' : 'q']]
      .filter(([s, right]) => this.rights.includes(String(right)) && this.cells[Number(s)]?.type === 'r' && this.cells[Number(s)]?.color === color)
      .map(([s]) => Number(s));
  }
  after(move: Move) {
    const next = this.copy(), from = squareIndex(move.from), to = squareIndex(move.to);
    next.cells[from] = null;
    next.cells[to] = { type: move.promotion ?? move.piece, color: move.color };
    if (move.isEnPassant()) next.cells[to + (move.color === 'w' ? -8 : 8)] = null;
    if (move.isKingsideCastle() || move.isQueensideCastle()) {
      const home = move.color === 'w' ? 0 : 56, rook = home + (move.isKingsideCastle() ? 7 : 0);
      next.cells[home + (move.isKingsideCastle() ? 5 : 3)] = next.cells[rook]; next.cells[rook] = null;
    }
    if (move.piece === 'k') next.rights = next.rights.replace(move.color === 'w' ? /[KQ]/g : /[kq]/g, '');
    for (const [s, right] of [[0,'Q'],[7,'K'],[56,'q'],[63,'k']] as const) if (from === s || to === s) next.rights = next.rights.replace(right, '');
    next.turn = other(this.turn);
    return next;
  }
}
