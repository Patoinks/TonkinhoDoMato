import type { Color } from 'chess.js';

/** Parse FEN explicitly: a1=0, black flips ranks only. Preserve even non-capturable EP. */
export function encode(fen: string, self: Color): Float64Array {
  const [placement, turn, rights, ep, clock] = fen.split(' ');
  const x = new Float64Array(782);
  let rank = 7, file = 0;
  for (const c of placement) {
    if (c === '/') { rank--; file = 0; continue; }
    if (/\d/.test(c)) { file += Number(c); continue; }
    const side = c === c.toUpperCase() ? 'w' : 'b';
    const square = rank * 8 + file++;
    x[(side === self ? 0 : 384) + 'pnbrqk'.indexOf(c.toLowerCase()) * 64
      + (self === 'w' ? square : square ^ 56)] = 1;
  }
  x[768] = Number(turn === self);
  const castling = self === 'w' ? 'KQkq' : 'kqKQ';
  [...castling].forEach((c, i) => { x[769 + i] = Number(rights.includes(c)); });
  if (ep !== '-') x[773 + ep.charCodeAt(0) - 97] = 1;
  x[781] = Math.min(Number(clock), 100) / 100;
  return x;
}

export function softmax(logits: number[]): number[] {
  if (!logits.length) return [];
  const max = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

