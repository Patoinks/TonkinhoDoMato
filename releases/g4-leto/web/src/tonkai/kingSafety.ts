import { Chess, type Color, type PieceSymbol } from 'chess.js';

type Piece = { type: PieceSymbol; color: Color };
type Position = (Piece | null)[];
export type KingSense = { square: string; risk: number; shelter: number; force: number; contacts: number;
  open_rays: number; uncovered: number; in_check: boolean; attackers: string[]; defenders: string[]; escapes: string[] };
const directions = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
const name = (sq: number) => String.fromCharCode(97 + sq % 8) + (1 + Math.floor(sq / 8));
const other = (colour: Color): Color => colour === 'w' ? 'b' : 'w';
const valid = (file: number, rank: number) => file >= 0 && file < 8 && rank >= 0 && rank < 8;

// Attack maps include the first occupied square, including friendly pieces.
function attacks(position: Position, sq: number): number[] {
  const piece = position[sq]!, file = sq % 8, rank = Math.floor(sq / 8), result: number[] = [];
  if (piece.type === 'p' || piece.type === 'n' || piece.type === 'k') {
    const steps = piece.type === 'p' ? [[-1, piece.color === 'w' ? 1 : -1], [1, piece.color === 'w' ? 1 : -1]]
      : piece.type === 'n' ? [[1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]] : directions;
    for (const [dx, dy] of steps) if (valid(file + dx, rank + dy)) result.push((rank + dy) * 8 + file + dx);
  } else {
    const rays = piece.type === 'b' ? directions.slice(4) : piece.type === 'r' ? directions.slice(0, 4) : directions;
    for (const [dx, dy] of rays) for (let distance = 1; distance < 8; distance++) {
      const f = file + dx * distance, r = rank + dy * distance;
      if (!valid(f, r)) break;
      const target = r * 8 + f; result.push(target);
      if (position[target]) break;
    }
  }
  return result;
}

export function kingSenses(board: Chess): Record<Color, KingSense> {
  const position: Position = Array.from({ length: 64 }, () => null);
  for (const piece of board.board().flat()) if (piece) {
    position[(Number(piece.square[1]) - 1) * 8 + piece.square.charCodeAt(0) - 97] = piece;
  }
  const kings = { w: position.findIndex(p => p?.type === 'k' && p.color === 'w'),
    b: position.findIndex(p => p?.type === 'k' && p.color === 'b') };
  if (kings.w < 0 || kings.b < 0) throw new Error('King safety requires both kings');
  const raw = position.map((p, sq) => p ? attacks(position, sq) : []);
  const effective = raw.map(a => [...a]);
  // Detect absolute pins by scanning outwards from each king.
  for (const colour of ['w', 'b'] as const) {
    const king = kings[colour], file = king % 8, rank = Math.floor(king / 8);
    for (const [dx, dy] of directions) {
      let candidate = -1;
      for (let distance = 1; distance < 8; distance++) {
        const f = file + dx * distance, r = rank + dy * distance;
        if (!valid(f, r)) break;
        const sq = r * 8 + f, piece = position[sq];
        if (!piece) continue;
        if (candidate < 0 && piece.color === colour) { candidate = sq; continue; }
        if (candidate >= 0 && piece.color !== colour && (piece.type === 'q' || piece.type === (dx && dy ? 'b' : 'r'))) {
          effective[candidate] = effective[candidate].filter(target =>
            (target % 8 - file) * dy === (Math.floor(target / 8) - rank) * dx);
        }
        break;
      }
    }
  }
  const result = {} as Record<Color, KingSense>;
  for (const colour of ['w', 'b'] as const) {
    const enemy = other(colour), king = kings[colour], ring = raw[king], zone = new Set([...ring, king]);
    const attackers: number[] = [], defenders: number[] = [], hostile = new Set<number>(), friendly = new Set<number>();
    let contacts = 0;
    position.forEach((piece, sq) => {
      if (!piece || piece.type === 'k') return;
      const covered = effective[sq].filter(target => zone.has(target));
      if (!covered.length) return;
      if (piece.color === enemy) { attackers.push(sq); contacts += covered.length; covered.forEach(s => hostile.add(s)); }
      else { defenders.push(sq); covered.forEach(s => friendly.add(s)); }
    });
    const threatened = ring.filter(sq => hostile.has(sq));
    const uncovered = threatened.filter(sq => !friendly.has(sq)).length / Math.max(1, threatened.length);
    const file = king % 8, rank = Math.floor(king / 8), files = [file - 1, file, file + 1].filter(f => f >= 0 && f < 8);
    let shelter = 0, openRays = 0;
    for (const f of files) for (const [distance, credit] of [[1, 1], [2, .5]]) {
      const r = rank + (colour === 'w' ? distance : -distance);
      const piece = valid(f, r) ? position[r * 8 + f] : null;
      if (piece?.type === 'p' && piece.color === colour) { shelter += credit; break; }
    }
    shelter /= files.length;
    for (const [dx, dy] of directions) {
      let empty = 0;
      for (let distance = 1; distance < 4; distance++) {
        const f = file + dx * distance, r = rank + dy * distance;
        if (!valid(f, r) || position[r * 8 + f]) break;
        empty++;
      }
      if (empty >= 2) openRays++;
    }
    const escapedPosition = [...position]; escapedPosition[king] = null;
    const enemyAttacks = new Set<number>();
    escapedPosition.forEach((piece, sq) => {
      if (piece?.color === enemy) attacks(escapedPosition, sq).forEach(target => enemyAttacks.add(target));
    });
    const escapes = ring.filter(sq => position[sq]?.color !== colour && sq !== kings[enemy] && !enemyAttacks.has(sq));
    const force = Math.min(1, position.reduce((sum, p) => sum + (p?.color === enemy
      ? ({ q: 4, r: 2, b: 1, n: 1, p: 0, k: 0 }[p.type]) : 0), 0) / 8);
    const pressure = Math.min(1, contacts / 12), coordination = Math.min(1, Math.max(0, attackers.length - 1) / 3);
    const gate = Math.min(1, attackers.length / 2) * (.35 + .65 * force);
    const checked = position.some((p, sq) => p?.color === enemy && raw[sq].includes(king));
    const risk = Math.min(1, gate * (.25 * pressure + .15 * coordination + .18 * (1 - shelter) * force
      + .12 * openRays / 8 + .18 * uncovered + .12 * (1 - Math.min(1, escapes.length / 3))) + .15 * Number(checked));
    result[colour] = { square: name(king), risk, shelter, force, contacts, open_rays: openRays, uncovered, in_check: checked,
      attackers: attackers.map(name).sort(), defenders: defenders.map(name).sort(), escapes: escapes.map(name).sort() };
  }
  return result;
}

export function kingBalance(senses: Record<Color, KingSense>, self: Color) {
  return senses[other(self)].risk - senses[self].risk;
}
