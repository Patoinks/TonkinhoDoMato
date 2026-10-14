import { Chess } from 'chess.js';
import { principleScores, repertoireKey, weightAt, type OpeningRepertoire } from '../src/tonkai/abertura';
import { LetoPosition, squareIndex } from '../src/tonkai/letoPosition';
import { hasRunners, lostOn, pawnBalance, pawnomics } from '../src/tonkai/pawnomics';

export type NumericFixture = {
  fen: string;
  numeric: { root: { has_runners: boolean; white: number; black: number; balance: number };
    candidates: Record<string, { principles: number[]; has_runners: boolean; pawns: number }>;
    book_key: string; book_move: string | null; fade: Record<string, number> };
  pawn_v2?: number;
  lost_on?: Record<string, number>;
  v3_scales?: Record<string, number[]>;
  scale_moves?: string[];
};
export type NumericResult = { section: string; field: string; expected: number | boolean | string | null; actual: number | boolean | string | null; same: boolean; difference: number };

export function checkNumeric(row: NumericFixture, values: number[], repertoire: OpeningRepertoire): NumericResult[] {
  const board = new Chess(row.fen), position = LetoPosition.from(board), me = board.turn();
  const ply = 2 * (Number(row.fen.split(' ')[5]) - 1) + Number(me === 'b');
  const results: NumericResult[] = [];
  const check = (section: string, field: string, expected: NumericResult['expected'], actual: NumericResult['actual']) =>
    results.push({ section, field, expected, actual, same: expected === actual, difference: typeof actual === 'number' && typeof expected === 'number' ? Math.abs(actual - expected) : Number(expected !== actual) });
  check('pawnomics', 'root.runners', row.numeric.root.has_runners, hasRunners(position));
  check('pawnomics', 'root.white', row.numeric.root.white, pawnomics(position, 'w', me === 'w', values));
  check('pawnomics', 'root.black', row.numeric.root.black, pawnomics(position, 'b', me === 'b', values));
  check('pawnomics', 'root.balance', row.numeric.root.balance, pawnBalance(position, me, values));
  const legal = new Map(board.moves({ verbose: true }).map(move => [move.from + move.to + (move.promotion ?? ''), move]));
  const ucis = Object.keys(row.numeric.candidates), moves = ucis.map(uci => {
    const move = legal.get(uci); if (!move) throw new Error(`Numeric fixture has illegal move ${uci}: ${row.fen}`); return move;
  });
  const scores = [1,2,3].map(version => principleScores(position, moves, ply, version));
  ucis.forEach((uci, i) => {
    const expected = row.numeric.candidates[uci], after = position.after(moves[i]), runners = hasRunners(after);
    for (let version = 0; version < 3; version++) check('principles', `${uci}.v${version + 1}`, expected.principles[version], scores[version][i]);
    check('pawnomics', `${uci}.runners`, expected.has_runners, runners);
    check('pawnomics', `${uci}.balance`, expected.pawns, runners ? pawnBalance(after, me, values) : 0);
  });
  check('book', 'key', row.numeric.book_key, repertoireKey(board));
  const theory = repertoire[repertoireKey(board)];
  check('book', 'move', row.numeric.book_move, theory && legal.has(theory) ? theory : null);
  for (const [fade, expected] of Object.entries(row.numeric.fade)) check('fade', fade, expected, weightAt(ply, 30, fade));
  if (row.pawn_v2 !== undefined) check('pawnomics-v2', 'balance', row.pawn_v2, pawnBalance(position, me, values, true));
  for (const [square, expected] of Object.entries(row.lost_on ?? {})) check('see', square, expected, lostOn(position, squareIndex(square), values));
  for (const [scale, expected] of Object.entries(row.v3_scales ?? {})) {
    const selected = row.scale_moves!.map(uci => legal.get(uci)!);
    principleScores(position, selected, ply, 3, Number(scale)).forEach((actual, i) => check('principles', `${row.scale_moves![i]}.scale${scale}`, expected[i], actual));
  }
  return results;
}
