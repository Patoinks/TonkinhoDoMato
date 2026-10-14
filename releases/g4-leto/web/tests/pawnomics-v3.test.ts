import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { Chess } from 'chess.js';
import { PawnBoard } from '../src/tonkai/pawnBoard';
import { assessPawns, legalPawnLoss } from '../src/tonkai/pawnomicsV3';
import { LetoPosition, squareIndex } from '../src/tonkai/letoPosition';
import { lostOn, pawnomics } from '../src/tonkai/pawnomics';
import { initializePawnProofs, loadPawnProofs, probePawnProof, provenPawnMoves } from '../src/tonkai/pawnProofs';

const manifest = JSON.parse(readFileSync('public/tonkai/leto.json', 'utf8'));
const values = [0, ...[1, 2, 3, 4, 5].map(type => manifest.kd_values[type])];
const uci = (square: number) => String.fromCharCode(97 + (square & 7)) + (1 + (square >> 3));
const fixtures = JSON.parse(readFileSync('tests/fixtures/leto-numeric.json', 'utf8'));
initializePawnProofs(readFileSync('public/tonkai/pawn-kpk-v3.bin'));

test('the lightweight legal board agrees with chess.js on real positions, promotions and en passant', () => {
  const selected = fixtures.filter((_: unknown, index: number) => index % Math.max(1, Math.floor(fixtures.length / 24)) === 0).map((row: { fen: string }) => row.fen);
  selected.push('7k/8/8/8/1p6/8/2P5/K7 w - - 0 1', '7k/8/8/r4pPK/8/8/8/8 w - f6 0 1',
    '7k/P7/8/8/8/8/8/K7 w - - 0 1', '4r2k/8/8/2p5/3P4/8/4N3/4K3 b - - 0 1');
  for (const fen of selected) {
    const board = new Chess(fen), before = board.fen(), position = PawnBoard.from(board);
    const expected = board.moves({ verbose: true }).filter(move => !move.isKingsideCastle() && !move.isQueensideCastle());
    const actual = position.moves();
    assert.deepEqual(actual.map(move => uci(move.from) + uci(move.to) + (move.promotion ? ' pnbrqk'[move.promotion] : '')).sort(),
      expected.map(move => move.from + move.to + (move.promotion ?? '')).sort(), fen);
    for (const move of actual) {
      board.move({ from: uci(move.from), to: uci(move.to), promotion: move.promotion ? ' pnbrqk'[move.promotion] : undefined });
      assert.deepEqual(position.make(move).cells, PawnBoard.from(board).cells, fen);
      board.undo();
    }
    assert.equal(board.fen(), before);
  }
});

test('a double push can skip a square attacked by a pinned pawn whose en passant is illegal', () => {
  const board = new Chess('1k6/8/8/8/1p6/8/2P5/KR6 w - - 0 1');
  assert.ok(board.moves().includes('c4'));
  assert.equal(pawnomics(LetoPosition.from(board), 'w', true, values), 0);
  assert.ok(assessPawns(board, values, { race: false }).white > 0);
  board.move('c4'); assert.ok(!board.moves().includes('bxc3'));
});

test('a pawn capture can open a promotion route through a blocked structure', () => {
  const board = new Chess('7k/8/3pp3/4P3/8/8/8/K7 w - - 0 1');
  assert.equal(pawnomics(LetoPosition.from(board), 'w', true, values), 0);
  assert.ok(assessPawns(board, values, { race: false }).white > 0);
});

test('an own blocker must actually move; it cannot disappear for an invented tempo', () => {
  const board = new Chess('7k/8/8/8/8/N7/P7/K7 w - - 0 1');
  assert.ok(pawnomics(LetoPosition.from(board), 'w', true, values) > 0);
  assert.equal(assessPawns(board, values, { race: false }).white, 0);
});

test('a pinned attacker cannot capture a pawn by exposing its king', () => {
  const board = new Chess('4k3/4n3/8/5P2/8/8/8/K3R3 w - - 0 1'), square = squareIndex('f5');
  assert.equal(lostOn(LetoPosition.from(board), square, values), values[1]);
  assert.equal(legalPawnLoss(PawnBoard.from(board), square, values), 0);
});

test('a pinned defender cannot recapture and make a lost pawn look safe', () => {
  const board = new Chess('4r2k/8/8/2p5/3P4/8/4N3/4K3 b - - 0 1'), square = squareIndex('d4');
  assert.equal(lostOn(LetoPosition.from(board), square, values), 0);
  assert.equal(legalPawnLoss(PawnBoard.from(board), square, values), values[1]);
});

test('all scoring and joint searches preserve the original board and history', () => {
  const board = new Chess('7k/8/2p5/4P3/8/8/8/K7 w - - 0 1'), before = board.pgn();
  const result = assessPawns(board, values);
  assert.ok(result.raceNodes > 0 && result.raceNodes <= 384);
  assert.ok(result.raceDepth >= 2);
  assert.ok(Number.isFinite(result.balance));
  assert.equal(board.pgn(), before);
  assert.deepEqual(assessPawns(board, values), result);
});

test('colour symmetry and the actual side to move are respected', () => {
  const white = new Chess('7k/8/3pp3/4P3/8/8/8/K7 w - - 0 1');
  const black = new Chess('k7/8/8/8/4p3/3PP3/8/7K b - - 0 1');
  assert.equal(assessPawns(white, values, { race: false }).balance, -assessPawns(black, values, { race: false }).balance);
});

test('a king can catch a pawn after its arrival and before its next push', () => {
  const losingTempo = new Chess('8/4p3/1K6/8/8/8/3k4/8 w - - 1 2');
  const supportingRoute = new Chess('8/4p3/1K6/8/8/3k4/8/8 w - - 1 2');
  assert.ok(assessPawns(supportingRoute, values, { race: false }).black > assessPawns(losingTempo, values, { race: false }).black);
});

test('single-pawn outcomes are exact and respect symmetry, opposition and the 50-move boundary', () => {
  const win = new Chess('4k3/8/4K3/4P3/8/8/8/8 w - - 0 1');
  assert.equal(probePawnProof(PawnBoard.from(win))?.winning, true);
  assert.equal(assessPawns(win, values).exact, true);
  const draw = new Chess('k7/P7/8/8/8/8/8/7K w - - 0 1');
  assert.equal(probePawnProof(PawnBoard.from(draw))?.winning, false);
  assert.equal(assessPawns(draw, values).balance, 0);
  const closeToClaim = new Chess('4k3/8/4K3/4P3/8/8/8/8 w - - 99 1');
  assert.equal(probePawnProof(PawnBoard.from(closeToClaim)), null);
  assert.throws(() => initializePawnProofs(new Uint8Array(4)));
});

test('all legal KPK children agree with an independent oracle and preserve the best outcome', () => {
  const oracle = JSON.parse(readFileSync('tests/fixtures/pawn-endgames-oracle.json', 'utf8'));
  for (const row of oracle.positions) {
    const board = new Chess(row.fen), original = board.pgn(), choices = provenPawnMoves(board);
    assert.ok(choices?.length, row.fen);
    assert.equal(choices.length, board.moves().length);
    for (const choice of choices) assert.equal(choice.wdl, row.moves[choice.uci], row.fen + ' ' + choice.uci);
    assert.equal(choices[0].wdl, row.best, row.fen);
    assert.equal(board.pgn(), original);
  }
  assert.equal(provenPawnMoves(new Chess()), null);
});

test('proof loading validates its descriptor and the actual bytes before activation', async () => {
  await assert.rejects(loadPawnProofs({ file: '../other.bin', bytes: 196608, sha256: '0'.repeat(64) }), /descriptor/);
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response(new Uint8Array(196608));
    await assert.rejects(loadPawnProofs({ file: 'pawn-kpk-v3.bin', bytes: 196608, sha256: '0'.repeat(64) }), /Checksum/);
  } finally { globalThis.fetch = originalFetch; }
});

test('advancing into an enemy pawn blockade does not invent a promotion route', () => {
  const board = new Chess('7k/4p3/8/4P3/8/8/8/K7 w - - 0 1');
  assert.ok(board.moves().includes('e6'));
  const value = assessPawns(board, values, { race: false }).white;
  assert.equal(value, 0);
});

test('a knight can prepare an interception before the pawn promotes', () => {
  const board = new Chess('n7/8/8/4P3/8/8/8/K6k w - - 0 1');
  assert.ok(assessPawns(board, values, { race: false }).white < values[1]);
  for (const move of ['e6', 'Nc7', 'e7', 'Kg1', 'e8=Q', 'Nxe8']) assert.ok(board.move(move));
  assert.ok(board.isInsufficientMaterial());
});

test('the defending king may approach before promotion creates new attacked squares', () => {
  const board = new Chess('3N4/8/5p2/1k5n/8/6p1/8/7K w - - 0 1');
  assert.ok(assessPawns(board, values, { race: false }).black < values[1]);
  for (const move of ['Kg2', 'f5', 'Kf3', 'f4', 'Nc6']) assert.ok(board.move(move));
  assert.ok(!board.moves().includes('f3'));
});

test('a pawn cannot escape a profitable capture when its opponent moves first', () => {
  const onTurn = new Chess('7k/3n4/8/4P3/8/8/8/K7 w - - 0 1');
  const waiting = new Chess('7k/3n4/8/4P3/8/8/8/K7 b - - 0 1');
  assert.ok(assessPawns(onTurn, values, { race: false }).white > 0);
  assert.equal(assessPawns(waiting, values, { race: false }).white, 0);
  assert.ok(waiting.move('Nxe5'));
});

test('legal SEE handles en passant on a different capture square and rejects a pinned capture', () => {
  const board = new Chess('7k/8/8/8/3pP3/8/8/K7 b - e3 0 1');
  assert.equal(legalPawnLoss(PawnBoard.from(board), squareIndex('e4'), values), values[1]);
  assert.ok(board.move('dxe3'));
  const pinned = new Chess('7k/8/8/r4pPK/8/8/8/8 w - f6 0 1');
  assert.equal(legalPawnLoss(PawnBoard.from(pinned), squareIndex('f5'), values), 0);
  const exposedDoublePush = new Chess('7k/8/8/8/1p6/8/2P5/K7 w - - 0 1');
  assert.equal(assessPawns(exposedDoublePush, values, { race: false }).white, 0);
  exposedDoublePush.move('c4'); assert.ok(exposedDoublePush.moves().includes('bxc3'));
});
