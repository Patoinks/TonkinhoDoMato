import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { Chess } from 'chess.js';
import { checkNumeric, type NumericFixture } from './leto-numeric-helper';
import { principleScores, repertoireKey, weightAt } from '../src/tonkai/abertura';
import { LetoPosition } from '../src/tonkai/letoPosition';

const folder = 'public/tonkai/';
const manifest = JSON.parse(readFileSync(folder + 'leto.json', 'utf8'));
const g4 = JSON.parse(readFileSync(folder + 'g4.json', 'utf8'));
const repertoire = JSON.parse(readFileSync(folder + 'leto-book-v2.json', 'utf8'));
const values = [0, ...[1,2,3,4,5].map(i => manifest.kd_values[i])];
const fixtures: NumericFixture[] = JSON.parse(readFileSync('tests/fixtures/leto-numeric.json', 'utf8'));

test('Leto reuses G4 weights, architecture, values and native rules with the exact v3 settings', () => {
  assert.equal(manifest.model_name, 'G4 Leto');
  for (const field of ['binary','sha256','bytes','tensors','architecture','kd_values','native']) assert.deepEqual(manifest[field], g4[field], field);
  assert.equal(manifest.claudio_config.syzygy, '');
  const expected = { reply_top: 4, mate_gate: .28, pawn_weight: .3, pawn_convert: .5, opening_book: true,
    opening_book_version: 2, opening_weight: .15, opening_plies: 30, opening_principles_version: 3, opening_v3_scale: 1,
    opening_fade: 'hold', pawn_v2: false, pawn_version: 3 };
  for (const [field,value] of Object.entries(expected)) assert.equal(manifest.claudio_config[field], value, field);
  assert.equal(createHash('sha256').update(readFileSync(folder + manifest.binary)).digest('hex'), manifest.sha256);
  assert.equal(createHash('sha256').update(readFileSync(folder + manifest.opening_repertoire.file)).digest('hex'), manifest.opening_repertoire.sha256);
  assert.equal(Object.keys(repertoire).length, 2589);
  assert.equal(manifest.pawn_proofs.file, 'pawn-kpk-v3.bin');
  const proofs = readFileSync(folder + manifest.pawn_proofs.file);
  assert.equal(proofs.length, manifest.pawn_proofs.bytes);
  assert.equal(createHash('sha256').update(proofs).digest('hex'), manifest.pawn_proofs.sha256);
});

test('legacy Python numeric fixtures remain exact for principles v1/v2/v3, Pawnomics, SEE and fade', () => {
  for (const fixture of fixtures) {
    const mismatches = checkNumeric(fixture, values, repertoire).filter(result => !result.same);
    assert.deepEqual(mismatches, [], fixture.fen);
  }
});

test('opening keys omit a non-legal en passant target and retain a legal one', () => {
  const initial = new Chess();
  assert.equal(repertoire[repertoireKey(initial)], 'e2e4');
  initial.move('e4');
  assert.equal(repertoireKey(initial).split(' ')[3], '-');
  const ep = new Chess('8/8/8/1pP5/8/6k1/8/6K1 w - b6 0 40');
  assert.equal(repertoireKey(ep).split(' ')[3], 'b6');
  const pinned = new Chess('k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 20');
  assert.equal(repertoireKey(pinned).split(' ')[3], '-');
});

test('principle scoring leaves the board untouched and scale zero is exactly v2', () => {
  for (const fixture of fixtures) {
    const board = new Chess(fixture.fen), position = LetoPosition.from(board), before = JSON.stringify(position), moves = board.moves({ verbose: true });
    assert.deepEqual(principleScores(position, moves, 20, 3, 0), principleScores(position, moves, 20, 2));
    assert.equal(JSON.stringify(position), before);
    for (const move of moves) {
      board.move(move);
      assert.deepEqual(position.after(move), LetoPosition.from(board), move.lan);
      board.undo();
    }
  }
});

test('all fade modes respect the opening cutoff, including the hold boundary', () => {
  assert.equal(weightAt(19, 30), 1);
  assert.equal(weightAt(20, 30), 1);
  assert.equal(weightAt(29, 30), .1);
  assert.equal(weightAt(29, 30, 'hold'), 1);
  assert.equal(weightAt(29, 30, 'linear'), 1/30);
  for (const mode of ['legacy','linear','hold']) assert.equal(weightAt(30, 30, mode), 0);
  assert.throws(() => weightAt(5, 30, 'unknown'));
});
