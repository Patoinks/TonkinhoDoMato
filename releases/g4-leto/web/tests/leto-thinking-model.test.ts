import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { Chess } from 'chess.js';
import { Network } from '../src/tonkai/network';
import { G16Network, type LetoSearchLimits } from '../src/tonkai/g16v2';
import { initializePawnProofs } from '../src/tonkai/pawnProofs';

const directory = new URL('../public/tonkai/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('leto.json', directory), 'utf8'));
const loader = (await import(new URL(manifest.native.loader, directory).href)).default;
const rules = await loader({ wasmBinary: readFileSync(new URL(manifest.native.wasm, directory)) });
const bytes = readFileSync(new URL(manifest.binary, directory));
const binary = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const repertoire = JSON.parse(readFileSync(new URL(manifest.opening_repertoire.file, directory), 'utf8'));
initializePawnProofs(readFileSync(new URL('pawn-kpk-v3.bin', directory)));
const create = () => new Network(structuredClone(manifest), binary, rules, repertoire);

test('transferring worker data never detaches or mutates the active G4 model', () => {
  const network = create(), seed = network.workerSeed();
  const transported = structuredClone(seed, { transfer: [seed.binary] });
  assert.equal(seed.binary.byteLength, 0);
  assert.equal(createHash('sha256').update(new Uint8Array(transported.binary)).digest('hex'), manifest.sha256);
  transported.manifest.sha256 = 'modified'; new Uint8Array(transported.binary)[0] ^= 255;
  const next = network.workerSeed();
  assert.equal(next.binary.byteLength, manifest.bytes);
  assert.equal(createHash('sha256').update(new Uint8Array(next.binary)).digest('hex'), manifest.sha256);
  assert.equal(network.manifest.sha256, manifest.sha256);
});

test('provisional search leaves history/settings intact and calibrates the model only once per decision', () => {
  const board = new Chess('r2q1rk1/ppp1bppp/2n5/3p1b2/2PP4/P1P2N2/4BPPP/R1BQ1RK1 w - - 1 13');
  const baseline = create(), timed = create(), settings = structuredClone(timed.manifest), before = board.pgn();
  assert.ok(baseline.temporal instanceof G16Network && timed.temporal instanceof G16Network);
  const expected = baseline.rank(board, 'w')[0].uci, spread = baseline.temporal.searchCalibration();
  const limits: LetoSearchLimits = { depth: 2, nodes: 4000, exchangeNodes: 4000, replyTop: 0,
    tacticTop: 6, rounds: 1, scanDepth: 1, mateNodes: 1000 };
  timed.rank(board, 'w', limits);
  assert.ok(Math.abs(timed.temporal.searchCalibration() - spread) < 1e-12);
  assert.equal(timed.rank(board, 'w')[0].uci, expected);
  assert.ok(Math.abs(timed.temporal.searchCalibration() - spread) < 1e-12);
  assert.deepEqual(timed.manifest, settings); assert.equal(board.pgn(), before);
  timed.temporal.restoreSearchCalibration(spread); assert.equal(timed.temporal.searchCalibration(), spread);
});
