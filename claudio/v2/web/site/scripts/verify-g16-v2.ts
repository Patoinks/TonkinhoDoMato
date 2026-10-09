import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { Chess } from 'chess.js';
import { Network } from '../src/tonkai/network';
import { G16Network } from '../src/tonkai/g16v2';

const folder = 'public/tonkai/';
const manifest = JSON.parse(readFileSync(folder + 'g16-v2.json', 'utf8'));
const { default: create } = await import(pathToFileURL(process.cwd() + '/public/tonkai/g16-rules.mjs').href);
const rules = await create({ wasmBinary: readFileSync(folder + 'g16-rules.wasm') });
const weights = readFileSync(folder + manifest.binary);
assert.equal(createHash('sha256').update(weights).digest('hex'), manifest.sha256);
const net = new Network(manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength), rules);
const reference = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const rows: object[] = []; let different = 0, maxMemoryError = 0;
for (const ref of reference) {
  const board = new Chess(ref.root);
  for (const uci of ref.moves) board.move({ from: uci.slice(0,2), to: uci.slice(2,4), promotion: uci[4] });
  const before = board.fen(), history = board.history();
  const start = performance.now(), ranking = net.rank(board, board.turn()), seconds = (performance.now()-start)/1000;
  assert.equal(board.fen(), before); assert.deepEqual(board.history(),history);
  const chosen = ranking[0].uci;
  const memory = net.temporal!.snapshot(board,board.turn()).memory;
  const memoryError = Math.max(...memory.map((v, i) => Math.abs(v-ref.memory[i])));
  maxMemoryError = Math.max(maxMemoryError,memoryError);
  different += Number(chosen !== ref.move);
  rows.push({ expected:ref.move, chosen, seconds, memoryError, fen:ref.fen, plies:ref.moves.length });
}
const g16 = net.temporal as G16Network;
for (const fen of ['8/8/8/8/8/8/4K3/7k w - - 0 1','8/8/8/8/8/8/4K3/5B1k w - - 0 1','7k/8/8/8/8/8/8/KQ6 w - - 100 60']) {
  const board = new Chess(fen); assert.equal(g16.terminal(board,'w'),.5);
}
const repetition = new Chess();
for(const move of ['Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1']) repetition.move(move);
assert.equal(g16.terminal(repetition,'b'),.5);
const result={ decisions:rows.length,different,maxMemoryError,stats:g16.diagnostics(),rows };
writeFileSync(process.argv[3],JSON.stringify(result,null,2));
console.log(JSON.stringify({decisions:rows.length,different,maxMemoryError,stats:g16.diagnostics()}));
assert.equal(different,0,'Browser moves must agree with frozen Python G16 v2');
assert.ok(maxMemoryError<1e-4,'Browser GRU must preserve the real game history');
