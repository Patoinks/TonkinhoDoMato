import assert from 'node:assert/strict';
import test from 'node:test';
import { Chess } from 'chess.js';
import { Game } from '../src/tonkai/game';
import type { Network } from '../src/tonkai/network';
import { LetoThinker, type LetoWorkerPort } from '../src/tonkai/letoThinking';
import { searchLetoUntil } from '../src/tonkai/letoTimedSearch';
import { MAX_LETO_THINK_MS, type LetoWorkerReply, type LetoWorkerRequest } from '../src/tonkai/letoThinkingProtocol';

class FakeWorker implements LetoWorkerPort {
  onmessage: LetoWorkerPort['onmessage'] = null;
  onerror: LetoWorkerPort['onerror'] = null;
  requests: LetoWorkerRequest[] = [];
  terminated = 0;
  postMessage(request: LetoWorkerRequest) { this.requests.push(request); }
  terminate() { this.terminated++; }
  emit(data: LetoWorkerReply) { this.onmessage?.({ data }); }
  get request() { return this.requests.findLast(request => request.type === 'think')!; }
}
class FakeClock {
  time = 0; serial = 0;
  callbacks = new Map<number, { at: number; callback: () => void }>();
  now = () => this.time;
  set = (callback: () => void, ms: number) => {
    const id = ++this.serial; this.callbacks.set(id, { at: this.time + ms, callback });
    return id as unknown as ReturnType<typeof setTimeout>;
  };
  clear = (id: ReturnType<typeof setTimeout>) => { this.callbacks.delete(id as unknown as number); };
  advance(ms: number) {
    const end = this.time + ms;
    for (;;) {
      const next = [...this.callbacks].filter(([, item]) => item.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
      if (!next) break;
      this.time = next[1].at; this.callbacks.delete(next[0]); next[1].callback();
    }
    this.time = end;
  }
}
const moves = (board: Chess) => board.moves({ verbose: true }).map(move => ({ uci: move.from + move.to + (move.promotion ?? ''), san: move.san, p: 1 }));
const network = { manifest: { sha256: 'fixed-g4', revision: 'same-g4', claudio_config: { tactic_depth: 6 } },
  workerSeed: () => ({ manifest: { sha256: 'fixed-g4' }, binary: new ArrayBuffer(4) }),
  predict: moves, rank: moves, createPlayer() { return { ...this }; } } as unknown as Network;
const opened = () => { const board = new Chess(); board.move('e4'); return board; };
const flush = async () => { await Promise.resolve(); await Promise.resolve(); };
const best = { uci: 'g8f6', score: .6, depth: 6, iterations: 2 };

test('the 5 s watchdog stops an unresponsive worker and retains the last completed G4 result', async () => {
  const port = new FakeWorker(), clock = new FakeClock(), thinker = new LetoThinker(() => port, clock);
  const board = opened(), original = board.pgn();
  const promise = thinker.think(network, board, 'b', 'game', new AbortController().signal, 20000);
  port.emit({ type: 'ready' }); await flush();
  assert.equal(port.request.type, 'think');
  assert.equal(port.request.remainingMs, MAX_LETO_THINK_MS - 50);
  port.emit({ type: 'best', id: port.request.id, fen: board.fen(), best, proved: false });
  clock.advance(4949); assert.equal(port.terminated, 0);
  clock.advance(1);
  const result = await promise;
  assert.equal(result.uci, best.uci); assert.equal(result.depth, 6); assert.equal(result.iterations, 2);
  assert.equal(result.elapsedMs, 4950); assert.equal(result.reason, 'deadline'); assert.equal(result.fallback, false);
  assert.equal(port.terminated, 1); assert.equal(board.pgn(), original);
  port.emit({ type: 'done', id: port.request.id, fen: board.fen(), best: { ...best, uci: 'a7a6' }, proved: false });
  assert.equal(result.uci, best.uci);
});

test('the deadline includes worker startup and returns a legal reserve if no pass finishes', async () => {
  const port = new FakeWorker(), clock = new FakeClock(), thinker = new LetoThinker(() => port, clock), board = opened();
  const promise = thinker.think(network, board, 'b', 'game', new AbortController().signal);
  clock.advance(5000); const result = await promise;
  assert.equal(result.fallback, true); assert.equal(result.elapsedMs, 4950);
  assert.ok(moves(board).some(move => move.uci === result.uci));
  assert.equal(port.terminated, 1);
});

test('proven answers return early, while aborting navigation ignores late replies', async () => {
  const port = new FakeWorker(), clock = new FakeClock(), thinker = new LetoThinker(() => port, clock), board = opened();
  const first = thinker.think(network, board, 'b', 'game', new AbortController().signal);
  port.emit({ type: 'ready' }); await flush(); clock.advance(40);
  port.emit({ type: 'best', id: port.request.id, fen: board.fen(), best, proved: true });
  assert.equal((await first).elapsedMs, 40); assert.equal(port.terminated, 0);
  const controller = new AbortController(), second = thinker.think(network, board, 'b', 'game', controller.signal);
  await flush(); const rejected = assert.rejects(second, { name: 'AbortError' });
  controller.abort(); await rejected; assert.equal(port.terminated, 1);
  port.emit({ type: 'done', id: port.request.id, fen: board.fen(), best, proved: true });
  assert.equal(board.history().length, 1);
});

test('illegal and stale worker results cannot be applied', async () => {
  const port = new FakeWorker(), clock = new FakeClock(), thinker = new LetoThinker(() => port, clock), board = opened();
  const promise = thinker.think(network, board, 'b', 'game', new AbortController().signal);
  port.emit({ type: 'ready' }); await flush();
  port.emit({ type: 'best', id: port.request.id - 1, fen: board.fen(), best, proved: true });
  port.emit({ type: 'best', id: port.request.id, fen: new Chess().fen(), best, proved: true });
  const rejected = assert.rejects(promise, /illegal move/);
  port.emit({ type: 'best', id: port.request.id, fen: board.fen(), best: { ...best, uci: 'e8e4' }, proved: true });
  await rejected; assert.equal(port.terminated, 1);
});

test('the worker receives retained move history, en passant, castling and an isolated model copy', async () => {
  const port = new FakeWorker(), clock = new FakeClock(), thinker = new LetoThinker(() => port, clock);
  const board = new Chess('r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 7 20'), root = board.fen();
  board.move('exd6'); board.move('O-O');
  const controller = new AbortController(), promise = thinker.think(network, board, 'w', 'branch', controller.signal);
  port.emit({ type: 'ready' }); await flush();
  assert.equal(port.request.root, root); assert.deepEqual(port.request.moves, ['e5d6', 'e8g8']);
  assert.equal(port.request.gameId, 'branch'); assert.equal(port.request.fen, board.fen());
  const rejected = assert.rejects(promise, { name: 'AbortError' }); controller.abort(); await rejected;
});

test('iterative search deepens actual work and discards a pass that crosses its deadline', () => {
  let elapsed = 0; const depths: number[] = [], published: number[] = [];
  const fake = { manifest: network.manifest, rank(_board: Chess, _self: string, limits: { depth: number }) {
    depths.push(limits.depth); elapsed += depths.length === 1 ? 20 : 200;
    return [{ uci: 'g8f6', utility: .6 }, { uci: 'a7a6', utility: .5 }];
  } } as unknown as Network;
  const result = searchLetoUntil(fake, opened(), 'b', 300, best => published.push(best.depth), () => elapsed);
  assert.deepEqual(depths, [2, 6, 8]); assert.deepEqual(published, [2, 6]);
  assert.equal(result.best?.depth, 6);
});

test('a delayed reply cannot move a resigned game or a different position, and legal promotion commits', () => {
  const game = new Game(network, 'w'); game.playHuman('e2e4'); const fen = game.board.fen();
  assert.equal(game.playTonkAIChoice('g8f6', new Chess().fen()), false);
  assert.throws(() => game.playTonkAIChoice('e8e4', fen), /illegal/);
  assert.equal(game.playTonkAIChoice('g8f6', fen), true);
  assert.equal(game.board.history().length, 2); assert.ok(game.humanTurn);
  game.playHuman('d2d4'); game.resigned = true;
  assert.equal(game.playTonkAIChoice('d7d5'), false);
  const promotion = new Game(network, 'b', '7k/P7/8/8/8/8/8/K7 w - - 0 1');
  assert.ok(promotion.playTonkAIChoice('a7a8q'));
  assert.equal(promotion.board.get('a8')?.type, 'q');
});

test('calibration survives a hard worker restart and stays isolated between training branches', async () => {
  const ports: FakeWorker[] = [], clock = new FakeClock();
  const thinker = new LetoThinker(() => { const port = new FakeWorker(); ports.push(port); return port; }, clock);
  const board = opened(), first = thinker.think(network, board, 'b', 'original', new AbortController().signal);
  ports[0].emit({ type: 'ready' }); await flush();
  ports[0].emit({ type: 'best', id: ports[0].request.id, fen: board.fen(), best: { ...best, spread: .013 }, proved: false });
  clock.advance(4950); assert.equal((await first).spread, .013);
  const secondAbort = new AbortController(), second = thinker.think(network, board, 'b', 'original', secondAbort.signal);
  ports[1].emit({ type: 'ready' }); await flush();
  assert.equal(ports[1].request.spread, .013);
  const secondRejected = assert.rejects(second, { name: 'AbortError' }); secondAbort.abort(); await secondRejected;
  const branchAbort = new AbortController(), branch = thinker.think(network, board, 'b', 'training', branchAbort.signal);
  ports[2].emit({ type: 'ready' }); await flush();
  assert.equal(ports[2].request.spread, undefined);
  const branchRejected = assert.rejects(branch, { name: 'AbortError' }); branchAbort.abort(); await branchRejected;
});
