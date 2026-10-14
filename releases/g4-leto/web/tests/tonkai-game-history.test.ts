import assert from 'node:assert/strict';
import test from 'node:test';
import { Chess } from 'chess.js';
import { Game } from '../src/tonkai/game';
import type { Network } from '../src/tonkai/network';
import { createGameRecord, GAME_HISTORY_STORAGE_KEY, loadGameHistory, parseSavedGame, pgnDownload, saveGameHistory } from '../src/tonkai/gameHistory';

const moves = (board: Chess) => board.moves({ verbose: true }).map(move => ({
  uci: move.from + move.to + (move.promotion ?? ''), san: move.san, p: 1 / board.moves().length,
}));
const network = { manifest: { sha256: 'history-test', model_name: 'G4 Leto' }, predict: moves, rank: moves, createPlayer() { return { ...this }; } } as unknown as Network;
const startedAt = '2026-10-14T12:00:00.000Z';
function storage() {
  const values = new Map<string, string>();
  return { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => { values.set(key, value); } };
}

test('untimed play remains legal after a very long delay and exports TimeControl "-"', () => {
  const game = new Game(network, 'w');
  assert.equal(game.clock, null);
  assert.equal(game.checkClock(86_400_000), false);
  assert.ok(game.playHuman('e2e4'));
  assert.equal(game.timeoutColor, null);
  const record = createGameRecord(game, 'ongoing', 'Martim', startedAt);
  assert.equal(record.result, '*');
  assert.match(record.pgn, /\[TimeControl "-"\]/);
  const restored = new Chess(); restored.loadPgn(record.pgn, { strict: true });
  assert.equal(restored.fen(), game.board.fen());
  assert.deepEqual(restored.history(), ['e4']);
});

for (const human of ['w', 'b'] as const) {
  test(`resignation as ${human} keeps correct players and winner through browser reload`, () => {
    const store = storage();
    const game = new Game(network, human);
    if (human === 'w') game.playHuman('e2e4'); else game.playTonkAI();
    game.resigned = true;
    const before = game.board.pgn();
    const record = createGameRecord(game, human, 'Martim', startedAt);
    assert.equal(game.board.pgn(), before, 'recording must not mutate the board');
    assert.equal(record.result, human === 'w' ? '0-1' : '1-0');
    assert.equal(record.tonkaiColor, game.self);
    assert.equal(record.opponent, 'Martim');
    assert.match(record.pgn, /\[Termination "resignation"\]/);
    assert.ok(saveGameHistory(record, [], store).persisted);
    assert.deepEqual(loadGameHistory(store), [record]);
  });
}

test('human victory, draw and custom initial FEN all export replayable games', () => {
  const game = new Game(network, 'w', '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1');
  assert.ok(game.playHuman('f7g7'));
  const mate = createGameRecord(game, 'mate', 'Tu', startedAt);
  assert.equal(mate.result, '1-0');
  assert.match(mate.pgn, /\[SetUp "1"\]/);
  assert.equal(mate.fens[0], '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1');
  assert.equal(mate.fens.at(-1), game.board.fen());
  const draw = createGameRecord(new Game(network, 'b', '7k/8/8/8/8/8/8/K7 w - - 0 1'), 'draw', 'Tu', startedAt);
  assert.equal(draw.result, '1/2-1/2');
  assert.equal(draw.plies, 0);
});

test('every move updates one match while preserving previous matches and other tabs', () => {
  const store = storage();
  const game = new Game(network, 'w'); game.playHuman('e2e4');
  const first = createGameRecord(game, 'first', 'Tu', startedAt, startedAt);
  const oldTab = saveGameHistory(first, [], store).games;
  const other = createGameRecord(game, 'other', 'Tu', '2026-10-14T13:00:00.000Z');
  saveGameHistory(other, [], store);
  game.playTonkAI(); game.resigned = true;
  const finished = createGameRecord(game, 'first', 'Tu', startedAt, '2026-10-14T14:00:00.000Z');
  const saved = saveGameHistory(finished, oldTab, store);
  assert.equal(saved.games.length, 2);
  assert.deepEqual(saved.games.map(record => record.id), ['other', 'first']);
  assert.equal(loadGameHistory(store)[1].plies, 2);
  assert.notEqual(loadGameHistory(store)[1].result, '*');
});

test('damaged entries do not discard valid history; quotas retain the downloadable game in memory', () => {
  const store = storage();
  const game = new Game(network, 'w'); game.playHuman('e2e4');
  const record = createGameRecord(game, 'valid', 'Tu', startedAt);
  store.setItem(GAME_HISTORY_STORAGE_KEY, JSON.stringify([null, { ...record, pgn: 'damaged' }, record]));
  assert.deepEqual(loadGameHistory(store), [record]);
  const denied = { getItem: store.getItem, setItem() { throw new Error('QuotaExceededError'); } };
  const saved = saveGameHistory(record, [], denied);
  assert.equal(saved.persisted, false);
  assert.deepEqual(saved.games, [record]);
  assert.deepEqual(loadGameHistory({ getItem() { throw new Error('SecurityError'); } }), []);
});

test('downloaded PGN is the exact saved legal game, with its MIME type', async () => {
  const game = new Game(network, 'w'); game.playHuman('e2e4');
  const record = createGameRecord(game, 'download', 'Martim "The player"', startedAt);
  const file = pgnDownload(record.pgn);
  assert.equal(file.type, 'application/x-chess-pgn;charset=utf-8');
  assert.equal(await file.text(), record.pgn);
  const board = new Chess(); board.loadPgn(await file.text(), { strict: true });
  assert.equal(board.fen(), game.board.fen());
  assert.equal(board.getHeaders().Result, '*');
});

test('contradictory PGN metadata is rejected', () => {
  const game = new Game(network, 'w'); game.playHuman('e2e4');
  const record = createGameRecord(game, 'invalid', 'Tu', startedAt);
  assert.throws(() => parseSavedGame({ ...record, pgn: record.pgn.replace('[Result "*"]', '[Result "1-0"]') }));
});
