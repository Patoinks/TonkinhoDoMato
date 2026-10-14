import assert from 'node:assert/strict';
import test from 'node:test';
import { Chess } from 'chess.js';
import { Game } from '../src/tonkai/game';
import type { Network } from '../src/tonkai/network';
import { createGameRecord, loadGameHistory, saveGameHistory } from '../src/tonkai/gameHistory';

const moves = (board: Chess) => board.moves({ verbose: true }).map(move => ({
  uci: move.from + move.to + (move.promotion ?? ''), san: move.san, p: 1 / board.moves().length,
}));
const network = { manifest: { sha256: 'training-test', model_name: 'G4 Leto' }, predict: moves, rank: moves, createPlayer() { return { ...this }; } } as unknown as Network;
const opened = '2026-10-14T20:00:00.000Z';

test('each position reconstructs the exact legal board and correct player turn', () => {
  const board = new Chess();
  const fens = [board.fen()];
  for (const move of ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6']) { board.move(move); fens.push(board.fen()); }
  const original = board.pgn();
  for (const human of ['w', 'b'] as const) {
    for (let ply = 0; ply < fens.length; ply++) {
      const game = Game.fromPgn(network, human, original, ply);
      assert.equal(game.board.fen(), fens[ply]);
      assert.equal(game.board.history().length, ply);
      assert.equal(game.humanTurn, game.board.turn() === human);
      assert.equal(game.predictions.length, game.humanTurn ? game.board.moves().length : 0);
      assert.equal(game.observations.length, 0);
      assert.equal(game.clock, null);
    }
  }
  assert.equal(board.pgn(), original);
});

test('a new move from an old position creates a legal branch without retaining later moves', () => {
  const board = new Chess(); for (const move of ['e4', 'e5', 'Nf3', 'Nc6']) board.move(move);
  const original = board.pgn();
  const game = Game.fromPgn(network, 'w', original, 2);
  assert.ok(game.playHuman('d2d4'));
  assert.deepEqual(game.board.history(), ['e4', 'e5', 'd4']);
  assert.equal(board.pgn(), original);
  assert.equal(game.resigned, false); assert.equal(game.timeoutColor, null);
  game.playTonkAI();
  assert.equal(game.board.history().length, 4);
});

test('custom starts retain en passant and castling through rewind', () => {
  const fen = 'r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 7 20';
  const board = new Chess(fen);
  board.move('exd6'); board.move('O-O');
  const zero = Game.fromPgn(network, 'w', board.pgn(), 0);
  assert.equal(zero.board.fen(), fen);
  assert.ok(zero.playHuman('e5d6'));
  assert.equal(zero.board.get('d5'), undefined);
  const one = Game.fromPgn(network, 'b', board.pgn(), 1);
  assert.ok(one.playHuman('e8g8'));
  assert.equal(one.board.get('f8')?.type, 'r');
  assert.equal(one.board.fen(), board.fen());
});

test('promotion and repetition history survive position reconstruction', () => {
  const promotion = new Chess('7k/P7/8/8/8/8/8/K7 w - - 0 1'); promotion.move('a8=Q+');
  const promoted = Game.fromPgn(network, 'w', promotion.pgn(), 1);
  assert.equal(promoted.board.get('a8')?.type, 'q');
  const repeated = new Chess();
  for (const move of ['Nf3', 'Nf6', 'Ng1', 'Ng8', 'Nf3', 'Nf6', 'Ng1', 'Ng8']) repeated.move(move);
  const game = Game.fromPgn(network, 'w', repeated.pgn(), 8);
  assert.ok(game.board.isThreefoldRepetition());
  assert.equal(game.result, 'draw');
  assert.equal(Game.fromPgn(network, 'w', repeated.pgn(), 7).over, false);
});

test('training branches are marked in PGN and leave the original stored game intact', () => {
  const board = new Chess(); for (const move of ['e4', 'e5', 'Nf3', 'Nc6']) board.move(move);
  const original = createGameRecord(Game.fromPgn(network, 'w', board.pgn(), 4), 'original', 'Tu', opened);
  const data = new Map<string, string>();
  const store = { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => { data.set(key, value); } };
  saveGameHistory(original, [], store);
  const branch = Game.fromPgn(network, 'w', original.pgn, 2); branch.playHuman('d2d4');
  const record = createGameRecord(branch, 'branch', 'Tu', opened, undefined, true);
  saveGameHistory(record, [], store);
  const games = loadGameHistory(store);
  assert.equal(games.length, 2);
  assert.equal(games.find(game => game.id === 'original')?.pgn, original.pgn);
  assert.equal(games.find(game => game.id === 'original')?.training, false);
  assert.equal(games.find(game => game.id === 'branch')?.training, true);
  assert.match(record.pgn, /\[Mode "training"\]/);
});

test('invalid position indices and malformed PGNs cannot create a training game', () => {
  const board = new Chess(); board.move('e4');
  for (const ply of [-1, 2, 0.5, NaN]) assert.throws(() => Game.fromPgn(network, 'w', board.pgn(), ply));
  assert.throws(() => Game.fromPgn(network, 'w', 'broken PGN', 0));
});
