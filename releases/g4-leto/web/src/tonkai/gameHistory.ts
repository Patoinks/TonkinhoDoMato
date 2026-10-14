import { Chess, type Color } from 'chess.js';
import type { Game } from './game';

export const GAME_HISTORY_STORAGE_KEY = 'tonkai.gameHistory.v1';
type GameResult = '1-0' | '0-1' | '1/2-1/2' | '*';
type GameStorage = Pick<Storage, 'getItem' | 'setItem'>;

export interface SavedGame {
  id: string;
  startedAt: string;
  savedAt: string;
  pgn: string;
  fens: string[];
  moves: string[];
  tonkaiColor: Color;
  tonkaiName: string;
  opponent: string;
  result: GameResult;
  plies: number;
  date: string;
  training: boolean;
}

function browserStorage(): GameStorage | undefined {
  try { return typeof window === 'undefined' ? undefined : window.localStorage; }
  catch { return undefined; }
}

/** Reconstruct the replay from the PGN instead of trusting cached positions. */
export function parseSavedGame(record: Pick<SavedGame, 'id' | 'startedAt' | 'savedAt' | 'pgn' | 'tonkaiColor'>): SavedGame {
  if (!record.id || !Number.isFinite(Date.parse(record.startedAt)) || !Number.isFinite(Date.parse(record.savedAt))
    || (record.tonkaiColor !== 'w' && record.tonkaiColor !== 'b')) throw new Error('Invalid saved game.');
  const board = new Chess();
  board.loadPgn(record.pgn, { strict: true });
  const headers = board.getHeaders();
  const result = headers.Result as GameResult;
  const declarations = [...record.pgn.matchAll(/\[Result\s+"([^"]*)"\s*\]/g)];
  const marker = record.pgn.replace(/\{[^}]*\}|;[^\r\n]*/g, '').trim().match(/(?:^|\s)(1-0|0-1|1\/2-1\/2|\*)$/)?.[1];
  if (declarations.length !== 1 || declarations[0][1] !== result || marker !== result
    || !['1-0', '0-1', '1/2-1/2', '*'].includes(result)
    || (headers.FEN && headers.SetUp !== '1') || !headers.White || !headers.Black) throw new Error('Invalid saved PGN.');
  if (board.isCheckmate() && result !== (board.turn() === 'w' ? '0-1' : '1-0')) throw new Error('Result contradicts checkmate.');
  const history = board.history({ verbose: true });
  return {
    id: record.id, startedAt: record.startedAt, savedAt: record.savedAt,
    pgn: record.pgn.trim() + '\n', tonkaiColor: record.tonkaiColor,
    tonkaiName: (record.tonkaiColor === 'w' ? headers.White : headers.Black)!,
    opponent: (record.tonkaiColor === 'w' ? headers.Black : headers.White)!, result,
    fens: [history[0]?.before ?? board.fen(), ...history.map(move => move.after)],
    moves: history.map(move => move.san), plies: history.length, date: headers.Date ?? '????.??.??',
    training: headers.Mode === 'training',
  };
}

/** Snapshot this match, including unfinished games and the actual final result. */
export function createGameRecord(game: Game, id: string, playerName: string, startedAt: string,
  savedAt = new Date().toISOString(), training = false): SavedGame {
  const copy = new Chess();
  copy.loadPgn(game.board.pgn(), { strict: true });
  const manifest = game.network.manifest;
  const botName = 'model_name' in manifest && typeof manifest.model_name === 'string' ? manifest.model_name : 'TonkAI';
  const clean = (name: string) => name.replace(/[\r\n"\\]/g, ' ').trim() || 'Tu';
  const result: GameResult = !game.over ? '*' : game.result === 'draw' ? '1/2-1/2'
    : (game.result === 'win' ? game.self : game.human) === 'w' ? '1-0' : '0-1';
  copy.setHeader('Event', `${clean(botName)} vs ${clean(playerName)}`);
  copy.setHeader('Site', typeof location === 'undefined' ? 'browser' : location.origin + '/tonkai');
  const date = new Date(startedAt);
  copy.setHeader('Date', `${date.getFullYear()}.${String(date.getMonth() + 1).padStart(2, '0')}.${String(date.getDate()).padStart(2, '0')}`);
  copy.setHeader('Round', '?');
  copy.setHeader('White', clean(game.self === 'w' ? botName : playerName));
  copy.setHeader('Black', clean(game.self === 'b' ? botName : playerName));
  copy.setHeader('Result', result);
  copy.setHeader('TimeControl', game.clock ? `${game.clock.control.initialMs / 1000}+${game.clock.control.incrementMs / 1000}` : '-');
  copy.setHeader('Termination', !game.over ? 'unterminated' : game.resigned ? 'resignation'
    : game.timeoutColor ? 'time forfeit' : game.board.isCheckmate() ? 'checkmate' : 'normal');
  copy.setHeader('PlyCount', String(copy.history().length));
  copy.setHeader('TonkAIWeights', game.network.manifest.sha256);
  if (training) copy.setHeader('Mode', 'training');
  return parseSavedGame({ id, startedAt, savedAt, pgn: copy.pgn(), tonkaiColor: game.self });
}

export function loadGameHistory(storage: Pick<Storage, 'getItem'> | undefined = browserStorage()): SavedGame[] {
  try {
    const raw = storage?.getItem(GAME_HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const records: unknown = JSON.parse(raw);
    if (!Array.isArray(records)) return [];
    const games: SavedGame[] = [];
    for (const record of records) {
      try { games.push(parseSavedGame(record)); } catch { /* Retain all other valid games. */ }
    }
    return games.sort((a, b) => b.startedAt.localeCompare(a.startedAt));
  } catch { return []; }
}

/** Replace this match's snapshot; preserve other matches, including other tabs. */
export function saveGameHistory(record: SavedGame, inMemory: SavedGame[] = [], storage = browserStorage()) {
  const gamesById = new Map<string, SavedGame>();
  for (const game of [...loadGameHistory(storage), ...inMemory]) {
    const previous = gamesById.get(game.id);
    if (!previous || game.savedAt > previous.savedAt) gamesById.set(game.id, game);
  }
  gamesById.set(record.id, parseSavedGame(record));
  const games = [...gamesById.values()].sort((a, b) => b.startedAt.localeCompare(a.startedAt));
  try {
    if (!storage) return { games, persisted: false };
    storage.setItem(GAME_HISTORY_STORAGE_KEY, JSON.stringify(games.map(({ id, startedAt, savedAt, pgn, tonkaiColor }) =>
      ({ id, startedAt, savedAt, pgn, tonkaiColor }))));
    return { games, persisted: true };
  } catch { return { games, persisted: false }; }
}

export function pgnDownload(pgn: string) {
  return new Blob([pgn], { type: 'application/x-chess-pgn;charset=utf-8' });
}
