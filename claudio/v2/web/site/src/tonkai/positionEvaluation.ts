import type { Game } from './game';
import { terminal } from './temporal';

/** Read the active game's checkpoint and real recurrent history, never a fresh FEN. */
export function positionEvaluation(game: Game) {
  const temporal = game.network.temporal;
  if (!temporal) return null;
  const snapshot = temporal.snapshot(game.board, game.self);
  const pieces = temporal.sensePieces(game.board, game.self);
  const pieceBalance = pieces.reduce((sum, piece) => sum + (piece.value ?? 0), 0);
  const nativeEnding = game.network.terminal(game.board, game.self);
  const ending = game.resigned ? 1 : nativeEnding === undefined ? terminal(game.board, game.self) : nativeEnding;
  return { fen: game.board.fen(), self: game.self, value: ending ?? snapshot.value,
    terminal: ending !== null, result: game.result, pieces, pieceBalance };
}

export function pieceBalanceLabel(value: number) {
  return `${value < -.005 ? '−' : '+'}${Math.abs(value) < .005 ? '0,00' : Math.abs(value).toFixed(2).replace('.', ',')}`;
}

/** Same learned balance, expressed from White's perspective for adjacent bars. */
export function tonkinhoWhiteBalance(evaluation: { self: 'w' | 'b'; pieceBalance: number }) {
  return evaluation.pieceBalance * (evaluation.self === 'w' ? 1 : -1);
}

export function pieceValueLabel(value: number | null) {
  if (value === null) return '∞';
  return (Math.abs(value) < .05 ? 0 : value).toFixed(1).replace('.', ',');
}
