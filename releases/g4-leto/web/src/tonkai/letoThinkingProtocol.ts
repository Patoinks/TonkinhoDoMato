import type { Color } from 'chess.js';
import type { LetoWorkerSeed } from './network';

export const MAX_LETO_THINK_MS = 5000;
export type LetoBest = { uci: string; depth: number; iterations: number; score: number; spread?: number };
export type LetoThinkRequest = { type: 'think'; id: number; gameId: string; root: string; moves: string[];
  fen: string; self: Color; remainingMs: number; spread?: number };
export type LetoWorkerRequest = { type: 'init'; seed: LetoWorkerSeed } | LetoThinkRequest;
export type LetoWorkerReply = { type: 'ready' } | { type: 'error'; id?: number; message: string }
  | { type: 'best' | 'done'; id: number; fen: string; best: LetoBest; proved: boolean };
export type LetoThinkingResult = LetoBest & { elapsedMs: number; reason: 'proved' | 'budget' | 'deadline'; fallback: boolean };
