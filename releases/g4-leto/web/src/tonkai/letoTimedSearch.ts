import { type Chess, type Color } from 'chess.js';
import type { Network } from './network';
import { G16Network, type G16Manifest, type LetoSearchLimits } from './g16v2';
import { MAX_LETO_THINK_MS, type LetoBest } from './letoThinkingProtocol';

/** Complete progressively deeper G4 passes. The owner enforces the hard worker deadline. */
export function searchLetoUntil(network: Network, board: Chess, self: Color, remainingMs: number,
  publish: (best: LetoBest, proved: boolean) => void, now = () => performance.now()) {
  const config = (network.manifest as G16Manifest).claudio_config;
  const start = now(), deadline = start + Math.max(0, Math.min(MAX_LETO_THINK_MS, remainingMs));
  let best: LetoBest | null = null;
  for (let iteration = 0; now() < deadline - 20; iteration++) {
    const baseDepth = Math.max(4, Number(config.tactic_depth) || 6);
    const depth = iteration === 0 ? 2 : Math.min(24, baseDepth + 2 * (iteration - 1));
    const growth = Math.max(1, 1 + (iteration - 1) * .5);
    const limits: LetoSearchLimits = iteration === 0
      ? { depth, nodes: 4000, exchangeNodes: 4000, replyTop: 0, tacticTop: 6, rounds: 1, scanDepth: 1, mateNodes: 1000 }
      : { depth, nodes: Math.floor((Number(config.tactic_nodes) || 559745) * growth),
        exchangeNodes: Number(config.exchange_nodes) || 69277,
        replyTop: Math.min(10, (Number(config.reply_top) || 4) + 2 * (iteration - 1)),
        tacticTop: Math.min(24, (Number(config.tactic_top) || 10) + 2 * (iteration - 1)),
        rounds: Number(config.tactic_rounds) || 3, scanDepth: Number(config.scan_depth) || 3,
        mateNodes: Math.floor((Number(config.mate_nodes) || 9241) * growth) };
    const ranked = network.rank(board, self, limits);
    if (!ranked.length) throw new Error('No legal G4 move.');
    // An overrun is discarded; only results completed inside the allotted time are eligible.
    if (now() >= deadline) break;
    best = { uci: ranked[0].uci, score: ranked[0].utility, depth, iterations: iteration + 1,
      spread: network.temporal instanceof G16Network ? network.temporal.searchCalibration() : undefined };
    const proved = ranked.length === 1; // Exact final, opening repertoire, forced move or proven mate.
    publish(best, proved);
    if (proved) return { best, proved: true };
  }
  return { best, proved: false };
}
