import { Chess } from 'chess.js';
import { Network } from './network';
import { G16Network, loadG16Rules, type G16Manifest } from './g16v2';
import { loadPawnProofs } from './pawnProofs';
import { clearPawnCache } from './pawnomicsV3';
import { searchLetoUntil } from './letoTimedSearch';
import type { LetoWorkerRequest, LetoWorkerReply } from './letoThinkingProtocol';

const scope = globalThis as unknown as { onmessage: ((event: { data: LetoWorkerRequest }) => void) | null;
  postMessage: (reply: LetoWorkerReply) => void };
let network: Network | null = null, player: Network | null = null, board: Chess | null = null;
let gameId = '', root = '', self = '';
scope.onmessage = event => { void handle(event.data); };
async function handle(request: LetoWorkerRequest) {
  try {
    if (request.type === 'init') {
      const { manifest, binary, repertoire } = request.seed;
      if ((manifest as G16Manifest).model_name !== 'G4 Leto') throw new Error('Expected the active G4 Leto model.');
      if (manifest.pawn_proofs) await loadPawnProofs(manifest.pawn_proofs);
      const rules = await loadG16Rules(manifest as G16Manifest);
      network = new Network(manifest, binary, rules, repertoire);
      clearPawnCache(); player = null; board = null;
      scope.postMessage({ type: 'ready' }); return;
    }
    if (!network) throw new Error('G4 Leto is not ready.');
    const history = board?.history({ verbose: true }).map(move => move.from + move.to + (move.promotion ?? '')) ?? [];
    if (!board || gameId !== request.gameId || root !== request.root || self !== request.self
      || history.length > request.moves.length || !history.every((move, index) => move === request.moves[index])) {
      board = new Chess(request.root); player = network.createPlayer();
      gameId = request.gameId; root = request.root; self = request.self;
    }
    for (const uci of request.moves.slice(board.history().length)) {
      board.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
    }
    if (board.fen() !== request.fen || board.turn() !== request.self || !player) throw new Error('Stale G4 position.');
    if (request.spread !== undefined && player.temporal instanceof G16Network) player.temporal.restoreSearchCalibration(request.spread);
    const result = searchLetoUntil(player, board, request.self, request.remainingMs,
      (best, proved) => scope.postMessage({ type: 'best', id: request.id, fen: request.fen, best, proved }));
    if (result.best) scope.postMessage({ type: 'done', id: request.id, fen: request.fen, best: result.best, proved: result.proved });
  } catch (error) {
    scope.postMessage({ type: 'error', id: request.type === 'think' ? request.id : undefined, message: String(error) });
  }
}
