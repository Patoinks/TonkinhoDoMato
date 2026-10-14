import { PawnBoard, pawnFile, pawnRank } from './pawnBoard';
import { type Chess } from 'chess.js';

export type PawnProofDescriptor = { file: string; bytes: number; sha256: string };
export type PawnProof = { owner: number; winning: boolean; resetDistance: number };
let proofs: Uint8Array | null = null;
let loadedHash = '';

/** Packed rule-based K+P/K outcomes and distance to the next pawn move/capture. */
export function initializePawnProofs(bytes: Uint8Array) {
  if (bytes.byteLength !== 24 * 64 * 64 * 2) throw new Error('Invalid pawn proof table.');
  proofs = bytes.slice();
}

export async function loadPawnProofs(descriptor: PawnProofDescriptor) {
  if (descriptor.file !== 'pawn-kpk-v3.bin' || descriptor.bytes !== 196608 || !/^[a-f0-9]{64}$/.test(descriptor.sha256)) {
    throw new Error('Invalid Pawnomics proof descriptor.');
  }
  if (loadedHash === descriptor.sha256 && proofs) return;
  const response = await fetch('/tonkai/' + descriptor.file, { cache: 'no-cache' });
  if (!response.ok) throw new Error('Não foi possível carregar os finais de peões.');
  const bytes = await response.arrayBuffer();
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))).map(value => value.toString(16).padStart(2, '0')).join('');
  if (hash !== descriptor.sha256) throw new Error('Checksum dos finais de peões inválido.');
  initializePawnProofs(new Uint8Array(bytes)); loadedHash = descriptor.sha256;
}

export function probePawnProof(board: PawnBoard): PawnProof | null {
  if (!proofs) return null;
  let pawn = -1, owner = 0, count = 0;
  for (let square = 0; square < 64; square++) {
    const piece = board.cells[square];
    if (!piece) continue;
    count++;
    if (Math.abs(piece) === 1) { pawn = square; owner = piece > 0 ? 1 : -1; }
    else if (Math.abs(piece) !== 6) return null;
  }
  if (count !== 3 || pawn < 0) return null;
  let white = board.king(owner), black = board.king(-owner);
  const turn = board.turn * owner;
  if (white < 0 || black < 0) return null;
  if (owner === -1) { pawn ^= 56; white ^= 56; black ^= 56; }
  if (pawnFile(pawn) >= 4) { pawn ^= 7; white ^= 7; black ^= 7; }
  const rank = pawnRank(pawn);
  if (rank < 1 || rank > 6) return null;
  const slot = (rank - 1) * 4 + pawnFile(pawn);
  const result = proofs[((slot * 64 + white) * 64 + black) * 2 + Number(turn === -1)];
  if (result === 255) return null;
  const resetDistance = Math.max(0, result - 1);
  // Near the 50-move boundary use the legal search; DTZ may be rounded by one ply.
  if (result && board.halfmove + resetDistance + 1 >= 100) return null;
  return { owner, winning: result > 0, resetDistance };
}

/** Exact outcome-preserving moves in K+P/K, including captures and underpromotion. */
export function provenPawnMoves(board: Chess) {
  const root = probePawnProof(PawnBoard.from(board));
  if (!root) return null;
  const me = board.turn() === 'w' ? 1 : -1;
  const choices: { uci: string; wdl: number; resetDistance: number }[] = [];
  for (const move of board.moves({ verbose: true })) {
    board.move(move);
    try {
      let wdl: number, resetDistance = 0;
      const next = probePawnProof(PawnBoard.from(board));
      if (board.isCheckmate()) wdl = 2;
      else if (board.isDraw()) wdl = 0;
      else if (next) {
        wdl = next.winning ? 2 * root.owner * me : 0;
        // A pawn move resets the clock now; its child's next reset may be much farther away.
        resetDistance = move.piece === 'p' || move.captured ? 0 : next.resetDistance + 1;
      }
      else if (move.promotion) {
        const canTake = board.moves({ verbose: true }).some(reply => reply.to === move.to && reply.isCapture());
        wdl = canTake || move.promotion === 'n' || move.promotion === 'b' ? 0 : 2;
      } else if (move.captured === 'p') wdl = 0;
      else return null; // Missing/50-move-sensitive proof: leave the choice to the engine.
      choices.push({ uci: move.from + move.to + (move.promotion ?? ''), wdl, resetDistance });
    } finally { board.undo(); }
  }
  return choices.sort((a, b) => b.wdl - a.wdl || (a.wdl > 0 ? a.resetDistance - b.resetDistance : b.resetDistance - a.resetDistance)
    || a.uci.localeCompare(b.uci));
}
