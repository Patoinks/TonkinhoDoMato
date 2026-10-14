import { Chess, type Color } from 'chess.js';
import { TemporalNetwork, type TemporalManifest } from './temporal';
import { encode, softmax } from './encoding';
import { BOT_NAME } from './brand';
import { G16Network, loadG16Rules, type G16Manifest, type G16Rules, type LetoSearchLimits } from './g16v2';
import type { OpeningRepertoire } from './abertura';
import { loadPawnProofs, type PawnProofDescriptor } from './pawnProofs';
import { clearPawnCache } from './pawnomicsV3';

export interface Manifest extends TemporalManifest {
  pawn_proofs?: PawnProofDescriptor;
  revision?: string;
  binary?: string;
  format: string; bytes: number; sha256: string; draw_weight: number;
  tensors: { name: string; shape: number[]; offset: number; count: number }[];
}

export { encode, softmax } from './encoding';
export type LetoWorkerSeed = { manifest: Manifest; binary: ArrayBuffer; repertoire?: OpeningRepertoire };

export class Network {
  readonly manifest: Manifest;
  private tensors = new Map<string, { data: Float32Array; shape: number[] }>();
  readonly temporal?: TemporalNetwork;
  private binary: ArrayBuffer;
  private rules?: G16Rules;
  private repertoire?: OpeningRepertoire;

  constructor(manifest: Manifest, binary: ArrayBuffer, rules?: G16Rules, repertoire?: OpeningRepertoire) {
    if (manifest.format !== 'float32-le' || binary.byteLength !== manifest.bytes) {
      throw new Error('Invalid TonkAI weights format or length.');
    }
    this.binary = binary; this.rules = rules; this.repertoire = repertoire;
    this.manifest = manifest;
    const view = new DataView(binary);
    for (const t of manifest.tensors) {
      const data = Float32Array.from({ length: t.count }, (_, i) => view.getFloat32(t.offset + i * 4, true));
      this.tensors.set(t.name, { data, shape: t.shape });
    }
    if (manifest.model_kind === 'claudio-v2-web') {
      if (!rules) throw new Error('Missing G16 v2 native rules');
      this.temporal = new G16Network(manifest as G16Manifest, this.tensors, rules, repertoire);
    } else if (manifest.model_kind) this.temporal = new TemporalNetwork(manifest, this.tensors);
  }

  /** A fresh recurrent player and value calibration for each game, using the loaded model. */
  createPlayer() { return new Network(this.manifest, this.binary, this.rules, this.repertoire); }
  /** Transfer a copy to a worker; the active game keeps its validated model and tensors. */
  workerSeed(): LetoWorkerSeed {
    return { manifest: structuredClone(this.manifest), binary: this.binary.slice(0),
      repertoire: this.repertoire ? structuredClone(this.repertoire) : undefined };
  }

  private linear(input: Float64Array, name: string, activation: boolean): Float64Array {
    const weight = this.tensors.get(`${name}.weight`)!;
    const bias = this.tensors.get(`${name}.bias`)!.data;
    const [rows, cols] = weight.shape;
    if (cols !== input.length) throw new Error(`Invalid layer ${name}`);
    const out = new Float64Array(rows);
    for (let r = 0; r < rows; r++) {
      let v = bias[r];
      for (let c = 0; c < cols; c++) v += weight.data[r * cols + c] * input[c];
      out[r] = activation ? v / (1 + Math.exp(-v)) : v;
    }
    return out;
  }

  forward(fen: string, self: Color) {
    return this.forwardBatch([fen], self)[0];
  }

  forwardBatch(fens: string[], self: Color) {
    const layer = (batch: Float64Array[], name: string, activation: boolean) =>
      batch.map(input => this.linear(input, name, activation));
    const inputs = fens.map(fen => encode(fen, self));
    const latent = layer(layer(inputs, 'encoder.0', true), 'encoder.2', true);
    const values = layer(layer(latent, 'value_head.0', true), 'value_head.2', false);
    const predictions = layer(layer(latent, 'predict_head.0', true), 'predict_head.2', false);
    return values.map((value, i) => {
      const logits = Array.from(value);
      const wdl = softmax(logits);
      return { logits, predictLogit: predictions[i][0], wdl,
        utility: wdl[0] + this.manifest.draw_weight * wdl[1] - wdl[2] };
    });
  }

  /** A batch of all legal successor positions; no lookahead beyond each candidate. */
  private candidates(board: Chess, self: Color) {
    const moves = board.moves({ verbose: true });
    const positions = moves.map(move => {
      board.move(move);
      try {
        const fields = board.fen({ forceEnpassantSquare: true }).split(' ');
        // chess.js drops EP internally unless an adjacent pawn exists. Python
        // retains every double-push target, which is part of the trained input.
        fields[3] = move.isBigPawn()
          ? move.from[0] + String((Number(move.from[1]) + Number(move.to[1])) / 2)
          : '-';
        return fields.join(' ');
      } finally { board.undo(); }
    });
    const outputs = this.forwardBatch(positions, self);
    return moves.map((move, i) => ({ uci: move.from + move.to + (move.promotion ?? ''),
      san: move.san, ...outputs[i] }));
  }

  rank(board: Chess, self: Color, limits?: LetoSearchLimits) {
    if (board.turn() !== self) throw new Error('Value ranking requires TonkAI to move.');
    if (this.temporal instanceof G16Network) return this.temporal.rank(board, self, limits);
    if (this.temporal) return this.temporal.rank(board, self);
    return this.candidates(board, self).sort((a, b) => b.utility - a.utility);
  }

  terminal(board: Chess, self: Color) {
    return this.temporal instanceof G16Network ? this.temporal.terminal(board, self) : undefined;
  }

  predict(board: Chess, self: Color) {
    if (board.turn() === self) throw new Error('Prediction requires the opponent to move.');
    if (this.temporal) return this.temporal.predict(board, self);
    const candidates = this.candidates(board, self);
    const probs = softmax(candidates.map(m => m.predictLogit));
    return candidates.map((m, i) => ({ uci: m.uci, san: m.san, p: probs[i] }))
      .sort((a, b) => b.p - a.p);
  }
}

export type RankedMove = ReturnType<Network['rank']>[number];
export type Prediction = ReturnType<Network['predict']>[number];

export async function loadNetwork(manifestPath = '/tonkai/tonkai.json'): Promise<Network> {
  const meta = await fetch(manifestPath, { cache: 'no-cache' });
  if (!meta.ok) throw new Error(`Não foi possível carregar o ${BOT_NAME}. Tenta novamente.`);
  const manifest: Manifest = await meta.json();
  if (Number((manifest as G16Manifest).claudio_config?.pawn_version) === 3) {
    if (!manifest.pawn_proofs) throw new Error('Missing Pawnomics proof descriptor.');
    await loadPawnProofs(manifest.pawn_proofs); clearPawnCache();
  }
  const name = manifest.binary ?? 'tonkai.bin';
  if (!/^tonkai(?:-[a-f0-9]{64})?\.bin$/.test(name)) throw new Error('Invalid model filename');
  const weights = await fetch(`/tonkai/${name}`, { cache: 'no-cache' });
  if (!weights.ok) throw new Error(`Não foi possível carregar o ${BOT_NAME}. Tenta novamente.`);
  const binary = await weights.arrayBuffer();
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', binary)))
    .map(b => b.toString(16).padStart(2, '0')).join('');
  if (hash !== manifest.sha256) throw new Error('Weights checksum mismatch. Please reload.');
  const rules = manifest.model_kind === 'claudio-v2-web' ? await loadG16Rules(manifest as G16Manifest) : undefined;
  let repertoire: OpeningRepertoire | undefined;
  const opening = (manifest as G16Manifest).opening_repertoire;
  if (opening) {
    if (!/^leto-book-v[12]\.json$/.test(opening.file)) throw new Error('Invalid opening repertoire filename');
    const response = await fetch(`/tonkai/${opening.file}`, { cache: 'no-cache' });
    if (!response.ok) throw new Error('Não foi possível carregar o livro do G4 Leto.');
    const bytes = await response.arrayBuffer();
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))).map(b => b.toString(16).padStart(2, '0')).join('');
    if (hash !== opening.sha256) throw new Error('Leto opening repertoire checksum mismatch. Please reload.');
    repertoire = JSON.parse(new TextDecoder().decode(bytes));
    if (!repertoire || Object.keys(repertoire).length !== opening.positions) throw new Error('Invalid opening repertoire size');
  }
  return new Network(manifest, binary, rules, repertoire);
}
