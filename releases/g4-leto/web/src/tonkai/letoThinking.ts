import { type Chess, type Color } from 'chess.js';
import type { Network } from './network';
import { MAX_LETO_THINK_MS, type LetoBest, type LetoThinkingResult, type LetoWorkerReply, type LetoWorkerRequest } from './letoThinkingProtocol';

export type LetoWorkerPort = { onmessage: ((event: { data: LetoWorkerReply }) => void) | null;
  onerror: (() => void) | null; postMessage: (message: LetoWorkerRequest, transfer?: Transferable[]) => void; terminate: () => void };
type Clock = { now: () => number; set: (callback: () => void, ms: number) => ReturnType<typeof setTimeout>;
  clear: (id: ReturnType<typeof setTimeout>) => void };
type Pending = { id: number; fen: string; onReply: (reply: LetoWorkerReply) => void; cancel: () => void };
const clock: Clock = { now: () => performance.now(), set: (callback, ms) => setTimeout(callback, ms), clear: id => clearTimeout(id) };
const aborted = () => new DOMException('Thinking cancelled.', 'AbortError');

/** One background G4 instance; a main-thread watchdog can stop even a synchronous WASM search. */
export class LetoThinker {
  private worker: LetoWorkerPort | null = null;
  private modelKey = '';
  private ready: Promise<void> | null = null;
  private readyResolve: (() => void) | null = null;
  private readyReject: ((error: Error) => void) | null = null;
  private bootTimer: ReturnType<typeof setTimeout> | null = null;
  private pending: Pending | null = null;
  private serial = 0;
  private calibrations = new Map<string, number>();
  private factory: () => LetoWorkerPort;
  private timer: Clock;
  constructor(factory: () => LetoWorkerPort = () => new Worker(new URL('./leto.worker.ts', import.meta.url), { type: 'module' }) as unknown as LetoWorkerPort,
    timer: Clock = clock) { this.factory = factory; this.timer = timer; }

  prepare(network: Network) {
    const key = network.manifest.revision ?? network.manifest.sha256;
    if (this.worker && this.modelKey === key && this.ready) return this.ready;
    this.dispose();
    const worker = this.factory(); this.worker = worker; this.modelKey = key;
    const ready = new Promise<void>((resolve, reject) => { this.readyResolve = resolve; this.readyReject = reject; });
    this.ready = ready;
    worker.onmessage = event => {
      if (this.worker !== worker) return;
      const reply = event.data;
      if (reply.type === 'ready') {
        if (this.bootTimer !== null) this.timer.clear(this.bootTimer);
        this.bootTimer = null; this.readyResolve?.(); this.readyResolve = null; this.readyReject = null;
      } else if (reply.type === 'error' && reply.id === undefined) {
        this.readyReject?.(new Error(reply.message)); this.stopWorker();
      } else {
        const pending = this.pending;
        if (pending && 'id' in reply && reply.id === pending.id) pending.onReply(reply);
      }
    };
    worker.onerror = () => {
      this.readyReject?.(new Error('Não foi possível iniciar o cálculo do G4 Leto.'));
      this.pending?.onReply({ type: 'error', id: this.pending.id, message: 'O cálculo do G4 Leto foi interrompido.' });
      this.stopWorker();
    };
    this.bootTimer = this.timer.set(() => { this.readyReject?.(new Error('G4 Leto não ficou pronto.')); this.stopWorker(); }, 15000);
    try {
      const seed = network.workerSeed();
      worker.postMessage({ type: 'init', seed }, [seed.binary]);
    } catch (error) { this.readyReject?.(error instanceof Error ? error : new Error(String(error))); this.stopWorker(); }
    return ready;
  }

  think(network: Network, board: Chess, self: Color, gameId: string, signal: AbortSignal,
    budgetMs = MAX_LETO_THINK_MS): Promise<LetoThinkingResult> {
    this.pending?.cancel();
    if (signal.aborted) return Promise.reject(aborted());
    if (board.turn() !== self || board.isGameOver()) return Promise.reject(new Error('G4 Leto is not on turn.'));
    // Leave a small scheduling margin so the watchdog does not normally cross the 5 s ceiling.
    const requested = Math.max(1, Math.min(MAX_LETO_THINK_MS, budgetMs));
    const start = this.timer.now(), budget = Math.max(1, requested - Math.min(50, requested * .01));
    const fen = board.fen(), history = board.history({ verbose: true });
    const legal = new Set(board.moves({ verbose: true }).map(move => move.from + move.to + (move.promotion ?? '')));
    // A legal reserve exists even if a slow device cannot complete its first pass before the deadline.
    const fallback = legal.values().next().value!;
    const id = ++this.serial;
    const calibrationKey = (network.manifest.revision ?? network.manifest.sha256) + ':' + gameId;
    return new Promise((resolve, reject) => {
      let settled = false, best: LetoBest | null = null;
      const finish = (reason: LetoThinkingResult['reason'], stop: boolean, error?: Error) => {
        if (settled) return;
        settled = true; this.timer.clear(deadline); signal.removeEventListener('abort', cancel);
        if (this.pending?.id === id) this.pending = null;
        if (stop) this.stopWorker();
        if (error) reject(error);
        else {
          if (best?.spread !== undefined && Number.isFinite(best.spread) && best.spread > 0) {
            if (this.calibrations.size >= 8 && !this.calibrations.has(calibrationKey)) this.calibrations.delete(this.calibrations.keys().next().value!);
            this.calibrations.set(calibrationKey, best.spread);
          }
          resolve({ ...(best ?? { uci: fallback, depth: 0, iterations: 0, score: 0 }),
          elapsedMs: this.timer.now() - start, reason, fallback: !best });
        }
      };
      const cancel = () => finish('deadline', true, aborted());
      const deadline = this.timer.set(() => finish('deadline', true), budget);
      signal.addEventListener('abort', cancel, { once: true });
      // Preparation precedes registration: prepare may replace a worker from another model.
      let preparation: Promise<void>;
      try { preparation = this.prepare(network); }
      catch (error) { finish('deadline', true, error instanceof Error ? error : new Error(String(error))); return; }
      this.pending = { id, fen, cancel, onReply: reply => {
        if (settled || !('fen' in reply) || reply.fen !== fen) {
          if (reply.type === 'error') finish('deadline', true, best ? undefined : new Error(reply.message));
          return;
        }
        if (this.timer.now() - start >= budget) { finish('deadline', true); return; }
        if (!legal.has(reply.best.uci)) { finish('deadline', true, new Error('G4 Leto returned an illegal move.')); return; }
        best = reply.best;
        if (reply.type === 'done' || reply.proved) finish(reply.proved ? 'proved' : 'budget', false);
      } };
      void preparation.then(() => {
        if (settled || signal.aborted) return;
        const remainingMs = budget - (this.timer.now() - start);
        if (remainingMs <= 0) { finish('deadline', true); return; }
        this.worker?.postMessage({ type: 'think', id, gameId, fen, self, remainingMs,
          root: history[0]?.before ?? board.fen({ forceEnpassantSquare: true }),
          moves: history.map(move => move.from + move.to + (move.promotion ?? '')), spread: this.calibrations.get(calibrationKey) });
      }).catch(error => { if (!settled) finish('deadline', true, error instanceof Error ? error : new Error(String(error))); });
    });
  }

  private stopWorker() {
    this.worker?.terminate(); this.worker = null; this.modelKey = ''; this.ready = null;
    this.readyReject?.(aborted()); this.readyResolve = null; this.readyReject = null;
    if (this.bootTimer !== null) this.timer.clear(this.bootTimer);
    this.bootTimer = null;
  }
  cancel() { this.pending?.cancel(); }
  dispose() { this.cancel(); this.stopWorker(); }
}
