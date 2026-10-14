import { Chess, type Color } from 'chess.js';
import { Network, type Prediction, type RankedMove } from './network';
import { terminal } from './temporal';
import { ChessClock, type TimeControl } from './clock';

export interface FinishedGame {
  result: 'win' | 'draw' | 'loss'; // always TonkAI's perspective
  plies: number; tonkai_colour: Color; mean_surprise: number;
  top1_hits: number; moves_played: number; client_timestamp: string; weights_sha256: string;
}

export class Game {
  readonly board: Chess;
  readonly network: Network;
  readonly human: Color;
  readonly self: Color;
  predictions: Prediction[] = [];
  ranking: RankedMove[] = [];
  observations: { san: string; surprise: number; hit: boolean; p: number }[] = [];
  resigned = false;
  readonly clock: ChessClock | null;
  timeoutColor: Color | null = null;

  constructor(network: Network, human: Color, fen?: string, timeControl?: TimeControl) {
    this.network = network; this.human = human; this.self = human === 'w' ? 'b' : 'w';
    this.board = new Chess(fen);
    this.refreshPrediction();
    this.clock = timeControl ? new ChessClock(this.board.turn(), timeControl) : null;
    if (this.over) this.clock?.stop();
  }
  /** Rebuild a legal prefix with a fresh player and the complete retained history. */
  static fromPgn(network: Network, human: Color, pgn: string, ply: number): Game {
    const source = new Chess();
    source.loadPgn(pgn, { strict: true });
    const history = source.history({ verbose: true });
    if (!Number.isInteger(ply) || ply < 0 || ply > history.length) throw new Error('Invalid training position.');
    const game = new Game(network.createPlayer(), human, history[0]?.before ?? source.fen());
    for (const move of history.slice(0, ply)) game.board.move(move.san);
    game.refreshPrediction();
    return game;
  }
  get over() {
    if (this.resigned || this.timeoutColor !== null) return true;
    const native = this.network.terminal?.(this.board, this.self);
    return native === undefined ? (this.network.temporal ? terminal(this.board, this.self) !== null : this.board.isGameOver()) : native !== null;
  }
  get humanTurn() { return !this.over && this.board.turn() === this.human; }
  get hits() { return this.observations.filter(o => o.hit).length; }
  get meanSurprise() {
    return this.observations.reduce((s, o) => s + o.surprise, 0) / (this.observations.length || 1);
  }
  get result(): FinishedGame['result'] | null {
    if (!this.over) return null;
    if (this.timeoutColor) {
      const winner = this.timeoutColor === 'w' ? 'b' : 'w';
      // A lone king cannot win on time.
      if (!this.board.board().flat().some(piece => piece?.color === winner && piece.type !== 'k')) return 'draw';
      return winner === this.self ? 'win' : 'loss';
    }
    if (this.resigned) return 'win';
    if (!this.board.isCheckmate()) return 'draw';
    return this.board.turn() === this.self ? 'loss' : 'win';
  }
  private refreshPrediction() {
    this.predictions = this.humanTurn ? this.network.predict(this.board, this.self) : [];
  }
  checkClock(at = this.clock?.now()): boolean {
    if (!this.clock) return false;
    if (this.over) { this.clock.stop(at); return false; }
    const flagged = this.clock.flaggedColor(at);
    if (!flagged) return false;
    this.timeoutColor = flagged;
    this.predictions = []; this.ranking = [];
    this.clock.stop(at);
    return true;
  }
  playHuman(uci: string): boolean {
    const at = this.clock?.now();
    if (this.checkClock(at)) return false;
    if (!this.humanTurn) return false;
    const prediction = this.predictions.find(m => m.uci === uci);
    if (!prediction) return false;
    this.board.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
    this.observations.push({ san: prediction.san, p: prediction.p,
      surprise: -Math.log(Math.max(prediction.p, Number.MIN_VALUE)), hit: prediction === this.predictions[0] });
    this.predictions = [];
    this.clock?.completeMove(this.human, at);
    if (this.over) this.clock?.stop(at);
    return true;
  }
  playTonkAI() {
    if (this.checkClock()) return;
    if (this.over || this.humanTurn) return;
    this.ranking = this.network.rank(this.board, this.self);
    // Ranking is synchronous; charge its real elapsed time before committing the move.
    const at = this.clock?.now();
    if (this.checkClock(at)) return;
    this.playTonkAIChoice(this.ranking[0].uci);
  }
  /** Commit a worker result only to its original, still-active position. */
  playTonkAIChoice(uci: string, expectedFen = this.board.fen()): boolean {
    const at = this.clock?.now();
    if (this.checkClock(at) || this.over || this.humanTurn || this.board.fen() !== expectedFen) return false;
    if (!this.board.moves({ verbose: true }).some(move => move.from + move.to + (move.promotion ?? '') === uci)) {
      throw new Error('G4 Leto returned an illegal move.');
    }
    this.board.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
    this.clock?.completeMove(this.self, at);
    if (this.over) this.clock?.stop(at);
    this.refreshPrediction();
    return true;
  }
  finish(): FinishedGame {
    if (!this.result) throw new Error('Cannot record an unfinished game.');
    return { result: this.result, plies: this.board.history().length, tonkai_colour: this.self,
      mean_surprise: this.meanSurprise, top1_hits: this.hits, moves_played: this.observations.length,
      client_timestamp: new Date().toISOString(), weights_sha256: this.network.manifest.sha256 };
  }
}
