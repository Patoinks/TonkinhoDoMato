import type { Color } from 'chess.js';

export const BLITZ_TIME_CONTROL = { initialMs: 300_000, incrementMs: 3_000 } as const;
export type TimeControl = { initialMs: number; incrementMs: number; now?: () => number };

/** Store elapsed time at move boundaries, rather than counting interval callbacks. */
export class ChessClock {
  readonly control: TimeControl;
  readonly now: () => number;
  private balance: Record<Color, number>;
  private since: number;
  activeColor: Color | null;

  constructor(turn: Color, control: TimeControl = BLITZ_TIME_CONTROL) {
    this.control = control;
    this.now = control.now ?? Date.now;
    this.balance = { w: control.initialMs, b: control.initialMs };
    this.activeColor = turn;
    this.since = this.now();
  }
  remaining(color: Color, at = this.now()) {
    const elapsed = this.activeColor === color ? Math.max(0, at - this.since) : 0;
    return Math.max(0, this.balance[color] - elapsed);
  }
  flaggedColor(at = this.now()): Color | null {
    return this.activeColor && this.remaining(this.activeColor, at) <= 0 ? this.activeColor : null;
  }
  completeMove(color: Color, at = this.now()) {
    if (this.activeColor !== color || this.remaining(color, at) <= 0) return false;
    this.balance[color] = this.remaining(color, at) + this.control.incrementMs;
    this.activeColor = color === 'w' ? 'b' : 'w';
    this.since = at;
    return true;
  }
  stop(at = this.now()) {
    if (this.activeColor) this.balance[this.activeColor] = this.remaining(this.activeColor, at);
    this.activeColor = null;
  }
}

export function clockLabel(ms: number) {
  if (ms <= 0) return '0:00';
  if (ms < 10_000) return `0:${(Math.ceil(ms / 100) / 10).toFixed(1).padStart(4, '0')}`;
  const seconds = Math.ceil(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}
