"""Counters, timers and the one compact log line.

The point of the timers is to answer a single question: where does the time
actually go? Never optimise before reading this.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

import numpy as np

from .encoding import PIECE_NAMES


class Timer:
    """`with metrics.timer('inference'): ...`"""

    def __init__(self, store: dict, key: str):
        self.store = store
        self.key = key

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.store[self.key] += time.perf_counter() - self.t0
        return False


class Window:
    """Mean over the last N values, cheaply."""

    def __init__(self, n: int = 2000):
        self.values = deque(maxlen=n)

    def add(self, v) -> None:
        self.values.append(float(v))

    def extend(self, vs) -> None:
        for v in vs:
            self.values.append(float(v))

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.values)) if self.values else 0.0

    def __len__(self) -> int:
        return len(self.values)


class Metrics:
    def __init__(self, window: int = 2000):
        self.t_start = time.perf_counter()
        self.times = defaultdict(float)

        self.games = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.plies = 0
        self.tonkai_positions = 0      # positions pushed through the encoder
        self.opponent_moves = 0
        self.train_batches = 0
        self.sacrifices = 0
        self.theory_events = 0

        self.game_length = Window(window)
        self.legal_moves = Window(window)
        self.enemy_moves = Window(window)

        self.value_loss = Window(window)
        self.predict_loss = Window(window)
        self.piece_loss = Window(window)

        self.top1 = Window(window)
        self.top3 = Window(window)
        self.surprise = Window(window)
        self.entropy = Window(window)
        self.true_move_prob = Window(window)

        self.piece_value = {name: Window(window) for name in PIECE_NAMES}
        self.termination = defaultdict(int)

        self._last_report = time.perf_counter()
        self._last_games = 0
        self._last_positions = 0

    def timer(self, key: str) -> Timer:
        return Timer(self.times, key)

    def add_game(self, result: int, plies: int, reason: str) -> None:
        self.games += 1
        self.plies += plies
        self.game_length.add(plies)
        self.termination[reason] += 1
        if result == 0:
            self.wins += 1
        elif result == 1:
            self.draws += 1
        else:
            self.losses += 1

    # --------------------------------------------------------------- report

    def rates(self):
        """Rate since the last report, falling back to the overall average when
        two reports land on top of each other (otherwise it reads as zero)."""
        now = time.perf_counter()
        dt = now - self._last_report
        dg = self.games - self._last_games
        if dt < 0.05 or dg <= 0:
            total = max(now - self.t_start, 1e-9)
            return self.games / total, self.tonkai_positions / total
        return dg / dt, (self.tonkai_positions - self._last_positions) / dt

    def mark_report(self) -> None:
        self._last_report = time.perf_counter()
        self._last_games = self.games
        self._last_positions = self.tonkai_positions

    def wdl_percent(self):
        n = max(self.games, 1)
        return 100 * self.wins / n, 100 * self.draws / n, 100 * self.losses / n

    def timing_breakdown(self) -> str:
        total = sum(self.times.values())
        if total <= 0:
            return "  (no timings yet)"
        rows = []
        for k in sorted(self.times, key=lambda k: -self.times[k]):
            v = self.times[k]
            rows.append(f"  {k:16s} {v:8.1f}s  {100 * v / total:5.1f}%")
        return "\n".join(rows)

    def bottleneck(self) -> str:
        if not self.times:
            return "unknown"
        k = max(self.times, key=lambda k: self.times[k])
        total = sum(self.times.values())
        return f"{k} ({100 * self.times[k] / total:.0f}% of measured time)"

    def report(self, extra: str = "") -> str:
        gps, pps = self.rates()
        w, d, l = self.wdl_percent()
        lines = [
            f"Games: {self.games}   W/D/L: {w:.1f} / {d:.1f} / {l:.1f}",
            f"Games/s: {gps:.1f}   Positions/s: {pps:.0f}   "
            f"Batches: {self.train_batches}",
            f"Loss  value {self.value_loss.mean:.4f}  "
            f"predict {self.predict_loss.mean:.4f}  "
            f"piece {self.piece_loss.mean:.5f}",
            f"Predict  top1 {100 * self.top1.mean:.1f}%  "
            f"top3 {100 * self.top3.mean:.1f}%  "
            f"p(true) {self.true_move_prob.mean:.3f}  "
            f"surprise {self.surprise.mean:.2f}  "
            f"entropy {self.entropy.mean:.2f}",
            f"Avg len {self.game_length.mean:.0f} plies   "
            f"legal {self.legal_moves.mean:.1f}   "
            f"enemy {self.enemy_moves.mean:.1f}",
        ]
        pv = "  ".join(
            f"{n[:2]} {self.piece_value[n].mean:+.2f}/{self.piece_value[n].std:.2f}"
            for n in PIECE_NAMES
            if len(self.piece_value[n])
        )
        if pv:
            lines.append(f"Piece values  {pv}")
        lines.append(
            f"Sacrifices: {self.sacrifices}   Theory events: {self.theory_events}"
        )
        if extra:
            lines.append(extra)
        self.mark_report()
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "plies": self.plies,
            "tonkai_positions": self.tonkai_positions,
            "opponent_moves": self.opponent_moves,
            "train_batches": self.train_batches,
            "sacrifices": self.sacrifices,
            "theory_events": self.theory_events,
            "times": dict(self.times),
            "actor_times": dict(getattr(self, "actor_times", {})),
            "inference_refresh_seconds": getattr(self, "inference_refresh_seconds", 0.0),
            "termination": dict(self.termination),
        }

    def snapshot(self) -> dict:
        """Report-time snapshot only; no extra work in the per-position loop."""
        data = self.to_dict()
        elapsed = max(time.perf_counter() - self.t_start, 1e-9)
        data.update(elapsed_seconds=elapsed, games_per_second=self.games / elapsed,
                    positions_per_second=self.tonkai_positions / elapsed)
        for name in ("game_length", "legal_moves", "enemy_moves", "value_loss",
                     "predict_loss", "piece_loss", "top1", "top3", "surprise",
                     "entropy", "true_move_prob"):
            window = getattr(self, name)
            data[name] = {"mean": window.mean, "std": window.std, "count": len(window)}
        data["piece_value"] = {
            name: {"mean": window.mean, "std": window.std, "count": len(window)}
            for name, window in self.piece_value.items()
        }
        return data
