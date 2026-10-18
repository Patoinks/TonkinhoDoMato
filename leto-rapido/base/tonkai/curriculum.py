"""Keep the opponent inside the band where TonkAI can actually learn.

The problem this solves, measured over 5000 games: against the default
opponent TonkAI loses 99.8% of its games, so "you lose" is the correct answer
in every position, the value head saturates at P(loss) = 1.0, the counterfactual
piece targets all collapse to zero, and nothing except Predict learns anything.

A constant result is not a training signal. So the opponent is moved up and
down a ladder to hold TonkAI's score somewhere it still has something to
learn from.

This teaches TonkAI nothing about chess. It only changes how hard the opponent
plays - the same three knobs that were always configurable. TonkAI never sees
which rung it is on.
"""

from __future__ import annotations

from collections import deque

# weakest first. (quiescence, depth, nodes, randomness)
LADDER = [
    # The bottom rung must be WEAK, not INCOHERENT. Two rungs with randomness
    # 1.4 and 2.0 were tried and removed: an opponent that moves at random
    # cannot convert a won position and cannot be predicted, so games ran to
    # 207 plies, 41% ended in stalemate, Predict was fitting noise and got
    # worse (14.1% -> 13.0% top1), and the contextual piece values separated
    # LESS than against a weak-but-coherent opponent (queen std 0.06 vs 0.13).
    # Weaken with depth, nodes and quiescence; keep randomness moderate.
    (False, 1, 50, 0.90),
    (False, 1, 100, 0.80),
    (False, 1, 200, 0.60),
    (False, 2, 500, 0.50),
    (True, 1, 200, 0.50),
    (True, 1, 1000, 0.40),
    (True, 2, 1000, 0.35),   # the spec's default settings
    (True, 2, 5000, 0.25),
    (True, 2, 20000, 0.15),
    (True, 3, 20000, 0.10),
]


class Curriculum:
    def __init__(self, opponent, start: int = 0, window: int = 200,
                 promote_at: float = 0.55, demote_at: float = 0.15,
                 patience: int = 200, stagnation: int = 2500):
        self.opponent = opponent
        self.rung = max(0, min(start, len(LADDER) - 1))
        self.window = window
        self.promote_at = promote_at
        self.demote_at = demote_at
        self.patience = patience
        # A draw-heavy regime parks the score near 0.4 forever: draws count
        # half, so `promote_at` is unreachable without real wins and the ladder
        # never moves. Measured: 3000 games pinned at exactly 0.41. If the
        # score is healthy but static for this long, move up anyway.
        self.stagnation = stagnation
        self.scores: deque[float] = deque(maxlen=window)
        self.since_change = 0
        self.changes = 0
        self.history: list[tuple[int, int, float]] = []
        self.apply()

    def apply(self) -> None:
        quiesce, depth, nodes, randomness = LADDER[self.rung]
        self.opponent.quiescence = quiesce
        self.opponent.depth = depth
        self.opponent.max_nodes = nodes
        self.opponent.randomness = randomness

    @property
    def score(self) -> float:
        """TonkAI's score over the recent window: win = 1, draw = 0.5."""
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def describe(self) -> str:
        quiesce, depth, nodes, randomness = LADDER[self.rung]
        return (f"rung {self.rung}/{len(LADDER) - 1} "
                f"(d{depth} n{nodes} r{randomness} "
                f"q{'on' if quiesce else 'off'})")

    def update(self, result: int, games: int) -> str | None:
        """Feed one game result. Returns a message when the rung changes."""
        self.scores.append(1.0 if result == 0 else 0.5 if result == 1 else 0.0)
        self.since_change += 1
        if len(self.scores) < self.window or self.since_change < self.patience:
            return None

        s = self.score
        old = self.rung
        stuck = (self.since_change >= self.stagnation
                 and s > self.demote_at
                 and self.rung < len(LADDER) - 1)
        if s >= self.promote_at and self.rung < len(LADDER) - 1:
            self.rung += 1
        elif stuck:
            self.rung += 1
        elif s <= self.demote_at and self.rung > 0:
            self.rung -= 1
        else:
            return None

        self.apply()
        self.scores.clear()
        self.since_change = 0
        self.changes += 1
        self.history.append((games, self.rung, s))
        arrow = "UP" if self.rung > old else "DOWN"
        return (f"[curriculum] {arrow} score {s:.2f} -> {self.describe()} "
                f"@ {games} games")
