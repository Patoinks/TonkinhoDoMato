"""Where TonkAI was wrong about the opponent.

V0 only collects. Nothing in here influences a decision - that is deliberate,
Theory comes later and will be built out of these recurring mistakes.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class TheoryEvent:
    fen: str                     # position the opponent was about to move in
    ply: int
    tonkai_move: str             # the move TonkAI had just played
    actual_move: str             # what the opponent actually did
    actual_prob: float           # what TonkAI gave it
    surprise: float              # -log(actual_prob)
    entropy: float
    top_predictions: list        # [(uci, prob), ...] a few of them
    result: int = -1             # filled in when the game ends
    latent: np.ndarray | None = field(default=None, repr=False)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("latent")
        return d


class TheoryBuffer:
    def __init__(self, capacity: int = 10_000, keep_latents: bool = True):
        self.capacity = capacity
        self.events: deque[TheoryEvent] = deque(maxlen=capacity)
        self.keep_latents = keep_latents
        self.total_seen = 0

    def __len__(self) -> int:
        return len(self.events)

    def extend(self, events, result: int) -> None:
        """Add a game's surprising moments once the result is known."""
        for ev in events:
            ev.result = result
            if not self.keep_latents:
                ev.latent = None
            self.events.append(ev)
            self.total_seen += 1

    def mean_surprise(self) -> float:
        if not self.events:
            return 0.0
        return float(np.mean([e.surprise for e in self.events]))

    def dump_json(self, path: str, limit: int | None = None) -> int:
        items = list(self.events)
        if limit:
            items = items[-limit:]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([e.to_json() for e in items], fh, indent=1)
        return len(items)


@dataclass
class SacrificeEvent:
    """A capture where TonkAI gave up a piece it valued more than the one it took.

    The values are TonkAI's own contextual estimates, not chess piece values.
    """

    fen: str
    ply: int
    move: str
    self_piece: str
    enemy_piece: str
    self_piece_value: float
    enemy_piece_value: float
    utility_before: float
    utility_after: float
    win_before: float
    win_after: float
    result: int = -1

    def to_json(self) -> dict:
        return asdict(self)


class SacrificeLog:
    def __init__(self, capacity: int = 5_000):
        self.events: deque[SacrificeEvent] = deque(maxlen=capacity)
        self.total = 0

    def __len__(self) -> int:
        return len(self.events)

    def extend(self, events, result: int) -> None:
        for ev in events:
            ev.result = result
            self.events.append(ev)
            self.total += 1

    def dump_json(self, path: str, limit: int | None = None) -> int:
        items = list(self.events)
        if limit:
            items = items[-limit:]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([e.to_json() for e in items], fh, indent=1)
        return len(items)
