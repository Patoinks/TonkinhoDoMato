"""A frozen opponent population selected by age and fixed mixture probabilities.

No result, rating or promotion affects retention or sampling. Snapshots retain
explicit network weights only; recurrent game memory remains with each actor.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import torch

from .temporal_model import SurvivalNet


@dataclass(frozen=True)
class PopulationSnapshot:
    id: int
    games: int
    net: SurvivalNet

    @property
    def name(self) -> str:
        return f"historical-{self.id}@{self.games}"


class Population:
    def __init__(self, cfg, initial_net: SurvivalNet, seed: int = 42):
        self.population_size = getattr(cfg, "population_size", 8)
        self.external_probability = float(getattr(cfg, "external_probability", 0.5))
        self.historical_probability = float(getattr(cfg, "historical_probability", 0.3))
        if (not isinstance(self.population_size, int) or isinstance(self.population_size, bool)
                or self.population_size < 1):
            raise ValueError("population_size must be a positive integer")
        probabilities = (self.external_probability, self.historical_probability)
        if (not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                or sum(probabilities) > 1):
            raise ValueError("opponent mixture probabilities must sum to at most one")
        self.rng = np.random.default_rng(seed)
        self.snapshots = [PopulationSnapshot(0, 0, self.current(initial_net))]
        self._next_id = 1

    def __len__(self) -> int:
        return len(self.snapshots)

    @staticmethod
    def current(net: SurvivalNet) -> SurvivalNet:
        """A separate CPU opponent, without sharing mutable weights with the learner."""
        frozen = deepcopy(net).to("cpu")
        frozen.requires_grad_(False)
        frozen.eval()
        return frozen

    def snapshot(self, net: SurvivalNet, games: int) -> None:
        if not isinstance(games, int) or isinstance(games, bool) or games < 0:
            raise ValueError("snapshot games must be a nonnegative integer")
        self.snapshots.append(PopulationSnapshot(self._next_id, games, self.current(net)))
        self._next_id += 1
        self.snapshots.sort(key=lambda entry: (entry.games, entry.id))
        self._trim()

    def _trim(self) -> None:
        while len(self.snapshots) > self.population_size:
            if self.population_size == 1:
                self.snapshots = [next(entry for entry in self.snapshots if entry.id == 0)]
                break
            # Preserve the original anchor and the most recent age. An interior
            # with closely spaced neighbours contributes least age coverage.
            # Unlike dropping the oldest, this retains different training eras.
            index = min(
                range(1, len(self.snapshots) - 1),
                key=lambda i: (
                    self.snapshots[i + 1].games - self.snapshots[i - 1].games,
                    self.snapshots[i].id,
                ),
            )
            self.snapshots.pop(index)

    def choose_kind(self, rng: np.random.Generator | None = None) -> str:
        source = self.rng if rng is None else rng
        draw = float(source.random())
        if draw < self.external_probability:
            return "external"
        if draw < self.external_probability + self.historical_probability:
            return "historical"
        return "current"

    def historical(self, rng: np.random.Generator | None = None) -> tuple[SurvivalNet, str]:
        """Uniform selection among retained ages, including the initial anchor.

        Returned models are frozen shared population entries for inference only;
        call current(model) if a separate mutable owner is needed.
        """
        source = self.rng if rng is None else rng
        entry = self.snapshots[int(source.integers(len(self.snapshots)))]
        return entry.net, entry.name

    def state_dict(self) -> dict:
        """Independent CPU tensors and complete private RNG state for checkpoints."""
        return {
            "schema_version": 1,
            "next_id": self._next_id,
            "numpy_rng": deepcopy(self.rng.bit_generator.state),
            "snapshots": [
                {
                    "id": entry.id,
                    "games": entry.games,
                    "model": {key: value.detach().cpu().clone()
                              for key, value in entry.net.state_dict().items()},
                }
                for entry in self.snapshots
            ],
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore against this architecture and configured population capacity.

        Mixture probabilities remain configuration, not learned state. A smaller
        configured capacity applies the same age-only retention rule on restore.
        Validation completes before replacing the live population.
        """
        if not isinstance(state, Mapping) or state.get("schema_version") != 1:
            raise ValueError("unsupported population checkpoint")
        rows = state.get("snapshots")
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError("population checkpoint must contain its initial anchor")
        restored = []
        ids = set()
        template = self.snapshots[0].net
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("invalid population snapshot")
            identifier, games = row.get("id"), row.get("games")
            if (not isinstance(identifier, int) or isinstance(identifier, bool) or identifier < 0
                    or identifier in ids or not isinstance(games, int) or isinstance(games, bool)
                    or games < 0 or not isinstance(row.get("model"), Mapping)):
                raise ValueError("invalid population snapshot metadata")
            net = self.current(template)
            net.load_state_dict(row["model"], strict=True)
            restored.append(PopulationSnapshot(identifier, games, net))
            ids.add(identifier)
        restored.sort(key=lambda entry: (entry.games, entry.id))
        if restored[0].id != 0 or restored[0].games != 0:
            raise ValueError("population checkpoint is missing its original anchor")
        next_id = state.get("next_id")
        if not isinstance(next_id, int) or isinstance(next_id, bool) or next_id <= max(ids):
            raise ValueError("invalid next population snapshot id")
        rng = np.random.default_rng(0)
        rng.bit_generator.state = deepcopy(state["numpy_rng"])
        self.snapshots = restored
        self._next_id = next_id
        self.rng = rng
        self._trim()
