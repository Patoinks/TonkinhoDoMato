"""Compact episode replay for recurrent survival learning.

An episode contains the initial state followed by every played half-move. No
win/draw labels or discounted targets are stored here: the learner combines
``remaining``, ``terminal_lost`` and ``terminated`` with its own discount.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import operator
import re

import numpy as np

from .encoding import PACKED_SIZE


@dataclass(frozen=True)
class SequenceEpisode:
    packed: np.ndarray                 # uint8 [T, 99], initial state + all plies
    lost: bool
    terminated: bool                   # False means a truncated, unfinished game
    opponent: str = ""
    surprises: np.ndarray | None = None  # nonnegative [T], aligned with states
    actions: tuple[str, ...] = ()        # optional UCI [T-1], chronological
    initial_fen: str = ""                # empty means the standard initial board
    result_value: float | None = None    # observed win=1, draw=.5, loss=0; never truncation
    proxies: np.ndarray | None = None    # "shaped" objective: victory measure in [0, 1] per state


def _integer(value, name, minimum=0):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _copy_episode(episode):
    if not isinstance(episode, SequenceEpisode):
        raise TypeError("add expects a SequenceEpisode")
    packed = np.asarray(episode.packed)
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] != PACKED_SIZE or not len(packed):
        raise ValueError(f"packed must be a nonempty uint8 array [T, {PACKED_SIZE}]")
    if not isinstance(episode.lost, (bool, np.bool_)) or not isinstance(episode.terminated, (bool, np.bool_)):
        raise ValueError("lost and terminated must be booleans")
    if episode.lost and not episode.terminated:
        raise ValueError("a lost episode must be terminated, not truncated")
    result = getattr(episode, "result_value", None)
    if result is not None and (result not in (0.0, .5, 1.0) or not episode.terminated
                               or episode.lost != (result == 0)):
        raise ValueError("result_value must match the observed outcome")
    if not isinstance(episode.opponent, str) or not isinstance(episode.initial_fen, str):
        raise ValueError("opponent and initial_fen must be strings")
    actions = tuple(episode.actions)
    if actions and (len(actions) != len(packed) - 1 or not all(
            isinstance(action, str) and re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", action)
            for action in actions)):
        raise ValueError("actions must contain one UCI string per played half-move")
    surprises = None
    if episode.surprises is not None:
        surprises = np.array(episode.surprises, dtype=np.float32, copy=True)
        if surprises.shape != (len(packed),) or not np.isfinite(surprises).all() or (surprises < 0).any():
            raise ValueError("surprises must be finite nonnegative values aligned with all T states")
        surprises.setflags(write=False)
    proxies = None
    if getattr(episode, "proxies", None) is not None:
        proxies = np.array(episode.proxies, dtype=np.float32, copy=True)
        if (proxies.shape != (len(packed),) or not np.isfinite(proxies).all()
                or (proxies < 0).any() or (proxies > 1).any()):
            raise ValueError("proxies must be finite values in [0, 1] aligned with all T states")
        proxies.setflags(write=False)
    packed = np.array(packed, dtype=np.uint8, order="C", copy=True)
    packed.setflags(write=False)
    return SequenceEpisode(packed, bool(episode.lost), bool(episode.terminated),
                           episode.opponent, surprises, actions, episode.initial_fen, result, proxies)


class SequenceReplay:
    """Whole-episode FIFO storage with bounded, mixed-priority start sampling.

    Every retained state is an eligible learning start, including a real
    terminal state. Sampling probability is uniform_floor / N plus the
    remaining mass proportional to 1 + clipped surprise/error. Default weights
    therefore lie in [1, 11]; even low-priority and newly added states retain
    probability at least 0.2 / N. No sampling operation crosses episodes.
    """

    def __init__(self, capacity_positions=200_000, seed=42, rng=None,
                 uniform_floor=0.2, priority_cap=10.0, importance_beta=1.0):
        self.capacity_positions = _integer(capacity_positions, "capacity_positions", 1)
        self.uniform_floor = float(uniform_floor)
        self.priority_cap = float(priority_cap)
        self.importance_beta = float(importance_beta)
        if not np.isfinite(self.uniform_floor) or not 0 < self.uniform_floor <= 1:
            raise ValueError("uniform_floor must be in (0, 1]")
        if not np.isfinite(self.priority_cap) or self.priority_cap < 0:
            raise ValueError("priority_cap must be finite and nonnegative")
        if not np.isfinite(self.importance_beta) or not 0 <= self.importance_beta <= 1:
            raise ValueError("importance_beta must be in [0, 1]")
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        if not isinstance(self.rng, np.random.Generator):
            raise TypeError("rng must be a numpy Generator")
        self._episodes: dict[int, SequenceEpisode] = {}
        self._priorities: dict[int, np.ndarray] = {}
        self._next_episode_id = 0
        self.size = 0
        self.total_added = 0
        self._sampling_cache = None
        self._layout_cache = None

    def __len__(self):
        return self.size

    @property
    def episode_count(self):
        return len(self._episodes)

    def add(self, episode):
        """Copy one episode and return its monotonic ID; evict whole old episodes."""
        stored = _copy_episode(episode)
        n = len(stored.packed)
        if n > self.capacity_positions:
            raise ValueError("episode exceeds replay capacity; it cannot be split without losing history")
        while self.size + n > self.capacity_positions:
            oldest = next(iter(self._episodes))
            self.size -= len(self._episodes.pop(oldest).packed)
            del self._priorities[oldest]
        episode_id = self._next_episode_id
        self._next_episode_id += 1
        self._episodes[episode_id] = stored
        priorities = np.ones(n, dtype=np.float64)
        if stored.surprises is not None:
            priorities += np.minimum(stored.surprises.astype(np.float64), self.priority_cap)
        self._priorities[episode_id] = priorities
        self.size += n
        self.total_added += n
        self._sampling_cache = None
        self._layout_cache = None
        return episode_id

    def _distribution(self):
        if self._sampling_cache is None:
            if self._layout_cache is None:
                ids = np.fromiter(self._episodes, dtype=np.int64)
                lengths = np.array([len(episode.packed) for episode in self._episodes.values()], dtype=np.int64)
                ends = np.cumsum(lengths)
                offsets = ends - lengths
                priorities = np.concatenate(list(self._priorities.values()))
                positions = dict(zip(map(int, ids), map(int, offsets)))
                self._layout_cache = ids, ends, offsets, priorities, positions
            ids, ends, offsets, priorities, _ = self._layout_cache
            probabilities = self.uniform_floor / self.size + (1.0 - self.uniform_floor) * priorities / priorities.sum()
            self._sampling_cache = probabilities, ids, ends, offsets
        return self._sampling_cache

    def sample(self, batch_size, burn_in=16, sequence_length=32, future_horizon=8):
        """Sample padded windows and metadata; no survival targets are invented.

        ``learn_offset`` is always burn_in. ``starts`` indexes each episode's
        first learning state; ``state_indices`` and ``remaining`` use -1 for
        padding. ``mask`` includes actual terminal states. ``future_mask`` is
        [B, sequence_length] for the target exactly future_horizon states ahead,
        including a real terminal target but never padding.

        Recurrent state starts at zero at the first available burn-in state.
        When memory_is_truncated is true this is only a finite-history
        approximation; no hidden state from collection time is stored or reused.
        ``episodes`` contains read-only stored references for legal action replay.
        """
        b = _integer(batch_size, "batch_size", 1)
        burn_in = _integer(burn_in, "burn_in")
        sequence_length = _integer(sequence_length, "sequence_length", 1)
        future_horizon = _integer(future_horizon, "future_horizon")
        if not self.size:
            raise ValueError("cannot sample an empty sequence replay")
        probabilities, ids, ends, offsets = self._distribution()
        flat_indices = self.rng.choice(self.size, size=b, p=probabilities)
        slots = np.searchsorted(ends, flat_indices, side="right")
        episode_ids = ids[slots]
        starts = flat_indices - offsets[slots]
        episodes = [self._episodes[int(episode_id)] for episode_id in episode_ids]
        lengths = np.array([len(episode.packed) for episode in episodes], dtype=np.int64)
        total_length = burn_in + sequence_length + future_horizon
        indices = starts[:, None] + np.arange(total_length)[None, :] - burn_in
        mask = (indices >= 0) & (indices < lengths[:, None])
        packed = np.zeros((b, total_length, PACKED_SIZE), dtype=np.uint8)
        for row, episode in enumerate(episodes):
            packed[row, mask[row]] = episode.packed[indices[row, mask[row]]]
        state_indices = np.where(mask, indices, -1).astype(np.int64)
        remaining = np.where(mask, lengths[:, None] - 1 - indices, -1).astype(np.int64)
        learning_mask = mask[:, burn_in:burn_in + sequence_length].copy()
        future_indices = starts[:, None] + np.arange(sequence_length)[None, :] + future_horizon
        future_mask = learning_mask & (future_indices < lengths[:, None])
        selected_probabilities = probabilities[flat_indices]
        importance_weights = (probabilities.min() / selected_probabilities) ** self.importance_beta
        return {
            "packed": packed, "mask": mask, "state_indices": state_indices,
            "remaining": remaining, "episode_id": episode_ids, "starts": starts,
            "episode_lengths": lengths, "learn_offset": burn_in,
            "prefix_lengths": np.minimum(starts, burn_in),
            "memory_is_truncated": starts > burn_in,
            "learning_mask": learning_mask, "future_mask": future_mask,
            "terminal_lost": np.array([episode.lost for episode in episodes], dtype=bool),
            "terminated": np.array([episode.terminated for episode in episodes], dtype=bool),
            "sampling_probabilities": selected_probabilities.copy(),
            "importance_weights": importance_weights.astype(np.float32),
            "sample_keys": [(int(episode_id), int(start)) for episode_id, start in zip(episode_ids, starts)],
            "episodes": episodes,
        }

    def update_priorities(self, keys, errors):
        """Update learning-start priorities; stale evicted keys are harmless.

        Signed errors use their absolute magnitude. Repeated keys in one batch
        use the largest error, independent of their order. No RNG is consumed.
        """
        keys = list(keys)
        errors = np.asarray(errors, dtype=np.float64)
        if errors.shape != (len(keys),) or not np.isfinite(errors).all():
            raise ValueError("errors must be one finite scalar per sample key")
        updates = {}
        for key, error in zip(keys, errors):
            if len(key) != 2:
                raise ValueError("sample keys must be (episode_id, start)")
            episode_id = _integer(key[0], "episode_id")
            start = _integer(key[1], "start")
            if episode_id not in self._episodes:
                continue
            if start >= len(self._episodes[episode_id].packed):
                raise ValueError("sample start is outside its retained episode")
            weight = 1.0 + min(abs(float(error)), self.priority_cap)
            updates[(episode_id, start)] = max(updates.get((episode_id, start), 1.0), weight)
        for (episode_id, start), weight in updates.items():
            self._priorities[episode_id][start] = weight
            if self._layout_cache is not None:
                _, _, _, flat, positions = self._layout_cache
                flat[positions[episode_id] + start] = weight
        if updates:
            self._sampling_cache = None

    def state_dict(self):
        """Independent, pickle-friendly snapshot including exact sampling RNG."""
        episodes = []
        for episode_id, episode in self._episodes.items():
            episodes.append({
                "id": episode_id, "packed": episode.packed.copy(),
                "lost": episode.lost, "terminated": episode.terminated,
                "opponent": episode.opponent, "initial_fen": episode.initial_fen,
                "surprises": None if episode.surprises is None else episode.surprises.copy(),
                "actions": episode.actions, "priorities": self._priorities[episode_id].copy(),
                "result_value": getattr(episode, "result_value", None),
                "proxies": None if episode.proxies is None else episode.proxies.copy(),
            })
        return {
            "version": 1, "capacity_positions": self.capacity_positions,
            "uniform_floor": self.uniform_floor, "priority_cap": self.priority_cap,
            "importance_beta": self.importance_beta,
            "next_episode_id": self._next_episode_id, "total_added": self.total_added,
            "episodes": episodes, "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        """Validate before replacing storage; preserve shared RNG object if possible."""
        if state.get("version") != 1:
            raise ValueError("unsupported sequence replay state version")
        rng_state = copy.deepcopy(state["rng"])
        bit_generator_type = getattr(np.random, rng_state.get("bit_generator", ""), None)
        if not isinstance(bit_generator_type, type) or not issubclass(bit_generator_type, np.random.BitGenerator):
            raise ValueError("unsupported replay RNG bit generator")
        generator = np.random.Generator(bit_generator_type())
        generator.bit_generator.state = rng_state
        restored = SequenceReplay(state["capacity_positions"], rng=generator,
                                  uniform_floor=state["uniform_floor"], priority_cap=state["priority_cap"],
                                  importance_beta=state["importance_beta"])
        for row in state["episodes"]:
            episode_id = _integer(row["id"], "episode_id")
            if episode_id in restored._episodes:
                raise ValueError("duplicate episode ID in replay state")
            episode = _copy_episode(SequenceEpisode(row["packed"], row["lost"], row["terminated"],
                                                    row["opponent"], row["surprises"], tuple(row["actions"]), row["initial_fen"], row.get("result_value"),
                                                    row.get("proxies")))
            priorities = np.array(row["priorities"], dtype=np.float64, copy=True)
            if priorities.shape != (len(episode.packed),) or not np.isfinite(priorities).all() or (priorities < 1).any() or (priorities > 1 + restored.priority_cap).any():
                raise ValueError("invalid priorities in replay state")
            restored._episodes[episode_id] = episode
            restored._priorities[episode_id] = priorities
            restored.size += len(episode.packed)
        restored._next_episode_id = _integer(state["next_episode_id"], "next_episode_id")
        restored.total_added = _integer(state["total_added"], "total_added")
        if restored.size > restored.capacity_positions or restored.total_added < restored.size:
            raise ValueError("replay state exceeds its capacity or total count")
        if restored._episodes and restored._next_episode_id <= max(restored._episodes):
            raise ValueError("next_episode_id must exceed all retained IDs")
        if type(self.rng.bit_generator) is bit_generator_type:
            self.rng.bit_generator.state = rng_state
            restored.rng = self.rng
        self.__dict__.update(restored.__dict__)
