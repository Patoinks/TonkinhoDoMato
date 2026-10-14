"""Compact ring buffers.

Positions are stored packed (99 bytes each), never as Python objects:

    value replay   200k x 99 B          ~ 19 MB
    predict replay  20k x 48 x 99 B     ~ 95 MB
"""

from __future__ import annotations

import numpy as np

from .encoding import INPUT_SIZE, PACKED_SIZE, unpack_many


class ReplayBuffer:
    """Positions seen by TonkAI, each labelled with the game's real result."""

    def __init__(self, capacity: int, rng: np.random.Generator | None = None):
        self.capacity = int(capacity)
        self.pos = np.zeros((self.capacity, PACKED_SIZE), dtype=np.uint8)
        self.result = np.zeros(self.capacity, dtype=np.uint8)
        self.size = 0
        self.cursor = 0
        self.total_added = 0
        self.rng = rng or np.random.default_rng()

    def __len__(self) -> int:
        return self.size

    def add(self, packed: np.ndarray, result: int) -> None:
        i = self.cursor
        self.pos[i] = packed
        self.result[i] = result
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def add_many(self, packed: np.ndarray, result: int) -> None:
        n = packed.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            packed = packed[-self.capacity:]
            n = packed.shape[0]
        end = self.cursor + n
        if end <= self.capacity:
            self.pos[self.cursor:end] = packed
            self.result[self.cursor:end] = result
        else:
            first = self.capacity - self.cursor
            self.pos[self.cursor:] = packed[:first]
            self.result[self.cursor:] = result
            self.pos[: n - first] = packed[first:]
            self.result[: n - first] = result
        self.cursor = end % self.capacity
        self.size = min(self.size + n, self.capacity)
        self.total_added += n

    def sample(self, batch_size: int):
        """-> (x float32 [B, 782], result int64 [B])"""
        idx = self.rng.integers(0, self.size, size=batch_size)
        x = unpack_many(self.pos[idx])
        return x, self.result[idx].astype(np.int64)


class PredictBuffer:
    """One sample = every legal opponent reply, plus which one was played.

    Positions with more than `max_candidates` replies are subsampled (the real
    move is always kept). `truncated` counts how often that happened.
    """

    def __init__(self, capacity: int, max_candidates: int,
                 rng: np.random.Generator | None = None):
        self.capacity = int(capacity)
        self.k = int(max_candidates)
        self.cand = np.zeros((self.capacity, self.k, PACKED_SIZE), dtype=np.uint8)
        self.count = np.zeros(self.capacity, dtype=np.int16)
        self.true_idx = np.zeros(self.capacity, dtype=np.int16)
        self.size = 0
        self.cursor = 0
        self.total_added = 0
        self.truncated = 0
        self.rng = rng or np.random.default_rng()

    def __len__(self) -> int:
        return self.size

    def add(self, packed_candidates: np.ndarray, true_index: int) -> None:
        n = packed_candidates.shape[0]
        if n < 1:
            return
        if n > self.k:
            self.truncated += 1
            others = np.delete(np.arange(n), true_index)
            keep = self.rng.choice(others, size=self.k - 1, replace=False)
            order = np.concatenate([[true_index], keep])
            self.rng.shuffle(order)
            packed_candidates = packed_candidates[order]
            true_index = int(np.where(order == true_index)[0][0])
            n = self.k

        i = self.cursor
        self.cand[i, :n] = packed_candidates
        if n < self.k:
            self.cand[i, n:] = 0
        self.count[i] = n
        self.true_idx[i] = true_index
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def sample(self, batch_size: int):
        """-> (x float32 [B, K, 782], mask bool [B, K], true_idx int64 [B])"""
        idx = self.rng.integers(0, self.size, size=batch_size)
        flat = self.cand[idx].reshape(-1, PACKED_SIZE)
        x = unpack_many(flat).reshape(batch_size, self.k, INPUT_SIZE)
        counts = self.count[idx].astype(np.int64)
        mask = np.arange(self.k)[None, :] < counts[:, None]
        return x, mask, self.true_idx[idx].astype(np.int64)

    def sample_valid(self, batch_size: int):
        """Decode only present candidates, in the same row order as sample()."""
        idx = self.rng.integers(0, self.size, size=batch_size)
        mask = np.arange(self.k)[None, :] < self.count[idx, None]
        rows, cols = np.nonzero(mask)
        x = unpack_many(self.cand[idx[rows], cols])
        return x, mask, self.true_idx[idx].astype(np.int64)
