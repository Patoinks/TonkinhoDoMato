from __future__ import annotations

import numpy as np
import pytest

from tonkai.encoding import INPUT_SIZE, PACKED_SIZE
from tonkai.replay import PredictBuffer, ReplayBuffer


def _packed(seed, n=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, PACKED_SIZE), dtype=np.uint8)


def test_replay_grows_then_caps():
    b = ReplayBuffer(10)
    for i in range(25):
        b.add(_packed(i)[0], i % 3)
    assert len(b) == 10
    assert b.total_added == 25


def test_replay_wraparound_keeps_the_newest():
    b = ReplayBuffer(4)
    rows = _packed(0, 6)
    for i, r in enumerate(rows):
        b.add(r, 0)
    stored = {bytes(b.pos[i]) for i in range(4)}
    assert stored == {bytes(r) for r in rows[2:]}


def test_add_many_wraps_correctly():
    b = ReplayBuffer(5)
    b.add_many(_packed(1, 3), 0)
    assert len(b) == 3 and b.cursor == 3
    rows = _packed(2, 4)
    b.add_many(rows, 2)
    assert len(b) == 5 and b.cursor == 2
    assert b.result[b.cursor - 1] == 2
    stored = {bytes(b.pos[i]) for i in range(5)}
    assert {bytes(r) for r in rows} <= stored


def test_add_many_larger_than_capacity():
    b = ReplayBuffer(3)
    rows = _packed(5, 10)
    b.add_many(rows, 1)
    assert len(b) == 3
    assert {bytes(b.pos[i]) for i in range(3)} == {bytes(r) for r in rows[-3:]}


def test_sample_shapes_and_labels():
    b = ReplayBuffer(50)
    b.add_many(_packed(3, 20), 1)
    x, y = b.sample(8)
    assert x.shape == (8, INPUT_SIZE) and x.dtype == np.float32
    assert y.shape == (8,) and set(y.tolist()) == {1}


def test_predict_buffer_roundtrip():
    b = PredictBuffer(8, max_candidates=10)
    cands = _packed(7, 4)
    b.add(cands, true_index=2)
    assert len(b) == 1 and b.count[0] == 4 and b.true_idx[0] == 2
    x, mask, ti = b.sample(1)
    assert x.shape == (1, 10, INPUT_SIZE)
    assert mask[0].tolist() == [True] * 4 + [False] * 6
    assert ti[0] == 2


def test_predict_buffer_truncates_but_keeps_the_real_move():
    b = PredictBuffer(4, max_candidates=5, rng=np.random.default_rng(0))
    cands = _packed(9, 20)
    b.add(cands, true_index=13)
    assert b.truncated == 1
    assert b.count[0] == 5
    kept = b.true_idx[0]
    assert bytes(b.cand[0, kept]) == bytes(cands[13])


def test_predict_buffer_wraps():
    b = PredictBuffer(3, max_candidates=4)
    for i in range(7):
        b.add(_packed(i, 2), 0)
    assert len(b) == 3 and b.total_added == 7


def test_empty_candidate_set_is_ignored():
    b = PredictBuffer(3, max_candidates=4)
    b.add(np.zeros((0, PACKED_SIZE), dtype=np.uint8), 0)
    assert len(b) == 0
