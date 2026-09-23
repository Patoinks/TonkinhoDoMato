from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from tonkai.encoding import PACKED_SIZE
from tonkai.sequence_replay import SequenceEpisode, SequenceReplay


def episode(length, marker=1, *, lost=False, terminated=True, surprises=None, actions=()):
    packed = np.full((length, PACKED_SIZE), marker, dtype=np.uint8)
    packed[:, 1] = np.arange(length, dtype=np.uint8)
    return SequenceEpisode(packed, lost, terminated, f"opponent-{marker}", surprises, actions)


def test_windows_never_cross_episodes_and_return_stored_readonly_references():
    replay = SequenceReplay(100, seed=5)
    original = episode(5, 11, lost=True)
    ids = [replay.add(original), replay.add(episode(8, 22, terminated=False))]
    original.packed[:] = 77
    batch = replay.sample(100, burn_in=3, sequence_length=6, future_horizon=2)
    assert batch["packed"].shape == (100, 11, PACKED_SIZE)
    assert batch["packed"].dtype == np.uint8
    assert batch["mask"].dtype == bool
    assert set(batch["episode_id"].tolist()) == set(ids)
    for row, stored in enumerate(batch["episodes"]):
        valid = batch["mask"][row]
        indices = batch["state_indices"][row, valid]
        np.testing.assert_array_equal(batch["packed"][row, valid], stored.packed[indices])
        assert np.all(batch["packed"][row, ~valid] == 0)
        assert np.all(batch["state_indices"][row, ~valid] == -1)
        assert np.all(batch["remaining"][row, ~valid] == -1)
        np.testing.assert_array_equal(batch["remaining"][row, valid], len(stored.packed) - 1 - indices)
        assert batch["terminal_lost"][row] == stored.lost
        assert batch["terminated"][row] == stored.terminated
        assert stored.packed[0, 0] in (11, 22)
        assert not stored.packed.flags.writeable
        same_episode_rows = np.flatnonzero(batch["episode_id"] == batch["episode_id"][row])
        assert all(batch["episodes"][int(other)] is stored for other in same_episode_rows)


def test_burnin_left_padding_end_padding_and_terminal_future_target():
    replay = SequenceReplay(10, seed=17)
    replay.add(episode(4, lost=True))
    batch = replay.sample(128, burn_in=3, sequence_length=4, future_horizon=2)
    assert batch["learn_offset"] == 3
    first = int(np.flatnonzero(batch["starts"] == 0)[0])
    last = int(np.flatnonzero(batch["starts"] == 3)[0])
    assert batch["state_indices"][first].tolist() == [-1, -1, -1, 0, 1, 2, 3, -1, -1]
    assert batch["state_indices"][last].tolist() == [0, 1, 2, 3, -1, -1, -1, -1, -1]
    assert batch["prefix_lengths"][first] == 0
    assert batch["prefix_lengths"][last] == 3
    assert batch["learning_mask"][first].tolist() == [True, True, True, True]
    assert batch["learning_mask"][last].tolist() == [True, False, False, False]
    # State 1's +2 target is the real terminal state 3, and remains valid.
    assert batch["future_mask"][first].tolist() == [True, True, False, False]
    assert batch["remaining"][first, 3:7].tolist() == [3, 2, 1, 0]
    assert not batch["memory_is_truncated"].any()


def test_reused_layout_preserves_sampling_through_priority_updates_eviction_and_restore():
    quick, reference = SequenceReplay(70, seed=834), SequenceReplay(70, seed=834)
    rng = np.random.default_rng(272)
    for step in range(50):
        if step % 3 == 0:
            item = episode(int(rng.integers(6, 24)), step + 1)
            assert quick.add(item) == reference.add(item)
        reference._layout_cache = reference._sampling_cache = None
        a, b = quick.sample(8, 4, 12, 4), reference.sample(8, 4, 12, 4)
        for key in a:
            if isinstance(a[key], np.ndarray):
                np.testing.assert_array_equal(a[key], b[key])
        assert a['sample_keys'] == b['sample_keys']
        errors = rng.random(8) * 20
        quick.update_priorities(a['sample_keys'], errors)
        reference.update_priorities(b['sample_keys'], errors)
        if step == 25:
            quick.load_state_dict(reference.state_dict())


def test_finite_burnin_explicitly_marks_approximate_recurrent_history():
    replay = SequenceReplay(30, seed=42)
    replay.add(episode(20))
    batch = replay.sample(100, burn_in=2, sequence_length=3, future_horizon=0)
    np.testing.assert_array_equal(batch["memory_is_truncated"], batch["starts"] > 2)
    np.testing.assert_array_equal(batch["prefix_lengths"], np.minimum(batch["starts"], 2))
    np.testing.assert_array_equal(batch["future_mask"], batch["learning_mask"])
    assert batch["memory_is_truncated"].any()
    assert not batch["memory_is_truncated"].all()


def test_whole_episode_eviction_retains_capacity_and_never_reuses_ids():
    replay = SequenceReplay(7)
    first = replay.add(episode(4, 1))
    second = replay.add(episode(4, 2))
    assert first == 0 and second == 1
    assert len(replay) == 4 and replay.episode_count == 1 and replay.total_added == 8
    replay.update_priorities([(first, 0)], [999])  # evicted asynchronous sample
    assert set(replay.sample(20)["episode_id"]) == {second}
    with pytest.raises(ValueError, match="capacity"):
        replay.add(episode(8, 3))
    assert len(replay) == 4 and replay.total_added == 8
    assert replay.add(episode(3, 4)) == 2
    assert len(replay) == 7 and replay.episode_count == 2


def test_bounded_priorities_keep_a_uniform_floor_and_correct_importance_weights():
    replay = SequenceReplay(20, seed=21)
    episode_id = replay.add(episode(10))
    replay.update_priorities([(episode_id, 0)], [-100_000])
    batch = replay.sample(6000, burn_in=0, sequence_length=1, future_horizon=0)
    high = batch["starts"] == 0
    assert 0.43 < high.mean() < 0.49
    # Priority weights are [11, 1, ...], total 20, mixed with 20% uniform.
    np.testing.assert_allclose(batch["sampling_probabilities"][high], 0.46)
    np.testing.assert_allclose(batch["sampling_probabilities"][~high], 0.06)
    np.testing.assert_allclose(batch["importance_weights"][high], 0.06 / 0.46, rtol=1e-6)
    np.testing.assert_allclose(batch["importance_weights"][~high], 1)
    assert np.all(batch["sampling_probabilities"] >= 0.2 / len(replay))
    replay.update_priorities([(episode_id, 0)], [0])
    uniform = replay.sample(20, burn_in=0, sequence_length=1, future_horizon=0)
    np.testing.assert_allclose(uniform["sampling_probabilities"], 0.1)
    np.testing.assert_array_equal(uniform["importance_weights"], np.ones(20))


def test_initial_surprises_are_bounded_and_new_data_can_be_sampled():
    replay = SequenceReplay(30, seed=8, priority_cap=2)
    replay.add(episode(3, surprises=np.array([0, 1, 100_000])))
    initial = replay.sample(50, burn_in=0, sequence_length=1, future_horizon=0)
    expected = np.array([0.2, 1 / 3, 7 / 15])
    np.testing.assert_allclose(initial["sampling_probabilities"], expected[initial["starts"]])
    new_id = replay.add(episode(2, 4))
    mixed = replay.sample(1000, burn_in=0, sequence_length=1, future_horizon=0)
    assert (mixed["episode_id"] == new_id).any()
    assert np.all(mixed["sampling_probabilities"] >= 0.2 / len(replay))


def test_priority_updates_consume_no_rng_and_repeated_keys_use_maximum_error():
    replay = SequenceReplay(20)
    episode_id = replay.add(episode(5))
    before = copy.deepcopy(replay.rng.bit_generator.state)
    replay.update_priorities([(episode_id, 2), (episode_id, 2)], [3, 1])
    assert replay.rng.bit_generator.state == before
    assert replay.state_dict()["episodes"][0]["priorities"][2] == 4
    replay.update_priorities([(episode_id, 2)], [0])
    assert replay.state_dict()["episodes"][0]["priorities"][2] == 1
    for errors in ([float("nan")], [float("inf")], []):
        with pytest.raises(ValueError, match="errors"):
            replay.update_priorities([(episode_id, 2)], errors)


def test_state_restore_preserves_storage_metadata_priorities_and_exact_rng_continuation():
    replay = SequenceReplay(9, seed=11, uniform_floor=0.3, importance_beta=0.5)
    replay.add(episode(5, 1))
    kept = replay.add(episode(6, 2, lost=True, surprises=np.arange(6)))
    replay.add(replace(episode(3, 3, terminated=False, actions=("e2e4", "e7e5")), initial_fen="custom-root"))
    replay.update_priorities([(kept, 2)], [3.4])
    replay.sample(7)
    snapshot = replay.state_dict()
    shared_rng = np.random.default_rng(999)
    restored = SequenceReplay(1, rng=shared_rng)
    restored.load_state_dict(snapshot)
    assert restored.rng is shared_rng
    assert len(restored) == len(replay) == 9
    assert restored.episode_count == replay.episode_count == 2
    assert restored.total_added == replay.total_added == 14
    assert restored.capacity_positions == 9 and restored.uniform_floor == 0.3
    expected = replay.sample(40, burn_in=2, sequence_length=4, future_horizon=3)
    actual = restored.sample(40, burn_in=2, sequence_length=4, future_horizon=3)
    for key in expected:
        if key == "episodes":
            for a, b in zip(expected[key], actual[key]):
                np.testing.assert_array_equal(a.packed, b.packed)
                assert (a.lost, a.terminated, a.opponent, a.actions, a.initial_fen) == (b.lost, b.terminated, b.opponent, b.actions, b.initial_fen)
        elif isinstance(expected[key], np.ndarray):
            np.testing.assert_array_equal(expected[key], actual[key])
        else:
            assert expected[key] == actual[key]
    assert replay.rng.bit_generator.state == restored.rng.bit_generator.state
    assert restored.add(episode(2)) == replay.add(episode(2)) == 3
    # Snapshots and caller arrays do not alias retained replay storage.
    snapshot["episodes"][0]["packed"][:] = 99
    assert not np.all(restored.state_dict()["episodes"][0]["packed"] == 99)


def test_snapshot_restores_nondefault_numpy_bit_generator():
    replay = SequenceReplay(20, rng=np.random.Generator(np.random.Philox(7)))
    replay.add(episode(9))
    restored = SequenceReplay()
    restored.load_state_dict(replay.state_dict())
    np.testing.assert_array_equal(replay.sample(20)["starts"], restored.sample(20)["starts"])


def test_invalid_restored_state_leaves_existing_replay_unchanged():
    replay = SequenceReplay(20)
    replay.add(episode(5, 3))
    before = replay.state_dict()
    invalid = copy.deepcopy(before)
    invalid["episodes"][0]["priorities"][0] = 1000
    with pytest.raises(ValueError, match="priorities"):
        replay.load_state_dict(invalid)
    np.testing.assert_array_equal(replay.state_dict()["episodes"][0]["packed"], before["episodes"][0]["packed"])
    assert replay.rng.bit_generator.state == before["rng"]


@pytest.mark.parametrize("invalid", [
    SequenceEpisode(np.zeros((0, PACKED_SIZE), dtype=np.uint8), False, True),
    SequenceEpisode(np.zeros((3, PACKED_SIZE), dtype=np.float32), False, True),
    SequenceEpisode(np.zeros((3, 10), dtype=np.uint8), False, True),
    episode(3, lost=True, terminated=False),
    episode(3, surprises=np.array([1.0, 2.0])),
    episode(3, surprises=np.array([0.0, -1.0, 2.0])),
    episode(3, surprises=np.array([0.0, np.nan, 2.0])),
    episode(3, actions=("e2e4",)),
    episode(3, actions=("e2e4", "not-uci")),
])
def test_invalid_episode_is_rejected_without_changing_replay(invalid):
    replay = SequenceReplay(20)
    with pytest.raises(ValueError):
        replay.add(invalid)
    assert len(replay) == replay.total_added == replay.episode_count == 0


def test_default_shapes_and_truncated_tail_are_distinct_from_terminal_loss():
    replay = SequenceReplay()
    episode_id = replay.add(episode(1, terminated=False))
    batch = replay.sample(2)
    assert batch["packed"].shape == (2, 56, 99)
    assert batch["mask"].sum() == 2
    assert batch["mask"][:, 16].all()
    assert not batch["terminated"].any() and not batch["terminal_lost"].any()
    assert batch["remaining"][:, 16].tolist() == [0, 0]
    assert not batch["future_mask"].any()
    assert batch["sample_keys"] == [(episode_id, 0), (episode_id, 0)]


def test_empty_replay_and_invalid_sample_sizes():
    replay = SequenceReplay()
    with pytest.raises(ValueError, match="empty"):
        replay.sample(1)
    replay.add(episode(2))
    for kwargs in ({"batch_size": 0}, {"batch_size": 1, "burn_in": -1},
                   {"batch_size": 1, "sequence_length": 0}, {"batch_size": 1, "future_horizon": -1}):
        with pytest.raises(ValueError):
            replay.sample(**kwargs)
