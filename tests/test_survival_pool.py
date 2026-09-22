"""Real spawned actors; the optional UCI binary is never a training teacher."""

from copy import deepcopy
import os
from pathlib import Path
import time

import chess
import numpy as np
import pytest
import torch

from tonkai.encoding import encode, pack_many
from tonkai.survival_agent import NeuralOpponent, SurvivalAgent, play_episode
from tonkai.survival_config import SurvivalConfig
from tonkai.survival_pool import SurvivalActorError, SurvivalActorPool
from tonkai.temporal_model import SurvivalNet


@pytest.fixture
def cfg():
    return SurvivalConfig(hidden_size=24, latent_size=16, head_hidden=8,
                          memory_size=8, ensemble_size=3, future_horizons=(1, 3),
                          max_plies=6, win_pgn_path="")


@pytest.fixture
def engine_path():
    configured = os.environ.get("TONKAI_UCI_TEST_ENGINE")
    default = (Path(__file__).resolve().parents[1] / "engines" / "stockfish-11"
               / "stockfish-11-win" / "Windows" / "stockfish_20011801_x64.exe")
    path = Path(configured) if configured else default
    if not path.is_file():
        if configured:
            pytest.fail(f"TONKAI_UCI_TEST_ENGINE does not exist: {path}")
        pytest.skip("Install the optional engine with scripts/setup_opponent.ps1")
    return path


@pytest.fixture
def net(cfg):
    # Scope the thread limit to this test process, never the live learner.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(192)
    yield SurvivalNet(cfg).eval()
    torch.set_num_threads(previous)


def assert_legal_episode(result, cfg):
    worker_id, episode, board, self_color, stats = result
    assert isinstance(worker_id, int)
    assert isinstance(self_color, bool)
    replay = chess.Board()
    positions = [encode(replay, self_color)]
    for action in episode.actions:
        move = chess.Move.from_uci(action)
        assert move in replay.legal_moves
        replay.push(move)
        positions.append(encode(replay, self_color))
    assert replay.fen() == board.fen()
    assert replay.move_stack == board.move_stack
    assert len(board.move_stack) == stats["plies"] <= cfg.max_plies
    np.testing.assert_array_equal(episode.packed, pack_many(np.stack(positions)))
    assert stats["opponent"] == episode.opponent
    assert stats["actor_seconds"] > 0
    assert bool(board.outcome(claim_draw=True)) == episode.terminated
    assert np.isfinite(episode.surprises).all()


def test_two_external_actors_return_legal_complete_history_and_close(cfg, net, engine_path):
    with SurvivalActorPool(cfg, engine_path, 2) as pool:
        assert pool.idle_ids == (0, 1)
        assert pool.startup_seconds > 0
        assert len({info["pid"] for info in pool.worker_info.values()}) == 2
        assert len({info["engine_pid"] for info in pool.worker_info.values()}) == 2
        for info in pool.worker_info.values():
            assert info["torch_threads"] == 1
            assert info["engine"]["elo"] == 1600
            assert info["engine"]["move_time"] == 0.02
            assert info["engine"]["threads"] == 1
        for worker_id in pool.idle_ids:
            pool.submit(worker_id, net.state_dict(), "external", None, "Stockfish", 81 + worker_id)
        assert pool.idle_ids == ()
        results = [pool.receive(timeout=30), pool.receive(timeout=30)]
        assert {result[0] for result in results} == {0, 1}
        for result in results:
            assert_legal_episode(result, cfg)
            assert result[-1]["opponent_kind"] == "external"
        assert pool.idle_ids == (0, 1)
        processes = list(pool.processes)
    assert all(not process.is_alive() and process.exitcode == 0 for process in processes)
    assert pool.idle_ids == ()
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.receive()


def test_neural_jobs_reset_history_across_workers_and_snapshot_inputs(cfg, net, engine_path):
    state = deepcopy(net.state_dict())
    rng = np.random.default_rng(499)
    local = play_episode(SurvivalAgent(net, cfg), NeuralOpponent(deepcopy(net), cfg, rng),
                         cfg, rng, opponent_name="frozen")
    with SurvivalActorPool(cfg, engine_path, 2) as pool:
        engine_ids = [info["engine_pid"] for info in pool.worker_info.values()]
        for worker_id in pool.idle_ids:
            pool.submit(worker_id, state, "historical", state, "frozen", 499)
        # submit must take a by-value snapshot before the learner can mutate it.
        for value in state.values():
            value.fill_(float("nan"))
        first = [pool.receive(timeout=30), pool.receive(timeout=30)]
        pool.submit(0, net.state_dict(), "current", net.state_dict(), "frozen", 499)
        repeated = pool.receive(timeout=30)
        for result in [*first, repeated]:
            assert_legal_episode(result, cfg)
            assert result[1].actions == local[0].actions
            assert result[3] == local[2]
            np.testing.assert_array_equal(result[1].packed, local[0].packed)
            np.testing.assert_allclose(result[1].surprises, local[0].surprises, rtol=0, atol=1e-6)
        assert [info["engine_pid"] for info in pool.worker_info.values()] == engine_ids


def test_busy_actor_timeout_validation_and_cancellation_are_bounded(cfg, net, engine_path):
    cfg.max_plies = 1000
    pool = SurvivalActorPool(cfg, engine_path, 1)
    try:
        with pytest.raises(RuntimeError, match="No survival games"):
            pool.receive()
        with pytest.raises(ValueError, match="opponent_kind"):
            pool.submit(0, net.state_dict(), "invalid", None, "invalid", 1)
        with pytest.raises(TypeError, match="opponent_state"):
            pool.submit(0, net.state_dict(), "historical", None, "missing", 1)
        pool.submit(0, net.state_dict(), "external", None, "Stockfish", 199)
        with pytest.raises(RuntimeError, match="in flight"):
            pool.submit(0, net.state_dict(), "external", None, "Stockfish", 200)
        with pytest.raises(TimeoutError):
            pool.receive(timeout=0)
        assert pool.idle_ids == ()
        started = time.monotonic()
        pool.close()
        assert time.monotonic() - started < 12
        assert all(not process.is_alive() and process.exitcode == 0 for process in pool.processes)
    finally:
        pool.close()


def test_worker_load_failure_is_identified_and_cleanup_stops_only_pool(cfg, net, engine_path):
    with SurvivalActorPool(cfg, engine_path, 2) as pool:
        pool.submit(1, {}, "external", None, "Stockfish", 1)
        with pytest.raises(SurvivalActorError, match="actor 1") as failure:
            pool.receive(timeout=30)
        assert failure.value.worker_id == 1
        assert "Missing key" in str(failure.value)
        processes = list(pool.processes)
    assert all(not process.is_alive() for process in processes)


def test_startup_failure_is_identified_and_bounded(cfg, tmp_path):
    started = time.monotonic()
    with pytest.raises(SurvivalActorError, match="actor 0") as failure:
        SurvivalActorPool(cfg, tmp_path / "missing-engine.exe", 1)
    assert failure.value.worker_id == 0
    assert "FileNotFoundError" in str(failure.value)
    assert time.monotonic() - started < 20


@pytest.mark.parametrize("workers", [0, 9, True, 1.5])
def test_invalid_worker_count_fails_before_startup(cfg, workers):
    with pytest.raises(ValueError, match="workers"):
        SurvivalActorPool(cfg, "unused", workers)
