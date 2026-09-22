"""UCI contracts and an optional smoke against the locally installed engine."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import chess
import chess.engine
import pytest

from tonkai.uci_opponent import UCIOpponent


def spin(name, default, minimum, maximum):
    return chess.engine.Option(name, "spin", default, minimum, maximum, [])


class FakeEngine:
    def __init__(self):
        self.id = {"name": "Fake Stockfish", "author": "test"}
        self.options = {
            "Threads": spin("Threads", 1, 1, 512),
            "Hash": spin("Hash", 16, 1, 131072),
            "UCI_Elo": spin("UCI_Elo", 1350, 1350, 2850),
            "UCI_LimitStrength": chess.engine.Option(
                "UCI_LimitStrength", "check", False, None, None, []),
        }
        self.calls = []
        self.configured = None
        self.error = None
        self.move = chess.Move.from_uci("e2e4")
        self.quit_calls = self.close_calls = self.ping_calls = 0

    def configure(self, options):
        self.configured = dict(options)

    def ping(self):
        self.ping_calls += 1

    def play(self, board, limit, **kwargs):
        self.calls.append((board, limit, kwargs))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(move=self.move)

    def quit(self):
        self.quit_calls += 1

    def close(self):
        self.close_calls += 1


@pytest.fixture
def engine_factory(monkeypatch, tmp_path):
    binary = tmp_path / "fake-engine.exe"
    binary.write_bytes(b"not executed: fake UCI engine fixture")
    engine = FakeEngine()
    launches = []

    def launch(path, **kwargs):
        launches.append((path, kwargs))
        return engine

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", launch)
    return binary, engine, launches


def test_configures_nominal_strength_and_reuses_one_process(engine_factory):
    binary, engine, launches = engine_factory
    with UCIOpponent(binary) as opponent:
        opponent.choose(chess.Board())
        opponent.choose(chess.Board(), rng=object())
        assert len(launches) == 1
        assert launches[0][1]["timeout"] == 5.0
        assert engine.configured == {
            "Threads": 1, "Hash": 16, "UCI_Elo": 1600, "UCI_LimitStrength": True,
        }
        assert engine.ping_calls == 1
        assert engine.calls[0][1].time == 0.02
        assert engine.calls[0][2]["info"] == chess.engine.INFO_NONE
        assert engine.calls[0][2]["ponder"] is False
        assert engine.calls[0][2]["game"] is engine.calls[1][2]["game"]
        assert "nominal Elo 1600" in opponent.description
        assert opponent.info["elo_range"] == [1350, 2850]
        assert opponent.info["seedable"] is False
        metadata = opponent.info
        metadata["options"]["UCI_Elo"] = 9000
        metadata["elo_range"][0] = 0
        assert opponent.info["options"]["UCI_Elo"] == 1600
        assert opponent.info["elo_range"] == [1350, 2850]
    opponent.close()
    assert engine.quit_calls == engine.close_calls == 1


def test_passes_move_history_and_reset_marks_new_game(engine_factory):
    binary, engine, _ = engine_factory
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    before = board.fen(), list(board.move_stack)
    with UCIOpponent(binary) as opponent:
        opponent.choose(board)
        opponent.reset()
        opponent.choose(board)
    assert engine.calls[0][0] is board
    assert len(engine.calls[0][0].move_stack) == 4
    assert (board.fen(), board.move_stack) == before
    assert engine.calls[0][2]["game"] is not engine.calls[1][2]["game"]


@pytest.mark.parametrize("kwargs", [
    {"elo": 0}, {"elo": True}, {"elo": 1600.5}, {"threads": 0},
    {"hash_mb": -1}, {"move_time": 0}, {"move_time": float("nan")},
    {"move_time": float("inf")}, {"move_time": 0.0001}, {"timeout": -1},
])
def test_bad_configuration_fails_before_launch(engine_factory, kwargs):
    binary, _, launches = engine_factory
    with pytest.raises(ValueError):
        UCIOpponent(binary, **kwargs)
    assert not launches


@pytest.mark.parametrize("kwargs", [{"elo": 1200}, {"elo": 3000}, {"threads": 513}])
def test_advertised_option_ranges_are_enforced(engine_factory, kwargs):
    binary, engine, _ = engine_factory
    with pytest.raises(ValueError, match="outside engine range"):
        UCIOpponent(binary, **kwargs)
    assert engine.configured is None
    assert engine.close_calls == 1


@pytest.mark.parametrize("option", ["UCI_Elo", "UCI_LimitStrength", "Threads", "Hash"])
def test_missing_option_never_silently_plays_at_full_strength(engine_factory, option):
    binary, engine, _ = engine_factory
    del engine.options[option]
    with pytest.raises(ValueError, match="must expose"):
        UCIOpponent(binary)
    assert engine.configured is None
    assert engine.close_calls == 1


@pytest.mark.parametrize("fen", [
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate
    "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",  # mate
    "7k/8/6K1/8/8/8/8/8 w - - 0 1",  # insufficient material
    "8/8/8/8/8/8/8/8 w - - 0 1",  # invalid
])
def test_rejects_finished_or_invalid_positions(engine_factory, fen):
    binary, engine, _ = engine_factory
    with UCIOpponent(binary) as opponent:
        with pytest.raises(ValueError):
            opponent.choose(chess.Board(fen))
    assert not engine.calls


@pytest.mark.parametrize("failure", [TimeoutError(), chess.engine.EngineTerminatedError("gone")])
def test_failed_engine_is_closed_and_does_not_fall_back_to_random_moves(engine_factory, failure):
    binary, engine, _ = engine_factory
    opponent = UCIOpponent(binary)
    engine.error = failure
    with pytest.raises(type(failure)):
        opponent.choose(chess.Board())
    assert engine.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        opponent.choose(chess.Board())
    opponent.close()
    assert engine.close_calls == 1


@pytest.mark.parametrize("move", [None, chess.Move.null(), chess.Move.from_uci("e2e5")])
def test_illegal_bestmove_closes_engine(engine_factory, move):
    binary, engine, _ = engine_factory
    opponent = UCIOpponent(binary)
    engine.move = move
    with pytest.raises(chess.engine.EngineError, match="no legal move"):
        opponent.choose(chess.Board())
    assert engine.close_calls == 1


def test_close_and_reset_after_close(engine_factory):
    binary, _, _ = engine_factory
    opponent = UCIOpponent(binary)
    opponent.close()
    with pytest.raises(RuntimeError, match="closed"):
        opponent.reset()


def test_real_engine_twenty_legal_plies():
    default_path = (Path(__file__).resolve().parents[1] / "engines" / "stockfish-11"
                    / "stockfish-11-win" / "Windows" / "stockfish_20011801_x64.exe")
    configured_path = os.environ.get("TONKAI_UCI_TEST_ENGINE")
    binary = Path(configured_path) if configured_path else default_path
    if not binary.is_file():
        if configured_path:
            pytest.fail(f"TONKAI_UCI_TEST_ENGINE does not exist: {binary}")
        pytest.skip("Install the optional engine with scripts/setup_opponent.ps1")
    board = chess.Board()
    with UCIOpponent(binary) as opponent:
        for _ in range(20):
            stack = list(board.move_stack)
            move = opponent.choose(board)
            assert move in board.legal_moves
            assert board.move_stack == stack
            board.push(move)
        opponent.reset()
        assert opponent.choose(chess.Board()) in chess.Board().legal_moves
