from __future__ import annotations

from dataclasses import replace

import chess
import chess.pgn
import numpy as np
import pytest

from tonkai.agent import GameRecord, GameWorker
from tonkai.config import Config
from tonkai.metrics import Metrics
from tonkai.win_history import LongestWinRecorder


def white_win(longer=False):
    board = chess.Board()
    opening = ["Nf3", "Nf6", "Ng1", "Ng8"] if longer else []
    for san in opening + ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]:
        board.push_san(san)
    return board


def black_win():
    board = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        board.push_san(san)
    return board


def read_one(path):
    with path.open(encoding="utf-8") as stream:
        game = chess.pgn.read_game(stream)
        assert game is not None and not game.errors
        assert chess.pgn.read_game(stream) is None
    board = game.board()
    for move in game.mainline_moves():
        assert move in board.legal_moves
        board.push(move)
    return game, board


@pytest.mark.parametrize("color,make_board,result", [
    (chess.WHITE, white_win, "1-0"),
    (chess.BLACK, black_win, "0-1"),
])
def test_full_win_pgn_roundtrips_legal_moves_and_colors(tmp_path, color, make_board, result):
    path = tmp_path / "wins" / "longest.pgn"
    board = make_board()
    before = board.fen(), tuple(board.move_stack)
    assert LongestWinRecorder(path).consider(board, color, "TinyOpponent", result, "checkmate")
    game, restored = read_one(path)
    assert restored.fen() == board.fen()
    assert tuple(restored.move_stack) == tuple(board.move_stack)
    assert game.headers["White" if color else "Black"] == "TonkAI"
    assert game.headers["Black" if color else "White"] == "TinyOpponent"
    assert game.headers["Result"] == result
    assert game.headers["Termination"] == "checkmate"
    assert game.headers["PlyCount"] == str(len(board.move_stack))
    assert (board.fen(), tuple(board.move_stack)) == before
    assert not list(path.parent.glob("*.tmp"))


def test_only_strictly_longer_wins_replace_existing_game_across_restarts(tmp_path):
    path = tmp_path / "longest.pgn"
    recorder = LongestWinRecorder(path)
    assert recorder.consider(white_win(), True, "First", "1-0", "checkmate")
    first = path.read_bytes()
    assert not recorder.consider(white_win(), True, "Equal", "1-0", "checkmate")
    assert not LongestWinRecorder(path).consider(black_win(), False, "Shorter", "0-1", "checkmate")
    assert path.read_bytes() == first
    assert LongestWinRecorder(path).consider(white_win(True), True, "Longer", "1-0", "checkmate")
    game, restored = read_one(path)
    assert game.headers["Black"] == "Longer"
    assert len(restored.move_stack) == 11


@pytest.mark.parametrize("result,color", [("0-1", True), ("1-0", False), ("1/2-1/2", True)])
def test_losses_and_draws_write_nothing(tmp_path, result, color):
    path = tmp_path / "absent" / "longest.pgn"
    assert not LongestWinRecorder(path).consider(chess.Board(), color, "Human", result, "resignation")
    assert not path.parent.exists()


def test_disabled_recorder_writes_nothing_and_resignation_is_supported(tmp_path):
    board = chess.Board()
    board.push_san("e4")
    assert not LongestWinRecorder("").consider(board, True, "Human", "1-0", "resignation")
    path = tmp_path / "human.pgn"
    assert LongestWinRecorder(path).consider(board, True, "Human", "1-0", "resignation")
    game, restored = read_one(path)
    assert game.headers["Termination"] == "resignation"
    assert restored.fen() == board.fen()


@pytest.mark.parametrize("result", ["*", "win", "2-0", None])
def test_invalid_results_are_rejected_without_a_file(tmp_path, result):
    path = tmp_path / "longest.pgn"
    with pytest.raises(ValueError, match="result"):
        LongestWinRecorder(path).consider(white_win(), True, "Human", result, "checkmate")
    assert not path.exists()


def test_illegal_history_and_contradictory_terminal_result_are_rejected(tmp_path):
    recorder = LongestWinRecorder(tmp_path / "longest.pgn")
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e5"))  # push itself does not check legality
    with pytest.raises(ValueError, match="illegal move"):
        recorder.consider(board, True, "Human", "1-0", "resignation")
    with pytest.raises(ValueError, match="contradicts"):
        recorder.consider(black_win(), True, "Human", "1-0", "checkmate")
    assert not recorder.path.exists()


def test_comparison_counts_moves_instead_of_trusting_plycount_header(tmp_path):
    path = tmp_path / "longest.pgn"
    recorder = LongestWinRecorder(path)
    assert recorder.consider(white_win(True), True, "Human", "1-0", "checkmate")
    text = path.read_text(encoding="utf-8").replace('[PlyCount "11"]', '[PlyCount "1"]')
    path.write_text(text, encoding="utf-8")
    assert not LongestWinRecorder(path).consider(white_win(), True, "Human", "1-0", "checkmate")
    assert path.read_text(encoding="utf-8") == text


def test_unrelated_or_invalid_existing_pgn_is_preserved(tmp_path):
    path = tmp_path / "longest.pgn"
    path.write_text('[White "Someone"]\n[Black "Else"]\n[Result "1-0"]\n\n1. e4 1-0\n', encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="not a TonkAI win"):
        LongestWinRecorder(path).consider(white_win(), True, "Human", "1-0", "checkmate")
    assert path.read_bytes() == original


def test_parse_errors_in_existing_pgn_are_not_overwritten(tmp_path):
    path = tmp_path / "longest.pgn"
    path.write_text('[White "TonkAI"]\n[Black "Human"]\n[Result "1-0"]\n\n1. e5 1-0\n', encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="valid game"):
        LongestWinRecorder(path).consider(white_win(), True, "Human", "1-0", "checkmate")
    assert path.read_bytes() == original


def test_failed_atomic_replace_preserves_previous_win_and_cleans_temp(tmp_path, monkeypatch):
    import tonkai.win_history as history

    path = tmp_path / "longest.pgn"
    recorder = LongestWinRecorder(path)
    assert recorder.consider(white_win(), True, "Human", "1-0", "checkmate")
    original = path.read_bytes()

    def fail_replace(*args):
        raise OSError("write interrupted")

    monkeypatch.setattr(history.os, "replace", fail_replace)
    with pytest.raises(OSError, match="write interrupted"):
        recorder.consider(white_win(True), True, "Human", "1-0", "checkmate")
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_parent_record_rebuilds_a_full_win_and_rejects_invalid_moves(tmp_path):
    board = white_win()
    record = GameRecord(np.empty((0, 99), dtype=np.uint8), 0, "checkmate", 7,
                        self_color=True, moves_uci=tuple(m.uci() for m in board.move_stack))
    path = tmp_path / "longest.pgn"
    assert LongestWinRecorder(path).consider_record(record, "TinyOpponent")
    assert read_one(path)[1].fen() == board.fen()
    with pytest.raises(ValueError):
        LongestWinRecorder(path).consider_record(replace(record, moves_uci=("e2e5",) * 7), "TinyOpponent")
    with pytest.raises(ValueError, match="lacks"):
        LongestWinRecorder(path).consider_record(replace(record, moves_uci=()), "TinyOpponent")
    assert not LongestWinRecorder(path).consider_record(replace(record, result=2), "TinyOpponent")


@pytest.mark.parametrize("color", [True, False])
def test_worker_retains_uci_history_only_for_wins_without_writing(tmp_path, monkeypatch, color):
    cfg = Config(win_pgn_path=str(tmp_path / "must-not-be-written.pgn"))
    worker = GameWorker(None, None, cfg, Metrics(), np.random.default_rng(42))
    # Scripted legal game tests the real worker loop/record path without inference.
    scripted = iter(white_win().move_stack)

    def turn(board, *args):
        board.push(next(scripted))
        return 1

    monkeypatch.setattr(worker, "_tonkai_turn", turn)
    monkeypatch.setattr(worker, "_opponent_turn", turn)
    record = worker.play_game(self_color=color, collect=False)
    assert record.plies == 7
    if color:
        assert record.result == 0 and record.self_color is True
        assert record.moves_uci == tuple(m.uci() for m in white_win().move_stack)
    else:
        assert record.result == 2 and record.self_color is None
        assert record.moves_uci == ()
    assert not (tmp_path / "must-not-be-written.pgn").exists()


@pytest.mark.parametrize("entrypoint", ["train", "evaluate"])
def test_parent_entrypoint_records_only_its_longest_win(tmp_path, monkeypatch, entrypoint):
    import importlib
    import signal
    import sys

    import tonkai

    entry = importlib.import_module(entrypoint)
    original_build = tonkai.build
    records = []
    for longer in (False, True, False):
        board = white_win(longer)
        records.append(GameRecord(
            np.empty((0, 99), dtype=np.uint8), 0, "checkmate", len(board.move_stack),
            self_color=True, moves_uci=tuple(move.uci() for move in board.move_stack),
        ))
    served = []

    def scripted_build(cfg, device=None):
        components = original_build(cfg, device)
        worker, metrics = components[3], components[5]
        remaining = iter(records)

        def play_game(*args, **kwargs):
            record = next(remaining)
            served.append(record.plies)
            metrics.add_game(record.result, record.plies, record.reason)
            return record

        monkeypatch.setattr(worker, "play_game", play_game)
        return components

    path = tmp_path / "longest.pgn"
    args = [entrypoint + ".py", "--games", "3", "--device", "cpu",
            "--inference_device", "cpu", "--win_pgn_path", str(path)]
    if entrypoint == "train":
        args += ["--no-train", "--no-history", "--checkpoint_dir", str(tmp_path / "checkpoints")]
        monkeypatch.setattr(entry, "_stop", False)
    else:
        args += ["--checkpoint", str(tmp_path / "absent-checkpoint.pt")]
    monkeypatch.setattr(tonkai, "build", scripted_build)
    monkeypatch.setattr(sys, "argv", args)
    previous_sigint = signal.getsignal(signal.SIGINT)
    try:
        assert entry.main() == 0
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    assert served == [7, 11, 7]
    game, final_board = read_one(path)
    assert len(final_board.move_stack) == 11
    assert final_board.fen() == white_win(True).fen()
    assert game.headers["White"] == "TonkAI"
    assert game.headers["Black"] == "TinyOpponent"
    assert game.headers["Result"] == "1-0"
    assert list(tmp_path.glob("*.pgn")) == [path]
