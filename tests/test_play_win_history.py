"""Exercise human-play win recording without models, training or real history files."""

import argparse

import chess
import chess.pgn
import numpy as np
import pytest

import play


def run_human_game(monkeypatch, tmp_path, commands, tonkai_moves, *extra_args):
    responses = iter(commands)
    replies = iter(tonkai_moves)

    class Agent:
        def predict_enemy(self, board, color):
            moves = list(board.legal_moves)
            return moves, np.full(len(moves), 1 / len(moves)), None

        def evaluate_moves(self, board, color):
            move = board.parse_san(next(replies))
            return argparse.Namespace(
                moves=[move], utilities=np.array([0.0]),
                probs=np.array([[0.2, 0.3, 0.5]]),
            )

    def answer(_):
        command = next(responses)
        if isinstance(command, BaseException):
            raise command
        return command

    longest = tmp_path / "longest.pgn"
    requested = tmp_path / "requested.pgn"
    net = argparse.Namespace(eval=lambda: None)
    monkeypatch.setattr(play.tonkai, "build", lambda *_: (net, Agent(), None, None, None, None))
    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr("sys.argv", [
        "play.py", "--quiet", "--checkpoint", str(tmp_path / "missing.pt"),
        "--win_pgn_path", str(longest), "--pgn", str(requested), *extra_args,
    ])
    result = play.main()
    return result, longest, requested


def read_game(path):
    with path.open(encoding="utf-8") as stream:
        game = chess.pgn.read_game(stream)
    assert game is not None and not game.errors
    return game


@pytest.mark.parametrize("commands,replies,args,expected_result", [
    (["f3", "g4"], ["e5", "Qh4#"], (), "0-1"),
    (["e5", "Nc6", "Nf6"], ["e4", "Bc4", "Qh5", "Qxf7#"], ("--black",), "1-0"),
])
def test_human_checkmate_records_complete_win(monkeypatch, tmp_path, commands, replies, args, expected_result):
    code, longest, requested = run_human_game(monkeypatch, tmp_path, commands, replies, *args)
    assert code == 0
    recorded = read_game(longest)
    exported = read_game(requested)
    assert recorded.headers["Result"] == expected_result
    assert recorded.end().board().is_checkmate()
    assert list(recorded.mainline_moves()) == list(exported.mainline_moves())
    assert recorded.headers["White" if args else "Black"] == "TonkAI"
    assert recorded.headers["Black" if args else "White"] == "Human"


def test_human_resignation_records_only_retained_moves_after_undo(monkeypatch, tmp_path):
    code, longest, requested = run_human_game(
        monkeypatch, tmp_path, ["e4", "undo", "d4", "resign"], ["e5", "d5"]
    )
    assert code == 0
    recorded = read_game(longest)
    assert recorded.headers["Result"] == "0-1"
    assert [move.uci() for move in recorded.mainline_moves()] == ["d2d4", "d7d5"]
    assert list(recorded.mainline_moves()) == list(read_game(requested).mainline_moves())


@pytest.mark.parametrize("commands,replies,args,expected_result", [
    (["quit"], [], (), "*"),
    ([EOFError()], [], (), "*"),
    (["e4"], ["e5"], ("--max_plies", "2"), "1/2-1/2"),
    (["e4", "Bc4", "Qh5", "Qxf7#"], ["e5", "Nc6", "Nf6"], (), "1-0"),
])
def test_quit_draw_and_human_win_never_record_tonkai_win(monkeypatch, tmp_path, commands, replies, args, expected_result):
    code, longest, requested = run_human_game(monkeypatch, tmp_path, commands, replies, *args)
    assert code == 0
    assert not longest.exists()
    assert read_game(requested).headers["Result"] == expected_result


def test_shorter_human_win_keeps_record_and_still_exports_requested_game(monkeypatch, tmp_path):
    code, longest, _ = run_human_game(monkeypatch, tmp_path, ["f3", "g4"], ["e5", "Qh4#"])
    assert code == 0
    original = longest.read_bytes()
    code, _, requested = run_human_game(monkeypatch, tmp_path, ["e4", "resign"], ["e5"])
    assert code == 0
    assert longest.read_bytes() == original
    assert len(list(read_game(requested).mainline_moves())) == 2


@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_longest_win_failure_preserves_explicit_pgn_export(monkeypatch, tmp_path, error_type):
    class FailingRecorder:
        def __init__(self, path):
            pass

        def consider(self, *args):
            raise error_type("test record failure")

    monkeypatch.setattr(play, "LongestWinRecorder", FailingRecorder)
    code, longest, requested = run_human_game(monkeypatch, tmp_path, ["e4", "resign"], ["e5"])
    assert code == 1
    assert not longest.exists()
    assert read_game(requested).headers["Result"] == "0-1"
