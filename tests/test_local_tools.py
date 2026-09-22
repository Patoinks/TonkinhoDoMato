import argparse
import io
import json
from unittest.mock import patch

import chess
import chess.pgn
import numpy as np
import pytest

from tonkai.history import RunHistory
from tonkai.metrics import Metrics
from plot_history import load_series, render
from play import export_pgn, undo_human_turn, print_summary, main
from calibrate_opponent import anchor_match


def test_history_records_config_windows_and_final_sample(tmp_path):
    metrics = Metrics()
    metrics.add_game(0, 12, "checkmate")
    metrics.top1.extend([0, 1])
    metrics.surprise.extend([1, 3])
    metrics.times["inference"] = 0.25
    history = RunHistory(tmp_path, {"seed": 42}, resume="old.pt", initial_games=300)
    history.record(metrics)
    metrics.add_game(1, 30, "draw")
    history.record(metrics, final=True)
    rows = [json.loads(line) for line in history.path.read_text().splitlines()]
    assert rows[0]["config"] == {"seed": 42}
    assert rows[0]["initial_games"] == 300
    assert rows[-1]["final"] is True
    assert rows[-1]["top1"]["mean"] == 0.5
    assert rows[-1]["surprise"]["mean"] == 2
    assert rows[-1]["times"]["inference"] == 0.25
    assert rows[-1]["wins"] == rows[-1]["draws"] == 1
    values = load_series(history.path, "top1.mean")
    assert values == [(1, 0.5), (2, 0.5)]
    svg = render([("run <one>", values)], "top1.mean")
    assert "run &lt;one&gt;" in svg
    assert "polyline" in svg
    assert RunHistory(tmp_path, {}).path != history.path


@pytest.mark.parametrize("human", [chess.WHITE, chess.BLACK])
def test_undo_removes_only_latest_human_observation(human):
    board = chess.Board()
    moves = ["e4", "e5"] if human == chess.WHITE else ["e4", "e5", "Nf3"]
    for move in moves:
        board.push_san(move)
    observations = [(1.2, True)]
    assert undo_human_turn(board, human, observations)
    assert board.turn == human
    assert len(board.move_stack) == (0 if human else 1)
    assert not observations
    assert not undo_human_turn(board, human, observations)


def test_pgn_round_trip_and_summary(tmp_path, capsys):
    board = chess.Board()
    for move in ["f3", "e5", "g4", "Qh4#"]:
        board.push_san(move)
    path = tmp_path / "game.pgn"
    export_pgn(board, chess.WHITE, path, "0-1")
    restored = chess.pgn.read_game(io.StringIO(path.read_text()))
    assert restored.end().board().fen() == board.fen()
    assert restored.headers["Result"] == "0-1"
    assert restored.headers["Black"] == "TonkAI"
    print_summary(board, [(1, True), (3, False)], "TonkAI wins")
    text = capsys.readouterr().out
    assert "1 of your 2" in text and "2.00" in text


def test_quiet_still_scores_hint_and_undo(tmp_path, monkeypatch, capsys):
    class Agent:
        def predict_enemy(self, board, color):
            moves = list(board.legal_moves)
            return moves, np.full(len(moves), 1 / len(moves)), None

        def evaluate_moves(self, board, color):
            moves = list(board.legal_moves)
            return argparse.Namespace(moves=moves, utilities=np.zeros(len(moves)),
                                      probs=np.tile([0.2, 0.3, 0.5], (len(moves), 1)))

    net = argparse.Namespace(eval=lambda: None)
    commands = iter(["hint", "e4", "moves", "pgn", "undo", "resign"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    path = tmp_path / "quiet.pgn"
    monkeypatch.setattr("sys.argv", ["play.py", "--quiet", "--checkpoint", str(tmp_path / "absent.pt"), "--pgn", str(path)])
    with patch("play.tonkai.build", return_value=(net, Agent(), None, None, None, None)):
        assert main() == 0
    output = capsys.readouterr().out
    assert "Prediction hint" in output and "surprise" in output
    assert "0 of your 0" in output
    assert chess.pgn.read_game(io.StringIO(path.read_text())).headers["Result"] == "0-1"


def anchor_args(**kwargs):
    return argparse.Namespace(anchor="definitely-absent-reference.exe", anchor_elo=None,
                              anchor_time=0.05, anchor_options="{}", settings="1",
                              nodes=100, randomness=0.3, seed=42, games=4, **kwargs)


def test_anchor_missing_is_optional(capsys):
    assert anchor_match(anchor_args()) is None
    assert "Anchor skipped" in capsys.readouterr().out


def test_anchor_only_measures_tiny_and_alternates_colours(monkeypatch, capsys):
    import chess.engine
    args = anchor_args()
    args.anchor_elo = 1300
    monkeypatch.setattr("calibrate_opponent.shutil.which", lambda _: "reference.exe")
    called = []

    class Reference:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def configure(self, options): assert options == {}
        def play(self, board, limit, game):
            called.append((limit.time, game))
            return argparse.Namespace(move=chess.Move.from_uci("e2e4"))

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _: Reference())
    pairs = []
    def match(white, black):
        pairs.append((white(chess.Board()), black(chess.Board())))
        return 0.5
    monkeypatch.setattr("calibrate_opponent.play", match)
    assert anchor_match(args) == 1300
    assert len(pairs) == len(called) == 4
    assert len({id(token) for _, token in called}) == 4
    assert "conditional Elo estimate 1300" in capsys.readouterr().out
