"""Every TonkAI win is archived (PGN + JSON for the site); nothing else is."""
import io
import json

import chess
import chess.pgn
import pytest

from tonkai import web_publish
from tonkai.win_history import WinArchive


def mate(moves):
    board = chess.Board()
    for san in moves:
        board.push_san(san)
    return board


SCHOLAR = ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#")      # white wins
FOOL = ("f3", "e5", "g4", "Qh4#")                                 # black wins


def test_every_win_is_appended_and_indexed(tmp_path):
    archive = WinArchive(tmp_path / "wins")
    assert archive.add(mate(SCHOLAR), chess.WHITE, "Rodent IV", "1-0", "checkmate", game=12)
    assert archive.add(mate(FOOL), chess.BLACK, "Rodent IV", "0-1", "checkmate", game=40)
    wins = archive.load()
    assert [w["number"] for w in wins] == [1, 2]
    assert [w["plies"] for w in wins] == [7, 4]
    assert wins[1]["tonkai_color"] == "black" and wins[1]["game"] == 40
    text = (tmp_path / "wins/wins.pgn").read_text(encoding="utf-8")
    stream, games = io.StringIO(text), []
    while (game := chess.pgn.read_game(stream)) is not None:
        games.append(game)
    assert [g.headers["Result"] for g in games] == ["1-0", "0-1"]
    assert games[1].headers["Black"] == "TonkAI"
    assert json.loads((tmp_path / "wins/wins.json").read_text())["format"] == "tonkai-wins-v1"


def test_losses_draws_and_disabled_archive_write_nothing(tmp_path):
    archive = WinArchive(tmp_path / "wins")
    assert not archive.add(mate(SCHOLAR), chess.BLACK, "Rodent IV", "1-0", "checkmate")   # TonkAI lost
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert not archive.add(stalemate, chess.WHITE, "Rodent IV", "1/2-1/2", "stalemate")
    assert not (tmp_path / "wins").exists()
    assert not WinArchive("").add(mate(SCHOLAR), chess.WHITE, "Rodent IV", "1-0", "checkmate")


def test_contradictory_results_are_rejected(tmp_path):
    archive = WinArchive(tmp_path / "wins")
    with pytest.raises(ValueError):
        archive.add(mate(FOOL), chess.WHITE, "Rodent IV", "1-0", "checkmate")     # board says 0-1
    with pytest.raises(ValueError):
        archive.add(mate(SCHOLAR), chess.WHITE, "TonkAI", "1-0", "checkmate")     # needs an opponent
    assert archive.load() == []


def test_only_the_real_archive_is_ever_published(tmp_path, monkeypatch):
    config = tmp_path / "web.json"
    config.write_text(json.dumps({"enabled": True}))
    launched = []
    monkeypatch.setattr(web_publish.subprocess, "Popen", lambda *a, **k: launched.append(a) or "process")
    assert web_publish.publish_wins(tmp_path / "wins", config_path=config) is None
    assert web_publish.publish_wins(web_publish.ROOT / "history/wins", config_path=tmp_path / "none.json") is None
    assert web_publish.publish_wins(web_publish.ROOT / "history/wins", config_path=config) == "process"
    config.write_text(json.dumps({"enabled": False}))
    assert web_publish.publish_wins(web_publish.ROOT / "history/wins", config_path=config) is None
    assert len(launched) == 1
