"""Training writes wins.json in batches: every win reaches wins.pgn at once,
wins.json on flush, and wins other processes added meanwhile are kept."""
import json

import chess

from tonkai.win_history import FeaturedGameRecorder, LongestWinRecorder, WinArchive


def fools_mate():
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(uci)
    return board                                         # TonkAI (black) won, 0-1


def stored(directory):
    path = directory / "wins.json"
    return json.loads(path.read_text(encoding="utf-8"))["wins"] if path.exists() else []


def test_deferred_wins_reach_the_json_on_flush(tmp_path):
    training = WinArchive(tmp_path, flush_seconds=3600)
    assert training.add(fools_mate(), chess.BLACK, "Tiny", "0-1", "checkmate")
    assert (tmp_path / "wins.pgn").read_text(encoding="utf-8").count("[Event ") == 1
    assert stored(tmp_path) == []                         # only in memory so far
    WinArchive(tmp_path).add(fools_mate(), chess.BLACK, "Other", "0-1", "checkmate")   # another process
    training.flush()
    wins = stored(tmp_path)
    assert [(w["number"], w["opponent"]) for w in wins] == [(1, "Other"), (2, "Tiny")]
    training.flush()                                      # nothing pending: no duplicate
    assert len(stored(tmp_path)) == 2


def test_default_archive_still_writes_every_win(tmp_path):
    WinArchive(tmp_path).add(fools_mate(), chess.BLACK, "Tiny", "0-1", "checkmate")
    assert len(stored(tmp_path)) == 1


def test_recorders_skip_games_that_cannot_replace_the_stored_one(tmp_path):
    longest = LongestWinRecorder(tmp_path / "longest.pgn")
    assert longest.consider(fools_mate(), chess.BLACK, "Tiny", "0-1", "checkmate")
    assert not longest.consider(fools_mate(), chess.BLACK, "Tiny", "0-1", "checkmate")    # not longer
    featured = FeaturedGameRecorder(tmp_path / "featured.pgn")
    assert featured.consider(fools_mate(), chess.BLACK, "Stockfish", 1800, "0-1", "checkmate")
    assert not featured.consider(fools_mate(), chess.BLACK, "Tiny", 800, "0-1", "checkmate")  # weaker rival
    assert featured.stored_key()[0] == 1800
