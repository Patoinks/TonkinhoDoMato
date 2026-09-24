"""Piece values measured on one kind of games only (e.g. "external@1800"):
the same level-weighted K/D rule, a separate who-killed-whom record."""
import chess

from tonkai.captures import CaptureStats

# Scholar's-mate style game with captures on both sides.
GAME = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]


def test_detail_kind_keeps_its_own_record():
    stats = CaptureStats()
    stats.add(GAME, chess.WHITE, "win", opponent_kind="external", detail_kind="external@1800")
    row = stats.by_kind["external@1800"]
    assert row["games"] == 1 and row["pair_games"] == 1
    assert row["pairs"]["self"]["Queen>Pawn"] == 1                 # Qxf7 was TonkAI's capture
    assert stats.by_kind["external"]["games"] == 1                 # the usual record is unchanged


def test_values_need_enough_deaths_of_that_kind():
    stats = CaptureStats()
    stats.add_pairs("external@1800", GAME, chess.WHITE)
    assert stats.piece_values("external@1800") is None             # too few deaths
    assert stats.piece_values("external@2200") is None             # no record at all
