"""Owner's rule: piece values take the opponent side only from Fairy 1000 (35%)
and Stockfish 1500 (65%); TonkAI's own pieces count from every game."""
import chess
import pytest

from tonkai.captures import CaptureStats, parse_weights

NAMES = ("Pawn", "Knight", "Bishop", "Rook", "Queen")


def filled(weights=None):
    stats = CaptureStats(opponent_weights=weights)
    stats.games = 100
    for name in NAMES:
        stats.kills["self"][name], stats.deaths["self"][name] = 100, 200          # 1 kill, 2 deaths per game
        for measure in ("mobility", "threats"):
            stats.activity["self"][measure][name] = 1000
            stats.activity["self"][measure + "_n"][name] = 1000
    for kind, games, kills in (("fairy", 40, 40), ("external", 60, 120), ("tiny", 50, 5000)):
        row = stats._kind(kind)
        row["games"] = games
        for name in NAMES:
            row["kills"][name] = kills                  # tiny's absurd numbers must be ignored
            row["deaths"][name] = games
            for measure in ("mobility", "threats"):
                row["activity"][measure][name] = games * 10
                row["activity"][measure + "_n"][name] = games * 10
    stats.kills["self"]["Queen"] = 300                  # the queen kills 3 per game
    return stats


def test_parse_weights_normalises():
    assert parse_weights("fairy=0.35,external=0.65") == pytest.approx({"fairy": .35, "external": .65})
    assert parse_weights("fairy=35,external=65") == pytest.approx({"fairy": .35, "external": .65})
    assert parse_weights("") is None


def test_weighted_kd_uses_only_fairy_and_stockfish_per_game():
    stats = filled({"fairy": .35, "external": .65})
    values = stats.weighted_kd()
    # Per game: self kills 1 (queen 3), deaths 2; fairy kills 1, deaths 1; stockfish kills 2, deaths 1.
    opponent_kills, opponent_deaths = .35 * 1 + .65 * 2, .35 * 1 + .65 * 1
    pawn = (1 + opponent_kills) / (2 + opponent_deaths)
    queen = (3 + opponent_kills) / (2 + opponent_deaths)
    assert values[chess.PAWN] == pytest.approx(1.0)
    assert values[chess.QUEEN] == pytest.approx(queen / pawn)
    assert stats.value_source() == "weighted"


def test_tiny_and_self_play_never_count_on_the_opponent_side():
    base = filled({"fairy": .35, "external": .65})
    changed = filled({"fairy": .35, "external": .65})
    changed.by_kind["tiny"]["kills"]["Knight"] = 10 ** 6
    changed._kind("self")["games"] = 7
    assert changed.weighted_kd() == base.weighted_kd()


def test_falls_back_to_pooling_until_both_opponents_have_data():
    stats = filled({"fairy": .35, "external": .65})
    stats.by_kind["external"]["games"] = 5                  # too few Stockfish games yet
    assert stats.value_source() == "pooled"
    assert stats.measures()["kd"] == stats.pooled_kd()


def test_add_records_the_opponent_side_per_kind_and_state_roundtrips():
    stats = CaptureStats(opponent_weights={"fairy": .35, "external": .65})
    board = chess.Board()
    for san in ("e4", "d5", "exd5", "Qxd5"):
        board.push_san(san)
    stats.add(board.move_stack, chess.WHITE, "unfinished", opponent_kind="fairy")
    stats.add(board.move_stack, chess.WHITE, "unfinished", opponent_kind="tiny")
    assert stats.by_kind["fairy"]["games"] == 1
    assert stats.by_kind["fairy"]["kills"]["Queen"] == 1 and stats.by_kind["fairy"]["deaths"]["Pawn"] == 1
    restored = CaptureStats(opponent_weights={"fairy": .35, "external": .65})
    restored.load_state_dict(stats.state_dict())
    assert restored.by_kind["fairy"]["kills"] == stats.by_kind["fairy"]["kills"]
    assert restored.by_kind["tiny"]["games"] == 1
