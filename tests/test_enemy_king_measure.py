"""Shaped objective with victory_measure="enemy_king": the measure of a real
position is 1 - KS of the enemy king; finished games still override it."""
import chess
import pytest

from tonkai import fast_rules
from tonkai.cognitive_config import CognitiveConfig
from test_mate_calculation import agent

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")


class KingSafety:
    """KS 0.9 for the white king, 0.3 for the black one."""

    def ks_rows(self, rows):
        raise AssertionError("not used by the measure")

    def ks_both(self, board):
        return 0.9, 0.3

    def advantage(self, board, colour):
        return 0.6 if colour else -0.6


def measured(**settings):
    tonkai = agent(**settings)
    tonkai.king_model = KingSafety()
    return tonkai


def test_measure_is_how_unsafe_the_enemy_king_is():
    board = chess.Board("r3k3/8/8/8/8/8/8/4K2R w - - 0 1")
    tonkai = measured(victory_measure="enemy_king")
    assert tonkai.position_proxy(board, chess.WHITE) == pytest.approx(0.7)       # 1 - KS(black king)
    assert tonkai.position_proxy(board, chess.BLACK) == pytest.approx(0.1)       # 1 - KS(white king)
    owner = measured()                                                          # default: kings + pieces
    assert owner.position_proxy(board, chess.WHITE) != pytest.approx(0.7)


def test_finished_games_still_decide():
    mated = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        mated.push_uci(uci)
    tonkai = measured(victory_measure="enemy_king")
    assert tonkai.position_proxy(mated, chess.BLACK) == 1.0
    assert tonkai.position_proxy(mated, chess.WHITE) == 0.0


def test_settings_are_validated():
    with pytest.raises(ValueError):
        CognitiveConfig(victory_measure="kings").validate()
    with pytest.raises(ValueError):
        CognitiveConfig(victory_measure="enemy_king").validate()                 # needs a KS model
    CognitiveConfig(victory_measure="enemy_king", king_safety_model="ks.pt").validate()


def test_enemy_box_counts_the_squares_we_control_around_the_enemy_king():
    tonkai = measured(victory_measure="enemy_box")
    assert tonkai.position_proxy(chess.Board(), chess.WHITE) == pytest.approx(1 - 5 / 8)     # e8: 5 free squares
    centre = chess.Board("8/8/8/3k4/8/8/8/K6R w - - 0 1")
    assert tonkai.position_proxy(centre, chess.WHITE) == pytest.approx(0.0)                 # all 8 free
    boxed = chess.Board("k7/8/1Q6/8/8/8/8/4K3 w - - 0 1")                                   # a8: a7 b7 b8 covered
    assert tonkai.position_proxy(boxed, chess.WHITE) == pytest.approx(1.0)


def test_mixed_measure_is_the_weighted_sum_of_its_parts():
    board = chess.Board("8/8/8/3k4/8/8/8/K6R w - - 0 1")                                    # box 0, enemy KS 0.3
    mixed = measured(victory_measure="enemy_king:1,enemy_box:1")
    assert mixed.position_proxy(board, chess.WHITE) == pytest.approx(0.5 * 0.7 + 0.5 * 0.0)
    material = measured(victory_measure="material")
    assert material.position_proxy(chess.Board(), chess.WHITE) == pytest.approx(0.5)        # equal material


def test_simplify_rewards_equal_trades_and_punishes_bad_ones():
    tonkai = measured(victory_measure="simplify:1,material:1")
    start = tonkai.position_proxy(chess.Board(), chess.WHITE)
    knights_off = chess.Board("r1bqkb1r/pppppppp/8/8/8/8/PPPPPPPP/R1BQKB1R w KQkq - 0 1")   # N for N
    knight_for_pawn = chess.Board("rnbqkbnr/ppppppp1/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1")
    assert tonkai.position_proxy(knights_off, chess.WHITE) > start
    assert tonkai.position_proxy(knight_for_pawn, chess.WHITE) < start
    five = chess.Board("8/8/8/4k3/8/8/2P5/2R1K3 w - - 0 1")                               # 4 pieces
    assert measured(victory_measure="simplify").position_proxy(five, chess.WHITE) == 1.0
