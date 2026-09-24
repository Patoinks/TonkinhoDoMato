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
