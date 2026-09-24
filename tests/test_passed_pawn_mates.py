"""Owner's rules: mate searches that do not move the attacker's pawns
(mate_pawn_reach) and a bonus for safely pushing an own passed pawn."""
import chess
import numpy as np
import pytest

from tonkai import fast_rules
from tonkai.cognitive_agent import passed_pawn
from test_tactics_endgame import tactical

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")


def test_mate_search_can_leave_the_pawns_out():
    promotion = fast_rules.board(chess.Board("7k/5P2/6K1/8/8/8/8/8 w - - 0 1"))   # f8=Q#, pawn 2 squares away
    assert promotion.forced_mate(True, 1, 100_000)[0] == 1
    assert promotion.forced_mate(True, 1, 100_000, 1)[0] == 0                       # no pawn moves at all
    assert promotion.forced_mate(True, 1, 100_000, 3)[0] == 1                       # pawns closer than 3 still move
    rook = fast_rules.board(chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"))        # Ra8#: pieces still mate
    assert rook.forced_mate(True, 1, 100_000, 1)[0] == 1


def test_passed_pawns_are_recognised():
    board = chess.Board("4k3/3p4/8/4P3/P7/8/8/4K3 w - - 0 1")
    assert passed_pawn(board, chess.A4)                      # nothing ahead on a/b
    assert not passed_pawn(board, chess.E5)                  # d7 guards the adjacent file ahead


def test_safe_passed_pawn_pushes_get_the_bonus():
    board = chess.Board("4k3/8/8/8/P7/8/8/4K3 w - - 0 1")
    plain = tactical(planning_candidates=64).evaluate(board, chess.WHITE, explore=False)
    bonus = tactical(planning_candidates=64, passed_pawn_bonus=0.07).evaluate(board, chess.WHITE, explore=False)
    push = plain.moves.index(chess.Move.from_uci("a4a5"))
    assert bonus.scores[push] - plain.scores[push] == pytest.approx(0.07 * 4 / 7, abs=1e-6)
    king = plain.moves.index(chess.Move.from_uci("e1e2"))
    assert bonus.scores[king] == pytest.approx(plain.scores[king])
