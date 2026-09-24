"""Owner's rules: mate searches only where the King Safety model sees a king in
danger, and repetitions broken (never go back to a position already seen)."""
import chess
import numpy as np
import pytest

from tonkai import fast_rules
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from test_mate_calculation import MATE_IN_TWO, agent

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")
# Black to move: a rook leaving the 8th rank allows Re8# (mate in 1); ...Rc8
# allows Re8+ Rxe8 Rxe8# (mate in 2); ...h6 is safe.
DEFEND = "3r2k1/5ppp/8/8/8/8/4RPPP/4R1K1 b - - 0 1"


class ConstantKS:
    """A King Safety model that sees every king as equally (un)safe."""

    def __init__(self, value):
        self.value = value

    def ks_rows(self, rows):
        return np.full(len(np.atleast_2d(rows)), self.value, dtype=np.float32)

    def ks_both(self, board):
        return self.value, self.value


def gated(ks, **settings):
    tonkai = agent(king_safety_weight=0.05, **settings)
    tonkai.king_model = ConstantKS(ks)
    return tonkai


def test_attack_search_only_when_the_enemy_king_is_unsafe():
    safe = gated(1.0, mate_search_moves=3, mate_attack_ks=0.9)
    choice = safe.choose(chess.Board(MATE_IN_TWO), chess.WHITE, np.random.default_rng(0), explore=False)
    assert choice.forced_mate == 0 and safe.diagnostics["mate_attack_skipped"] == 1
    unsafe = gated(0.2, mate_search_moves=3, mate_attack_ks=0.9)
    choice = unsafe.choose(chess.Board(MATE_IN_TWO), chess.WHITE, np.random.default_rng(0), explore=False)
    assert choice.forced_mate == 2 and unsafe.diagnostics["mate_attack_skipped"] == 0


def test_defence_checks_mate_in_one_always_and_deeper_only_when_unsafe():
    safe = gated(1.0, mate_defense_moves=2, mate_defense_ks=0.9)
    safe.evaluate(chess.Board(DEFEND), chess.BLACK, explore=False)
    found = set(safe._evaluation["allows_mate"].values())
    assert found == {1} and safe.diagnostics["mate_defence_shallow"] == 1
    unsafe = gated(0.2, mate_defense_moves=2, mate_defense_ks=0.9)
    unsafe.evaluate(chess.Board(DEFEND), chess.BLACK, explore=False)
    assert set(unsafe._evaluation["allows_mate"].values()) == {1, 2}
    ungated = gated(1.0, mate_defense_moves=2)                           # 0 = always the full search
    ungated.evaluate(chess.Board(DEFEND), chess.BLACK, explore=False)
    assert set(ungated._evaluation["allows_mate"].values()) == {1, 2}


def shuffled():
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):                         # back to the start
        board.push_uci(uci)
    return board


def test_repeated_moves_are_the_returns_to_a_seen_position():
    board = shuffled()
    moves = list(board.legal_moves)
    repeated = {moves[i].uci() for i in CognitiveAgent._repeated_moves(board, moves)}
    assert repeated == {"g1f3"}                                          # 1.Nf3 was already played here
    board.push_uci("e2e4")                                               # a pawn move: nothing before it counts
    assert CognitiveAgent._repeated_moves(board, list(board.legal_moves)) == set()


def test_break_repetitions_never_repeats_while_another_move_exists():
    for seed in range(4):
        board = shuffled()
        tonkai = agent(seed, break_repetitions=True)
        choice = tonkai.choose(board, chess.WHITE, np.random.default_rng(seed), explore=True)
        assert choice.move.uci() != "g1f3"
        best = choice.scores.argmax()
        assert choice.moves[best].uci() != "g1f3"


def test_gate_settings_are_validated():
    with pytest.raises(ValueError):
        CognitiveConfig(mate_attack_ks=1.5).validate()
