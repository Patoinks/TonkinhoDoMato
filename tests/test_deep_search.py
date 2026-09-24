"""Deep material search with prunings (deep_after_moves) and the network's
value at the end of each candidate's main line (leaf_network)."""
import chess
import numpy as np
import pytest

from tonkai import fast_rules
from tonkai.cognitive_config import CognitiveConfig
from test_tactics_endgame import FORK, VALUES, tactical

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")
TACTICS = ["6k1/8/8/8/3n4/8/1R5R/6K1 w - - 0 1",                                  # knight fork
           "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",    # Qxf7# on the board
           "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"]


def test_pruned_search_agrees_with_the_full_search():
    for fen in TACTICS:
        board = chess.Board(fen)
        fast = fast_rules.board(board)
        codes = [fast_rules.code(m) for m in board.legal_moves]
        full = fast.search_after_moves(codes, VALUES, 3, 5, 20_000_000)
        pruned = [loss for loss, _ in fast.deep_after_moves(codes, VALUES, 3, 5, 20_000_000)]
        exact = [loss for loss, _ in fast.deep_after_moves(codes, VALUES, 3, 5, 20_000_000, True)]
        mate = lambda values: [np.sign(v) * 1000 if abs(v) >= 900 else v for v in values]   # prunings count mate distance
        assert mate(pruned) == pytest.approx(mate(full)), fen
        assert mate(exact) == pytest.approx(mate(full)), fen


def test_main_lines_are_legal_and_budget_gives_nan():
    board = chess.Board(FORK)
    fast = fast_rules.board(board)
    codes = [fast_rules.code(chess.Move.from_uci(u)) for u in ("b2a2", "h2h3")]
    found = fast.deep_after_moves(codes, VALUES, 6, 5, 5_000_000)
    assert [loss for loss, _ in found] == [2.0, 0.0]
    for code, (_, line) in zip(codes, found):
        assert 0 < len(line) <= 6
        fast.push(code)
        for step in line:
            assert step in fast.legal_moves()
            fast.push(step)
        for _ in range(len(line) + 1):
            fast.pop()
    assert all(np.isnan(loss) and not line for loss, line in fast.deep_after_moves(codes, VALUES, 6, 5, 10))


def test_agent_uses_the_pruned_search_and_the_leaf_value():
    board = chess.Board(FORK)
    tonkai = tactical(tactic_candidates=64, tactic_depth=4, tactic_search="pruned", leaf_network=1.0)
    moves = tonkai.evaluate(board, chess.WHITE, explore=False).moves
    assert tonkai._evaluation["exchange_nets"][moves.index(chess.Move.from_uci("b2a2"))] == pytest.approx(-2.0)
    assert tonkai.diagnostics["leaf_evaluations"] >= 1


def test_settings_are_validated():
    with pytest.raises(ValueError):
        CognitiveConfig(tactic_search="deep").validate()
    CognitiveConfig(tactic_search="exact").validate()
    with pytest.raises(ValueError):
        CognitiveConfig(leaf_network=2.0).validate()
