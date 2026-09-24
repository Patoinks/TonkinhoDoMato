"""One-move tactics (forks, double attacks) in the exact exchange result, and
endgame technique: long mates with few pieces, no draws when ahead."""
import chess
import numpy as np
import pytest

from tonkai import fast_rules
from tonkai.cognitive_config import CognitiveConfig
from test_mate_calculation import agent

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")
VALUES = [0.0, 1.0, 3.0, 3.0, 5.0, 9.0]
KD = {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0}
# Black's knight forks with ...Nf3+ (king g1 and rook h2) unless White prevents it.
FORK = "6k1/8/8/8/3n4/8/1R5R/6K1 w - - 0 1"


def test_rules_see_what_one_opponent_move_wins():
    board = chess.Board(FORK)
    fast = fast_rules.board(board)
    moves = [chess.Move.from_uci(u) for u in ("b2a2", "h2h3", "g1g2")]
    codes = [fast_rules.code(m) for m in moves]
    assert fast.exchange_after_moves(codes, VALUES, 5, 200_000) == [0.0, 0.0, 0.0]   # captures only: blind
    assert fast.tactic_after_moves(codes, VALUES, 5, 200_000) == [2.0, 0.0, 0.0]     # rook for knight


def tactical(**settings):
    tonkai = agent(exchange_depth=5, piece_safety_weight=0.2, **settings)
    tonkai.kd_values = KD
    return tonkai


def test_agent_counts_forks_for_its_best_candidates():
    board = chess.Board(FORK)
    tonkai = tactical(tactic_candidates=64)                     # every move checked
    moves = tonkai.evaluate(board, chess.WHITE, explore=False).moves
    nets = tonkai._evaluation["exchange_nets"]
    assert nets[moves.index(chess.Move.from_uci("b2a2"))] == pytest.approx(-2.0)
    assert nets[moves.index(chess.Move.from_uci("h2h3"))] == pytest.approx(0.0)
    assert tonkai.diagnostics["tactics_found"] >= 1
    blind = tactical()                                          # default: captures only
    blind.evaluate(board, chess.WHITE, explore=False)
    assert blind._evaluation["exchange_nets"][moves.index(chess.Move.from_uci("b2a2"))] == pytest.approx(0.0)


def test_endgame_mate_search_looks_deeper_with_few_pieces():
    board = chess.Board("7k/8/8/8/6K1/8/8/5R2 w - - 0 1")     # mate in 5
    short = agent(mate_search_moves=3)
    assert short.choose(board, chess.WHITE, np.random.default_rng(0), explore=False).forced_mate == 0
    deep = agent(mate_search_moves=3, endgame_pieces=6, endgame_mate_moves=5)
    choice = deep.choose(board, chess.WHITE, np.random.default_rng(0), explore=False)
    assert choice.forced_mate == 5 and deep.diagnostics["endgame_mate_searches"] == 1


def test_no_stalemate_when_ahead():
    board = chess.Board("k7/2K5/8/8/8/8/8/1Q6 w - - 0 1")      # Qb6 and Qg1 stalemate, a queen up
    tonkai = tactical(avoid_draws_ahead=3.0)
    choice = tonkai.evaluate(board, chess.WHITE, explore=False)
    stalemates = [i for i, move in enumerate(choice.moves)
                  if board.copy(stack=False).is_stalemate() is False and _stalemates(board, move)]
    assert {choice.moves[i].uci() for i in stalemates} == {"b1b6", "b1g1"}
    others = np.delete(choice.scores, stalemates)
    assert choice.scores[stalemates].max() < others[others > -1e8].min()
    assert tonkai.diagnostics["draws_refused_ahead"] == 1


def _stalemates(board, move):
    board.push(move)
    try:
        return board.is_stalemate()
    finally:
        board.pop()


def test_settings_are_validated():
    with pytest.raises(ValueError):
        CognitiveConfig(tactic_candidates=-1).validate()
    with pytest.raises(ValueError):
        CognitiveConfig(endgame_mate_moves=0).validate()
