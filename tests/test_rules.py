"""Terminal conditions, as seen by the environment TonkAI actually plays in."""

from __future__ import annotations

import chess
import pytest

from tonkai import chess_env as env
from tonkai.chess_env import RESULT_DRAW, RESULT_LOSS, RESULT_WIN


def test_checkmate_delivered_by_tonkai_is_a_win():
    # fool's mate, black has just mated white
    b = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4"):
        b.push_san(san)
    assert b.is_checkmate()
    res, why = env.terminal(b, chess.BLACK, 300)
    assert (res, why) == (RESULT_WIN, env.TERM_CHECKMATE)
    res, why = env.terminal(b, chess.WHITE, 300)
    assert (res, why) == (RESULT_LOSS, env.TERM_CHECKMATE)


def test_stalemate_is_a_draw():
    b = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert b.is_stalemate()
    for color in (chess.WHITE, chess.BLACK):
        assert env.terminal(b, color, 300) == (RESULT_DRAW, env.TERM_STALEMATE)


def test_insufficient_material_is_a_draw():
    b = chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")
    assert env.terminal(b, chess.WHITE, 300)[1] == env.TERM_INSUFFICIENT


def test_fifty_move_rule():
    b = chess.Board("8/8/4k3/8/8/4K3/8/6R1 w - - 99 80")
    assert env.terminal(b, chess.WHITE, 300)[0] is None
    b.push_san("Rg2")
    assert b.halfmove_clock == 100
    assert env.terminal(b, chess.WHITE, 300) == (RESULT_DRAW, env.TERM_FIFTY)


def test_threefold_repetition():
    b = chess.Board("4k3/8/8/8/8/8/8/R3K2R w - - 0 1")
    # shuffle the rooks back and forth: the start position repeats twice more
    for _ in range(2):
        for san in ("Ra2", "Ke7", "Ra1", "Ke8"):
            b.push_san(san)
    assert b.is_repetition(3)
    assert env.terminal(b, chess.WHITE, 300) == (RESULT_DRAW, env.TERM_REPETITION)


def test_ply_limit_is_a_draw():
    b = chess.Board()
    for _ in range(3):
        for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
            b.push_san(san)
    assert env.terminal(b, chess.WHITE, 8)[1] in (
        env.TERM_PLY_LIMIT, env.TERM_REPETITION
    )
    assert env.terminal(b, chess.WHITE, 4)[1] is not None


def test_ongoing_game_is_not_terminal():
    assert env.terminal(chess.Board(), chess.WHITE, 300) == (None, None)


def test_illegal_opponent_move_is_rejected():
    b = chess.Board()
    with pytest.raises(env.IllegalOpponentMove):
        env.apply_opponent_move(b, "e2e5")
    with pytest.raises(env.IllegalOpponentMove):
        env.apply_opponent_move(b, "0000")
    with pytest.raises(env.IllegalOpponentMove):
        env.apply_opponent_move(b, "not-a-move")
    assert b.fen() == chess.STARTING_FEN


def test_legal_opponent_move_is_played():
    b = chess.Board()
    mv = env.apply_opponent_move(b, "e2e4")
    assert mv.uci() == "e2e4"
    assert b.move_stack[-1].uci() == "e2e4"


def test_result_from_white_pov():
    assert env.result_from_white_pov(RESULT_WIN, chess.WHITE) == "1-0"
    assert env.result_from_white_pov(RESULT_WIN, chess.BLACK) == "0-1"
    assert env.result_from_white_pov(RESULT_LOSS, chess.WHITE) == "0-1"
    assert env.result_from_white_pov(RESULT_DRAW, chess.BLACK) == "1/2-1/2"
