"""The tactical calculation's options (claudio_fast.Board.search_options) and the shared table
(claudio_fast.Board.table_session): off, the search is the original one; on, they see what they promise.

    python -m pytest claudio/test_tactics.py -q
"""
import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401

import claudio_fast  # noqa: E402
from claudio_agent import native_board  # noqa: E402
from tonkai import fast_rules  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(claudio_fast.Board, "search_options"), reason="old claudio_fast build")
VALUES = [0.0, 1.0, 3.0, 3.0, 5.0, 9.0]


def gain(fen, plies, promotions=False, checks=False, extensions=0):
    claudio_fast.Board.search_options(promotions, checks, extensions)
    try:
        return native_board(chess.Board(fen)).forced_gain(VALUES, plies, 5, 2_000_000)
    finally:
        claudio_fast.Board.search_options(False, False, 0)


def test_promotions_see_a_pawn_promoting_past_the_horizon():
    fen = "8/P5k1/8/8/8/8/6K1/8 w - - 0 1"
    assert gain(fen, 0) == 0.0
    assert gain(fen, 0, promotions=True) == 8.0         # a queen for a pawn


def test_quiet_checks_see_a_fork_with_check():
    fen = "r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1"           # Nc7+ and Nxa8
    assert gain(fen, 0) == 0.0
    assert gain(fen, 0, checks=True) == 5.0


def test_check_extension_sees_a_mate_one_check_deeper():
    fen = "6k1/6pp/8/8/8/8/2Q5/K3R3 w - - 0 1"
    assert gain(fen, 3) == 0.0
    assert gain(fen, 3, extensions=2) == 1000.0


def test_options_off_again_after_use():
    fen = "8/P5k1/8/8/8/8/6K1/8 w - - 0 1"
    gain(fen, 0, promotions=True, checks=True, extensions=2)
    assert gain(fen, 0) == 0.0


def test_a_shared_table_gives_the_same_values_and_lines():
    games = ["e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7".split(),
             "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8".split()]
    for moves in games:
        board = chess.Board()
        for uci in moves:
            board.push_uci(uci)
        fast = native_board(board)
        codes = list(fast.legal_moves())
        alone = fast.deep_after_moves(codes, VALUES, 3, 5, 2_000_000, True)
        claudio_fast.Board.table_session(True)
        try:
            fast.deep_after_moves(codes[::-1], VALUES, 2, 5, 2_000_000, True)     # fills the table first
            shared = fast.deep_after_moves(codes, VALUES, 3, 5, 2_000_000, True)
        finally:
            claudio_fast.Board.table_session(False)
        assert [value for value, _ in shared] == [value for value, _ in alone]
        for (_, line), code in zip(shared, codes):         # each line is legal from its move
            replay = native_board(board)
            replay.push(code)
            for move in line:
                assert move in replay.legal_moves()
                replay.push(move)


def test_claudio_with_every_option_plays_legal_moves():
    import numpy as np
    import torch
    from claudio_agent import Claudio, ClaudioConfig
    from tonkai.temporal_model import SurvivalNet
    torch.manual_seed(0)
    cfg = ClaudioConfig.with_settings(["shared_table=true", "quiesce_promotions=true", "quiesce_checks=true",
                                       "check_extension=1", "tactic_top=4"])
    me = Claudio(SurvivalNet(None), {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0}, cfg, rng=np.random.default_rng(0))
    board = chess.Board()
    for _ in range(12):
        move = me.choose(board)
        assert move in board.legal_moves
        board.push(move)
        board.push(next(iter(board.legal_moves)))
    assert gain("8/P5k1/8/8/8/8/6K1/8 w - - 0 1", 0) == 0.0     # options reset after each decision
    assert fast_rules.AVAILABLE
