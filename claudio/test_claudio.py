"""Claudio's rules: legal moves, its own mates, never a move that allows mate,
no free pieces, and a memory that sees each real position exactly once.

    python -m pytest claudio/test_claudio.py -q
"""
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401

from tonkai import fast_rules  # noqa: E402
from tonkai.encoding import encode  # noqa: E402
from tonkai.temporal_model import SurvivalNet  # noqa: E402
from claudio_agent import Claudio, ClaudioConfig, move_of  # noqa: E402

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")
VALUES = {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0}


def player(**settings):
    torch.manual_seed(0)
    cfg = ClaudioConfig.with_settings([f"{k}={v}" for k, v in settings.items()])
    return Claudio(SurvivalNet(None), VALUES, cfg, rng=np.random.default_rng(0))


def test_rules_moves_are_the_legal_moves():
    rng = np.random.default_rng(1)
    for _ in range(30):
        board = chess.Board()
        for _ in range(int(rng.integers(0, 80))):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[int(rng.integers(len(moves)))])
        fast = fast_rules.board(board)
        assert {move_of(code) for code in fast.legal_moves()} == set(board.legal_moves)


def test_takes_its_own_mate():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert player().choose(board) == chess.Move.from_uci("a1a8")


def test_never_plays_a_move_that_allows_mate():
    # Black threatens ...Qxg2# (queen g5? no: bishop b7 + queen g2 battery); only moves covering g2 hold.
    board = chess.Board("6k1/1b3ppp/8/8/8/6q1/5PP1/5RK1 w - - 0 1")
    tonkai = player(mate_attack=0)
    move = tonkai.choose(board)
    board.push(move)
    fast = fast_rules.board(board)
    assert fast.forced_mate(False, 1, 20_000)[0] == 0


@pytest.mark.parametrize("mode,depth", [("min", 2), ("expect", 2), ("min", 3), ("expect", 3)])
def test_replies_alone_see_a_mate_in_one(mode, depth):
    # No mate defence: only the look one reply further can see that ...Qxg2# is coming.
    board = chess.Board("6k1/1b3ppp/8/8/8/6q1/5PP1/5RK1 w - - 0 1")
    move = player(mate_attack=0, mate_defence=0, reply_top=100, reply_mode=mode, reply_depth=depth).choose(board)
    board.push(move)
    assert fast_rules.board(board).forced_mate(False, 1, 20_000)[0] == 0


def test_replies_keep_the_choice_legal_and_off_changes_nothing():
    rng = np.random.default_rng(3)
    for _ in range(10):
        board = chess.Board()
        for _ in range(int(rng.integers(0, 40))):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[int(rng.integers(len(moves)))])
        if board.is_game_over():
            continue
        assert player(reply_top=6).choose(board) in board.legal_moves
        assert player(reply_top=6, reply_mode="expect", reply_depth=3).choose(board) in board.legal_moves
        assert player(reply_top=6, reply_mode="expect", leaf_weight=1.0).choose(board) in board.legal_moves
        assert player(reply_top=0).choose(board) == player().choose(board)


def test_leaf_value_still_never_leaves_its_queen_to_a_pawn():
    board = chess.Board("4k3/8/8/8/2p5/3Q4/8/4K3 w - - 0 1")      # ...c4 attacks the queen on d3
    move = player(mate_attack=0, leaf_weight=1.0, tactic_depth=5).choose(board)
    fast = fast_rules.board(board)
    loss = fast.exchange_after_moves([fast_rules.code(move)], [0, 1, 3, 3, 5, 9], 5, 200_000)[0]
    assert (1.0 if board.is_capture(move) else 0.0) - loss >= 0


def test_the_scan_calculates_every_move():
    # Every legal move gets the calculation at scan_depth, not only the best candidates.
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    me = player(scan_depth=2)
    fast = fast_rules.board(board)
    codes = list(fast.legal_moves())
    values = [0.0] + [VALUES[k] for k in range(1, 6)]
    gains = np.array([values[board.piece_type_at(move.to_square) or 0] if board.is_capture(move) else 0.0
                      for move in map(move_of, codes)])
    nets = gains - np.asarray(fast.exchange_after_moves(codes, values, 5, 200_000))
    scores = np.zeros(len(codes))
    me._scan(fast, codes, gains, nets, scores)
    deep = gains - np.array([loss for loss, _ in fast.deep_after_moves(codes, values, 2, 5, 2_000_000, True)])
    np.testing.assert_allclose(nets, deep)


def test_a_draw_is_worth_less_when_it_thinks_it_is_winning():
    board = chess.Board("6k1/8/8/8/8/8/5Q2/6K1 w - - 0 1")
    for threshold, expected in ((0.0, -1.0), (1.01, 0.4)):          # its evaluation is always in [0, 1]
        me = player(mate_attack=0, ahead_draws=True, ahead_value=threshold)
        me.choose(board)
        assert me._draw == expected


def test_endgame_finds_a_longer_mate():
    # King and rook against king: mate in 5, which a 3-move search cannot see.
    board = chess.Board("8/k3K3/8/8/8/8/1R6/8 w - - 0 1")
    me = player(mate_attack=3, endgame_pieces=6, endgame_mate=5)
    move = me.choose(board)
    board.push(move)
    fast = fast_rules.board(board)
    assert board.is_checkmate() or fast.forced_mate(True, 4, 400_000, 1)[0] > 0


def test_phase_values_follow_the_game():
    me = player(mate_attack=0, phase_values="1,2,3,4,5;1,3,3,5,9;1,6,7,11,22")
    me.choose(chess.Board())
    assert me.values[1:] == [1, 2, 3, 4, 5]                          # opening
    me.choose(chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 30"))
    assert me.values[1:] == [1, 6, 7, 11, 22]                        # ending (8 pieces)


def test_fifty_horizon_resets_the_count_when_winning():
    board = chess.Board("6k1/8/8/8/8/8/P4Q2/6K1 w - - 96 80")        # a queen up, 4 half-moves to the draw
    me = player(mate_attack=0, ahead_draws=True, ahead_value=0.0, fifty_horizon=10)
    assert board.is_zeroing(me.choose(board))


def test_book_prefers_the_move_that_won_in_its_own_games(tmp_path):
    import json
    from claudio_agent import _key
    fast = fast_rules.board(chess.Board())
    book = {"positions": {_key(fast): {"e2e4": [40, 32], "d2d4": [40, 4]}}}
    path = tmp_path / "book.json"
    path.write_text(json.dumps(book))
    assert player(mate_attack=0, book=str(path), book_weight=5.0).choose(chess.Board()).uci() == "e2e4"
    book["positions"][_key(fast)] = {"e2e4": [40, 4], "d2d4": [40, 32]}
    path2 = tmp_path / "book2.json"
    path2.write_text(json.dumps(book))
    assert player(mate_attack=0, book=str(path2), book_weight=5.0).choose(chess.Board()).uci() == "d2d4"


def test_critical_moment_calculates_deeper():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    me = player(mate_attack=0, critical_depth=5, critical_gap=1e9)
    me.choose(board)
    assert me.critical_moves == 1 and me._depth == 5


def test_does_not_leave_its_queen_to_a_pawn():
    board = chess.Board("4k3/8/8/8/2p5/3Q4/8/4K3 w - - 0 1")      # ...c4 attacks the queen on d3
    move = player(mate_attack=0).choose(board)
    fast = fast_rules.board(board)
    loss = fast.exchange_after_moves([fast_rules.code(move)], [0, 1, 3, 3, 5, 9], 5, 200_000)[0]
    gain = 1.0 if board.is_capture(move) else 0.0
    assert gain - loss >= 0


def test_memory_sees_each_real_position_once():
    net = SurvivalNet(None).eval()
    board = chess.Board()
    incremental = Claudio(net, VALUES, ClaudioConfig(), rng=np.random.default_rng(0))
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"):
        if board.turn == chess.WHITE:
            incremental.choose(board)
        board.push_uci(uci)
    incremental.choose(board)
    x = torch.from_numpy(np.stack([encode(b, chess.WHITE) for b in _positions(board)]))
    with torch.inference_mode():
        latent, memory = net.encode(x), net.initial_memory(1)
        for row in range(len(x)):
            memory = net.memory_cell(latent[row:row + 1], memory)
    assert torch.allclose(incremental.memory, memory)


def _positions(board):
    replay, out = board.root(), [board.root()]
    for move in board.move_stack:
        replay.push(move)
        out.append(replay.copy())
    return out
