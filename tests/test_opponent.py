"""TinyOpponent: always legal, reproducible, and not a random mover."""

from __future__ import annotations

import random

import chess
import pytest

import tiny_opponent
from tonkai import chess_env as env


def test_always_returns_a_legal_move_over_many_random_positions():
    eng = tiny_opponent.TinyOpponent(depth=2, max_nodes=500, seed=7)
    rng = random.Random(1234)
    checked = 0
    for _ in range(40):
        board = chess.Board()
        for _ in range(rng.randint(0, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if board.is_game_over():
                break
            uci = eng.choose_move(board.fen())
            assert uci != "0000"
            assert chess.Move.from_uci(uci) in board.legal_moves, board.fen()
            checked += 1
    assert checked > 100


def test_returns_null_move_only_when_there_is_none():
    eng = tiny_opponent.TinyOpponent()
    assert eng.choose_move("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1") == "0000"  # stalemate


def test_same_seed_gives_the_same_game():
    def play(seed):
        eng = tiny_opponent.TinyOpponent(depth=2, max_nodes=500, seed=seed,
                                         randomness=0.5)
        board = chess.Board()
        for _ in range(30):
            if board.is_game_over():
                break
            env.apply_opponent_move(board, eng.choose_move(board.fen()))
        return board.fen()

    assert play(99) == play(99)


def test_different_seeds_diverge():
    def play(seed):
        eng = tiny_opponent.TinyOpponent(depth=2, max_nodes=500, seed=seed,
                                         randomness=0.6)
        board = chess.Board()
        for _ in range(30):
            if board.is_game_over():
                break
            env.apply_opponent_move(board, eng.choose_move(board.fen()))
        return board.fen()

    assert play(1) != play(123456)


def test_zero_randomness_is_deterministic():
    eng = tiny_opponent.TinyOpponent(depth=2, max_nodes=2000, randomness=0.0)
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    choices = {eng.choose_move(fen) for _ in range(20)}
    assert len(choices) == 1


def test_it_finds_mate_in_one():
    eng = tiny_opponent.TinyOpponent(depth=2, max_nodes=20000, randomness=0.0)
    # back-rank mate: Ra8#
    assert eng.choose_move("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1") == "a1a8"


def test_it_takes_a_free_queen():
    eng = tiny_opponent.TinyOpponent(depth=2, max_nodes=20000, randomness=0.0)
    # the black queen on d5 is hanging to the rook on d1
    assert eng.choose_move("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1") == "d1d5"


def test_it_does_not_hang_its_queen_for_nothing():
    """With quiescence on, a depth-2 search must not walk into a free capture."""
    eng = tiny_opponent.TinyOpponent(depth=2, max_nodes=50000, randomness=0.0,
                                     quiescence=True)
    fen = "4k3/8/8/8/8/5n2/8/3QK3 w - - 0 1"
    move = eng.choose_move(fen)
    board = chess.Board(fen)
    board.push(chess.Move.from_uci(move))
    # after the move the queen must not simply be capturable for free
    attackers = board.attackers(chess.BLACK, chess.Move.from_uci(move).to_square)
    if board.piece_at(chess.Move.from_uci(move).to_square).piece_type == chess.QUEEN:
        defenders = board.attackers(chess.WHITE,
                                    chess.Move.from_uci(move).to_square)
        assert not attackers or defenders


def _play(white, black, max_plies=160):
    board = chess.Board()
    while len(board.move_stack) < max_plies:
        if board.is_checkmate():
            return 0.0 if board.turn == chess.WHITE else 1.0
        if (not any(board.generate_legal_moves())
                or board.is_insufficient_material()
                or board.halfmove_clock >= 100):
            return 0.5
        eng = white if board.turn == chess.WHITE else black
        env.apply_opponent_move(board, eng.choose_move(board.fen()))
    return 0.5


def test_more_depth_beats_less_depth():
    """Measured over 80 games this pairing scores 0.94, so 7/10 is a safe bar.

    The node budget has to be big enough for depth 2 to actually complete -
    otherwise this measures the node cap, not the depth.
    """
    strong = tiny_opponent.TinyOpponent(depth=2, max_nodes=200_000,
                                        randomness=0.0, seed=1)
    weak = tiny_opponent.TinyOpponent(depth=1, max_nodes=200_000,
                                      randomness=0.5, seed=2)
    score = sum(_play(strong, weak) for _ in range(5))
    score += sum(1.0 - _play(weak, strong) for _ in range(5))
    assert score >= 7.0, f"depth 2 only scored {score}/10 against depth 1"


def test_node_cap_is_respected():
    eng = tiny_opponent.TinyOpponent(depth=4, max_nodes=200, randomness=0.0)
    eng.reset_stats()
    eng.choose_move(chess.STARTING_FEN)
    assert eng.nodes <= 400  # the cap plus a little slack for the abort path


def test_node_budget_controls_the_depth_actually_reached():
    """The cap must weaken the search smoothly, not corrupt it.

    Root iterative deepening means a depth is only used once every root move
    has been searched at it, so a small budget just means a shallower - but
    still coherent - ranking.
    """
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    depths = []
    for nodes in (1000, 5000, 200_000):
        eng = tiny_opponent.TinyOpponent(depth=3, max_nodes=nodes,
                                         randomness=0.0)
        eng.choose_move(fen)
        depths.append(eng.last_depth)
    assert depths == sorted(depths), depths
    assert depths[0] >= 1, "even a tiny budget must complete depth 1"
    assert depths[-1] == 3


def test_a_tiny_node_budget_still_plays_a_legal_sensible_move():
    eng = tiny_opponent.TinyOpponent(depth=3, max_nodes=50, randomness=0.0)
    fen = "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1"
    move = eng.choose_move(fen)
    assert chess.Move.from_uci(move) in chess.Board(fen).legal_moves
    # depth 1 is always completed, so the free queen is still taken
    assert move == "d1d5"
