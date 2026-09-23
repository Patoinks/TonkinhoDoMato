"""Exact exchange search against known tactics and the python-chess oracle."""
from __future__ import annotations

from pathlib import Path

import chess
import chess.pgn
import pytest

from tonkai import fast_rules
from tonkai import exchange


pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")
VALUES = [0.0, 1.0, 3.9, 4.6, 5.2, 10.3]
LIMIT = 10_000_000
PGN = Path(__file__).resolve().parents[1] / "benchmarks/benchmark-v3.1-vs-checkpoint1-500.pgn"


@pytest.mark.parametrize("fen,expected", [
    ("n6k/8/8/8/8/8/8/R6K w - - 0 1", 3.9),  # hanging knight
    ("rr6/7k/8/8/8/8/8/R6K w - - 0 1", 0.0),  # equal rook exchange
    ("q6k/p7/8/8/8/8/7K/R7 w - - 0 1", 0.0),  # pawn, then costly recapture
    ("r6k/rR6/8/8/8/8/8/R6K w - - 0 1", 5.2),  # x-ray rook battery
    ("7k/8/8/3pP3/8/8/8/7K w - d6 0 1", 1.0),  # en passant
    ("k6r/6P1/8/8/8/8/8/6K1 w - - 0 1", 14.5),  # capture promotion
    ("kb6/8/8/8/8/8/7r/7K w - - 0 1", 0.0),  # quiet evasion in check
    ("7k/6r1/5KQ1/8/8/8/8/8 w - - 0 1", 1005.2),  # capture checkmate
    ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", -1000.0),  # already mated
])
def test_hand_positions(fen: str, expected: float) -> None:
    board = chess.Board(fen)
    assert board.is_valid()
    original = board.fen(en_passant="fen")
    fast = fast_rules.board(board)
    gain, complete, nodes = fast.exchange_gain(VALUES, 8, LIMIT)
    assert complete and nodes > 0
    assert gain == pytest.approx(expected, abs=1e-9)
    assert gain == pytest.approx(exchange.exchange_gain(board, VALUES, 8), abs=1e-9)
    assert board.fen(en_passant="fen") == original
    assert fast.fen() == original


def test_after_moves_and_limits() -> None:
    board = chess.Board("q6k/p7/8/8/8/8/7K/R7 w - - 0 1")
    fast = fast_rules.board(board)
    original = board.fen(en_passant="fen")
    move = chess.Move.from_uci("a1a7")
    assert fast.exchange_after_moves([fast_rules.code(move)], VALUES, 8, LIMIT) == pytest.approx([5.2])
    assert exchange.exchange_after_moves(board, [move], VALUES, 8) == pytest.approx([5.2])
    assert fast.fen() == board.fen(en_passant="fen") == original
    assert fast.exchange_gain(VALUES, 8, 0)[1:] == (False, 1)
    with pytest.raises(RuntimeError, match="node_limit"):
        fast.exchange_after_moves([fast_rules.code(move)], VALUES, 8, 0)
    with pytest.raises(ValueError, match="illegal"):
        exchange.exchange_after_moves(board, [chess.Move.from_uci("a1a8")], VALUES, 8)
    with pytest.raises(ValueError, match="illegal"):
        fast.exchange_after_moves([fast_rules.code(chess.Move.from_uci("a1a8"))], VALUES, 8, LIMIT)
    assert fast.fen() == board.fen(en_passant="fen") == original


def test_all_after_moves_match_python_reference() -> None:
    board = chess.Board("n6k/8/8/8/8/8/8/R6K w - - 0 1")
    fast = fast_rules.board(board)
    moves = list(board.legal_moves)
    original = board.fen(en_passant="fen")
    actual = fast.exchange_after_moves([fast_rules.code(move) for move in moves], VALUES, 8, LIMIT)
    expected = exchange.exchange_after_moves(board, moves, VALUES, 8)
    assert actual == pytest.approx(expected, abs=1e-9)
    assert fast.fen() == board.fen(en_passant="fen") == original


def test_horizon_detects_mate() -> None:
    mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    check = chess.Board("kb6/8/8/8/8/8/7r/7K w - - 0 1")
    assert fast_rules.board(mate).exchange_gain(VALUES, 0, LIMIT)[0] == -1000.0
    assert exchange.exchange_gain(mate, VALUES, 0) == -1000.0
    assert fast_rules.board(check).exchange_gain(VALUES, 0, LIMIT)[0] == 0.0
    assert exchange.exchange_gain(check, VALUES, 0) == 0.0


def test_pgn_positions_match_python_reference() -> None:
    positions = []
    with PGN.open(encoding="utf-8") as stream:
        for game_index in range(300):
            game = chess.pgn.read_game(stream)
            assert game is not None
            moves = list(game.mainline_moves())
            assert moves
            board = game.board()
            for move in moves[:min(len(moves), 8 + 2 * (game_index % 8))]:
                board.push(move)
            positions.append(board)
    assert len(positions) == 300
    for board in positions:
        original = board.fen(en_passant="fen")
        fast = fast_rules.board(board)
        result, complete, _ = fast.exchange_gain(VALUES, 8, LIMIT)
        assert complete, original
        assert result == pytest.approx(exchange.exchange_gain(board, VALUES, 8), abs=1e-9), original
        assert board.fen(en_passant="fen") == original
        assert fast.fen() == original
