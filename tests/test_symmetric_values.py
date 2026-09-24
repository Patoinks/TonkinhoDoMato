"""symmetric_piece_values: the network values an enemy piece as it would the
same piece of its own on the mirrored square - one scale for both sides."""
import chess
import pytest

from test_mate_calculation import agent

ITALIAN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


def values(tonkai, board, colour):
    tonkai.reset()
    tonkai.observe(board, colour)
    value = tonkai.piece_valuer(board, colour)
    return {square: value(piece, square) for square, piece in board.piece_map().items()
            if piece.piece_type != chess.KING}


@pytest.mark.parametrize("colour", [chess.WHITE, chess.BLACK])
def test_mirrored_pieces_are_worth_the_same(colour):
    worth = values(agent(symmetric_piece_values=True), chess.Board(), colour)
    for square, value in worth.items():
        assert worth[square ^ 56] == pytest.approx(-value)            # the mirror belongs to the other side
    assert sum(worth.values()) == pytest.approx(0.0, abs=1e-6)       # equal material, equal value


def test_values_still_depend_on_the_position():
    worth = values(agent(symmetric_piece_values=True), chess.Board(ITALIAN), chess.WHITE)
    own_knight, enemy_knight = worth[chess.F3], worth[chess.C6]
    assert own_knight > 0 > enemy_knight


def test_off_by_default_keeps_the_side_embedding():
    worth = values(agent(), chess.Board(), chess.WHITE)
    assert any(abs(worth[square] + worth[square ^ 56]) > 1e-6 for square in worth)
