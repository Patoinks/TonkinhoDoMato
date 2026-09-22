"""The C++ rules must return EXACTLY what the Python references return."""
import random

import chess
import pytest

from tonkai import fast_rules
from tonkai.cognitive_agent import terminal_value
from tonkai.king_safety import king_senses
from tonkai.piece_safety import piece_senses, position_risk

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")

FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",   # kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "4k3/1P6/8/8/8/8/6p1/4K3 w - - 0 1",                                  # promotions both ways
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",                                  # en passant
    "6k1/8/8/8/8/8/8/R5K1 w - - 98 80",                                   # fifty-move edge
]


def games(seed, count, plies=160, shuffle=False):
    """Random games that favour captures (exchanges) or shuffle knights (repetitions)."""
    rng = random.Random(seed)
    for n in range(count):
        board = chess.Board(FENS[n % len(FENS)])
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over(claim_draw=False):
                break
            captures = [m for m in moves if board.is_capture(m)]
            knights = [m for m in moves if board.piece_type_at(m.from_square) == chess.KNIGHT]
            if shuffle and knights and rng.random() < .8:
                move = rng.choice(knights)
            elif captures and rng.random() < .35:
                move = rng.choice(captures)
            else:
                move = rng.choice(moves)
            board.push(move)
            yield board.copy(stack=True)


def valuer(seed):
    """Values depend only on (seed, piece, square), never on call order."""
    def value(piece, square):
        if piece.piece_type == chess.KING:
            return 10000.0
        rng = random.Random(seed * 4099 + piece.color * 2053 + piece.piece_type * 67 + square)
        # coarse values make exact ties (and their tie-breaks) common
        return rng.choice([1.0, 1.0, 3.0, 3.0, 3.25, 5.0, 9.0]) * (1 if piece.color else -1) \
            + rng.choice([0.0, 0.0, 0.5, -0.25])
    return value


def test_legal_moves_match_python_chess():
    for board in games(1, 60):
        fast = fast_rules.board(board)
        assert sorted(fast.legal_moves()) == sorted(fast_rules.code(m) for m in board.legal_moves), board.fen()


@pytest.mark.parametrize("shuffle", [False, True])
def test_terminal_value_matches(shuffle):
    seen = set()
    for board in games(2 + shuffle, 80, plies=220, shuffle=shuffle):
        fast = fast_rules.board(board)
        for objective, weight in (("survival", .5), ("victory", .5), ("shaped", .5), ("shaped", .3)):
            for colour in (chess.WHITE, chess.BLACK):
                expected = terminal_value(board, colour, objective, weight)
                assert fast_rules.terminal_value(fast, colour, objective, weight) == expected, board.fen()
        seen.add(terminal_value(board, chess.WHITE))
    assert None in seen and len(seen) > 1          # both ongoing and finished positions were exercised


def test_king_senses_match_field_by_field():
    names = lambda mask: sorted(chess.square_name(s) for s in chess.SquareSet(mask))
    for board in games(4, 60):
        fast = fast_rules.board(board)
        expected, got = king_senses(board), fast.king_detail()
        for side in ("w", "b"):
            row, mine = expected[side], got[side]
            assert chess.square_name(mine["square"]) == row["square"]
            for field in ("risk", "shelter", "force", "contacts", "open_rays", "uncovered", "in_check"):
                assert mine[field] == row[field], (field, board.fen())
            for field in ("attackers", "defenders", "escapes"):
                assert names(mine[field]) == row[field], (field, board.fen())
        white, black = fast.king_risks()
        assert (white, black) == (expected["w"]["risk"], expected["b"]["risk"])


@pytest.mark.parametrize("tolerance,depth", [(.5, 6), (.25, 3), (1.0, 1)])
def test_piece_risks_match_python(tolerance, depth):
    for n, board in enumerate(games(5, 50)):
        value = valuer(n)
        rows = piece_senses(board, value, tolerance=tolerance, depth=depth)
        expected = (position_risk(rows, chess.WHITE), position_risk(rows, chess.BLACK))
        fast = fast_rules.board(board)
        got = fast_rules.piece_risks(fast, valuer(n), fast_rules.items(fast), tolerance=tolerance, depth=depth)
        assert got == expected, board.fen()


def test_push_pop_restore_the_position():
    board = next(iter(games(6, 1)))
    fast = fast_rules.board(board)
    before = (fast.legal_moves(), fast.king_risks(), fast.terminal(), fast.ply)
    for move in list(board.legal_moves)[:10]:
        fast.push(fast_rules.code(move))
        fast.pop()
    assert (fast.legal_moves(), fast.king_risks(), fast.terminal(), fast.ply) == before


def test_encoding_after_moves_is_bit_identical():
    import numpy as np
    from tonkai.encoding import encode_after_moves
    for board in games(7, 60):
        moves = list(board.legal_moves)
        fast = fast_rules.board(board)
        codes = [fast_rules.code(m) for m in moves]
        for colour in (chess.WHITE, chess.BLACK):
            expected = encode_after_moves(board, moves, colour)
            got = fast_rules.encode_after_moves(fast, codes, colour)
            assert got.dtype == expected.dtype and got.shape == expected.shape
            assert np.array_equal(got, expected), board.fen()
