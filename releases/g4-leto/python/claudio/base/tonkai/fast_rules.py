"""Bridge to ``tonkai_fast``, the C++ port of the V3 agent's chess rules.

Each call returns exactly what the Python reference returns; the references
stay in ``king_safety``, ``piece_safety`` and ``cognitive_agent.terminal_value``
and tests/test_fast_rules.py holds the two against each other. When the
extension is not built, ``AVAILABLE`` is False and the agent uses Python.
"""
from __future__ import annotations

import chess
import os

try:
    import tonkai_fast
except ImportError:  # extension not built: python setup.py build_ext --inplace
    tonkai_fast = None

AVAILABLE = tonkai_fast is not None
REQUESTED_BACKEND = os.environ.get("TONKAI_RULES_BACKEND", "auto").lower()
if REQUESTED_BACKEND not in ("auto", "python", "cpp"):
    raise ValueError("TONKAI_RULES_BACKEND must be auto, python or cpp")
if REQUESTED_BACKEND == "cpp" and not AVAILABLE:
    raise RuntimeError("C++ rules requested but tonkai_fast is unavailable. Run python setup.py build_ext --inplace")
ENABLED = AVAILABLE and REQUESTED_BACKEND != "python"
BACKEND = "cpp" if ENABLED else "python"
NONE, DRAW, WHITE_WINS, BLACK_WINS = -1, 0, 1, 2
PIECES = {(kind, colour): chess.Piece(kind, colour) for kind in range(1, 7) for colour in (True, False)}


def code(move: chess.Move) -> int:
    return move.from_square | move.to_square << 6 | (move.promotion or 0) << 12


def board(position: chess.Board):
    """A C++ board with the same root position and complete move history."""
    if position.chess960:
        raise ValueError("tonkai_fast supports standard chess only")
    root = position.root()
    fast = tonkai_fast.Board(root.pawns, root.knights, root.bishops, root.rooks, root.queens, root.kings,
                             root.occupied_co[chess.WHITE], root.occupied_co[chess.BLACK], root.turn,
                             root.castling_rights, -1 if root.ep_square is None else root.ep_square,
                             root.halfmove_clock, root.fullmove_number)
    fast.push_many([code(move) for move in position.move_stack])
    return fast


def terminal_value(fast, colour, objective="survival", weight=0.5, draw=None, strict=False):
    """cognitive_agent.terminal_value on a C++ board (strict: repetition only
    once the position has occurred three times, see terminal_strict)."""
    result = fast.terminal_strict() if strict else fast.terminal()
    if result == NONE:
        return None
    if result == DRAW:
        if draw is not None:
            return float(draw)
        return .5 if objective == "victory" else 1.0 - weight / 2 if objective == "shaped" else 1.0
    return float((result == WHITE_WINS) == colour)


def king_balance(fast, colour):
    """king_safety.king_balance(king_senses(board), colour) on a C++ board."""
    white, black = fast.king_risks()
    return black - white if colour else white - black


def items(fast):
    """sorted(board.piece_map().items()) of a C++ board."""
    return [(square, PIECES[kind, colour]) for square, kind, colour in fast.pieces()]


def piece_risks(fast, value, pieces, *, tolerance, depth):
    """(position_risk(white), position_risk(black)) of piece_senses(board, value)."""
    values = [0.0] * 64
    for square, piece in pieces:
        values[square] = value(piece, square)
    return fast.piece_risks(values, lambda colour, kind, square: value(PIECES[kind, bool(colour)], square),
                            tolerance, depth)


def encode_after_moves(fast, codes, self_color):
    """encoding.encode_after_moves for the C++ board's position (bit-identical)."""
    return fast.encode_after(codes, self_color)[:len(codes)]
