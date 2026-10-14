"""The rules. python-chess is the single source of truth for legality.

TonkAI never gets to pick a move that did not come out of `board.legal_moves`,
so it cannot invent an illegal move or exploit a movegen bug. TinyOpponent has
its own generator for search speed; whatever it returns is checked against
python-chess before it is played (see `apply_opponent_move`).
"""

from __future__ import annotations

import chess

RESULT_WIN = 0
RESULT_DRAW = 1
RESULT_LOSS = 2
RESULT_NAMES = ("win", "draw", "loss")

# why the game ended, for logging
TERM_CHECKMATE = "checkmate"
TERM_STALEMATE = "stalemate"
TERM_INSUFFICIENT = "insufficient_material"
TERM_FIFTY = "fifty_move"
TERM_REPETITION = "threefold_repetition"
TERM_PLY_LIMIT = "ply_limit"


class IllegalOpponentMove(RuntimeError):
    pass


def new_board() -> chess.Board:
    return chess.Board()


def terminal(board: chess.Board, self_color: bool, max_plies: int):
    """Return (result, reason) if the game is over, else (None, None).

    `result` is always from TonkAI's point of view.
    """
    if board.is_checkmate():
        # the side to move has been mated
        res = RESULT_LOSS if board.turn == self_color else RESULT_WIN
        return res, TERM_CHECKMATE
    if not any(board.generate_legal_moves()):
        return RESULT_DRAW, TERM_STALEMATE
    if board.is_insufficient_material():
        return RESULT_DRAW, TERM_INSUFFICIENT
    if board.halfmove_clock >= 100:
        return RESULT_DRAW, TERM_FIFTY
    # a repetition needs at least four reversible plies, so this gate is free
    if board.halfmove_clock >= 4 and board.is_repetition(3):
        return RESULT_DRAW, TERM_REPETITION
    if len(board.move_stack) >= max_plies:
        return RESULT_DRAW, TERM_PLY_LIMIT
    return None, None


def apply_opponent_move(board: chess.Board, uci: str) -> chess.Move:
    """Play TinyOpponent's move after checking it against python-chess."""
    if uci == "0000":
        raise IllegalOpponentMove(f"TinyOpponent returned a null move: {board.fen()}")
    try:
        move = chess.Move.from_uci(uci)
    except ValueError as exc:
        raise IllegalOpponentMove(f"unparseable move {uci!r}") from exc
    if move not in board.legal_moves:
        raise IllegalOpponentMove(
            f"TinyOpponent played an illegal move {uci!r} in {board.fen()}"
        )
    board.push(move)
    return move


def result_from_white_pov(result: int, self_color: bool) -> str:
    """'1-0' / '0-1' / '1/2-1/2' for PGN-ish logging."""
    if result == RESULT_DRAW:
        return "1/2-1/2"
    self_won = result == RESULT_WIN
    white_won = self_won if self_color == chess.WHITE else not self_won
    return "1-0" if white_won else "0-1"


def flip_result(result: int) -> int:
    if result == RESULT_WIN:
        return RESULT_LOSS
    if result == RESULT_LOSS:
        return RESULT_WIN
    return RESULT_DRAW
