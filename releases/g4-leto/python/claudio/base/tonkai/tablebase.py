"""Endgame tablebases (Syzygy): the rules-only truth of every position with few
pieces. Used to end training/match games early with the exact result and to
value moves that enter a tablebase ending. Never an engine opinion."""
from __future__ import annotations

import chess
import chess.syzygy

MAX_PIECES = 5                       # the 3-4-5 tables (kings included)
_open: dict[str, chess.syzygy.Tablebase] = {}


def tablebase(path: str):
    if not path:
        return None
    tables = _open.get(path)
    if tables is None:
        tables = _open[path] = chess.syzygy.open_tablebase(path)
    return tables


def in_tables(board: chess.Board) -> bool:
    return chess.popcount(board.occupied) <= MAX_PIECES and not board.castling_rights


def verdict(board: chess.Board, path: str):
    """True/False = that colour wins, None = draw, "" = not in the tables.

    A cursed win / blessed loss (a result the 50-move rule turns into a draw)
    counts as a draw."""
    if not path or not in_tables(board):
        return ""
    try:
        wdl = tablebase(path).probe_wdl(board)
    except (KeyError, chess.syzygy.MissingTableError):
        return ""
    if wdl >= 2:
        return board.turn
    if wdl <= -2:
        return not board.turn
    return None


def enters_tables(board: chess.Board, move: chess.Move) -> bool:
    """Cheap test: after `move` the position has at most MAX_PIECES pieces."""
    return chess.popcount(board.occupied) - board.is_capture(move) <= MAX_PIECES
