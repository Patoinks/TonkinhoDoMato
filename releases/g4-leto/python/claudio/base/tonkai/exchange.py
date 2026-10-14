"""Reference capture search using python-chess's legal moves."""
from __future__ import annotations

import chess


MATE = 1000.0


def _move_gain(board: chess.Board, move: chess.Move, values: list[float]) -> float:
    if board.is_en_passant(move):
        victim = chess.PAWN
    else:
        piece = board.piece_at(move.to_square)
        victim = piece.piece_type if piece is not None else 0
    gain = values[victim] if victim else 0.0
    if move.promotion:
        gain += values[move.promotion] - values[chess.PAWN]
    return gain


def exchange_gain(board: chess.Board, values: list[float], max_depth: int) -> float:
    """Material the side to move can force in at most max_depth plies.

    ``values`` has six entries indexed by python-chess piece type 0..5;
    the king has no material value. The input board is left unchanged.
    """
    if len(values) != 6 or max_depth < 0:
        raise ValueError("exchange_gain requires six values and non-negative max_depth")

    def q(depth: int) -> float:
        check = board.is_check()
        moves = list(board.legal_moves) if check or depth else []
        if check and not moves:
            return -MATE
        if not depth:
            return 0.0
        if not check:
            moves = [move for move in moves if board.is_capture(move)]
        best = float("-inf") if check else 0.0
        for move in moves:
            gain = _move_gain(board, move, values)
            board.push(move)
            try:
                score = gain - q(depth - 1)
            finally:
                board.pop()
            best = max(best, score)
        return best

    return q(max_depth)


def exchange_after_moves(board: chess.Board, moves: list[chess.Move], values: list[float],
                         max_depth: int) -> list[float]:
    """For each legal move, return the opponent's subsequent exchange gain."""
    out = []
    for move in moves:
        if not board.is_legal(move):
            raise ValueError(f"illegal move in exchange_after_moves: {move}")
        board.push(move)
        try:
            out.append(exchange_gain(board, values, max_depth))
        finally:
            board.pop()
    return out
