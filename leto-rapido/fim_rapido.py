"""board.outcome(claim_draw=True) for a game loop, the same answer without replaying the game every ply.

python-chess's can_claim_threefold_repetition pops the game back to the last irreversible move, pushes it
again and then tries every legal move (~0.2 ms a ply, the tournament's own cost). The Judge keeps the same
count of positions since the last irreversible move as the game goes, and tries the legal moves only when
some position already stands twice (otherwise no move can make a third).

    judge = Judge(board)
    while judge.outcome(board) is None and ...:
        judge.push(board, move)
"""
import collections

import chess


class Judge:
    def __init__(self, board):
        self.counts = collections.Counter()
        b = board.copy()
        self.counts[b._transposition_key()] += 1
        while b.move_stack:                    # the same walk as can_claim_threefold_repetition
            move = b.pop()
            if b.is_irreversible(move):
                break
            self.counts[b._transposition_key()] += 1

    def push(self, board, move):
        if board.is_irreversible(move):        # (judged before the move, as python-chess does after pop)
            self.counts.clear()
        board.push(move)
        self.counts[board._transposition_key()] += 1

    def threefold(self, board):
        if self.counts[board._transposition_key()] >= 3:
            return True
        if max(self.counts.values()) < 2:
            return False
        for move in board.generate_legal_moves():
            board.push(move)
            try:
                if self.counts[board._transposition_key()] >= 2:
                    return True
            finally:
                board.pop()
        return False

    def outcome(self, board):
        outcome = board.outcome(claim_draw=False)
        if outcome is not None:
            return outcome
        if board.can_claim_fifty_moves():
            return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
        if self.threefold(board):
            return chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
        return None
