"""Actual terminal rules, plus a conservative certificate for dead locked-pawn positions.

A material advantage or a search timeout is never an adjudicated result.
"""
import chess


def dead_pawn_lock(board):
    if not board.is_valid() or board.is_check():
        return False
    if board.knights or board.bishops or board.rooks or board.queens or not board.pawns:
        return False
    pawn_attacks = {}
    for colour in (chess.WHITE, chess.BLACK):
        pawns = board.pawns & board.occupied_co[colour]
        enemy = board.pawns & board.occupied_co[not colour]
        attacks = 0
        for square in chess.scan_forward(pawns):
            target = square + (8 if colour else -8)
            if not 0 <= target < 64 or not board.pawns & chess.BB_SQUARES[target]:
                return False
            attacks |= chess.BB_PAWN_ATTACKS[colour][square]
        if attacks & enemy:
            return False
        pawn_attacks[colour] = attacks
    for colour in (chess.WHITE, chess.BLACK):
        king = board.king(colour)
        blocked = (board.pawns & board.occupied_co[colour]) | pawn_attacks[not colour]
        if king is None or blocked & chess.BB_SQUARES[king]:
            return False
        reachable, frontier = 0, chess.BB_SQUARES[king]
        while frontier:
            reachable |= frontier
            next_ = 0
            for square in chess.scan_forward(frontier):
                next_ |= chess.BB_KING_ATTACKS[square]
            frontier = next_ & ~blocked & ~reachable & chess.BB_ALL
        if reachable & board.pawns & board.occupied_co[not colour]:
            return False
    # All pawns stay fixed, neither king can reach a pawn, and moving kings cannot give check.
    return True


def adjudicate(board, claim_draw=True):
    outcome = board.outcome(claim_draw=claim_draw)
    if outcome is not None:
        return {'winner': outcome.winner, 'end': outcome.termination.name,
                'result': outcome.result(), 'draw_points': 0.5 if outcome.winner is None else None}
    if dead_pawn_lock(board):
        return {'winner': None, 'end': 'DEAD_PAWN_LOCK', 'result': '1/2-1/2', 'draw_points': 0.5}
    return None


def draw_config(config):
    # Winning positions still seek proved mates; all actual/prospective draws have the requested value.
    return dict(config, draw_value=0.5, draw_ahead=0.5)
