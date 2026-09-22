"""Per-piece exchange and legal-recapture awareness, in learned pawn units.

The bounded exchange probe follows the least valuable legal recapturer. It is
local tactical evidence, not a proof: an opponent may choose another square.
"""
import math

import chess


def captured_square(board, move):
    return move.to_square + (-8 if board.turn else 8) if board.is_en_passant(move) else move.to_square


def captures_to(board, square):
    return list(board.generate_legal_captures(to_mask=chess.BB_SQUARES[square]))


def exchange_gain(board, first, value, depth=6):
    """Gain for first mover, allowing either side to stop exchanging.

    Values are frozen to the evaluated position, but occupancy, pins, checks,
    x-rays, en passant and promotions are updated after every capture.
    """
    return _exchange_probe(board, first, value, depth)[0]


def _exchange_probe(board, first, value, depth, root_values=None, with_defenders=False):
    work = board.copy(stack=False)
    values = (root_values.copy() if root_values is not None else
              {sq: value(piece, sq) for sq, piece in work.piece_map().items()})
    defenders = []

    def follow(move, remaining):
        target = captured_square(work, move)
        gain = values.get(target, 0.0)
        moving_value = values[move.from_square]
        promoted = value(chess.Piece(move.promotion, work.turn), move.to_square) if move.promotion else moving_value
        if move.promotion:
            gain += promoted - moving_value
        values.pop(target, None)
        values.pop(move.from_square, None)
        values[move.to_square] = promoted
        work.push(move)
        if remaining <= 1 and not (remaining == depth and with_defenders):
            return gain
        replies = captures_to(work, move.to_square)
        if remaining == depth and with_defenders:
            defenders.extend(sorted({chess.square_name(m.from_square) for m in replies}))
        if remaining <= 1:
            return gain
        if not replies:
            return gain
        reply = min(replies, key=lambda m: (values[m.from_square], m.uci()))
        return gain - max(0.0, follow(reply, remaining - 1))

    return follow(first, depth), defenders


def piece_senses(board, value, *, tolerance=.5, depth=6):
    """Return all pieces, with priority 2/3 only when exchange is near equal.

    Multiple attackers are distinct pieces (promotion choices count once).
    Defenders are legal recapturers AFTER each potential capture, not nominal
    attack-map contacts. The worst branch determines the defensive requirement.
    Kings are reported as check threats, never as exchangeable material.
    """
    pieces = board.piece_map()
    root_values = {sq: value(piece, sq) for sq, piece in pieces.items()}
    captures = {}
    for colour in (chess.WHITE, chess.BLACK):
        work = board.copy(stack=False)
        work.turn = not colour
        if work.turn != board.turn:
            work.ep_square = None  # EP exists only for the actual side to move.
        by_square = {}
        for move in work.generate_legal_captures():
            square = captured_square(work, move)
            victim = work.piece_at(square)
            if victim and victim.color == colour and victim.piece_type != chess.KING:
                by_square.setdefault(square, []).append(move)
        captures[colour] = (work, by_square)

    result = []
    for square, piece in sorted(pieces.items()):
        work, available = captures[piece.color]
        attacks = available.get(square, [])
        row = dict(square=chess.square_name(square), color="w" if piece.color else "b", piece=piece.symbol(),
                   value=None if piece.piece_type == chess.KING else root_values[square],
                   attacked=False, attackers=0, defenders=0, required_defenders=0, defender_squares=[],
                   exchange_balance=None, priority=0, deficit=0, risk=0.0, branches=[])
        if piece.piece_type == chess.KING:
            row["attacked"] = board.is_attacked_by(not piece.color, square)
            row["priority"] = 1 if row["attacked"] else 0
            result.append(row)
            continue
        if not attacks:
            result.append(row)
            continue
        attacker_count = len({m.from_square for m in attacks})
        for move in attacks:
            gain, defenders = _exchange_probe(work, move, value, depth, root_values, True)
            row["branches"].append(dict(move=move.uci(), balance=-gain, defenders=defenders))
        worst = min(row["branches"], key=lambda branch: (branch["balance"], len(branch["defenders"])))
        balance = worst["balance"]
        row.update(attacked=True, attackers=attacker_count, exchange_balance=balance, priority=1,
                   defenders=len(worst["defenders"]), defender_squares=worst["defenders"])
        if balance < -tolerance:
            # Losing exchanges dominate any equal-exchange support deficit.
            row["risk"] = .5 + .5 * min(1.0, (-balance - tolerance) / 4)
        elif abs(balance) <= tolerance:
            row["priority"] = 2 if row["defenders"] == 0 else 3
            row["required_defenders"] = attacker_count
            row["deficit"] = max(0, attacker_count - row["defenders"])
            row["risk"] = .4 if not row["defenders"] else .3 * row["deficit"] / attacker_count
        result.append(row)
    return result


def position_risk(rows, colour):
    risks = [row["risk"] for row in rows if row["color"] == ("w" if colour else "b")]
    # Protect the worst exposed piece; other pieces are a bounded tie-break.
    return min(1.0, max(risks, default=0.0) + .05 * sum(risks))


def pawn_unit(raw_values, pieces, floor=.001):
    """Robust local scale; not a calibrated classical pawn/centipawn estimate."""
    values = sorted(abs(float(v)) for v, p in zip(raw_values, pieces) if p.piece_type == chess.PAWN and math.isfinite(v))
    if not values:
        return floor
    middle = len(values) // 2
    return max(floor, (values[middle] + values[(len(values) - 1) // 2]) / 2)
