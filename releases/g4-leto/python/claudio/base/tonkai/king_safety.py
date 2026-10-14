"""Bounded, geometric king-pressure prior; no material or mate evaluation.

Pinned pieces only contribute along their pin. Escape squares use raw attacks:
even a pinned enemy piece forbids a king move under chess rules.
"""
import chess


DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

# Geometry is immutable; occupancy and attacks are always recomputed.
OPEN_RAYS = []
SHELTER = {True: [], False: []}
for king in chess.SQUARES:
    kfile, krank = chess.square_file(king), chess.square_rank(king)
    rays = []
    for dx, dy in DIRECTIONS:
        if 0 <= kfile + 2 * dx < 8 and 0 <= krank + 2 * dy < 8:
            rays.append(chess.BB_SQUARES[chess.square(kfile + dx, krank + dy)]
                        | chess.BB_SQUARES[chess.square(kfile + 2 * dx, krank + 2 * dy)])
    OPEN_RAYS.append(tuple(rays))
    for colour in (True, False):
        columns = []
        for file in range(max(0, kfile - 1), min(8, kfile + 2)):
            column = []
            for distance in (1, 2):
                rank = krank + (distance if colour else -distance)
                column.append(chess.BB_SQUARES[chess.square(file, rank)] if 0 <= rank < 8 else 0)
            columns.append(tuple(column))
        SHELTER[colour].append(tuple(columns))


def king_senses(board):
    kings = {colour: board.king(colour) for colour in (True, False)}
    if None in kings.values():
        raise ValueError("King safety requires both kings")
    zones = {colour: chess.BB_KING_ATTACKS[sq] | chess.BB_SQUARES[sq] for colour, sq in kings.items()}
    relevant = zones[True] | zones[False]
    effective = {}
    for sq in chess.scan_reversed(board.occupied & ~board.kings):
        mask = board.attacks_mask(sq)
        if mask & relevant:
            colour = bool(board.occupied_co[True] & chess.BB_SQUARES[sq])
            effective[sq] = (mask & board.pin_mask(colour, sq), colour)
    result = {}
    for colour in (chess.WHITE, chess.BLACK):
        king = kings[colour]
        ring = chess.BB_KING_ATTACKS[king]
        zone = ring | chess.BB_SQUARES[king]
        attackers = [sq for sq, (mask, side) in effective.items() if side != colour and mask & zone]
        defenders = [sq for sq, (mask, side) in effective.items() if side == colour and mask & zone]
        hostile, friendly, contacts = 0, 0, 0
        for sq in attackers:
            hostile |= effective[sq][0] & zone
            contacts += (effective[sq][0] & zone).bit_count()
        for sq in defenders:
            friendly |= effective[sq][0] & zone
        threatened = hostile & ring
        uncovered = (threatened & ~friendly).bit_count() / max(1, threatened.bit_count())
        pawn_mask = board.pawns & board.occupied_co[colour]
        columns = SHELTER[colour][king]
        shelter = sum(1. if pawn_mask & first else .5 if pawn_mask & second else 0.
                      for first, second in columns) / len(columns)
        open_rays = sum(not (board.occupied & ray) for ray in OPEN_RAYS[king])
        # Removing the king reveals x-rays behind its original square. The
        # potential captured piece cannot attack its own square.
        occupancy = board.occupied & ~chess.BB_SQUARES[king]
        escapes = [sq for sq in chess.scan_forward(ring & ~board.occupied_co[colour])
                   if sq != kings[not colour]
                   and not board.attackers_mask(not colour, sq, occupancy)]
        enemy = board.occupied_co[not colour]
        force = min(1., (4 * (board.queens & enemy).bit_count() + 2 * (board.rooks & enemy).bit_count()
                        + ((board.bishops | board.knights) & enemy).bit_count()) / 8)
        pressure = min(1., contacts / 12)
        coordination = min(1., max(0, len(attackers) - 1) / 3)
        gate = min(1., len(attackers) / 2) * (.35 + .65 * force)
        checked = board.is_attacked_by(not colour, king)
        risk = min(1., gate * (.25 * pressure + .15 * coordination + .18 * (1 - shelter) * force
                              + .12 * open_rays / 8 + .18 * uncovered
                              + .12 * (1 - min(1., len(escapes) / 3))) + .15 * checked)
        result['w' if colour else 'b'] = dict(square=chess.square_name(king), risk=risk,
            shelter=shelter, force=force, contacts=contacts, open_rays=open_rays,
            uncovered=uncovered, in_check=checked,
            attackers=sorted(chess.square_name(sq) for sq in attackers),
            defenders=sorted(chess.square_name(sq) for sq in defenders),
            escapes=sorted(chess.square_name(sq) for sq in escapes))
    return result


def king_balance(senses, colour):
    own, enemy = ('w', 'b') if colour else ('b', 'w')
    return senses[enemy]['risk'] - senses[own]['risk']
