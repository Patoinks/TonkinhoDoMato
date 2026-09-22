"""Bounded, geometric king-pressure prior; no material or mate evaluation.

Pinned pieces only contribute along their pin. Escape squares use raw attacks:
even a pinned enemy piece forbids a king move under chess rules.
"""
import chess


DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def king_senses(board):
    pieces = board.piece_map()
    effective = {sq: board.attacks_mask(sq) & board.pin_mask(p.color, sq)
                 for sq, p in pieces.items() if p.piece_type != chess.KING}
    result = {}
    for colour in (chess.WHITE, chess.BLACK):
        king = board.king(colour)
        if king is None:
            raise ValueError("King safety requires both kings")
        ring = chess.BB_KING_ATTACKS[king]
        zone = ring | chess.BB_SQUARES[king]
        attackers = [sq for sq, mask in effective.items() if pieces[sq].color != colour and mask & zone]
        defenders = [sq for sq, mask in effective.items() if pieces[sq].color == colour and mask & zone]
        hostile, friendly, contacts = 0, 0, 0
        for sq in attackers:
            hostile |= effective[sq] & zone
            contacts += (effective[sq] & zone).bit_count()
        for sq in defenders:
            friendly |= effective[sq] & zone
        threatened = hostile & ring
        uncovered = (threatened & ~friendly).bit_count() / max(1, threatened.bit_count())
        kfile, krank = chess.square_file(king), chess.square_rank(king)
        files = [f for f in range(kfile - 1, kfile + 2) if 0 <= f < 8]
        shelter = 0.
        for file in files:
            for distance, credit in ((1, 1.), (2, .5)):
                rank = krank + (distance if colour else -distance)
                if 0 <= rank < 8 and pieces.get(chess.square(file, rank)) == chess.Piece(chess.PAWN, colour):
                    shelter += credit
                    break
        shelter /= len(files)
        open_rays = 0
        for dx, dy in DIRECTIONS:
            empty = 0
            for distance in range(1, 4):
                file, rank = kfile + dx * distance, krank + dy * distance
                if not (0 <= file < 8 and 0 <= rank < 8) or chess.square(file, rank) in pieces:
                    break
                empty += 1
            open_rays += empty >= 2
        # Removing the king reveals x-rays behind its original square. The
        # potential captured piece cannot attack its own square.
        occupancy = board.occupied & ~chess.BB_SQUARES[king]
        escapes = [sq for sq in chess.scan_forward(ring & ~board.occupied_co[colour])
                   if pieces.get(sq) != chess.Piece(chess.KING, not colour)
                   and not board.attackers_mask(not colour, sq, occupancy)]
        force = min(1., sum({chess.QUEEN: 4, chess.ROOK: 2, chess.BISHOP: 1, chess.KNIGHT: 1}.get(p.piece_type, 0)
                           for p in pieces.values() if p.color != colour) / 8)
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
