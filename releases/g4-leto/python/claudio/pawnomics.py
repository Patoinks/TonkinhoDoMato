"""Pawnomics: how far each pawn can go without being lost and without being stopped (the owner's idea).

For every pawn with no enemy pawn in front of it on its file, it walks the pawn's path square by square,
in its own piece values, and stops at the first square where:
- an enemy pawn controls it (a lever: it cannot go through without trading),
- the pawn would be lost there (the exact capture sequence on that square, x-rays included),
- the enemy king gets there (or next to it, undefended) before the pawn - the race, with the tempo,
- an enemy piece can step onto it first and sit there undisturbed (a blockade),
- an enemy piece already sits there and cannot be won or chased away (a permanent blockade);
  one that can be chased costs two tempi, an own piece in the way one.
A pawn that gets all the way is a new queen, worth what a queen is worth more than a pawn, less the further
away it is and the more enemy pieces are left that could still catch it; otherwise it is worth the rank it
reaches. pawnomics(board, colour, first) is that sum for one side, in pawns.
"""
import chess

KING_VALUE = 1000.0
RANK_WORTH = [0.0, 0.0, 0.0, 0.1, 0.25, 0.5, 0.9, 1.6]    # by the rank reached, counted from its own side
FILES = [chess.BB_FILES[f] for f in range(8)]


def _value(values, piece_type):
    return KING_VALUE if piece_type == chess.KING else values[piece_type]


def _cheapest(board, side, square, values):
    best, best_value = None, None
    for s in board.attackers(side, square):
        v = _value(values, board.piece_type_at(s))
        if best is None or v < best_value:
            best, best_value = s, v
    return best, best_value


def lost_on(board, square, values):
    """The best the side NOT owning the piece on `square` wins by starting captures there (0: nothing)."""
    target = board.piece_at(square)
    side = not target.color
    b = board.copy()
    gains, standing = [], _value(values, target.piece_type)
    while True:
        attacker, worth = _cheapest(b, side, square, values)
        if attacker is None:
            break
        if worth == KING_VALUE and b.attackers(not side, square):
            break                                   # the king cannot take a defended piece
        gains.append(standing - (gains[-1] if gains else 0.0))
        standing = worth
        piece = b.remove_piece_at(attacker)
        b.set_piece_at(square, piece)
        side = not side
    for i in range(len(gains) - 1, 0, -1):
        gains[i - 1] = -max(-gains[i - 1], gains[i])
    return max(0.0, gains[0]) if gains else 0.0


def _attacks_from(kind, square, occupied):
    if kind == chess.KNIGHT:
        return chess.BB_KNIGHT_ATTACKS[square]
    mask = 0
    if kind in (chess.BISHOP, chess.QUEEN):
        mask |= chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & occupied]
    if kind in (chess.ROOK, chess.QUEEN):
        mask |= (chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & occupied]
                 | chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & occupied])
    return mask


def two_move_reach(board, side):
    """Squares the side's pieces (not pawns, not the king) can stand on within two moves."""
    occupied, own = board.occupied, board.occupied_co[side]
    reach = 0
    for kind in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for s in board.pieces(kind, side):
            first = _attacks_from(kind, s, occupied) & ~own
            reach |= first
            for t in chess.scan_forward(first):
                reach |= _attacks_from(kind, t, occupied & ~chess.BB_SQUARES[s]) & ~own
    return reach


def _structure(board, square, colour):
    """v2: a real passed pawn (no enemy pawn in front on its own or the next files) protected by a pawn, connected,
    or with a rook behind it is worth more; with an enemy rook behind it, less."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    front = (chess.BB_ALL << (8 * (rank + 1))) & chess.BB_ALL if colour else (1 << (8 * rank)) - 1
    near = FILES[file] | (FILES[file - 1] if file > 0 else 0) | (FILES[file + 1] if file < 7 else 0)
    if board.pieces_mask(chess.PAWN, not colour) & near & front:
        return 1.0
    factor = 1.0
    own_pawns = board.pieces_mask(chess.PAWN, colour)
    if board.attackers_mask(colour, square) & own_pawns:
        factor *= 1.3
    sides = (FILES[file - 1] if file > 0 else 0) | (FILES[file + 1] if file < 7 else 0)
    band = chess.BB_RANKS[rank] | (chess.BB_RANKS[rank - 1] if rank > 0 else 0) | (chess.BB_RANKS[rank + 1] if rank < 7 else 0)
    if own_pawns & sides & band:
        factor *= 1.2
    behind = FILES[file] & ~front & ~chess.BB_SQUARES[square]
    for s in chess.scan_forward(board.attackers_mask(colour, square) & behind & board.rooks):
        factor *= 1.2
        break
    for s in chess.scan_forward(board.attackers_mask(not colour, square) & behind & board.rooks):
        factor *= 0.8
        break
    return factor


def _pawn(board, square, colour, first, values, enemy_material, reach2=None):
    """What one pawn is worth by how far it can get (pawns), or None if it is not on an open file."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    step = 8 if colour == chess.WHITE else -8
    ahead = [square + step * k for k in range(1, (7 - rank if colour else rank) + 1)]
    enemy_pawns = board.pieces_mask(chess.PAWN, not colour)
    if enemy_pawns & FILES[file] & chess.SquareSet(ahead).mask:
        return None
    rel = rank + 1 if colour else 8 - rank          # 2..7 from its own side
    free = board.copy(stack=False)
    free.remove_piece_at(square)
    king = board.king(not colour)
    tempo, reached = 0, rel
    for i, sq in enumerate(ahead):
        moves = i + 1 - (1 if rel == 2 and i >= 1 else 0) + tempo   # the double step from its first rank
        before = moves if not first else moves - 1                  # enemy moves before the pawn gets there
        occupant = board.piece_at(sq)
        if occupant is not None:
            if occupant.color == colour:
                if occupant.piece_type == chess.PAWN:
                    break                               # its own pawn in front: that one is the runner
                tempo += 1                              # an own piece in the way steps aside
                moves += 1
                before += 1
            elif occupant.piece_type == chess.KING:
                break
            else:
                chased = (lost_on(board, sq, values) > 0 or any(
                    _value(values, board.piece_type_at(s)) < _value(values, occupant.piece_type)
                    for s in board.attackers(colour, sq)))
                if not chased:
                    break                               # a blockade it cannot break
                tempo += 2
                moves += 2
                before += 2
        if any(board.piece_type_at(s) == chess.PAWN for s in board.attackers(not colour, sq)):
            break
        there = free.copy(stack=False)
        there.remove_piece_at(sq)
        there.set_piece_at(sq, chess.Piece(chess.PAWN, colour))
        if lost_on(there, sq, values) > 0:
            break
        defended = bool(there.attackers(colour, sq))
        if king is not None and not defended and chess.square_distance(king, sq) - 1 <= before:
            break                                       # the king gets there (or next to it) first
        if not defended and before >= 1 and occupant is None and any(
                board.piece_type_at(s) not in (chess.PAWN, chess.KING) for s in board.attackers(not colour, sq)):
            break                                       # a piece steps in front first, undisturbed
        if (reach2 is not None and not defended and before >= 2 and occupant is None
                and reach2 & chess.BB_SQUARES[sq]):
            break                                       # v2: or gets there in two moves
        reached = rel + i + 1
        if reached == 8:
            catch = max(0.2, 1.0 - enemy_material / 15.0)
            worth = (values[chess.QUEEN] - values[chess.PAWN]) * catch * 0.92 ** (moves - 1)
            return worth * (_structure(board, square, colour) if reach2 is not None else 1.0)
    worth = RANK_WORTH[min(reached, 7)]
    return worth * (_structure(board, square, colour) if reach2 is not None and worth else 1.0)


def pawnomics(board, colour, first, values, v2=False):
    """Sum over the side's pawns (pawns). first: this side moves next."""
    reach2 = two_move_reach(board, not colour) if v2 else None
    enemy_material = sum(values[t] * chess.popcount(board.pieces_mask(t, not colour))
                         for t in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
    total = 0.0
    for sq in board.pieces(chess.PAWN, colour):
        worth = _pawn(board, sq, colour, first, values, enemy_material, reach2)
        if worth:
            total += worth
    return total


def balance(board, me, values, v2=False):
    """Own pawnomics minus the enemy's, the side to move moving first (pawns)."""
    to_move = board.turn
    return (pawnomics(board, me, to_move == me, values, v2) - pawnomics(board, not me, to_move != me, values, v2))


def has_runners(board):
    """Any pawn with no enemy pawn in front of it on its file (otherwise there is nothing to compute)."""
    for colour in (chess.WHITE, chess.BLACK):
        enemy = board.pieces_mask(chess.PAWN, not colour)
        for sq in board.pieces(chess.PAWN, colour):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            span = chess.BB_FILES[f] & ((chess.BB_ALL << (8 * (r + 1))) & chess.BB_ALL if colour else
                                        (1 << (8 * r)) - 1)
            if not enemy & span:
                return True
    return False
