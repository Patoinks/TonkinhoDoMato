"""The ten Pantheon personalities on G4 Leto: their own openings and a style for the rest of the game.

A personality never changes what Leto computes. It acts in two places only:

1. Opening: its own repertoire (SAN lines below) is consulted before Leto's. While the game follows
   one of its lines, it plays the line's move; when the opponent leaves the lines, Leto's own
   repertoire and search take over.
2. Style: after Leto's full evaluation and tactical search, among the safe candidates whose score is
   within `window` of the best, a style bonus of at most `window` is added. Forced mates, moves with an
   exact value (game endings), moves that lose material by Leto's own exchange count, and the final
   mate-defence check all stay as they are: a style can only choose between moves Leto rates nearly
   equal. `window` sets how strong the personality is.
"""
from dataclasses import dataclass, field

import chess

CENTRE = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5
WIDE_CENTRE = CENTRE | chess.BB_C3 | chess.BB_D3 | chess.BB_E3 | chess.BB_F3 | chess.BB_C6 | chess.BB_D6 | \
    chess.BB_E6 | chess.BB_F6 | chess.BB_C4 | chess.BB_F4 | chess.BB_C5 | chess.BB_F5
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

# ---------------------------------------------------------------- openings
# Each line is played by whichever side the personality has; the other side's moves are what it expects.
OPENINGS = {   # W: lines it plays as White; B: its replies as Black (to 1.e4, 1.d4 and the flank openings)
    "nezha": """
        W: e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4 O-O
        W: e4 e5 Nf3 Nc6 Bc4 Nf6 d4 exd4 O-O
        B: e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Na5
        B: d4 d5 c4 dxc4 e4 e5 Nf3 exd4 Bxc4
        B: c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5
    """,
    "loki": """
        W: e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4 Nf3
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O
        B: d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6
        B: c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7
    """,
    "wukong": """
        W: e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4 O-O O-O
        B: e4 Nf6 e5 Nd5 d4 d6 Nf3 g6
        B: d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3
        B: c4 Nf6 Nc3 e5 Nf3 Nc6 g3 Bb4
    """,
    "susanoo": """
        W: e4 e5 f4 exf4 Nf3 g5 h4 g4 Ne5
        B: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6
        B: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7
        B: c4 e5 Nc3 Nf6 Nf3 Nc6 d4 exd4 Nxd4
    """,
    "tsukuyomi": """
        W: d4 d5 Bf4 Nf6 e3 c5 c3 Nc6 Nd2 e6 Ngf3
        B: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6
        B: d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5
        B: c4 c6 e4 d5 exd5 cxd5 cxd5 Nf6
    """,
    "anubis": """
        W: e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O f6 d4
        B: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3
        B: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3
        B: c4 e6 Nc3 d5 d4 Nf6 cxd5 exd5
    """,
    "amaterasu": """
        W: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O
        B: e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 Bd3 c5 c3 Nc6
        B: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4 Bd2 Be7
        B: c4 Nf6 Nc3 e6 e4 d5 e5 d4
    """,
    "ra": """
        W: d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 b6
        B: e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6 Bxc6 dxc6 dxe5 Nf5
        B: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5
        B: c4 g6 d4 Nf6 Nc3 Bg7 e4 d6
    """,
    "hades": """
        W: c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5 Bg2 Nb6
        B: e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3 bxc3 Ne7
        B: d4 f5 c4 Nf6 g3 e6 Bg2 d5 Nf3 c6 O-O Bd6
        B: Nf3 f5 g3 Nf6 Bg2 e6 O-O d5
    """,
    "zeus": """
        W: e4 e5 Nf3 Nc6 d4 exd4 Bc4 Bc5 c3 Nf6 e5 d5
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4
        B: d4 Nf6 c4 e5 dxe5 Ng4 Bf4 Nc6 Nf3 Bb4
        B: c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5 Bg2 Nb6
    """,
}


def _key(board):
    return board._transposition_key()


_BOOKS = {}


def repertoire(name):
    """{(colour, position key): move}: only the personality's own side of each line counts; the first
    line to reach a position decides it."""
    if name not in _BOOKS:
        book = {}
        for text in OPENINGS[name].strip().splitlines():
            side, _, line = text.strip().partition(":")
            colour = chess.WHITE if side.strip() == "W" else chess.BLACK
            board = chess.Board()
            for san in line.split():
                move = board.parse_san(san)          # an illegal line is a bug: fail loudly
                if board.turn == colour:
                    book.setdefault((colour, _key(board)), move)
                board.push(move)
        _BOOKS[name] = book
    return _BOOKS[name]


def opening_move(name, board):
    move = repertoire(name).get((board.turn, _key(board)))
    return move if move is not None and move in board.legal_moves else None


# ---------------------------------------------------------------- style signals
def _forward(colour, move):
    rank_from, rank_to = chess.square_rank(move.from_square), chess.square_rank(move.to_square)
    return (rank_to - rank_from) if colour == chess.WHITE else (rank_from - rank_to)


def _king_zone(board, colour):
    king = board.king(colour)
    return 0 if king is None else chess.BB_KING_ATTACKS[king] | chess.BB_SQUARES[king]


def features(board, move):
    """Signals of one move, all from the position itself (no engine)."""
    me = board.turn
    piece = board.piece_at(move.from_square)
    kind = piece.piece_type if piece else chess.PAWN
    captured = board.piece_at(move.to_square)
    capture = board.is_capture(move)
    check = board.gives_check(move)
    zone_before = _king_zone(board, not me)
    attacked_before = sum(1 for s in chess.scan_forward(zone_before) if board.is_attacked_by(me, s))
    after = board.copy(stack=False)
    after.push(move)
    zone = _king_zone(after, not me)
    attacked_after = sum(1 for s in chess.scan_forward(zone) if after.is_attacked_by(me, s))
    targets = sum(1 for s in chess.scan_forward(after.occupied_co[not me])
                  if after.is_attacked_by(me, s) and after.piece_type_at(s) != chess.KING
                  and not after.is_attacked_by(not me, s))
    enemy_mobility = after.legal_moves.count()
    own_defended = sum(1 for s in chess.scan_forward(after.occupied_co[me] & ~after.kings)
                       if after.is_attacked_by(me, s))
    forward = _forward(me, move)
    pawn_attacks_pawn = kind == chess.PAWN and any(
        after.piece_type_at(s) == chess.PAWN and after.color_at(s) != me
        for s in chess.scan_forward(chess.BB_PAWN_ATTACKS[me][move.to_square]))
    material = sum(VALUE[p.piece_type] * (1 if p.color == me else -1) for p in board.piece_map().values())
    return {
        "check": float(check),
        "capture": float(capture),
        "trade_ahead": float(capture and material > 0 and captured is not None and captured.piece_type != chess.PAWN),
        "quiet": float(not capture),
        "advance": float(max(0, forward)) if kind != chess.KING else 0.0,
        "retreat": float(max(0, -forward)) if kind != chess.KING else 0.0,
        "knight": float(kind == chess.KNIGHT),
        "outpost": float(kind == chess.KNIGHT and chess.square_rank(move.to_square) in ((4, 5) if me else (2, 3))
                         and not after.is_attacked_by(not me, move.to_square)),
        "pawn_push": float(kind == chess.PAWN and not capture),
        "pawn_break": float(pawn_attacks_pawn or (kind == chess.PAWN and capture)),
        "centre": float(bool(chess.BB_SQUARES[move.to_square] & CENTRE)) +
                  0.5 * float(bool(chess.BB_SQUARES[move.to_square] & WIDE_CENTRE)),
        "king_zone": float(attacked_after - attacked_before),
        "threats": float(targets),
        "restrict": -float(enemy_mobility),
        "coordination": float(own_defended),
        "castle": float(board.is_castling(move)),
        "king_move": float(kind == chess.KING and not board.is_castling(move)),
        "development": float(kind in (chess.KNIGHT, chess.BISHOP) and
                             chess.square_rank(move.from_square) == (0 if me == chess.WHITE else 7)),
    }


@dataclass
class Personality:
    name: str
    weights: dict
    window: float = 0.06          # 3x Leto's value_spread_ref (0.0215): visibly its own, within near-equal moves
    material_floor: float = -0.25  # never a move that loses more than this by Leto's own exchange count
    stats: dict = field(default_factory=lambda: {"decisions": 0, "changed": 0, "book": 0})


PERSONALITIES = {
    "nezha": {"development": 1.0, "advance": 0.6, "threats": 0.6, "retreat": -1.0, "check": 0.3},
    "loki": {"quiet": 0.8, "threats": 1.0, "king_zone": 0.3, "capture": -0.6},
    "wukong": {"knight": 1.0, "outpost": 1.2, "advance": 0.3, "retreat": -0.4},
    "susanoo": {"pawn_break": 1.2, "pawn_push": 0.4, "capture": 0.4, "king_zone": 0.4},
    "tsukuyomi": {"restrict": 0.08, "quiet": 0.6, "coordination": 0.3, "advance": -0.2},
    "anubis": {"trade_ahead": 1.5, "capture": 0.6, "coordination": 0.3, "check": -0.3},
    "amaterasu": {"coordination": 0.6, "castle": 1.5, "development": 0.6, "king_move": -1.0},
    "ra": {"centre": 1.2, "pawn_push": 0.5, "advance": 0.3, "retreat": -0.3},
    "hades": {"restrict": 0.12, "quiet": 0.4, "coordination": 0.4, "capture": -0.3},
    "zeus": {"check": 1.5, "king_zone": 0.8, "threats": 0.5, "advance": 0.3},
}


def make(name, window=None):
    personality = Personality(name, PERSONALITIES[name])
    if window is not None:
        personality.window = window
    repertoire(name)                 # load and validate the lines now
    return personality


def apply_style(personality, board, moves, scores, known, nets):
    """Adds the style bonus in place; returns True when it changed Leto's preferred move."""
    import numpy as np
    eligible = [i for i in range(len(moves))
                if not known[i] and scores[i] >= scores.max() - personality.window
                and (nets is None or nets[i] >= personality.material_floor)]
    personality.stats["decisions"] += 1
    if len(eligible) < 2:
        return False
    before = int(np.argmax(scores))
    raw = []
    for i in eligible:
        f = features(board, moves[i])
        raw.append(sum(weight * f.get(name, 0.0) for name, weight in personality.weights.items()))
    low, high = min(raw), max(raw)
    if high - low < 1e-12:
        return False
    for i, value in zip(eligible, raw):
        scores[i] += personality.window * (value - low) / (high - low)
    changed = int(np.argmax(scores)) != before
    personality.stats["changed"] += changed
    return changed
