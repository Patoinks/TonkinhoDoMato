"""The Seven Deadly Sins on G4 Leto: seven personalities, each with its own module, openings, style and engine
settings, at seven difficulty levels (Sloth, the easiest, to Pride, the boss).

A sin acts where the Pantheon personalities act (claudio_agent: the opening, then the style among near-equal
safe moves) and, through `settings`, on Leto's own engine options. Each module adds its own signals:

- Sloth     autopilot: a piece-placement plan (London / King's Indian setup) for 16 half-moves; lazy, quiet play.
- Gluttony  material appetite: its captures count for more, and it takes slightly losing captures (poisoned pawns).
- Lust      queen and king hunt: early queen activity, its pieces drawn to the enemy king.
- Wrath     king attack: attack units on the enemy king's zone, pawn storms, deeper mate searches.
- Envy      prophylaxis: what the opponent would take next is kept small; its mobility restricted; its most
            active piece traded off.
- Greed     hoard and convert: when ahead, trades pieces; never leaves a piece hanging; never sacrifices.
- Pride     contempt: draws are worth little, repetitions and drawish endgames (opposite bishops, symmetrical
            pawns) are avoided, tension kept.

    from pecados import make
    player.personality = make("wrath")       # and ClaudioConfig settings += SINS["wrath"].settings
"""
from dataclasses import dataclass, field

import chess

import personalidades as P

V = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
ATTACK_UNITS = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 5, chess.KING: 0}


def _material(board, colour):
    return sum(V[p.piece_type] * (1 if p.color == colour else -1) for p in board.piece_map().values())


def _zone(board, colour):
    k = board.king(colour)
    return 0 if k is None else chess.BB_KING_ATTACKS[k] | chess.BB_SQUARES[k]


def _distance(a, b):
    return max(abs(chess.square_file(a) - chess.square_file(b)), abs(chess.square_rank(a) - chess.square_rank(b)))


def _attack_units(board, me):
    """Stockfish-style king danger, crude: every attack of `me` on the enemy king's zone, by the attacker's kind."""
    units = 0
    for s in chess.scan_forward(_zone(board, not me)):
        for a in board.attackers(me, s):
            units += ATTACK_UNITS[board.piece_type_at(a)]
    return units


def _pawn_storm(board, me):
    k = board.king(not me)
    if k is None:
        return 0
    kf = chess.square_file(k)
    total = 0
    for s in board.pieces(chess.PAWN, me):
        if abs(chess.square_file(s) - kf) <= 1:
            total += chess.square_rank(s) if me == chess.WHITE else 7 - chess.square_rank(s)
    return total


def _tropism(board, me):
    k = board.king(not me)
    if k is None:
        return 0
    return sum(7 - _distance(s, k) for s in chess.scan_forward(board.occupied_co[me] & ~board.kings & ~board.pawns))


def _hanging(board, colour):
    """Pieces of `colour` attacked and not defended, or attacked by something cheaper (their value)."""
    total = 0
    for s in chess.scan_forward(board.occupied_co[colour] & ~board.kings):
        attackers = board.attackers(not colour, s)
        if not attackers:
            continue
        cheapest = min(V[board.piece_type_at(a)] for a in attackers)
        value = V[board.piece_type_at(s)]
        if not board.attackers(colour, s) or cheapest < value:
            total += value
    return total


def _threat(board, colour):
    """What `colour` (to move or not) could take next, crudely: its best capture's value net of the recapture."""
    best = 0
    for s in chess.scan_forward(board.occupied_co[not colour] & ~board.kings):
        attackers = board.attackers(colour, s)
        if not attackers:
            continue
        value = V[board.piece_type_at(s)]
        cheapest = min(V[board.piece_type_at(a)] for a in attackers)
        best = max(best, value if not board.attackers(not colour, s) else value - cheapest)
    return best


def _activity(board, square):
    return len(board.attacks(square))


def _drawish(board):
    """Opposite-coloured bishops alone, or symmetrical pawns with few pieces: the endgames Pride avoids."""
    pieces = board.occupied & ~board.kings & ~board.pawns
    wb, bb = board.pieces(chess.BISHOP, chess.WHITE), board.pieces(chess.BISHOP, chess.BLACK)
    if pieces == (board.bishops & pieces) and len(wb) == 1 and len(bb) == 1:
        a, b = next(iter(wb)), next(iter(bb))
        if (chess.square_file(a) + chess.square_rank(a)) % 2 != (chess.square_file(b) + chess.square_rank(b)) % 2:
            return 1.0
    if chess.popcount(pieces) <= 4:
        wf = sorted(chess.square_file(s) for s in board.pieces(chess.PAWN, chess.WHITE))
        bf = sorted(chess.square_file(s) for s in board.pieces(chess.PAWN, chess.BLACK))
        if wf == bf:
            return 1.0
    return 0.0


# ---------------------------------------------------------------- openings (by hand, from theory)
SLOTH_PLAN = {   # Sloth's autopilot: the same setup against everything, in this order (piece, square)
    chess.WHITE: [("P", "d4"), ("N", "f3"), ("B", "f4"), ("P", "e3"), ("B", "d3"), ("N", "d2"), ("P", "c3"),
                  ("K", "g1"), ("P", "h3"), ("Q", "e2"), ("R", "e1")],
    chess.BLACK: [("N", "f6"), ("P", "g6"), ("B", "g7"), ("K", "g8"), ("P", "d6"), ("N", "d7"), ("P", "e5"),
                  ("R", "e8"), ("P", "c6"), ("Q", "c7"), ("P", "h6")],
}

OPENINGS = {
    "sloth": "",
    "gluttony": """
        W: e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Ne5 e6 g4 Bg6 h4 Nbd7 Nxd7 Nxd7
        W: e4 e5 f4 exf4 Nf3 g5 Bc4 g4 O-O gxf3 Qxf3
        W: e4 e5 Nf3 Nc6 d4 exd4 c3 dxc3 Nxc3 Bb4 Bc4 d6 O-O Bxc3 bxc3
        W: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qb6 Qd2 Qxb2 Rb1 Qa3
        W: e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 Qg4 Qc7 Qxg7 Rg8 Qxh7 cxd4
        W: e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 O-O Nc6 c3 cxd4 cxd4
        W: d4 d5 c4 dxc4 e4 e5 Nf3 exd4 Bxc4 Nc6 O-O
        W: d4 Nf6 c4 c5 d5 b5 cxb5 a6 bxa6 Bxa6 Nc3 d6 e4 Bxf1 Kxf1 g6 g3 Bg7 Kg2
        W: d4 Nf6 c4 e5 dxe5 Ng4 Bf4 Nc6 Nf3 Bb4+ Nbd2 Qe7 e3 Ngxe5 Nxe5 Nxe5 Be2
        W: d4 f5 e4 fxe4 Nc3 Nf6 Bg5 Nc6 d5 Ne5 Qd4
        B: e4 e5 f4 exf4 Nf3 g5 h4 g4 Ne5 Nf6 Bc4 d5 exd5 Bd6
        B: e4 e5 Nf3 Nc6 d4 exd4 c3 dxc3 Nxc3 Bb4 Bc4 d6 O-O Bxc3 bxc3 Nf6
        B: e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4 O-O dxc3
        B: e4 e5 d4 exd4 c3 dxc3 Bc4 cxb2 Bxb2 d5 Bxd5 Nf6 Bxf7+ Kxf7 Qxd8 Bb4+
        B: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5 dxe5 Be6
        B: e4 c5 d4 cxd4 c3 dxc3 Nxc3 Nc6 Nf3 d6 Bc4 e6 O-O Nf6 Qe2 Be7 Rd1 e5
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qb6 Qd2 Qxb2 Rb1 Qa3
        B: e4 c5 Nf3 d6 c3 Nf6 Be2 Nc6 d4 cxd4 cxd4 Nxe4
        B: d4 d5 c4 dxc4 e3 b5 a4 c6 axb5 cxb5 b3 cxb3 Qxb3
        B: d4 d5 c4 dxc4 Nf3 Nf6 e3 b5 a4 c6 axb5 cxb5 b3 e6 bxc4 bxc4 Bxc4
        B: d4 d5 c4 dxc4 e4 e5 Nf3 exd4 Bxc4 Nc6 O-O Be6
        B: d4 Nf6 Bg5 c5 d5 Qb6 Nc3 Qxb2 Bd2 Qb6 e4 d6 f4
        B: d4 Nf6 Bf4 c5 e3 Qb6 Nc3 Qxb2 Nb5 Na6 a3 d5 Rb1 Qxa3 Ra1 Qb4+ c3 Qa5
        B: c4 e5 Nc3 Nf6 Nf3 Nc6 d4 exd4 Nxd4 Bb4 Bg5 h6 Bh4 Bxc3+ bxc3 Ne5
        B: Nf3 d5 c4 dxc4 e3 Nf6 Bxc4 e6 O-O c5
    """,
    "lust": """
        W: e4 e5 Qh5 Nc6 Bc4 g6 Qf3 Nf6 Ne2 Bg7 Nbc3 d6 d3 Bg4 Qg3
        W: e4 e5 f4 exf4 Bc4 Qh4+ Kf1 d5 Bxd5 g5 Nc3 Bg7 d4 Ne7 Nf3 Qh5 h4 h6 Kf2
        W: e4 e5 f4 exf4 Nf3 d5 exd5 Nf6 Bb5+ c6 dxc6 bxc6 Bc4 Nd5
        W: e4 e5 Bc4 Nf6 d4 exd4 Nf3 Nxe4 Qxd4 Nf6 Bg5 Be7 Nc3 c6 O-O-O
        W: e4 c5 Nf3 d6 d4 cxd4 Qxd4 Nc6 Bb5 Bd7 Bxc6 Bxc6 Nc3 Nf6 Bg5 e6 O-O-O Be7
        W: e4 e6 d4 d5 e5 c5 Qg4 cxd4 Nf3 Nc6 Bd3 f5 Qg3 Nge7 O-O Ng6
        W: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Qf3 Bxe4 Qxe4 Nf6 Qd3 e6 Nf3
        W: e4 d5 exd5 Qxd5 Nc3 Qa5 Qh5 Nf6 Qh4 Nc6 Bb5 Bd7 Nf3
        W: e4 Nf6 e5 Nd5 d4 d6 Nf3 g6 Bc4 Nb6 Bb3 Bg7 Qe2 Nc6 O-O O-O h3 a5 a4
        W: e4 d6 d4 Nf6 Nc3 g6 Be3 Bg7 Qd2 c6 f3 b5 Bh6 Bxh6 Qxh6
        W: e4 g6 d4 Bg7 Nc3 d6 Be3 a6 Qd2 Nd7 Bh6 Bxh6 Qxh6
        B: e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Bd2 e6 Qe2 Bb4 Ne5 Nbd7 O-O-O
        B: e4 d5 exd5 Qxd5 Nc3 Qd6 d4 Nf6 Nf3 a6 g3 Bg4 Bg2 Nc6 O-O O-O-O
        B: e4 d5 exd5 Qxd5 Nf3 Bg4 Be2 Nc6 d4 O-O-O c4 Qa5+ Bd2 Qf5
        B: e4 d5 exd5 Qxd5 d4 Nc6 Nf3 Bg4 Be2 O-O-O c3 e5
        B: e4 d5 Nc3 d4 Nce2 e5 Ng3 Be6 c3 c5 Bb5+ Nd7
        B: e4 d5 e5 Bf5 d4 e6 Nf3 c5 c3 Nc6
        B: e4 d5 d3 dxe4 dxe4 Qxd1+ Kxd1 Nc6
        B: d4 d5 c4 Nc6 Nc3 dxc4 Nf3 Nf6 e4 Bg4 Be3 e5 d5 Bxf3 gxf3 Ne7 Bxc4 Ng6
        B: d4 d5 c4 e5 dxe5 d4 Nf3 Nc6 g3 Bg4 Bg2 Qd7 O-O O-O-O
        B: d4 d5 Nf3 Bg4 Ne5 Bh5 c4 f6 Nf3 e6 Nc3 c6 Qb3 Qb6
        B: d4 d5 Bf4 c5 e3 Qb6 Nc3 Qxb2 Nb5 Na6
        B: c4 d5 cxd5 Qxd5 Nc3 Qd8 d4 Nf6 Nf3 e6
        B: Nf3 d5 c4 d4 b4 g6 e3 Bg7 Na3 Nc6
    """,
}
OPENINGS["wrath"] = P.OPENINGS.get("nezha", "")       # gambits and kingside attacks (the rebuilt Nezha repertoire)
OPENINGS["greed"] = P.OPENINGS.get("anubis", "")      # exchange variations, the Caro-Kann, the QGD
OPENINGS["pride"] = P.OPENINGS.get("hades", "")       # the most ambitious main lines
OPENINGS["envy"] = """
        W: c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 Nf3 Nf6 O-O O-O d4 cxd4 Nxd4 Nxd4 Qxd4 d6 Qd3
        W: c4 e5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6 e4 Nge7 Nge2 O-O O-O
        W: c4 Nf6 Nc3 e6 e4 d5 e5 d4 exf6 dxc3 bxc3 Qxf6 d4
        W: c4 Nf6 g3 g6 Bg2 Bg7 Nc3 O-O e4 d6 Nge2 e5 O-O c6 d3
        W: c4 e6 Nc3 d5 d4 Nf6 cxd5 exd5 Bg5 Be7 e3 c6 Qc2 Nbd7 Bd3 O-O Nge2
        W: c4 c6 e4 d5 exd5 cxd5 d4 Nf6 Nc3 Nc6 Bg5
        W: c4 g6 Nc3 Bg7 g3 c5 Bg2 Nc6 e3 e6 Nge2 Nge7 O-O O-O d4
        W: c4 b6 d4 e6 a3 Bb7 Nc3 f5 d5
        W: c4 f5 g3 Nf6 Bg2 g6 Nc3 Bg7 d3 O-O e4
        W: c4 d5 cxd5 Nf6 Nc3 Nxd5 g3 g6 Bg2 Nxc3 bxc3 Bg7 Rb1
        B: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Be7 c4 Nb4 Be2 O-O Nc3 Bf5
        B: e4 e5 Nf3 Nf6 Nc3 Nc6 Bb5 Bb4 O-O O-O d3 d6 Bg5 Bxc3 bxc3 Qe7
        B: e4 e5 Nf3 Nf6 d4 Nxe4 Bd3 d5 Nxe5 Nd7 Nxd7 Bxd7 O-O Bd6
        B: e4 e5 Nf3 Nf6 Bc4 Nxe4 Nc3 Nxc3 dxc3 f6 O-O d6
        B: e4 e5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6 f4 Nge7
        B: e4 e5 Bc4 Bc5 Nf3 d6 c3 Nf6 d4 exd4 cxd4 Bb4+
        B: e4 e5 d4 exd4 Qxd4 Nc6 Qe3 Nf6 Nc3 Bb4 Bd2 O-O O-O-O Re8
        B: e4 e5 f4 d5 exd5 exf4 Nf3 Nf6 c4 c6 d4 cxd5
        B: d4 d5 c4 c5 cxd5 Nf6 e4 Nxe4 dxc5 Nxc5 Nc3 e6
        B: d4 d5 c4 c6 cxd5 cxd5 Nc3 Nc6 Nf3 Nf6 Bf4 Bf5 e3 e6 Bb5 Bd6 Bxd6 Qxd6
        B: d4 d5 Nf3 Nf6 c4 c6 cxd5 cxd5 Nc3 Nc6 Bf4 Bf5 e3 e6
        B: d4 d5 Bf4 Bf5 e3 e6 Nf3 Nf6 c4 c6 Nc3 Nbd7
        B: d4 d5 e3 e6 Bd3 Bd6 Nf3 Nf6 O-O O-O c4 c6
        B: d4 d5 Nc3 Nc6 Bf4 Bf5 e3 e6 Nb5 Rc8 Nf3 Nf6
        B: d4 d5 g3 g6 Bg2 Bg7 Nf3 Nf6 O-O O-O c4 c6
        B: c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 Nf3 Nf6 O-O O-O d4 cxd4 Nxd4 Nxd4 Qxd4 d6
        B: c4 c5 Nf3 Nf6 Nc3 Nc6 g3 g6 Bg2 Bg7 O-O O-O
        B: c4 c5 g3 g6 Bg2 Bg7 Nc3 Nc6 e3 e6 Nge2 Nge7
        B: Nf3 Nf6 c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 O-O O-O
        B: Nf3 Nf6 g3 g6 Bg2 Bg7 O-O O-O d3 d6 e4 e5
        B: Nf3 Nf6 d4 d5 c4 c6 cxd5 cxd5
        B: g3 g6 Bg2 Bg7 Nf3 Nf6 O-O O-O d3 d6
        B: b3 b6 Bb2 Bb7 e3 e6 Nf3 Nf6 c4 c5
        B: f4 f5 Nf3 Nf6 e3 e6 d4 d5
        B: Nc3 Nc6 e4 e5 Nf3 Nf6 Bb5 Bb4
        B: e3 e6 d4 d5 c4 c5
    """


@dataclass
class Sin:
    name: str
    level: int                 # 1 (Beginner) .. 7 (Expert)
    label: str
    target: float              # intended score against Stockfish 2250 at 2 ms (calibrated by A/B)
    window: float              # how far below Leto's best a move may be for the style to choose it
    weights: dict              # personalidades.features signals and this module's own signals
    settings: list             # Leto's own engine options for this sin
    material_floor: float = -0.25
    stats: dict = field(default_factory=lambda: {"decisions": 0, "changed": 0, "book": 0, "plan": 0})

    # ------------------------------------------------------------ the module's own signals
    def signals(self, board, move, after):
        return {}

    def floor(self, board):
        return self.material_floor

    # ------------------------------------------------------------ the two hooks (claudio_agent)
    def opening(self, board, moves):
        book = _book(self.name)
        move = book.get((board.turn, board._transposition_key()))
        if move is not None and move in moves:
            self.stats["book"] += 1
            return move
        return None

    def adjust(self, board, moves, scores, known, nets):
        """Leto's near-equal safe moves, re-ranked by this sin's signals (as personalidades.apply_style)."""
        import numpy as np
        floor = self.floor(board)
        eligible = [i for i in range(len(moves)) if not known[i] and scores[i] >= scores.max() - self.window
                    and (nets is None or nets[i] >= floor)]
        self.stats["decisions"] += 1
        if len(eligible) < 2:
            return False
        before = int(np.argmax(scores))
        raw = []
        for i in eligible:
            f = P.features(board, moves[i])
            after = board.copy(stack=False)
            after.push(moves[i])
            f.update(self.signals(board, moves[i], after))
            raw.append(sum(w * f.get(k, 0.0) for k, w in self.weights.items()))
        low, high = min(raw), max(raw)
        if high - low < 1e-12:
            return False
        for i, value in zip(eligible, raw):
            scores[i] += self.window * (value - low) / (high - low)
        changed = int(np.argmax(scores)) != before
        self.stats["changed"] += changed
        return changed


_BOOKS = {}


def _book(name):
    if name not in _BOOKS:
        book = {}
        for text in OPENINGS.get(name, "").strip().splitlines():
            side, _, line = text.strip().partition(":")
            colour = chess.WHITE if side.strip() == "W" else chess.BLACK
            board = chess.Board()
            for san in line.split():
                move = board.parse_san(san)
                if board.turn == colour:
                    book.setdefault((colour, board._transposition_key()), move)
                board.push(move)
        _BOOKS[name] = book
    return _BOOKS[name]


class Sloth(Sin):
    plan_plies = 16                              # the autopilot runs this many half-moves of the game

    def opening(self, board, moves):
        """The autopilot: the first step of its setup not done yet that a legal move does, if Leto rates it sound."""
        if board.ply() > self.plan_plies:
            return None
        plan = SLOTH_PLAN[board.turn]
        for symbol, square in plan:
            target = chess.parse_square(square)
            kind = chess.Piece.from_symbol(symbol).piece_type
            have = board.piece_at(target)
            if have is not None and have.color == board.turn and have.piece_type == kind:
                continue
            for m in moves:
                if kind == chess.KING:
                    if board.is_castling(m) and m.to_square == target:
                        self.stats["plan"] += 1
                        return m
                elif board.piece_type_at(m.from_square) == kind and m.to_square == target:
                    after = board.copy(stack=False)
                    after.push(m)
                    attackers = after.attackers(not board.turn, target)
                    if attackers and (not after.attackers(board.turn, target) or
                                      min(V[after.piece_type_at(s)] for s in attackers) < V[kind]):
                        continue                 # even Sloth does not leave the piece to be taken
                    self.stats["plan"] += 1
                    return m
        return None

    def signals(self, board, move, after):
        return {"lazy": float(not board.is_capture(move) and not board.gives_check(move))}


class Gluttony(Sin):
    appetite_window = 0.6                        # a capture this far below Leto's best is still taken

    def floor(self, board):
        return -1.0                              # takes losing captures: the poisoned pawn

    def adjust(self, board, moves, scores, known, nets):
        """The appetite first: the most valuable capture Leto rates within appetite_window of its best, losing at
        most a pawn by its exchange count, becomes the move; otherwise the style as usual."""
        import numpy as np
        best = float(scores.max())
        food = []
        for i, m in enumerate(moves):
            if known[i] or not board.is_capture(m) or scores[i] < best - self.appetite_window:
                continue
            if nets is not None and nets[i] < self.floor(board):
                continue
            victim = chess.PAWN if board.is_en_passant(m) else board.piece_type_at(m.to_square)
            food.append((V[victim], scores[i], i))
        if food:
            _, _, i = max(food)
            self.stats["decisions"] += 1
            changed = int(np.argmax(scores)) != i
            scores[i] = best + 1e-6
            self.stats["changed"] += changed
            return changed
        return super().adjust(board, moves, scores, known, nets)

    def signals(self, board, move, after):
        gain = 0.0
        if board.is_capture(move):
            victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
            gain = V[victim]
        return {"appetite": gain}


class Lust(Sin):
    def signals(self, board, move, after):
        me = board.turn
        queen_out = float(board.piece_type_at(move.from_square) == chess.QUEEN and board.ply() < 30)
        return {"queen_out": queen_out, "tropism": float(_tropism(after, me) - _tropism(board, me))}


class Wrath(Sin):
    def signals(self, board, move, after):
        me = board.turn
        return {"attack_units": float(_attack_units(after, me) - _attack_units(board, me)),
                "storm": float(_pawn_storm(after, me) - _pawn_storm(board, me))}


class Envy(Sin):
    def signals(self, board, move, after):
        me = board.turn
        out = {"prophylaxis": -float(_threat(after, not me))}
        if board.is_capture(move) and not board.is_en_passant(move):
            # trading off the opponent's most active piece
            victims = [s for s in chess.scan_forward(board.occupied_co[not me] & ~board.kings & ~board.pawns)]
            if victims:
                busiest = max(victims, key=lambda s: _activity(board, s))
                out["trade_best"] = float(move.to_square == busiest)
        return out


class Greed(Sin):
    def floor(self, board):
        return 0.0                               # never gives material away

    def signals(self, board, move, after):
        me = board.turn
        ahead = _material(board, me) >= 1
        trade = 0.0
        if ahead and board.is_capture(move) and not board.is_en_passant(move):
            mover, victim = board.piece_type_at(move.from_square), board.piece_type_at(move.to_square)
            trade = float(victim not in (chess.PAWN, None) and V[victim] >= V[mover])
        return {"hoard": -float(_hanging(after, me)), "convert": trade}


class Pride(Sin):
    def signals(self, board, move, after):
        me = board.turn
        return {"no_draw": -_drawish(after), "no_repeat": -float(after.is_repetition(2)), "keep_pieces": -float(board.is_capture(move) and
                                                                     _material(board, me) <= 0.5 and
                                                                     _material(board, me) >= -0.5)}


# Difficulty: Sloth (1, Beginner) ... Pride (7, Expert). `target` is the measured score against Stockfish 2250
# at 2 ms per move (ab_stockfish.py, 300 games each, interleaved with neutral Leto at ~0.56); the windows and the
# engine settings are what put each sin there.
SINS = {
    "sloth": Sloth("sloth", 1, "Beginner", 0.22, 0.06,
                   {"lazy": 1.0, "quiet": 0.5, "coordination": 0.3, "castle": 1.0, "development": 0.5},
                   ["draw_value=0.5"]),
    "gluttony": Gluttony("gluttony", 3, "Easy+", 0.37, 0.10,
                         {"appetite": 1.0, "capture": 0.8, "trade_ahead": 0.3, "retreat": -0.2},
                         ["safety_weight=2.5", "simplify_weight=0.0"]),
    "lust": Lust("lust", 2, "Easy", 0.29, 0.10,
                 {"queen_out": 1.0, "tropism": 0.4, "king_zone": 0.4, "threats": 0.5, "check": 0.5, "retreat": -0.5},
                 ["tactic_depth=5"]),
    "wrath": Wrath("wrath", 4, "Intermediate", 0.41, 0.07,
                   {"attack_units": 0.3, "storm": 0.4, "check": 0.8, "threats": 0.6, "king_zone": 0.4,
                    "retreat": -1.0, "castle": -0.3},
                   ["mate_attack=3", "king_weight=0.15"]),
    "envy": Envy("envy", 5, "Advanced", 0.46, 0.012,
                 {"prophylaxis": 0.8, "restrict": 0.06, "trade_best": 1.0, "coordination": 0.2, "outpost": 0.4},
                 ["tactic_depth=7"]),
    "greed": Greed("greed", 6, "Advanced+", 0.55, 0.008,
                   {"hoard": 0.6, "convert": 1.2, "coordination": 0.3, "check": -0.3, "pawn_break": -0.3},
                   ["tactic_depth=7", "check_extension=1", "simplify_weight=0.08", "simplify_margin=0.5"]),
    "pride": Pride("pride", 7, "Expert", 0.62, 0.006,
                   {"no_draw": 1.5, "no_repeat": 2.0, "keep_pieces": 0.6, "centre": 0.4, "threats": 0.4,
                    "pawn_break": 0.3},
                   ["tactic_depth=8", "check_extension=1", "draw_value=0.35", "ahead_draws=True",
                    "ahead_value=0.6", "repeat_as_draw=True", "fifty_horizon=20"]),
}


def make(name, window=None):
    import copy
    sin = copy.deepcopy(SINS[name])
    if window is not None:
        sin.window = window
    _book(name)                      # parse the lines now: an illegal one fails loudly
    return sin
