"""Openings: a repertoire of main lines of opening theory, and the opening principles after it.

repertoire_move(board): the theory move for this position (first line listed wins), or None.
principles(board, move, ply): the move's worth by the classic principles, in units (about -2..+1.5):
develop the knights and bishops towards the centre, central pawns, castle, and not the rim knights,
early queen trips, rook or king walks before castling, the same piece twice, or wing pawns.
"""
import chess

# White: 1.e4. Black: 1...e5 against 1.e4, the Queen's Gambit Declined / Nimzo against 1.d4, 1...e5 against
# 1.c4, 1...d5 against the rest. Each string a line from the start; a position's move is the first line's.
LINES = """
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5 Bc2 c5 d4 Qc7
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5 dxe5 Be6 c3
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O b5 Bb3 Be7 Re1 d6 c3 O-O h3
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O d6 c3 Bd7 d4
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 d6 c3 Bd7 d4 Nf6 O-O
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 b5 Bb3 Nf6 O-O Be7 Re1 d6 c3 O-O h3
e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O f6 d4 exd4 Nxd4 c5 Nb3 Qxd1 Rxd1
e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6 Bxc6 dxc6 dxe5 Nf5 Qxd8+ Kxd8 Nc3
e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Be7 Re1 d6 d4
e4 e5 Nf3 Nc6 Bb5 Bc5 c3 Nf6 d4 exd4 e5
e4 e5 Nf3 Nc6 Bb5 d6 d4 Bd7 Nc3 Nf6 O-O
e4 e5 Nf3 Nc6 Bb5 f5 Nc3 fxe4 Nxe4 d5 Nxe5
e4 e5 Nf3 Nc6 Bb5 Nge7 O-O g6 c3 Bg7 d4
e4 e5 Nf3 Nc6 Bb5 Nd4 Nxd4 exd4 O-O
e4 e5 Nf3 Nc6 Bb5 g6 d4 exd4 Bg5
e4 e5 Nf3 Nc6 Bb5 Qf6 O-O
e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O Re1 a6 Bb3 Ba7 h3
e4 e5 Nf3 Nc6 Bc4 Bc5 O-O Nf6 d3 d6 c3 O-O
e4 e5 Nf3 Nc6 Bc4 Bc5 d3 Nf6 c3 d6 O-O O-O
e4 e5 Nf3 Nc6 Bc4 Bc5 Nc3 Nf6 d3 d6 O-O O-O
e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4 O-O Nge7
e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Be7 O-O O-O Re1 d6 c3
e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Ba6
e4 e5 Nf3 Nc6 d4 exd4 Bc4 Nf6 e5 d5 Bb5 Ne4 Nxd4 Bd7
e4 e5 Nf3 Nc6 Nc3 Nf6 d4 exd4 Nxd4 Bb4 Nxc6 bxc6 Bd3 d5
e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4 O-O O-O d3 d6
e4 e5 Nf3 Nc6 c3 Nf6 d4 Nxe4 d5 Ne7 Nxe5 Ng6
e4 e5 Nf3 Nc6 Be2 Nf6 d3 d5 Nbd2 Bc5
e4 e5 Nf3 Nc6 d3 Nf6 Be2 d5 Nbd2 Bc5
e4 e5 Nf3 d6 d4 Nf6 Nc3 Nbd7 Bc4 Be7 O-O O-O
e4 e5 Nf3 d6 d4 exd4 Nxd4 Nf6 Nc3 Be7 Bf4 O-O Qd2
e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Be7
e4 e5 Nf3 f6 Nxe5 fxe5 Qh5+ Ke7 Qxe5+ Kf7 Bc4+
e4 e5 Nf3 Qf6 Nc3 c6 d4
e4 e5 Nf3 Qe7 Nc3 c6 d4
e4 e5 Nf3 Bc5 Nxe5
e4 e5 Nf3 Bd6 d4
e4 e5 Nf3 d5 exd5 e4 Qe2 Nf6 d3
e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4 Nf3 Be7 d4 O-O
e4 e5 Nc3 Nf6 Bc4 Nc6 d3 Bb4 Nf3 d5
e4 e5 Nc3 Nf6 Nf3 Nc6 Bb5 Bb4
e4 e5 Nc3 Nf6 g3 d5 exd5 Nxd5 Bg2 Nxc3 bxc3 Bd6
e4 e5 Nc3 Nf6 d3 Bb4 Nf3 d5
e4 e5 f4 exf4 Nf3 d5 exd5 Nf6 Bb5+ c6 dxc6 Nxc6 d4 Bd6
e4 e5 f4 exf4 Bc4 Nf6 Nc3 c6 d4 Bb4
e4 e5 Bc4 Nf6 d3 c6 Nf3 d5 Bb3 Bd6 Nc3 dxe4
e4 e5 d4 exd4 Qxd4 Nc6 Qe3 Nf6 Nc3 Bb4 Bd2 O-O O-O-O Re8
e4 e5 d4 exd4 c3 d5 exd5 Qxd5 cxd4 Nc6 Nf3 Bg4
e4 e5 d3 Nf6 Nf3 Nc6 Be2 d5
e4 e5 Qh5 Nc6 Bc4 g6 Qf3 Nf6 Ne2 Bg7
e4 e5 Qf3 Nf6 Bc4 Nc6 Ne2 Bc5
e4 e5 c3 Nf6 d4 Nxe4 dxe5 d5
e4 e5 Nh3 Nf6 Nc3 Bc5
e4 e5 a3 Nf6 Nc3 d5 exd5 Nxd5
e4 e5 h3 Nf6 Nc3 Bb4
e4 e5 Ke2 d5 exd5 Qxd5
e4 e5 Na3 Nf6 Nc4 Nc6
e4 e5 Be2 Nf6 d3 d5
e4 e5 g3 Nf6 Bg2 d5 exd5 Nxd5
e4 e5 b3 Nf6 Bb2 Nc6
e4 e5 Ne2 Nf6 Ng3 Bc5
e4 e5 c4 Nf6 Nc3 Bc5
e4 e5 f3 Nc6 c3 Nf6
e4 e5 d3 Nc6
e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 Qd2 O-O O-O-O
e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6 Bc4
e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6 Qd2 a6 O-O-O
e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6 Be3 a6 f3 b5 Qd2
e4 c5 Nf3 d6 d4 Nf6 Nc3 cxd4 Nxd4
e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5 Nd5 Be7
e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 Nc3 Bg7 Be3 Nf6 Bc4 O-O Bb3
e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e6 Nc3 Qc7 Be3 a6 Qd2 Nf6 O-O-O
e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e5 Nb5 d6 c4
e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Qb6 Nb3 Nf6 Nc3 e6
e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be3 a6 Qd2 Nf6 O-O-O
e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3 Nf6 O-O Qc7 Qe2
e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Ndb5 Bb4 a3
e4 c5 Nf3 g6 d4 cxd4 Nxd4 Bg7 Nc3 Nc6 Be3 Nf6 Bc4 O-O Bb3
e4 c5 Nf3 a6 c3 Nf6 e5 Nd5 d4 cxd4 cxd4
e4 c5 Nf3 Nf6 e5 Nd5 Nc3 e6 Nxd5 exd5 d4
e4 c5 Nf3 b6 d4 cxd4 Nxd4 Bb7 Nc3
e4 c5 Nf3 Qc7 c3
e4 c5 Nf3 d5 exd5 Qxd5 Nc3 Qd8 Bc4
e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7 Bxe7 Qxe7 f4 O-O Nf3 c5
e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 Qg4 O-O Bd3
e4 e6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Nf3 Ngf6 Nxf6+ Nxf6 Bd3 c5 dxc5 Bxc5 O-O
e4 e6 d4 d5 Nc3 c5 exd5 exd5 Nf3 Nc6 Bb5
e4 e6 d4 d5 Nc3 Nc6 Nf3 Nf6 e5 Ne4 Bd3
e4 e6 d4 c5 Nf3 cxd4 Nxd4
e4 e6 d4 b6 Nf3 Bb7 Bd3
e4 e6 d4 Nf6 e5 Nd5 Nf3
e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3
e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Nf3 Ngf6 Nxf6+ Nxf6 Bd3
e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nf6 Nxf6+ exf6 Bc4
e4 c6 d4 d5 Nc3 g6 Nf3 Bg7 h3
e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 Be3
e4 c6 d4 g6 Nc3 Bg7 Nf3 d6 Be2
e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Bd2 e6 Qe2
e4 d5 exd5 Qxd5 Nc3 Qd6 d4 Nf6 Nf3 a6 g3
e4 d5 exd5 Qxd5 Nc3 Qd8 d4 Nf6 Nf3 Bg4 h3
e4 d5 exd5 Nf6 d4 Nxd5 Nf3 g6 Be2 Bg7 O-O O-O c4 Nb6
e4 d5 exd5 Nf6 d4 Bg4 Nf3 Qxd5 Be2
e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O c6 a4
e4 d6 d4 Nf6 Nc3 e5 Nf3 Nbd7 Bc4 Be7 O-O O-O
e4 d6 d4 g6 Nc3 Bg7 Nf3 Nf6 Be2 O-O O-O
e4 d6 d4 e5 Nf3 exd4 Nxd4 Nf6 Nc3
e4 g6 d4 Bg7 Nc3 d6 Nf3 Nf6 Be2 O-O O-O
e4 g6 d4 Bg7 Nc3 c6 Nf3 d5 h3
e4 g6 d4 d6 Nc3 Bg7 Nf3 Nf6 Be2
e4 Nf6 e5 Nd5 d4 d6 Nf3 Bg4 Be2 e6 O-O Be7 c4 Nb6
e4 Nf6 e5 Nd5 d4 d6 Nf3 g6 Bc4 Nb6 Bb3 Bg7
e4 Nf6 e5 Ng8 d4 d6 Nf3
e4 Nf6 e5 Ne4 d4 d5 Bd3
e4 Nc6 d4 d5 Nc3 dxe4 d5 Ne5 Qd4 Ng6 Qxe4
e4 Nc6 d4 e5 dxe5 Nxe5 Nf3 Nxf3+ Qxf3
e4 b6 d4 Bb7 Bd3 e6 Nf3 c5 c3
e4 a6 d4 b5 Nf3 Bb7 Bd3 e6 O-O
e4 b5 Bxb5 Bb7 Nc3
e4 f5 exf5 Nf6 Nf3 d5 d4
e4 a5 d4 e6 Nf3
e4 h6 d4 e6 Nf3 d5 Nc3
e4 g5 d4 h6 h4 g4 Bf4
e4 Na6 d4 c6 Nf3
e4 Nh6 d4 g6 Nf3
e4 f6 d4 e6 Nf3
e4 h5 d4 d5 exd5
e4 c5 Nf3 Nc6 Nc3
e4 c5 Nc3 Nc6 Nf3
d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6 Be2 Bb7
d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bf4 O-O e3 c5 dxc5 Bxc5 a3 Nc6
d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 cxd5 exd5 Bg5 c6 Qc2
d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 e3 O-O Bd3 dxc4 Bxc4 c5 O-O
d4 Nf6 c4 e6 Nf3 d5 g3 Be7 Bg2 O-O O-O dxc4 Qc2 a6 Qxc4 b5
d4 Nf6 c4 e6 Nf3 d5 cxd5 exd5 Nc3 c6 Bg5 Be7 e3 Nbd7
d4 Nf6 c4 e6 Nf3 d5 e3 Be7 Nc3 O-O Bd3 dxc4 Bxc4 c5
d4 Nf6 c4 e6 Nf3 d5 Bg5 Be7 e3 O-O Nc3 h6 Bh4
d4 Nf6 c4 e6 Nf3 d5 Qc2 c5 cxd5 exd5
d4 Nf6 c4 e6 Nf3 d5 a3 dxc4 e3 c5
d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nc6 a3 Bxc3 bxc3 dxc4 Bxc4 Qc7
d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3+ Qxc3 b6 Bg5 Bb7 f3 h6 Bh4 d5
d4 Nf6 c4 e6 Nc3 Bb4 a3 Bxc3+ bxc3 c5 e3 Nc6 Bd3 O-O Ne2 b6
d4 Nf6 c4 e6 Nc3 Bb4 f3 d5 a3 Be7 e4 dxe4 fxe4 e5 d5 Bc5
d4 Nf6 c4 e6 Nc3 Bb4 Bg5 h6 Bh4 c5 d5 d6 e3 exd5 cxd5
d4 Nf6 c4 e6 Nc3 Bb4 Nf3 O-O Bg5 c5 e3 cxd4 exd4 h6
d4 Nf6 c4 e6 Nc3 Bb4 Bd2 O-O e3 d5 Nf3 c5
d4 Nf6 c4 e6 Nc3 Bb4 g3 O-O Bg2 d5 Nf3 dxc4 O-O Nc6
d4 Nf6 c4 e6 Nc3 Bb4 e4 Bxc3+ bxc3 d6 Bd3 e5
d4 Nf6 c4 e6 Nc3 Bb4 Qb3 c5 dxc5 Nc6 Nf3 Ne4
d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7
d4 Nf6 c4 e6 g3 d5 Nf3 Be7 Bg2 O-O O-O dxc4
d4 Nf6 c4 e6 Bg5 h6 Bxf6 Qxf6 Nc3 Bb4
d4 Nf6 c4 e6 e3 d5 Nc3 Be7 Nf3 O-O Bd3 dxc4 Bxc4 c5
d4 Nf6 c4 e6 a3 d5 Nc3 Be7 cxd5 exd5 Bg5 O-O e3
d4 Nf6 c4 e6 Nd2 d5 Ngf3 c5
d4 Nf6 c4 e6 d5 exd5 cxd5 d6 Nc3 g6
d4 Nf6 c4 e6 f3 d5 Nc3 Be7 cxd5 exd5 Bf4 c5
d4 Nf6 Nf3 d5 c4 e6 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6
d4 Nf6 Nf3 d5 e3 e6 Bd3 c5 b3 Nc6 Bb2 Bd6 O-O O-O
d4 Nf6 Nf3 d5 Bf4 c5 e3 Nc6 c3 Qb6 Qb3 c4 Qc2 Bf5 Qc1 e6 Nbd2 Be7
d4 Nf6 Nf3 d5 g3 e6 Bg2 Be7 O-O O-O c4 dxc4
d4 Nf6 Nf3 d5 Bg5 Ne4 Bf4 c5 e3 Nc6
d4 Nf6 Nf3 d5 Nbd2 e6 e3 c5 c3 Nc6 Bd3 Bd6 O-O O-O
d4 Nf6 Nf3 d5 c3 e6 Bg5 c5 e3 Nc6 Nbd2 Be7
d4 Nf6 Nf3 e6 c4 d5 Nc3 Be7
d4 Nf6 Nf3 e6 Bg5 c5 e3 h6 Bxf6 Qxf6
d4 Nf6 Nf3 e6 e3 d5 Bd3 c5
d4 Nf6 Nf3 e6 g3 d5 Bg2 Be7 O-O O-O
d4 Nf6 Nf3 e6 Bf4 c5 e3 Nc6 c3 d5
d4 Nf6 Bf4 d5 e3 c5 c3 Nc6 Nd2 e6 Ngf3 Bd6 Bg3 O-O Bd3 b6
d4 Nf6 Bf4 d5 Nf3 c5 e3 Nc6 c3 Qb6 Qb3 c4 Qc2 Bf5
d4 Nf6 Bf4 d5 Nc3 e6 e3 Bd6 Bxd6 cxd6
d4 Nf6 Bf4 e6 e3 c5 c3 d5 Nd2 Nc6
d4 Nf6 Bg5 e6 e4 h6 Bxf6 Qxf6 Nf3 d6 Nc3 g6
d4 Nf6 Bg5 e6 Nf3 c5 e3 h6 Bxf6 Qxf6
d4 Nf6 Bg5 e6 Nd2 d5 e3 Be7
d4 Nf6 Bg5 e6 e3 h6 Bh4 c5 c3 Qb6 Qb3 cxd4 exd4
d4 Nf6 Bg5 e6 c4 h6 Bxf6 Qxf6 Nc3 Bb4
d4 Nf6 Nc3 d5 Bg5 Nbd7 Nf3 h6 Bh4 e6 e3 Be7 Bd3 c5
d4 Nf6 Nc3 d5 Bf4 e6 e3 Bd6 Bxd6 cxd6 Nf3 O-O
d4 Nf6 Nc3 d5 e4 dxe4 f3 exf3 Nxf3 Bg4 h3 Bxf3 Qxf3 c6
d4 Nf6 Nc3 d5 Nf3 e6 Bg5 Be7 e3 O-O
d4 Nf6 Nc3 d5 e3 e6 Nf3 c5
d4 Nf6 Nc3 d5 Qd3 c5 dxc5 e6
d4 Nf6 e3 d5 Nf3 e6 Bd3 c5 c3 Nc6 Nbd2 Bd6 O-O O-O
d4 Nf6 e3 d5 Bd3 c5 c3 Nc6 f4 Bg4
d4 Nf6 e3 d5 c4 e6 Nc3 Be7 Nf3 O-O
d4 Nf6 g3 d5 Bg2 e6 Nf3 Be7 O-O O-O c4 dxc4
d4 Nf6 f3 d5 e4 dxe4 Nc3 exf3 Nxf3 Bg4 h3 Bxf3 Qxf3 c6
d4 Nf6 Nd2 d5 Ngf3 e6 e3 c5 c3 Nc6
d4 Nf6 c3 d5 Bf4 c5 e3 Nc6 Nd2 e6
d4 Nf6 Qd3 d5 Nf3 e6 Bg5 c5
d4 Nf6 a3 d5 Nf3 e6 c4 dxc4 e3 c5
d4 Nf6 h3 d5 Nf3 e6 c4 c5
d4 Nf6 Nh3 d5 g3 e5
d4 Nf6 b3 e6 Bb2 d5 Nf3 c5 e3 Nc6
d4 Nf6 d5 e6 c4 exd5 cxd5 d6 Nc3 g6
d4 Nf6 Na3 d5 Nf3 e6
d4 Nf6 a4 d5 Nf3 e6
d4 Nf6 h4 d5 Nf3 e6
d4 Nf6 g4 Nxg4 e4 d6 h3 Nf6 Nc3 e5
d4 Nf6 f4 d5 Nf3 g6 e3 Bg7 Be2 O-O O-O c5 c3
d4 Nf6 b4 e6 a3 d5 Nf3 c5 bxc5 Bxc5
c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7 d3 O-O a3 Be6 b4 a5
c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5 Bg2 Nb6 Nf3 Nc6 O-O Be7 d3 O-O
c4 e5 Nc3 Nf6 e4 Bc5 Nf3 d6 d4 exd4 Nxd4 O-O Be2 Re8
c4 e5 Nc3 Nf6 e3 Bb4 Nge2 O-O a3 Be7 d4 exd4 Nxd4 d5
c4 e5 Nc3 Nf6 d3 Bb4 Bd2 O-O e4 c6
c4 e5 Nc3 Nf6 a3 d5 cxd5 Nxd5 e4 Nxc3 bxc3
c4 e5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6
c4 e5 g3 Nf6 Bg2 d5 cxd5 Nxd5 Nc3 Nb6 Nf3 Nc6 O-O Be7 d3 O-O
c4 e5 Nf3 e4 Nd4 Nc6 Nc2 Nf6 Nc3 d5 cxd5 Nxd5
c4 e5 e3 Nf6 Nc3 d5 cxd5 Nxd5 Nf3 Nc6
c4 e5 d3 Nc6 Nc3 f5 g3 Nf6 Bg2 Bc5
c4 e5 a3 Nf6 Nc3 d5 cxd5 Nxd5
c4 e5 e4 Nf6 Nc3 Bc5 g3
c4 e5 d4 exd4 Qxd4 Nc6 Qd1 Nf6 Nc3 Bb4
c4 e5 b3 Nf6 Bb2 Nc6 e3 d5 cxd5 Nxd5
c4 e5 Qc2 Nf6 Nc3 Bb4
c4 e5 f4 exf4 Nf3 Nf6 d4 d5
c4 e5 h3 Nf6 Nc3 d5
c4 e5 g4 d5 cxd5 Bxg4
c4 e5 Nf3 Nc6 Nc3 Nf6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7 d3 O-O
c4 e5 Nf3 Nc6 d4 e4 Nfd2 Nf6 Nc3 d5
c4 e5 Nf3 Nc6 g3 Nf6 Bg2 d5 cxd5 Nxd5 O-O Nb6 Nc3 Be7
c4 e5 Nf3 Nc6 e3 Nf6 d4 e4 Nfd2 d5
Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7 d3 O-O Nbd2 c5 e4 Nc6 Re1 b5
Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6
Nf3 d5 d4 Nf6 Bf4 c5 e3 Nc6 c3 Qb6 Qb3 c4 Qc2 Bf5
Nf3 d5 d4 Nf6 e3 e6 Bd3 c5 b3 Nc6 Bb2 Bd6 O-O O-O
Nf3 d5 d4 Nf6 g3 e6 Bg2 Be7 O-O O-O c4 dxc4
Nf3 d5 d4 Nf6 Bg5 Ne4 Bf4 c5 e3 Nc6
Nf3 d5 d4 Nf6 c3 e6 Bg5 c5 e3 Nc6
Nf3 d5 d4 Nf6 Nbd2 e6 e3 c5 c3 Nc6
Nf3 d5 d4 Nf6 Nc3 e6 Bg5 Be7 e3 O-O
Nf3 d5 c4 e6 d4 Nf6 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6
Nf3 d5 c4 e6 g3 Nf6 Bg2 Be7 O-O O-O b3 c5 Bb2 Nc6 e3 d4
Nf3 d5 c4 e6 b3 Nf6 Bb2 Be7 e3 O-O
Nf3 d5 c4 e6 cxd5 exd5 d4 Nf6 Nc3 c6
Nf3 d5 c4 e6 e3 Nf6 Nc3 Be7 d4 O-O
Nf3 d5 c4 d4 e3 Nc6 exd4 Nxd4 Nxd4 Qxd4 Nc3
Nf3 d5 e3 Nf6 c4 e6 Nc3 Be7 d4 O-O b3 b6
Nf3 d5 e3 Nf6 d4 e6 Bd3 c5 b3 Nc6
Nf3 d5 b3 Nf6 Bb2 e6 e3 Be7 c4 O-O Nc3 c5
Nf3 d5 d3 Nf6 g3 c5 Bg2 Nc6 O-O e5 Nbd2 Be7 e4 O-O
Nf3 d5 e4 dxe4 Ng5 Nf6 d3 exd3 Bxd3
Nf3 d5 Nc3 Nf6 d4 e6 Bg5 Be7 e3 O-O
Nf3 d5 b4 Nf6 Bb2 e6 a3 c5 bxc5 Bxc5
Nf3 d5 c3 Nf6 d4 e6 Bg5 c5
Nf3 d5 a3 Nf6 d4 e6 c4 c5
Nf3 d5 h3 Nf6 d4 e6 c4 c5
Nf3 d5 Ng1 Nf6 d4 e6
Nf3 d5 g4 Bxg4 Ne5 Bf5
Nf3 d5 a4 Nf6 d4 e6
Nf3 d5 Na3 Nf6 d4 e6
Nf3 d5 Rg1 Nf6 d4 e6
Nf3 d5 e3 Nf6 b3 e6 Bb2 Be7 c4 O-O
g3 d5 Bg2 Nf6 Nf3 e6 O-O Be7 d3 O-O Nbd2 c5 e4 Nc6
g3 d5 Bg2 Nf6 d3 e5 Nf3 Nc6 O-O Be7 c4 O-O
g3 d5 Bg2 Nf6 d4 e6 Nf3 Be7 O-O O-O c4 dxc4
g3 d5 Bg2 Nf6 c4 c6 Nf3 Bf5 cxd5 cxd5 Qb3 Qb6
g3 d5 Nf3 Nf6 Bg2 e6 O-O Be7
g3 d5 d4 Nf6 Bg2 e6 Nf3 Be7
g3 d5 c4 c6 Bg2 Nf6 Nf3 Bf5
g3 d5 e4 dxe4 Bg2 Nf6 Nc3 e5
g3 d5 Bg2 Nf6 e3 e5 Ne2 Bd6
b3 e5 Bb2 Nc6 e3 d5 Bb5 Bd6 Nf3 Qe7 c4 Nf6 Nc3 O-O
b3 e5 Bb2 Nc6 c4 Nf6 e3 d5 cxd5 Nxd5 a3 Bd6
b3 e5 Bb2 Nc6 Nf3 e4 Nd4 Nxd4 Bxd4 Nf6
b3 e5 Bb2 Nc6 e4 Nf6 Nc3 Bb4
b3 e5 Bb2 Nc6 g3 d5 Bg2 Nf6 d3 Be7
b3 e5 Bb2 Nc6 d4 exd4 Nf3 Bc5
b3 e5 e4 Nf6 Bb2 Nc6 Nc3 Bb4
b3 e5 Nf3 e4 Nd4 Nc6 Nxc6 dxc6 Bb2 Bd6
b3 e5 c4 Nf6 Bb2 Nc6 e3 d5 cxd5 Nxd5
b3 e5 e3 d5 Bb2 Nc6 Bb5 Bd6
f4 d5 Nf3 Nf6 e3 g6 Be2 Bg7 O-O O-O d3 c5 Qe1 Nc6
f4 d5 Nf3 Nf6 g3 g6 Bg2 Bg7 O-O O-O d3 c5
f4 d5 Nf3 Nf6 b3 g6 Bb2 Bg7 e3 O-O Be2 c5
f4 d5 e3 Nf6 Nf3 g6 Be2 Bg7 O-O O-O
f4 d5 e4 dxe4 Nc3 Nf6 d3 exd3 Bxd3 g6
f4 d5 d4 Nf6 Nf3 g6 e3 Bg7
f4 d5 g3 Nf6 Bg2 g6 Nf3 Bg7
f4 d5 b3 Nf6 Bb2 g6 Nf3 Bg7
f4 d5 c4 d4 Nf3 c5
e3 d5 d4 Nf6 Nf3 e6 Bd3 c5 c3 Nc6 Nbd2 Bd6 O-O O-O
e3 d5 Nf3 Nf6 c4 e6 d4 Be7 Nc3 O-O
e3 d5 c4 e6 d4 Nf6 Nc3 Be7 Nf3 O-O
e3 d5 b3 e5 Bb2 Bd6 c4 c6
e3 d5 f4 Nf6 Nf3 g6
e3 d5 g3 Nf6 Bg2 e5 Ne2 Bd6
e3 d5 Be2 e5 d4 e4
e3 d5 Qh5 Nf6 Qh4 e5
d3 d5 Nf3 Nf6 g3 c5 Bg2 Nc6 O-O e5 Nbd2 Be7 e4 O-O
d3 d5 e4 dxe4 dxe4 Qxd1+ Kxd1 Nc6
d3 d5 g3 e5 Bg2 Nf6 Nf3 Nc6 O-O Be7
d3 d5 Nd2 e5 e4 Nf6 Ngf3 Nc6
d3 d5 c3 e5 Nd2 Nf6 e4 Nc6
d3 d5 e3 e5 Nd2 Nf6
d3 d5 Nc3 Nf6 e4 dxe4 dxe4 Qxd1+ Kxd1 e5
d3 d5 Bf4 Nf6 Nf3 c5
Nc3 d5 e4 d4 Nce2 e5 Ng3 Be6 c3 c5 Bb5+ Nd7
Nc3 d5 d4 Nf6 Bg5 Nbd7 Nf3 h6 Bh4 e6 e3 Be7
Nc3 d5 d4 Nf6 Bf4 e6 e3 Bd6 Bxd6 cxd6
Nc3 d5 d4 Nf6 e4 dxe4 f3 exf3 Nxf3 Bg4 h3 Bxf3 Qxf3 c6
Nc3 d5 Nf3 Nf6 d4 e6 Bg5 Be7
Nc3 d5 e3 e5 d4 exd4 exd4 Nf6
Nc3 d5 f4 Nf6 Nf3 g6 e3 Bg7
Nc3 d5 g3 Nf6 Bg2 e6 d3 Be7
Nc3 d5 b3 e5 Bb2 Nc6 e3 Nf6
Nc3 d5 e4 dxe4 Nxe4 Nf6
a3 d5 d4 Nf6 Nf3 e6 c4 c5 e3 Nc6
a3 d5 Nf3 Nf6 d4 e6 c4 c5
a3 d5 c4 e6 d4 Nf6 Nc3 Be7
a3 d5 e4 dxe4 Nc3 Nf6
a3 d5 b4 e5 Bb2 Bd6
a3 d5 g3 Nf6 Bg2 e5
h3 d5 d4 Nf6 Nf3 e6 c4 c5
h3 d5 c4 e6 d4 Nf6
h3 d5 Nf3 Nf6 d4 e6
h3 d5 e4 dxe4 Nc3 Nf6
h3 d5 g4 e5 Bg2 c6
b4 e5 Bb2 Bxb4 Bxe5 Nf6 c4 O-O Nf3 d5 e3
b4 e5 a3 d5 Bb2 Bd6 e3 Nf6
b4 e5 Bb2 Bxb4 e4 Nc6
b4 e5 b5 d5 Bb2 Bd6
b4 e5 c3 d5 e3 Nf6
c3 e5 d4 exd4 cxd4 d5 Nc3 Nf6 Bg5 Be7 e3 O-O
c3 e5 e4 Nf6 d4 Nxe4 dxe5 d5
c3 e5 Nf3 e4 Nd4 c5 Nc2 d5 d3 Nf6
c3 e5 d3 d5 Nd2 Nf6 e4 Nc6
Na3 e5 Nc4 Nc6 e4 Nf6 d3 d5 exd5 Nxd5 Nf3
Na3 e5 e4 Nf6 Nc4 Nc6
Na3 e5 c4 Nf6 Nc2 d5
Na3 e5 d4 exd4 Qxd4 Nc6
Nh3 d5 g3 e5 f4 Bxh3 Bxh3 exf4 O-O
Nh3 d5 d4 Nf6 Nf4 e5
Nh3 d5 e3 e5 d4 Nc6
a4 e5 e4 Nf6 Nc3 Bc5 Nf3 Nc6
a4 e5 d4 exd4 Qxd4 Nc6 Qe4+ Be7
h4 d5 d4 c5 e3 Nc6 Nf3 Nf6 c4 e6
h4 d5 e4 dxe4 Nc3 Nf6
f3 e5 e4 Nc6 c3 Nf6 d4 d5
f3 e5 d4 exd4 Qxd4 Nc6
f3 e5 e3 d5 c4 c6
g4 d5 Bg2 Bxg4 c4 c6 cxd5 cxd5 Qb3 Nc6 Qxb7 Nd4 Kf1 e6
g4 d5 h3 e5 Bg2 c6 d4 e4 c4 Bd6
g4 d5 e4 dxe4 h3 Nc6
"""

_BOOK = None


def _key(board):
    return board.board_fen() + (" w " if board.turn else " b ") + board.castling_xfen()


def _book():
    global _BOOK
    if _BOOK is None:
        _BOOK = {}
        for text in LINES.strip().splitlines():
            b = chess.Board()
            for san in text.split():
                move = b.parse_san(san)          # an error here is a typo in a line: it must fail loudly
                _BOOK.setdefault(_key(b), move)
                b.push(move)
    return _BOOK


def repertoire_move(board):
    move = _book().get(_key(board))
    return move if move is not None and move in board.legal_moves else None


MINORS = (chess.KNIGHT, chess.BISHOP)
RIM = chess.BB_FILE_A | chess.BB_FILE_H


def _home(colour):
    return chess.BB_RANK_1 if colour == chess.WHITE else chess.BB_RANK_8


def principles(board, move, ply):
    """Classic opening principles for one move of the side to move, in units."""
    me = board.turn
    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0.0
    kind, units = piece.piece_type, 0.0
    capture = board.is_capture(move)
    to_bb = chess.BB_SQUARES[move.to_square]
    from_bb = chess.BB_SQUARES[move.from_square]
    can_castle = board.has_castling_rights(me)
    if board.is_castling(move):
        return 1.5
    if kind == chess.KING:
        return -2.0 if can_castle else -0.5
    if capture:
        return 0.0                                   # trades and gains belong to the calculation
    undeveloped = sum(1 for t in MINORS for s in board.pieces(t, me) if chess.BB_SQUARES[s] & _home(me))
    if kind in MINORS:
        if from_bb & _home(me):
            units += 0.8                             # a new piece in play
            if kind == chess.KNIGHT:
                if to_bb & RIM:
                    units -= 1.6                     # a knight on the rim
                elif chess.square_file(move.to_square) in (3, 4):
                    units -= 0.4                     # Ne2/Nd2: in front of its own pieces, not the centre
        else:
            units -= 0.6 if undeveloped else 0.2     # the same piece again, others still at home
            if to_bb & _home(me):
                units -= 0.6                         # and back home
        if kind == chess.BISHOP:
            # in front of its own central pawn still at home (Bd3 with the d-pawn on d2): blocks it
            front = move.to_square - 8 if me == chess.WHITE else move.to_square + 8
            if 0 <= front < 64 and chess.square_file(move.to_square) in (3, 4):
                p = board.piece_at(front)
                if p is not None and p.piece_type == chess.PAWN and p.color == me:
                    units -= 0.7
    elif kind == chess.QUEEN:
        units -= 0.8 if ply < 16 else 0.3
    elif kind == chess.ROOK:
        if can_castle:
            units -= 1.0                             # a rook walk before castling
        elif chess.square_file(move.to_square) not in (2, 3, 4):
            units -= 0.4                             # a rook belongs on the central files
    elif kind == chess.PAWN:
        f = chess.square_file(move.from_square)
        start = from_bb & (chess.BB_RANK_2 if me == chess.WHITE else chess.BB_RANK_7)
        if f in (3, 4):
            units += (0.9 if abs(move.to_square - move.from_square) == 16 else 0.4) if start else 0.1
        elif f == 2:
            units += 0.3 if start else 0.0
        elif f in (0, 7):
            units -= 0.6
        elif f == 5:
            units -= 0.2                             # the f-pawn opens its own king
        elif f in (1, 6):
            units -= 0.1 if start and abs(move.to_square - move.from_square) == 8 else 0.4
    return units


def weight_at(ply, plies):
    """Full weight for the first two thirds of the opening, then fading to nothing at `plies`."""
    if ply >= plies:
        return 0.0
    full = plies * 2 // 3
    return 1.0 if ply < full else (plies - ply) / max(1, plies - full)
