"""Openings: a repertoire of main lines of opening theory, and the opening principles after it.

repertoire_move(board): the theory move for this position (first line listed wins), or None.
principles(board, move, ply): the move's worth by the classic principles, in units (about -2..+1.5):
develop the knights and bishops towards the centre, central pawns, castle, and not the rim knights,
early queen trips, rook or king walks before castling, the same piece twice, or wing pawns.
"""
import chess

# White: 1.e4. Black: 1...e5 against 1.e4, the Queen's Gambit Declined / Nimzo against 1.d4, 1...e5 against
# 1.c4, 1...d5 against the rest. Each string a line from the start; a position's move is the first line's.
LINES_V1 = """
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

# Human opening theory only. The frozen lines come first, so an extension never changes
# an existing choice. Opponent alternatives add branches; there is still one move per key.
# Main lines reach moves 12--15; dubious early moves are met by centre/development,
# or established tactical refutations (Damiano, Latvian, elephant and queen raids).
LINES_EXTENSIONS = """
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5 Bc2 c5 d4 Qc7 Nbd2 cxd4 cxd4 Nc6 Nb3 a5 Be3 a4
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5 Bc2 c5 d4 Qc7 Nbd2 Nc6 d5 Nd8 Nf1 Ne8 Ng3 g6
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5 Bc2 c5 d4 Qc7 Nbd2 Bb7 Nf1 Rfe8 Ng3 Bf8 d5
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5 dxe5 Be6 c3 Bc5 Nbd2 O-O Bc2 Bf5 Nb3 Bb6 Nfd4 Nxd4 cxd4
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O b5 Bb3 Be7 Re1 d6 c3 O-O h3 Na5 Bc2 c5 d4 Qc7 Nbd2 cxd4 cxd4 Nc6 Nb3
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O d6 c3 Bd7 d4 Be7 Re1 O-O Nbd2 Re8 Nf1 Bf8 Ng3 g6 h3 Bg7
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 d6 c3 Bd7 d4 Nf6 O-O Be7 Re1 O-O Nbd2 Re8 Nf1 Bf8 Ng3 g6 h3 Bg7
e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O f6 d4 exd4 Nxd4 c5 Nb3 Qxd1 Rxd1 Be6 Nc3 Bd6 Be3 b6 Nd5 O-O-O a4 Ne7
e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6 Bxc6 dxc6 dxe5 Nf5 Qxd8+ Kxd8 Nc3 Be6 h3 Be7 Bg5 h6 Bxe7+ Kxe7 Rad1 Rad8
e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Be7 Re1 d6 d4 exd4 Nxd4 Bd7 Nc3 O-O Bf4 Re8 Bxc6 bxc6 Qd3 Bf8 Rad1
e4 e5 Nf3 Nc6 Bb5 Bc5 c3 Nf6 d4 exd4 e5 Nd5 O-O O-O cxd4 Bb6 Nc3 Nxc3 bxc3 d6 exd6 Qxd6 Re1
e4 e5 Nf3 Nc6 Bb5 d6 d4 Bd7 Nc3 Nf6 O-O Be7 Re1 exd4 Nxd4 O-O Bf4 Re8 Bxc6 bxc6 Qd3 Bf8 Rad1
e4 e5 Nf3 Nc6 Bb5 f5 Nc3 fxe4 Nxe4 d5 Nxe5 dxe4 Nxc6 Qg5 Qe2 Nf6 f4 Qxf4 Ne5+ c6 d4 Qh4+ g3 Qh3 Bc4
e4 e5 Nf3 Nc6 Bb5 Nge7 O-O g6 c3 Bg7 d4 exd4 cxd4 d5 exd5 Nxd5 Re1+ Be6 Nc3 O-O
e4 e5 Nf3 Nc6 Bb5 Nd4 Nxd4 exd4 O-O c6 Bc4 Nf6 Re1 d6 d3 Be7 Nd2 O-O Nf3
e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O Re1 a6 Bb3 Ba7 h3 h6 Nbd2 Be6 Bc2 d5 exd5 Qxd5 Nc4
e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O Re1 a6 Bb3 Ba7 h3 h6 Nh2 Be6 Bxe6 fxe6 Be3 Bxe3 fxe3 d5 exd5 exd5
e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O Re1 a6 Bb3 Ba7 h3 h6 Be3 Bxe3 Rxe3 Be6 Nbd2 d5
e4 e5 Nf3 Nc6 Bc4 Bc5 O-O Nf6 d3 d6 c3 O-O Re1 a6 Bb3 Ba7 h3 h6 Nbd2 Be6 Bc2
e4 e5 Nf3 Nc6 Bc4 Bc5 d3 Nf6 c3 d6 O-O O-O Re1 a6 Bb3 Ba7 h3 h6 Nbd2 Be6 Bc2
e4 e5 Nf3 Nc6 Bc4 Bc5 Nc3 Nf6 d3 d6 O-O O-O Bg5 h6 Bh4 g5 Bg3 Bg4 Nd5 Nxd5 Bxd5
e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4 O-O Nge7 cxd4 d5 exd5 Nxd5 Ba3 Be6 Qb3 Bb6 Re1
e4 e5 Nf3 Nc6 Bc4 Bc5 d4 exd4 O-O d6 c3 Nf6 cxd4 Bb6 Nc3 O-O Re1 Bg4 Be3
e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Ba6 b3 g6 Ba3 Qe6 Bxf8 Kxf8 Nd2 Kg7
e4 e5 Nf3 Nc6 d4 exd4 Bc4 Nf6 e5 d5 Bb5 Ne4 Nxd4 Bd7 Bxc6 bxc6 O-O Bc5 Be3 O-O
e4 e5 Nf3 Nc6 Nc3 Nf6 d4 exd4 Nxd4 Bb4 Nxc6 bxc6 Bd3 d5 exd5 cxd5 O-O O-O Bg5 c6 Qf3
e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4 O-O O-O d3 d6 Bg5 Bxc3 bxc3 Qe7 Re1 Nd8 d4 Ne6
e4 e5 Nf3 d6 d4 Nf6 Nc3 Nbd7 Bc4 Be7 O-O O-O Re1 c6 a4 Qc7 Ba2 h6 h3 Re8
e4 e5 Nf3 d6 d4 exd4 Nxd4 Nf6 Nc3 Be7 Bf4 O-O Qd2 Nc6 O-O-O Nxd4 Qxd4 Be6 f3 c6
e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Be7 c4 Nb4 Be2 O-O Nc3 Bf5 a3 Nxc3 bxc3 Nc6
e4 e5 Nf3 f6 Nxe5 fxe5 Qh5+ Ke7 Qxe5+ Kf7 Bc4+ d5 Bxd5+ Kg6 h4 h5 Bxb7 Bxb7 Qf5+ Kh6 d4+
e4 e5 Nf3 f6 Nxe5 Qe7 Nf3 Qxe4+ Be2 d5 O-O Nc6 Re1 Be7 Nc3 Qf5 d4
e4 e5 Nf3 f5 Nxe5 Qf6 d4 d6 Nc4 fxe4 Nc3 Qg6 Be2 Nf6 O-O Be7 f3 exf3 Bxf3
e4 e5 Nf3 d5 exd5 e4 Qe2 Nf6 d3 Be7 dxe4 O-O Nc3 Re8 Be3 Bb4 Qd3
e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4 Nf3 Be7 d4 O-O Bd3 f5 exf6 Nxf6 O-O c5
e4 e5 Nc3 Nf6 Bc4 Nc6 d3 Bb4 Nf3 d5 exd5 Nxd5 Bd2 Nxc3 bxc3 Bd6 O-O
e4 e5 f4 exf4 Nf3 d5 exd5 Nf6 Bb5+ c6 dxc6 Nxc6 d4 Bd6 O-O O-O Nc3 Bg4
e4 e5 f4 exf4 Bc4 Nf6 Nc3 c6 d4 Bb4 e5 d5 exf6 dxc4 Nf3 O-O O-O
e4 e5 Bc4 Nf6 d3 c6 Nf3 d5 Bb3 Bd6 Nc3 dxe4 Nxe4 Nxe4 dxe4 O-O O-O
e4 e5 d4 exd4 Qxd4 Nc6 Qe3 Nf6 Nc3 Bb4 Bd2 O-O O-O-O Re8 Qg3 d6 f3 Be6
e4 e5 d4 exd4 c3 d5 exd5 Qxd5 cxd4 Nc6 Nf3 Bg4 Be2 O-O-O Nc3 Bb4 O-O
e4 e5 Qh5 Nc6 Bc4 g6 Qf3 Nf6 Ne2 Bg7 d3 O-O Nbc3 d6 Bg5 h6
e4 e5 Qf3 Nf6 Bc4 Nc6 Ne2 Bc5 d3 d6 O-O Be6 Nbc3 Qd7
e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 Qd2 O-O O-O-O Nbd7 g4 b5 g5 b4 Nd5 Nxd5 exd5 Bf5
e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 Qd2 O-O O-O-O Qc7 g4 b5 g5 b4 Nd5 Nxd5 exd5 Bf5
e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6 Bc4 Bd7 O-O-O Rc8 Bb3 Ne5 h4 h5 Kb1
e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6 Qd2 a6 O-O-O Bd7 f4 Be7 Nf3 b5 Bxf6 gxf6 Kb1 Qb6
e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6 Be3 a6 f3 b5 Qd2 Nbd7 g4 h6 O-O-O Bb7 h4 b4 Na4 d5
e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5 Nd5 Be7 Bxf6 Bxf6 c3 O-O Nc2 Bg5 a4
e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 Nc3 Bg7 Be3 Nf6 Bc4 O-O Bb3 d6 f3 Bd7 Qd2 Rc8 O-O-O Ne5 h4 h5
e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e6 Nc3 Qc7 Be3 a6 Qd2 Nf6 O-O-O Bb4 f3 Ne5 Nb3 b5 Kb1 Bb7
e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3 Nf6 O-O Qc7 Qe2 d6 c4 Be7 Nc3 O-O b3 b6 Bb2 Bb7
e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e5 Nb5 d6 c4 Be7 N1c3 a6 Na3 Nf6 Be2 O-O O-O Be6
e4 c5 Nf3 a6 c3 Nf6 e5 Nd5 d4 cxd4 cxd4 d6 Bc4 Nb6 Bb3 Nc6 O-O dxe5 dxe5 Qxd1 Rxd1
e4 c5 Nf3 Nf6 e5 Nd5 Nc3 e6 Nxd5 exd5 d4 cxd4 Qxd4 Nc6 Qxd5 d6 exd6 Bxd6
e4 c5 Nf3 d5 exd5 Qxd5 Nc3 Qd8 Bc4 e6 O-O Nc6 d3 Nf6 Re1 Be7
e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7 Bxe7 Qxe7 f4 O-O Nf3 c5 Qd2 Nc6 O-O-O a6 dxc5 Nxc5 Bd3 b5 Kb1 Bb7 Ne2 Qc7 Rhf1 Rfc8
e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 Qg4 O-O Bd3 Nbc6 Nf3 f5 exf6 Rxf6 Bg5 Rf7 O-O
e4 e6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Nf3 Ngf6 Nxf6+ Nxf6 Bd3 c5 dxc5 Bxc5 O-O O-O Qe2 b6 Bg5 Bb7 Rad1
e4 e6 d4 d5 Nc3 c5 exd5 exd5 Nf3 Nc6 Bb5 Nf6 O-O Be7 dxc5 O-O Be3 Bg4 h3 Bh5
e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3 e6 Bf4 Ngf6 O-O-O Be7 Ne4 Nxe4 Qxe4
e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Nf3 Ngf6 Nxf6+ Nxf6 Bd3 Bg4 c3 e6 h3 Bxf3 Qxf3 Bd6 O-O O-O
e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nf6 Nxf6+ exf6 Bc4 Bd6 Ne2 O-O O-O Re8 Bf4
e4 c6 d4 d5 Nc3 g6 Nf3 Bg7 h3 Nf6 e5 Ne4 Nxe4 dxe4 Ng5 c5 Bc4
e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Bd2 e6 Qe2 Bb4 O-O-O Nbd7 a3 Bxc3 Bxc3 Qc7
e4 d5 exd5 Qxd5 Nc3 Qd6 d4 Nf6 Nf3 a6 g3 Bg4 Bg2 Nc6 O-O O-O-O Be3 e6 Qd2
e4 d5 exd5 Qxd5 Nc3 Qd8 d4 Nf6 Nf3 Bg4 h3 Bxf3 Qxf3 c6 Be3 e6 O-O-O Nbd7 Bd3
e4 d5 exd5 Nf6 d4 Nxd5 Nf3 g6 Be2 Bg7 O-O O-O c4 Nb6 Nc3 Nc6 Be3 Bg4 d5 Bxf3 Bxf3
e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O c6 a4 Qc7 h3 Nbd7 Be3 e5 dxe5 dxe5 Qd2
e4 g6 d4 Bg7 Nc3 d6 Nf3 Nf6 Be2 O-O O-O Nc6 d5 Ne5 Nxe5 dxe5 Be3 e6
e4 Nf6 e5 Nd5 d4 d6 Nf3 Bg4 Be2 e6 O-O Be7 c4 Nb6 Nc3 O-O Be3 Nc6 exd6 cxd6 b3
e4 Nf6 e5 Nd5 d4 d6 Nf3 g6 Bc4 Nb6 Bb3 Bg7 O-O O-O h3 Nc6 Re1 dxe5 dxe5 Qxd1 Rxd1
e4 Nf6 e5 Ng8 d4 d6 Nf3 Nf6 Nc3 g6 Bc4 Bg7 O-O O-O Re1 Nc6 exd6 cxd6 h3
e4 Nf6 e5 Ng8 d4 e6 Nf3 d5 Bd3 c5 c3 Nc6 O-O Nge7 Re1 Ng6 h4 Be7 h5
e4 Nf6 e5 Ng8 d4 d5 Nf3 Bf5 Bd3 Bxd3 Qxd3 e6 O-O c5 c3 Nc6 Be3
e4 Nc6 d4 d5 Nc3 dxe4 d5 Ne5 Qd4 Ng6 Qxe4 Nf6 Qa4+ Bd7 Bb5 a6 Bxd7+ Qxd7
e4 b6 d4 Bb7 Bd3 e6 Nf3 c5 c3 Nf6 Qe2 Be7 O-O O-O Nbd2 d5 e5
e4 a6 d4 b5 Nf3 Bb7 Bd3 e6 O-O c5 c3 Nf6 Re1 Be7 e5 Nd5 Nbd2
e4 a5 d4 e6 Nf3 d5 Nc3 Nf6 e5 Nfd7 Bd3 c5 O-O Nc6 Re1 Be7 h3 O-O
e4 a5 d4 d5 exd5 Qxd5 Nc3 Qd6 Nf3 Nf6 Bc4 e6 O-O Be7 Re1 O-O
e4 a5 d4 e5 Nf3 Nc6 Bb5 Nf6 O-O d6 d5 Bd7 Nc3 Be7 Be3 O-O
e4 a5 d4 Nc6 Nf3 e5 Bb5 Nf6 O-O d6 d5 Bd7 Nc3 Be7 Be3 O-O
e4 a5 d4 d6 Nf3 Nf6 Nc3 g6 Bc4 Bg7 O-O O-O Re1 Nc6 h3
e4 a5 d4 a4 Nf3 d6 Nc3 e5 dxe5 dxe5 Qxd8+ Kxd8 Nxe5 Be6 Bc4
e4 a5 d4 h5 Nf3 d6 Bc4 Nf6 Nc3 e6 O-O Be7 Re1 O-O
e4 a5 d4 b5 Bxb5 Bb7 Nc3 e6 Nf3 Nf6 Bd3 Bb4 O-O O-O Re1
e4 b5 Bxb5 Bb7 Nc3 e6 Nf3 Nf6 e5 Nd5 O-O Be7 d4 O-O
e4 f5 exf5 Nf6 Nf3 d5 d4 Bxf5 Bd3 Bxd3 Qxd3 Nc6 O-O Qd7 Re1
e4 g5 d4 h6 h4 g4 Bf4 d6 Nc3 Nf6 Qd2 Nc6 O-O-O e5 dxe5
e4 h6 d4 e6 Nf3 d5 Nc3 Nf6 e5 Nfd7 Bd3 c5 O-O Nc6 Re1
e4 f6 d4 e6 Nf3 d5 Nc3 Bb4 Bd3 Ne7 O-O O-O Re1
e4 h5 d4 d5 exd5 Qxd5 Nc3 Qa5 Nf3 Nf6 Bc4 e6 O-O
e4 Na6 d4 c6 Nf3 Nc7 Bd3 d5 e5 Bg4 O-O e6 h3
e4 Nh6 d4 g6 Nf3 Bg7 Nc3 d6 Bc4 O-O O-O Nc6 h3
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 h6 d4 Re8 Nbd2 Bf8 Nf1
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Re8 d4 Bf8 Nbd2 h6 Nf1
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7 Nbd2 Bb7 Bc2 Re8
d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6 Be2 Bb7 O-O Nbd7 Rc1 c5 cxd5 Nxd5 Bxe7 Qxe7 Nxd5 exd5
d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bf4 O-O e3 c5 dxc5 Bxc5 a3 Nc6 Qc2 Qa5 Rd1 Be7 Be2 dxc4 Bxc4 e5
d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 cxd5 exd5 Bg5 c6 Qc2 O-O e3 Nbd7 Bd3 Re8 O-O Nf8 Rab1
d4 Nf6 c4 e6 Nf3 d5 g3 Be7 Bg2 O-O O-O dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7 Bd2 Be4 Qc1 Nbd7
d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nc6 a3 Bxc3 bxc3 dxc4 Bxc4 Qc7 Bd3 e5 Qc2 Re8 d5 e4
d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3+ Qxc3 b6 Bg5 Bb7 f3 h6 Bh4 d5 e3 Nbd7 Bd3 c5 Ne2
d4 Nf6 c4 e6 Nc3 Bb4 a3 Bxc3+ bxc3 c5 e3 Nc6 Bd3 O-O Ne2 b6 e4 Ne8 O-O Ba6 f4 f5
d4 Nf6 c4 e6 Nc3 Bb4 f3 d5 a3 Be7 e4 dxe4 fxe4 e5 d5 Bc5 Nf3 Ng4 b4 Bf2+ Ke2 a5
d4 Nf6 c4 e6 Nc3 Bb4 Nf3 O-O Bg5 c5 e3 cxd4 exd4 h6 Bh4 d5 Rc1 Nc6 a3 Be7 Bd3
d4 Nf6 c4 e6 Nc3 Bb4 g3 O-O Bg2 d5 Nf3 dxc4 O-O Nc6 Qa4 Bd6 Qxc4 e5 Rd1
d4 Nf6 Nf3 d5 e3 e6 Bd3 c5 b3 Nc6 Bb2 Bd6 O-O O-O Nbd2 b6 Ne5 Bb7 f4
d4 Nf6 Nf3 d5 Bf4 c5 e3 Nc6 c3 Qb6 Qb3 c4 Qc2 Bf5 Qc1 e6 Nbd2 Be7 Be2 O-O O-O Nh5 Bg3
d4 Nf6 Bf4 d5 e3 c5 c3 Nc6 Nd2 e6 Ngf3 Bd6 Bg3 O-O Bd3 b6 O-O Bb7 Ne5 Qc7
d4 Nf6 Nc3 d5 Bg5 Nbd7 Nf3 h6 Bh4 e6 e3 Be7 Bd3 c5 O-O O-O Re1 b6
c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7 d3 O-O a3 Be6 b4 a5 b5 Nd4 Nxd4 exd4 Ne4 Bd5
c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5 Bg2 Nb6 Nf3 Nc6 O-O Be7 d3 O-O Be3 Be6 Rc1 f5
c4 e5 Nf3 e4 Nd4 Nc6 Nc2 Nf6 Nc3 d5 cxd5 Nxd5 Nxe4 Bf5 Ng3 Bg6 d4
Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7 d3 O-O Nbd2 c5 e4 Nc6 Re1 b5 e5 Nd7 Nf1 Bb7 Bf4
Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 O-O e3 h6 Bh4 b6 Be2 Bb7 Bd3 Nbd7 O-O c5
g3 d5 Bg2 Nf6 Nf3 e6 O-O Be7 d3 O-O Nbd2 c5 e4 Nc6 Re1 b5 e5 Nd7 Nf1 Bb7
f4 d5 Nf3 Nf6 e3 g6 Be2 Bg7 O-O O-O d3 c5 Qe1 Nc6 Nc3 d4 Ne4 Nxe4 dxe4
b3 e5 Bb2 Nc6 e3 d5 Bb5 Bd6 Nf3 Qe7 c4 Nf6 Nc3 O-O cxd5 Nb4 a3 Nbxd5 Nxd5 Nxd5
"""
LINES = LINES_V1 + LINES_EXTENSIONS
_BOOK = {}


def _key(board, version=2):
    key = board.board_fen() + (" w " if board.turn else " b ") + board.castling_xfen()
    if version == 2:
        key += " " + (chess.square_name(board.ep_square) if board.has_legal_en_passant() else "-")
    return key


def _book(version=2):
    if version not in (1, 2):
        raise ValueError("opening book version must be 1 or 2")
    if version not in _BOOK:
        book = {}
        for text in (LINES_V1 if version == 1 else LINES).strip().splitlines():
            b = chess.Board()
            for san in text.split():
                move = b.parse_san(san)          # an error here is a typo in a line: it must fail loudly
                book.setdefault(_key(b, version), move)
                b.push(move)
        _BOOK[version] = book
    return _BOOK[version]


def repertoire_move(board, version=2):
    move = _book(version).get(_key(board, version))
    return move if move is not None and move in board.legal_moves else None


MINORS = (chess.KNIGHT, chess.BISHOP)
RIM = chess.BB_FILE_A | chess.BB_FILE_H


def _home(colour):
    return chess.BB_RANK_1 if colour == chess.WHITE else chess.BB_RANK_8


def principles_v1(board, move, ply):
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


CENTRE = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5


def _pawn_structure(pawns):
    counts = [chess.popcount(pawns & file_bb) for file_bb in chess.BB_FILES]
    doubled = sum(max(0, n - 1) for n in counts)
    isolated = sum(n for f, n in enumerate(counts)
                   if not (counts[f - 1] if f else 0) and not (counts[f + 1] if f < 7 else 0))
    return doubled, isolated


def _connected_rooks(board, colour):
    rooks = list(board.pieces(chess.ROOK, colour))
    return any((chess.square_rank(a) == chess.square_rank(b) or
                chess.square_file(a) == chess.square_file(b)) and
               not (chess.between(a, b) & board.occupied)
               for i, a in enumerate(rooks) for b in rooks[i + 1:])


def _context(board):
    me = board.turn
    king = board.king(me)
    flank = None
    if king is not None and chess.BB_SQUARES[king] & _home(me) and not board.has_castling_rights(me):
        file = chess.square_file(king)
        flank = (5, 6, 7) if file >= 6 else (0, 1, 2) if file <= 2 else None
    return (_pawn_structure(board.pawns & board.occupied_co[me]),
            _connected_rooks(board, me), flank)


def _principles_v2(board, after, move, ply, context):
    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0.0
    me, kind = piece.color, piece.piece_type
    if board.is_castling(move) or kind == chess.KING:
        return principles_v1(board, move, ply)
    units = principles_v1(board, move, ply)
    structure, connected, flank = context
    capture = board.is_capture(move)
    if kind == chess.PAWN:
        doubled, isolated = _pawn_structure(after.pawns & after.occupied_co[me])
        # Recaptures may be necessary; use a softer preference, never a veto.
        scale = 0.5 if capture else 1.0
        units -= scale * (0.55 * (doubled - structure[0]) + 0.3 * (isolated - structure[1]))
        from_file, to_file = chess.square_file(move.from_square), chess.square_file(move.to_square)
        if flank is not None and from_file in flank:
            rank = chess.square_rank(move.to_square)
            relative = rank if me == chess.WHITE else 7 - rank
            if to_file != from_file:
                units -= 0.35                       # opening a file in the king's pawn cover
            elif relative >= 3:
                units -= 0.65                       # f4/g4/h4 or a5/b5/c5 in front of own castled king
            elif relative == 2:
                units -= 0.1                        # a single step can give the king an escape square
            if flank == (5, 6, 7) and from_file == 5:
                units -= 0.55                       # the f-pawn also opens the diagonal towards the king
    if kind in MINORS or kind == chess.PAWN:
        # Compare real attacks after the move, including opened bishop diagonals.
        before_control = chess.popcount(board.attacks_mask(move.from_square) & CENTRE)
        after_control = chess.popcount(after.attacks_mask(move.to_square) & CENTRE)
        units += 0.18 * (after_control - before_control)
        if kind == chess.PAWN:
            units += 0.25 * (bool(chess.BB_SQUARES[move.to_square] & CENTRE) -
                             bool(chess.BB_SQUARES[move.from_square] & CENTRE))
    if kind != chess.PAWN and kind != chess.BISHOP:
        behind = move.to_square - 8 if me == chess.WHITE else move.to_square + 8
        if 0 <= behind < 64 and chess.square_file(move.to_square) in (3, 4):
            pawn = board.piece_at(behind)
            rank = chess.square_rank(behind) if me == chess.WHITE else 7 - chess.square_rank(behind)
            if pawn is not None and pawn.color == me and pawn.piece_type == chess.PAWN and rank in (1, 2):
                units -= 0.7                       # knights/queen/rooks can block d/e pawns too
    if kind == chess.BISHOP and flank is not None:
        home = (chess.G2 if me else chess.G7) if flank == (5, 6, 7) else (chess.B2 if me else chess.B7)
        if move.from_square == home:
            target = board.piece_at(move.to_square)
            if capture and target is not None and target.piece_type in MINORS and board.is_attacked_by(not me, move.to_square):
                units -= 0.65                      # avoid exchanging the bishop protecting the castled king
            elif not capture:
                units -= 0.25                      # retain its king-cover diagonal where practical
    if not connected and _connected_rooks(after, me):
        units += 0.55                              # finishing development, including moving the queen
    return units


def principle_scores(board, moves, ply, version=2):
    """One shared copy/context for all legal candidates; the caller's board is untouched."""
    if version == 1:
        return [principles_v1(board, move, ply) for move in moves]
    if version != 2:
        raise ValueError("opening principles version must be 1 or 2")
    context, after = _context(board), board.copy(stack=False)
    scores = []
    for move in moves:
        after.push(move)
        scores.append(_principles_v2(board, after, move, ply, context))
        after.pop()
    return scores


def principles(board, move, ply, version=2):
    """Classic opening preferences, with frozen V1 available for paired comparisons."""
    return principle_scores(board, [move], ply, version)[0]


def weight_at(ply, plies):
    """Full weight for the first two thirds of the opening, then fading to nothing at `plies`."""
    if ply >= plies:
        return 0.0
    full = plies * 2 // 3
    return 1.0 if ply < full else (plies - ply) / max(1, plies - full)
