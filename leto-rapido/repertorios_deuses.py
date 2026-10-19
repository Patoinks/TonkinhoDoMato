"""The four deities' opening repertoires (2026-10-19), written by hand from theory, wide enough that an opponent
seldom leaves them at once. Each personality's profile decides its systems:

- Nezha  - aggression, chaos, tricks: 1.e4 gambits (Evans, Fried Liver, Morra, Fantasy, Alekhine-Chatard,
           Austrian, Four Pawns); as Black the Yugoslav Dragon and the Mar del Plata King's Indian.
- Anubis - positional, defensive, materialist, no tricks: 1.d4 exchange variations (QGD, Slav, KID, Gruenfeld),
           gambits accepted and kept; as Black the Caro-Kann and the Queen's Gambit Declined.
- Hades  - positional, aggression, tricks, chaos: 1.e4 main lines (Marshall Ruy Lopez, English Attack, Winawer
           poisoned pawn, 150 Attack); as Black the Najdorf, the Nimzo-Indian and the Benoni.
- Loki   - positional, tricks, chaos: the Trompowsky and the Catalan; as Black the Kan/Taimanov Sicilian and
           the Benko Gambit.

W: lines it plays as White, B: as Black (the other side's moves are the replies it expects).
"""

OPENINGS = {
    "nezha": """
        W: e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4 O-O Nge7 cxd4 d5 exd5 Nxd5 Ba3
        W: e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bb6 a4 a6 Nc3 Nf6 Nd5 Nxd5 exd5
        W: e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Be7 d4 Na5 Be2 exd4 Qxd4
        W: e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Na5 Bb5+ c6 dxc6 bxc6 Be2 h6 Nf3 e4 Ne5
        W: e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Nxd5 Nxf7 Kxf7 Qf3+ Ke6 Nc3
        W: e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 Bc5 Nxf7 Bxf2+ Kxf2 Nxe4+ Kg1
        W: e4 e5 Nf3 Nc6 Bc4 Be7 d4 d6 Nc3 Nf6 h3 O-O O-O
        W: e4 e5 Nf3 Nc6 Bc4 d6 d4 Bg4 dxe5 Nxe5 Nxe5 Bxd1 Bxf7+ Ke7 Nc3
        W: e4 e5 Nf3 d6 d4 exd4 Nxd4 Nf6 Nc3 Be7 Bf4 O-O Qd2 a6 O-O-O
        W: e4 e5 Nf3 d6 d4 Nf6 Nc3 Nbd7 Bc4 Be7 O-O O-O Re1 c6 a4
        W: e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 Nc3 Nxc3 dxc3 Be7 Be3 O-O Qd2 Nd7 O-O-O
        W: e4 e5 Nf3 f5 Nxe5 Qf6 d4 d6 Nc4 fxe4 Nc3 Qg6 f3
        W: e4 c5 d4 cxd4 c3 dxc3 Nxc3 Nc6 Nf3 d6 Bc4 e6 O-O Nf6 Qe2 Be7 Rd1 e5 Be3
        W: e4 c5 d4 cxd4 c3 dxc3 Nxc3 e6 Nf3 Nc6 Bc4 a6 O-O Nge7 Bg5 f6 Be3
        W: e4 c5 d4 cxd4 c3 Nf6 e5 Nd5 Nf3 Nc6 Bc4 Nb6 Bb3 d5 exd6 Qxd6 O-O
        W: e4 c5 d4 cxd4 c3 d5 exd5 Qxd5 cxd4 Nc6 Nf3 Bg4 Nc3 Bxf3 gxf3 Qxd4 Qxd4 Nxd4 Nb5
        W: e4 c5 d4 cxd4 c3 e5 Nf3 Nc6 Bc4 h6 O-O Nf6 cxd4
        W: e4 c5 d4 cxd4 c3 d3 Bxd3 Nc6 Nf3 d6 c4
        W: e4 c5 d4 e6 Nf3 cxd4 Nxd4 a6 Nc3 Qc7 Bd3 Nc6 Be3 Nf6 O-O
        W: e4 c5 d4 d6 Nf3 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qb6 Qd2 Qxb2 Rb1 Qa3 f5
        W: e4 c5 d4 Nc6 Nf3 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5 Nd5
        W: e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7 h4 Bxg5 hxg5 Qxg5 Nh3 Qe7 Nf4
        W: e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7 h4 c5 Bxe7 Kxe7 f4
        W: e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 Qg4 Qc7 Qxg7 Rg8 Qxh7 cxd4 Ne2
        W: e4 e6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Nf3 Ngf6 Nxf6+ Nxf6 Bg5 h6 Bh4 c5 Bb5+
        W: e4 e6 d4 d5 Nc3 Nf6 Bg5 dxe4 Nxe4 Be7 Bxf6 Bxf6 Nf3 O-O Qd2
        W: e4 c6 d4 d5 f3 dxe4 fxe4 e5 Nf3 exd4 Bc4 Bb4+ c3 dxc3 Bxf7+ Kxf7 Qxd8
        W: e4 c6 d4 d5 f3 e6 Nc3 Bb4 Bf4 Ne7 Qd3
        W: e4 c6 d4 d5 f3 g6 Nc3 Bg7 Be3 Qb6 Qd2 Qxb2 Rb1 Qa3
        W: e4 c6 d4 d5 f3 Nf6 e5 Nfd7 f4 c5 Nf3 Nc6 c3
        W: e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Bd2 e6 Nd5 Qd8 Nxf6+ gxf6
        W: e4 d5 exd5 Qxd5 Nc3 Qd6 d4 Nf6 Nf3 a6 Ne5 Nbd7 Bf4
        W: e4 d5 exd5 Nf6 d4 Nxd5 c4 Nb6 Nf3 g6 Nc3 Bg7 c5 Nd5 Bc4
        W: e4 d6 d4 Nf6 Nc3 g6 f4 Bg7 Nf3 O-O e5 Nfd7 h4 c5 h5
        W: e4 d6 d4 Nf6 Nc3 e5 Nf3 Nbd7 Bc4 Be7 O-O O-O Re1
        W: e4 g6 d4 Bg7 Nc3 d6 f4 Nf6 Nf3 O-O e5 Nfd7 h4
        W: e4 Nf6 e5 Nd5 d4 d6 c4 Nb6 f4 dxe5 fxe5 Nc6 Be3 Bf5 Nc3 e6 Nf3 Be7 d5
        W: e4 Nf6 e5 Nd5 d4 d6 c4 Nb6 f4 g6 Nc3 Bg7 Be3 O-O Nf3
        W: e4 Nc6 d4 d5 e5 Bf5 c3 e6 Nd2 f6 f4
        W: e4 b6 d4 Bb7 Bd3 e6 Nf3 c5 c3 Nf6 Qe2 Be7 O-O
        W: e4 a6 d4 b5 Nf3 Bb7 Bd3 e6 O-O
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6 Bc4 Bd7 O-O-O Rc8 Bb3 Ne5 h4 h5
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6 O-O-O d5 exd5 Nxd5 Nxc6 bxc6
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be2 Bg7 O-O O-O Be3 Nc6 Nb3 Be6 f4 Na5
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 f4 Nbd7 Nf3 Qc7 Bd3 a6
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 f3 e5 Bb5+ Nbd7 Nf5 d5
        B: e4 c5 Nf3 d6 d4 cxd4 Qxd4 Nc6 Bb5 Bd7 Bxc6 Bxc6 Nc3 Nf6 Bg5 e6 O-O-O Be7
        B: e4 c5 Nf3 d6 Bb5+ Bd7 Bxd7+ Qxd7 c4 Nc6 Nc3 g6 d4 cxd4 Nxd4 Bg7
        B: e4 c5 Nf3 d6 c3 Nf6 Be2 Nc6 d4 cxd4 cxd4 Nxe4 d5 Qa5+ Nc3 Nxc3 bxc3 Ne5
        B: e4 c5 Nf3 d6 Nc3 Nf6 d4 cxd4 Nxd4 g6
        B: e4 c5 Nf3 d6 g3 Nc6 Bg2 g6 O-O Bg7 c3 Nf6
        B: e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6 f4 e6 Nf3 Nge7 O-O O-O Be3 Nd4
        B: e4 c5 Nc3 Nc6 f4 g6 Nf3 Bg7 Bb5 Nd4 O-O a6 Bd3 d6
        B: e4 c5 c3 Nf6 e5 Nd5 d4 cxd4 Nf3 Nc6 cxd4 d6 Bc4 Nb6 Bb5 dxe5 Nxe5 Bd7
        B: e4 c5 d4 cxd4 c3 dxc3 Nxc3 Nc6 Nf3 d6 Bc4 e6 O-O Nf6 Qe2 Be7 Rd1 e5
        B: e4 c5 f4 d5 exd5 Nf6 Bb5+ Bd7 Bxd7+ Qxd7 c4 e6 dxe6 Qxe6+
        B: e4 c5 b4 cxb4 a3 d5 exd5 Qxd5 Nf3 e5 axb4 Bxb4 c3 Be7
        B: e4 c5 d3 Nc6 g3 g6 Bg2 Bg7 f4 d6 Nf3 e6 O-O Nge7
        B: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Ne1 Nd7 Nd3 f5 Bd2 Nf6 f3 f4
        B: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 d5 a5 Bg5 h6 Bh4 Na6
        B: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 dxe5 dxe5 Qxd8 Rxd8 Bg5 Re8
        B: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 d5 Nh5 Qd2 f5 O-O-O Nd7
        B: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f4 O-O Nf3 c5 d5 e6 Be2 exd5 cxd5 Re8
        B: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 h3 O-O Bg5 c5 d5 e6 Bd3 exd5 cxd5 Re8
        B: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5 Nbd7 Qd2 e5 d5 Nc5 f3 a5
        B: d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d6 O-O Nc6 Nc3 a6 d5 Na5 Nd2 c5 Qc2 Rb8
        B: d4 Nf6 c4 g6 g3 Bg7 Bg2 O-O Nc3 d6 Nf3 Nc6 O-O a6
        B: d4 Nf6 c4 g6 Nc3 Bg7 Bg5 O-O e3 d6 Nf3 c5
        B: d4 Nf6 Nf3 g6 Bf4 Bg7 e3 O-O Be2 d6 h3 c5 c3 b6 O-O Bb7
        B: d4 Nf6 Nf3 g6 Bg5 Bg7 Nbd2 O-O e4 d6 c3 c5
        B: d4 Nf6 Nf3 g6 g3 Bg7 Bg2 O-O O-O d6 c4 Nc6
        B: d4 Nf6 Nf3 g6 c4 Bg7 Nc3 O-O e4 d6 Be2 e5 O-O Nc6 d5 Ne7
        B: d4 Nf6 Bg5 Ne4 Bf4 c5 f3 Qa5+ c3 Nf6 d5 Qb6 Bc1 e5
        B: d4 Nf6 Bf4 g6 e3 Bg7 Nf3 O-O Be2 d6 h3 c5 c3 b6
        B: d4 Nf6 Nc3 d5 Bg5 Nbd7 f3 c6 e4 dxe4 fxe4 e5
        B: d4 Nf6 e3 g6 c4 Bg7 Nc3 O-O Nf3 d6
        B: d4 Nf6 c3 g6 Bg5 Bg7 Nd2 O-O e4 d6
        B: c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7 d3 O-O a3 Be6 b4 f6
        B: c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5 Bg2 Nb6 Nf3 Nc6 O-O Be7 d3 O-O
        B: c4 e5 g3 Nf6 Bg2 c6 d4 e4 Nc3 d5 cxd5 cxd5 Bg5 Nc6
        B: c4 e5 Nf3 e4 Nd4 Nc6 Nc3 Nf6 Nxc6 dxc6 e3 Bf5
        B: c4 e5 Nc3 Nf6 e4 Bb4 Nge2 O-O d3 c6
        B: c4 e5 d3 Nc6 Nc3 f5 g3 Nf6 Bg2 Bb4
        B: Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 O-O Nc6 d5 Ne7
        B: Nf3 Nf6 g3 g6 Bg2 Bg7 O-O O-O d3 d6 e4 e5 Nc3 Nc6
        B: Nf3 Nf6 d4 g6 Bf4 Bg7 e3 O-O Be2 d6 h3 c5
        B: Nf3 Nf6 b3 g6 Bb2 Bg7 g3 O-O Bg2 d6 O-O e5 d3 Nc6
        B: g3 e5 Bg2 d5 d3 Nf6 Nf3 Nc6 O-O Be7 c4 O-O
        B: b3 e5 Bb2 Nc6 e3 d5 Bb5 Bd6 f4 Qh4+ g3 Qe7
        B: f4 e5 fxe5 d6 exd6 Bxd6 Nf3 g5 d4 g4 Ne5 Bxe5 dxe5 Qxd1+ Kxd1
        B: Nc3 d5 e4 d4 Nce2 e5 Ng3 Be6 c3 c5
        B: e3 e5 d4 exd4 exd4 d5 Nf3 Nf6 Bd3 Bd6 O-O O-O
        B: d3 e5 e4 Nf6 Nf3 Nc6 g3 d5 exd5 Nxd5 Bg2 Be7
        B: a3 e5 e4 Nf6 Nc3 Bc5 Nf3 d6
        B: Nh3 d5 g3 e5 f4 Bxh3 Bxh3 exf4 O-O
    """,
    "anubis": """
        W: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 O-O Bd3 Nbd7 Qc2 Re8 Nf3 Nf8 O-O c6 Rab1
        W: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 c6 e3 Be7 Bd3 Nbd7 Qc2 O-O Nf3 Re8 O-O Nf8
        W: d4 d5 c4 e6 Nc3 Be7 cxd5 exd5 Bf4 c6 e3 Bf5 g4 Be6 h3
        W: d4 d5 c4 e6 Nc3 c5 cxd5 exd5 Nf3 Nc6 g3 Nf6 Bg2 Be7 O-O O-O Bg5 cxd4 Nxd4 h6 Be3 Re8
        W: d4 d5 c4 e6 Nc3 c6 e3 Nf6 Nf3 Nbd7 Qc2 Bd6 Bd3 O-O O-O dxc4 Bxc4 b5 Bd3 Bb7
        W: d4 d5 c4 e6 Nc3 Nf6 cxd5 Nxd5 e4 Nxc3 bxc3 c5 Nf3 cxd4 cxd4 Bb4+ Bd2 Bxd2+ Qxd2 O-O
        W: d4 d5 c4 c6 cxd5 cxd5 Nc3 Nf6 Bf4 Nc6 e3 Bf5 Nf3 e6 Bb5 Nd7 Qa4 Qb6
        W: d4 d5 c4 c6 cxd5 cxd5 Nc3 Nf6 Bf4 a6 e3 Nc6 Nf3 Bg4 Be2 e6 O-O
        W: d4 d5 c4 c6 cxd5 cxd5 Nc3 Nf6 Bf4 e6 e3 Nc6 Nf3 Bd6 Bxd6 Qxd6 Bd3 O-O O-O
        W: d4 d5 c4 dxc4 e3 Nf6 Bxc4 e6 Nf3 c5 O-O a6 dxc5 Qxd1 Rxd1 Bxc5 Kf1 b5 Be2
        W: d4 d5 c4 dxc4 e3 e5 Bxc4 exd4 exd4 Nf6 Nf3 Be7 O-O O-O
        W: d4 d5 c4 dxc4 e3 Nf6 Bxc4 e6 Nf3 a6 O-O c5 Bb3 Nc6 Nc3 cxd4 exd4 Be7
        W: d4 d5 c4 Nc6 Nf3 Bg4 cxd5 Bxf3 gxf3 Qxd5 e3 e5 Nc3 Bb4 Bd2 Bxc3 Bxc3
        W: d4 d5 c4 e5 dxe5 d4 Nf3 Nc6 Nbd2 Bg4 a3 Qe7 h3 Bxf3 Nxf3 O-O-O Qd3
        W: d4 d5 c4 Bf5 cxd5 Bxb1 Rxb1 Qxd5 Nf3 e6 e3 Nf6 a3 Nbd7 Bd3
        W: d4 d5 c4 c5 cxd5 Nf6 e4 Nxe4 dxc5 Nxc5 Nc3 e6 Nf3 exd5 Qxd5 Qxd5 Nxd5
        W: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 dxe5 dxe5 Qxd8 Rxd8 Bg5 Re8 Nd5 Nxd5 cxd5
        W: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 c5 O-O cxd4 Nxd4 Nc6 Be3 Bd7
        W: d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 Nbd7 O-O e5 Re1 c6 Bf1
        W: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 Be3 Qa5 Qd2 O-O Rc1 cxd4 cxd4 Qxd2+ Kxd2 Rd8 Kc2
        W: d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 Be3 Bg4 Rc1 cxd4 cxd4 Qa5+ Qd2 Qxd2+ Kxd2
        W: d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nc6 a3 Bxc3 bxc3 dxc4 Bxc4 Qc7
        W: d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Bd3 Nc6 Nf3 Bxc3+ bxc3 d6 O-O e5
        W: d4 Nf6 c4 e6 Nc3 Bb4 e3 b6 Ne2 Bb7 a3 Be7 d5 O-O
        W: d4 Nf6 c4 e6 Nc3 Bb4 e3 d5 a3 Bxc3+ bxc3 c5 cxd5 exd5 Bd3 O-O Ne2
        W: d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4+ Bd2 Be7 Bg2 c6 Bc3 d5 Nbd2 Nbd7 O-O O-O Re1
        W: d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7 O-O O-O Nc3 Ne4 Qc2 Nxc3 Qxc3
        W: d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bf4 O-O e3 c5 dxc5 Bxc5 a3 Nc6 Qc2 Qa5 Rd1
        W: d4 Nf6 c4 e6 Nf3 c5 d5 exd5 cxd5 d6 Nc3 g6 e4 Bg7 Be2 O-O O-O Re8 Nd2 Na6 f3
        W: d4 Nf6 c4 e6 Nf3 Bb4+ Bd2 Qe7 g3 Nc6 Nc3 Bxc3 Bxc3 Ne4 Rc1 O-O Bg2
        W: d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6 Nf3 Bg7 Be2 O-O O-O Re8 Nd2 Na6 f3
        W: d4 Nf6 c4 c5 d5 b5 cxb5 a6 bxa6 Bxa6 Nc3 d6 e4 Bxf1 Kxf1 g6 g3 Bg7 Kg2 O-O Nf3 Nbd7 Re1
        W: d4 Nf6 c4 c5 d5 b5 cxb5 a6 bxa6 g6 Nc3 Bxa6 e4 Bxf1 Kxf1 d6 Nf3 Bg7 g3 O-O Kg2
        W: d4 Nf6 c4 c5 d5 e5 Nc3 d6 e4 Be7 Nf3 O-O Be2 Ne8 O-O g6
        W: d4 Nf6 c4 e5 dxe5 Ng4 Bf4 Nc6 Nf3 Bb4+ Nbd2 Qe7 e3 Ngxe5 Nxe5 Nxe5 Be2 O-O O-O
        W: d4 Nf6 c4 e5 dxe5 Ne4 Nf3 Nc6 a3 d6 Qc2 Bf5 Nc3 Nxc3 Qxc3
        W: d4 Nf6 c4 d6 Nc3 e5 Nf3 Nbd7 e4 c6 Be2 Be7 O-O O-O
        W: d4 Nf6 c4 Nc6 Nf3 e6 a3 d5 Nc3 Be7 Bf4 O-O e3
        W: d4 Nf6 c4 b6 Nc3 Bb7 Qc2 e6 a3 d5 cxd5 exd5
        W: d4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7 O-O O-O c4 d6 Nc3 Qe8 d5 Na6 Rb1
        W: d4 f5 g3 Nf6 Bg2 e6 Nf3 d5 O-O Bd6 c4 c6 b3 Qe7 Ne5 O-O Bb2
        W: d4 e6 c4 d5 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 O-O Bd3 Nbd7 Qc2 Re8
        W: d4 e6 c4 Bb4+ Bd2 Bxd2+ Qxd2 Nf6 Nc3 d5 e3 O-O Nf3
        W: d4 d6 c4 e5 Nf3 e4 Ng5 f5 Nc3 Nf6 g3
        W: d4 g6 c4 Bg7 Nc3 d6 e4 Nf6 Nf3 O-O Be2 e5 dxe5 dxe5 Qxd8 Rxd8 Bg5
        W: d4 c5 d5 e5 e4 d6 Nc3 Be7 Nf3 Bg4 Be2 Nd7 O-O
        W: d4 c5 d5 Nf6 Nc3 d6 e4 g6 Nf3 Bg7 Be2 O-O O-O
        W: d4 Nc6 Nf3 d5 c4 Bg4 cxd5 Bxf3 gxf3 Qxd5 e3
        W: d4 b6 c4 e6 a3 Bb7 Nc3 f5 d5 Nf6 g3
        B: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3 e6 Bd2 Ngf6 O-O-O Be7
        B: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 Nf3 Nd7 Bd3 Bxd3 Qxd3 e6 O-O Ngf6
        B: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Nc5 b6 Nb3 e6 Nf3 Nd7
        B: e4 c6 d4 d5 Nd2 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7
        B: e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 Be3 Nd7 O-O Ne7 c4 dxc4 Na3 Nd5
        B: e4 c6 d4 d5 e5 Bf5 Nc3 e6 g4 Bg6 Nge2 c5 h4 h5 Nf4 Bh7
        B: e4 c6 d4 d5 e5 Bf5 c3 e6 Be2 c5 Nf3 Nc6 O-O Nge7
        B: e4 c6 d4 d5 e5 c5 dxc5 Nc6 Bb5 e6 Be3 Nge7 c3 Bd7
        B: e4 c6 d4 d5 exd5 cxd5 Bd3 Nc6 c3 Nf6 Bf4 Bg4 Qb3 Qd7 Nd2 e6 Ngf3 Bxf3 Nxf3 Bd6
        B: e4 c6 d4 d5 exd5 cxd5 c4 Nf6 Nc3 e6 Nf3 Be7 cxd5 Nxd5 Bd3 Nc6 O-O O-O Re1 Nf6
        B: e4 c6 d4 d5 f3 dxe4 fxe4 e5 Nf3 exd4 Bc4 Nf6 O-O Be7 e5 Nd5
        B: e4 c6 d4 d5 Be3 dxe4 Nd2 Nf6 c3 Nbd7 Qc2 Nb6
        B: e4 c6 Nc3 d5 Nf3 Bg4 h3 Bxf3 Qxf3 e6 d4 Nf6 Bd3 dxe4 Nxe4 Qxd4 c3 Qd8
        B: e4 c6 Nc3 d5 d4 dxe4 Nxe4 Bf5 Ng3 Bg6
        B: e4 c6 Nf3 d5 Nc3 Bg4 h3 Bxf3 Qxf3 e6 d4 Nf6
        B: e4 c6 Nf3 d5 e5 Bg4 d4 e6 Be2 c5 c3 Nc6
        B: e4 c6 c4 d5 exd5 cxd5 cxd5 Nf6 Nc3 Nxd5 Nf3 Nc6 d4 Bg4 Qb3 Bxf3 gxf3 e6
        B: e4 c6 d3 d5 Nd2 e5 Ngf3 Bd6 g3 Nf6 Bg2 O-O O-O Re8 Re1 Nbd7
        B: e4 c6 Bc4 d5 exd5 cxd5 Bb5+ Nc6 d4 Nf6
        B: e4 c6 g3 d5 Bg2 dxe4 Bxe4 Nf6 Bg2 e5
        B: e4 c6 b3 d5 Bb2 dxe4 Nc3 Nf6 g4 h6
        B: e4 c6 f4 d5 e5 Bf5 Nf3 e6 d4 c5
        B: d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 Ne4 Bxe7 Qxe7 cxd5 Nxc3 bxc3 exd5 Qb3 Rd8 c4 dxc4 Bxc4 Nc6
        B: d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bxf6 Bxf6 Qc2 c5 dxc5 Qa5 cxd5 exd5
        B: d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 Ne4 Bxe7 Qxe7 Rc1 c6 Bd3 Nxc3 Rxc3 dxc4 Bxc4 Nd7
        B: d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 c6 e3 Be7 Bd3 Nbd7 Qc2 O-O Nf3 Re8 O-O Nf8
        B: d4 d5 c4 e6 Nc3 Nf6 Nf3 Be7 Bf4 O-O e3 c5 dxc5 Bxc5 Qc2 Nc6 a3 Qa5 Rd1 Rd8
        B: d4 d5 c4 e6 Nc3 Nf6 Nf3 Be7 Bg5 O-O e3 h6 Bh4 Ne4 Bxe7 Qxe7
        B: d4 d5 c4 e6 Nc3 Nf6 e3 Be7 Nf3 O-O Bd3 dxc4 Bxc4 c5 O-O a6 dxc5 Qxd1 Rxd1 Bxc5
        B: d4 d5 c4 e6 Nc3 Nf6 Qc2 c5 cxd5 exd5 dxc5 Be7 Bg5 O-O e3 Nc6
        B: d4 d5 c4 e6 Nf3 Nf6 g3 Be7 Bg2 O-O O-O dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7 Bd2 Be4 Qc1 Bb7
        B: d4 d5 c4 e6 Nf3 Nf6 Nc3 Be7 Bg5 O-O e3 h6 Bh4 Ne4 Bxe7 Qxe7
        B: d4 d5 c4 e6 Nf3 Nf6 e3 Be7 Bd3 O-O O-O c5 Nc3 cxd4 exd4 dxc4 Bxc4 Nc6
        B: d4 d5 c4 e6 g3 Nf6 Bg2 Be7 Nf3 O-O O-O dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7
        B: d4 d5 c4 e6 cxd5 exd5 Nc3 Nf6 Bg5 c6 e3 Be7
        B: d4 d5 c4 e6 e3 Nf6 Nc3 Be7 Nf3 O-O Bd3 dxc4 Bxc4 c5 O-O a6
        B: d4 d5 Nf3 Nf6 c4 e6 Nc3 Be7 Bg5 O-O e3 h6 Bh4 Ne4 Bxe7 Qxe7
        B: d4 d5 Nf3 Nf6 Bf4 e6 e3 c5 c3 Nc6 Nbd2 Bd6 Bg3 O-O Bd3 b6 Ne5 Bb7
        B: d4 d5 Nf3 Nf6 g3 e6 Bg2 Be7 O-O O-O c4 dxc4 Qc2 a6 Qxc4 b5
        B: d4 d5 Nf3 Nf6 e3 e6 Bd3 c5 b3 Nc6 O-O Bd6 Bb2 O-O
        B: d4 d5 Nf3 Nf6 Bg5 e6 e3 Be7 Nbd2 O-O Bd3 c5 c3 Nbd7
        B: d4 d5 Bf4 Nf6 e3 e6 Nf3 c5 c3 Nc6 Nbd2 Bd6 Bg3 O-O Bd3 b6 Ne5 Bb7
        B: d4 d5 Bf4 Nf6 Nc3 e6 e3 Bb4 Bd3 c5
        B: d4 d5 e3 Nf6 Bd3 c5 c3 Nc6 f4 Bg4 Nf3 e6 O-O Bd6
        B: d4 d5 Nc3 Nf6 Bg5 e6 e4 Be7 e5 Nfd7 Bxe7 Qxe7 f4 O-O Nf3 c5
        B: d4 d5 Bg5 h6 Bh4 c6 e3 Qb6 Qc1 Nf6
        B: d4 d5 e4 dxe4 Nc3 Nf6 f3 exf3 Nxf3 Bg4 h3 Bxf3 Qxf3 c6
        B: d4 d5 c3 Nf6 Bf4 c5 e3 Nc6 Nd2 e6
        B: d4 d5 g3 Nf6 Bg2 e6 Nf3 Be7 O-O O-O c4 dxc4
        B: c4 e6 Nc3 d5 d4 Nf6 cxd5 exd5 Bg5 Be7 e3 c6 Bd3 O-O Qc2 Nbd7 Nf3 Re8
        B: c4 e6 Nc3 d5 e4 dxe4 Nxe4 Nf6 Nxf6+ Qxf6 d4 Nc6 Nf3 Bb4+ Bd2 Bxd2+ Qxd2 O-O
        B: c4 e6 Nc3 d5 Nf3 Nf6 d4 Be7 Bg5 O-O e3 h6 Bh4 Ne4 Bxe7 Qxe7
        B: c4 e6 Nc3 d5 cxd5 exd5 d4 Nf6 Bg5 c6 e3 Be7
        B: c4 e6 Nc3 d5 g3 Nf6 Bg2 Be7 Nf3 O-O O-O c6
        B: c4 e6 Nf3 d5 g3 Nf6 Bg2 Be7 O-O O-O b3 c5 Bb2 Nc6 e3 b6 Nc3 Bb7
        B: c4 e6 Nf3 d5 d4 Nf6 Nc3 Be7 Bg5 O-O e3 h6 Bh4 Ne4
        B: c4 e6 Nf3 d5 b3 Nf6 Bb2 Be7 e3 O-O Be2 c5 O-O Nc6 cxd5 exd5 d4
        B: c4 e6 g3 d5 Bg2 Nf6 Nf3 Be7 O-O O-O d4 dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7
        B: c4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 Ne4
        B: c4 e6 e4 d5 exd5 exd5 d4 Nf6 Nc3 Be7 Nf3 O-O
        B: Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 O-O e3 h6 Bh4 Ne4 Bxe7 Qxe7
        B: Nf3 d5 g3 Nf6 Bg2 c6 O-O Bg4 d3 Nbd7 Nbd2 e5 e4 dxe4 dxe4 Be7 h3 Bh5
        B: Nf3 d5 c4 e6 g3 Nf6 Bg2 Be7 O-O O-O d4 dxc4 Qc2 a6 Qxc4 b5
        B: Nf3 d5 c4 e6 d4 Nf6 Nc3 Be7 Bg5 O-O e3 h6 Bh4 Ne4
        B: Nf3 d5 c4 e6 b3 Nf6 Bb2 Be7 e3 O-O Be2 c5 O-O Nc6 cxd5 exd5 d4
        B: Nf3 d5 b3 Nf6 Bb2 e6 e3 Be7 c4 O-O Nc3 c5 cxd5 exd5 d4 Nc6
        B: Nf3 d5 e3 Nf6 b3 e6 Bb2 Be7 c4 O-O
        B: Nf3 d5 d3 Nf6 g3 c6 Bg2 Bg4 O-O e6 Nbd2 Be7
        B: g3 d5 Bg2 Nf6 Nf3 c6 O-O Bg4 d3 Nbd7 Nbd2 e5
        B: b3 d5 Bb2 Nf6 e3 e6 Nf3 Be7 c4 O-O
        B: f4 d5 Nf3 g6 e3 Bg7 d4 Nf6 Bd3 O-O O-O c5 c3 b6
        B: e3 e5 d4 exd4 exd4 d5 Nf3 Nf6 Bd3 Bd6 O-O O-O
        B: Nc3 d5 e4 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7
        B: Nc3 d5 d4 Nf6 Bg5 e6 e4 Be7
        B: d3 d5 Nf3 Nf6 g3 c6 Bg2 Bg4 O-O e6 Nbd2 Be7
        B: c3 d5 d4 Nf6 Bf4 e6 e3 c5 Nd2 Nc6
        B: a3 d5 d4 Nf6 Nf3 e6 c4 Be7
        B: b4 e5 Bb2 Bxb4 Bxe5 Nf6 c4 O-O Nf3 d5
        B: g4 d5 Bg2 Bxg4 c4 c6 cxd5 cxd5 Qb3 Nc6
    """,
    "hades": """
        W: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O c3 d5 exd5 Nxd5 Nxe5 Nxe5 Rxe5 c6 d4 Bd6 Re1 Qh4 g3 Qh3 Be3 Bg4 Qd3 Rae8 Nd2
        W: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5 Bc2 c5 d4 Qc7 Nbd2 cxd4 cxd4 Nc6 Nb3 a5 Be3 a4 Nbd2
        W: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Nbd2 Bf8 a4 h6 Bc2 exd4 cxd4 Nb4 Bb1 c5 d5 Nd7 Ra3
        W: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7 Nbd2 Bb7 Bc2 Re8 Nf1 Bf8 Ng3 g6 a4 c5 d5 c4 Bg5
        W: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5 dxe5 Be6 Nbd2 Nc5 c3 d4 Ng5
        W: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O b5 Bb3 Bc5 a4 Rb8 c3 d6 d4 Bb6 axb5 axb5 Na3 O-O Nxb5 Bg4
        W: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 d6 c3 Bd7 d4 Nge7 Bb3 h6 Nbd2 Ng6 Nc4 Be7 Ne3
        W: e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O f6 d4 exd4 Nxd4 c5 Nb3 Qxd1 Rxd1 Bg4 f3 Be6 Nc3
        W: e4 e5 Nf3 Nc6 Bb5 Nf6 d3 Bc5 c3 O-O O-O d6 h3 a6 Ba4 Ba7 Re1 Ne7 Nbd2 Ng6 Nf1 c6 Bc2 h6 Ng3
        W: e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 Re1 Nd6 Nxe5 Be7 Bf1 Nxe5 Rxe5 O-O d4 Bf6 Re1 Re8 c3 Rxe1 Qxe1 Ne8
        W: e4 e5 Nf3 Nc6 Bb5 Bc5 c3 Nf6 O-O O-O d4 Bb6 Bg5 h6 Bh4 d6 a4 a5 Na3
        W: e4 e5 Nf3 Nc6 Bb5 f5 Nc3 fxe4 Nxe4 d5 Nxe5 dxe4 Nxc6 Qg5 Qe2 Nf6 f4 Qxf4 Ne5+ c6 d4 Qh4+ g3 Qh3 Bc4
        W: e4 e5 Nf3 Nc6 Bb5 g6 c3 a6 Ba4 d6 d4 Bd7 O-O Bg7 dxe5 dxe5 Be3 Nf6 Nbd2
        W: e4 e5 Nf3 Nc6 Bb5 Nd4 Nxd4 exd4 O-O Bc5 d3 c6 Ba4 Ne7 f4 d5 Bb3 dxe4 dxe4 O-O Kh1
        W: e4 e5 Nf3 Nc6 Bb5 Nge7 c3 d6 d4 Bd7 O-O Ng6 Re1 Be7 Nbd2 O-O Nf1
        W: e4 e5 Nf3 Nf6 d4 Nxe4 Bd3 d5 Nxe5 Nd7 Nxd7 Bxd7 O-O Bd6 c4 c6 Nc3 Nxc3 bxc3 dxc4 Bxc4 O-O Qh5
        W: e4 e5 Nf3 Nf6 d4 exd4 e5 Ne4 Qxd4 d5 exd6 Nxd6 Nc3 Nc6 Qf4 g6 Be3 Bg7 O-O-O
        W: e4 e5 Nf3 d6 d4 Nf6 Nc3 Nbd7 Bc4 Be7 O-O O-O a4 c6 Re1 b6 Ba2 a6 Ng5
        W: e4 e5 Nf3 d6 d4 exd4 Nxd4 Nf6 Nc3 Be7 Bf4 O-O Qd2 a6 O-O-O d5 e5 Ne4 Nxe4 dxe4
        W: e4 e5 Nf3 f5 Nxe5 Qf6 d4 d6 Nc4 fxe4 Nc3 Qg6 f3 exf3 Qxf3 Nf6 Bd3 Qh5
        W: e4 e5 Nf3 Qe7 Nc3 c6 d4 d6 Be2 Nf6 O-O
        W: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 Qd2 O-O O-O-O Nbd7 g4 b5 g5 b4 Ne2 Ne8 f4 a5 f5 a4 Nbd4 exd4 Nxd4 b3 Kb1 bxc2+ Nxc2 Bb3
        W: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e6 f3 b5 g4 h6 Qd2 Nbd7 O-O-O Bb7 h4 b4 Na4 d5 Bh3
        W: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 Ng4 Bg5 h6 Bh4 g5 Bg3 Bg7 Be2 h5 Bxg4 Bxg4 f3 Bd7
        W: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6 Bc4 Bd7 O-O-O Rc8 Bb3 Ne5 h4 h5 Kb1 Nc4 Bxc4 Rxc4 g4
        W: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6 Qd2 Be7 O-O-O O-O f4 Nxd4 Qxd4 Qa5 Bc4 Bd7 e5 dxe5 fxe5 Bc6 Bd2
        W: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6 g4 h6 h4 Nc6 Rg1 h5 gxh5 Nxh5 Bg5 Nf6
        W: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Bb5+ Bd7 Bxd7+ Qxd7 Nf5 h5
        W: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5 Nd5 Be7 Bxf6 Bxf6 c3 O-O Nc2 Bg5 a4 bxa4 Rxa4 a5 Bc4 Rb8 b3
        W: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Bg5 e6 Qd2 a6 O-O-O Bd7 f4 b5 Bxf6 gxf6 Kb1 Qb6 Nxc6 Bxc6
        W: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 c4 Bg7 Be3 Nf6 Nc3 O-O Be2 d6 O-O Bd7 Qd2 Nxd4 Bxd4 Bc6 f3 a5 b3 Nd7
        W: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 e5 Nb5 d6 c4 Be7 N1c3 a6 Na3 f5 exf5 Bxf5 Bd3
        W: e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Qb6 Nb3 Nf6 Nc3 e6 Bd3 d6 O-O a6 Be3 Qc7 f4
        W: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be3 a6 Qd2 Nf6 O-O-O Bb4 f3 Ne5 Nb3 b5 Kb1 Be7 Qf2 d6 Bb6 Qb8 g4
        W: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3 Nf6 O-O Qc7 Qe2 d6 c4 g6 Nc3 Bg7 Rd1 O-O Nf3 Nc6 h3
        W: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Ndb5 Bb4 a3 Bxc3+ Nxc3 d5 exd5 exd5 Bd3 O-O O-O d4 Ne2
        W: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Bc5 Nb3 Bb6 Nc3 Ne7 Bf4 d6 Qd2
        W: e4 c5 Nf3 g6 d4 cxd4 Nxd4 Nc6 c4 Bg7 Be3 Nf6 Nc3 O-O Be2 d6 O-O Bd7 Qd2
        W: e4 c5 Nf3 a6 c3 d5 exd5 Qxd5 d4 Nf6 Be2 e6 O-O Nc6 Be3 cxd4 cxd4 Be7 Nc3 Qd6
        W: e4 c5 Nf3 b6 d4 cxd4 Nxd4 Bb7 Nc3 a6 Bd3 e6 O-O Nf6 Qe2
        W: e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 Qg4 Qc7 Qxg7 Rg8 Qxh7 cxd4 Ne2 Nbc6 f4 Bd7 Qd3 dxc3 Qxc3 O-O-O
        W: e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 Qg4 O-O Bd3 Nbc6 Qh5 Ng6 Nf3 Qc7 Be3 c4 Bxg6 fxg6 Qg4
        W: e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Ba5 b4 cxb4 Nb5 bxa3 c3 Bc7 Bxa3 Ne7 Bd3
        W: e4 e6 d4 d5 Nc3 Bb4 e5 b6 Qg4 Bf8 Nf3 Qd7 h4 Ba6 Bxa6 Nxa6 Bg5
        W: e4 e6 d4 d5 Nc3 Bb4 e5 Ne7 a3 Bxc3+ bxc3 b6 Qg4 Ng6 h4 h5 Qd1 Ba6 Bxa6 Nxa6 Ne2
        W: e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7 f4 c5 Nf3 Nc6 Be3 cxd4 Nxd4 Bc5 Qd2 O-O O-O-O a6 h4 Nxd4 Bxd4 b5 Rh3
        W: e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7 Bxe7 Qxe7 f4 O-O Nf3 c5 Qd2 Nc6 dxc5 Nxc5 O-O-O
        W: e4 e6 d4 d5 Nc3 Nf6 Bg5 Bb4 e5 h6 Bd2 Bxc3 bxc3 Ne4 Qg4 g6 Bd3 Nxd2 Kxd2 c5
        W: e4 e6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Nf3 Ngf6 Nxf6+ Nxf6 c3 c5 Ne5 a6 Be3 Qc7 Qa4+ Bd7 Nxd7 Nxd7 dxc5
        W: e4 e6 d4 d5 Nc3 dxe4 Nxe4 Bd7 Nf3 Bc6 Bd3 Nd7 O-O Ngf6 Ng3 Bxf3 Qxf3 c6 c3
        W: e4 e6 d4 d5 Nc3 c5 exd5 exd5 Nf3 Nc6 Bb5 Bd6 dxc5 Bxc5 O-O Nge7 Nxd5 Nxd5 Qxd5
        W: e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 Be3 Nd7 O-O Ne7 c4 dxc4 Na3 Nd5 Nxc4 Be7 Qb3
        W: e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 Nd7 O-O h6 Nbd2 Ne7 Nb3 g5 Ne1 Bg7 f4
        W: e4 c6 d4 d5 e5 c5 dxc5 Nc6 Bb5 e6 Be3 Nge7 c3 Bd7 Nf3 Nf5 Bd4 Nfxd4 cxd4 b6
        W: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3 e6 Bf4 Ngf6 O-O-O Be7 Kb1 O-O Ne5 c5
        W: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Ng5 Ngf6 Bd3 e6 N1f3 Bd6 Qe2 h6 Ne4 Nxe4 Qxe4 Nf6 Qh4
        W: e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nf6 Nxf6+ exf6 c3 Bd6 Bd3 O-O Qc2 Re8+ Ne2 g6 h4
        W: e4 c6 d4 d5 Nc3 g6 Nf3 Bg7 h3 Nf6 Bd3 O-O O-O dxe4 Nxe4 Nxe4 Bxe4
        W: e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Bd2 e6 Qe2 Bb4 Ne5 Nbd7 O-O-O Nxe5 dxe5 Nd5
        W: e4 d5 exd5 Qxd5 Nc3 Qd6 d4 Nf6 Nf3 a6 g3 Bg4 Bg2 Nc6 Be3 O-O-O O-O
        W: e4 d5 exd5 Qxd5 Nc3 Qd8 d4 Nf6 Nf3 Bg4 h3 Bxf3 Qxf3 c6 Be3 e6 Bd3
        W: e4 d5 exd5 Nf6 d4 Nxd5 Nf3 g6 c4 Nb6 Nc3 Bg7 h3 O-O Be3 Nc6 Qd2 e5 d5 Ne7 g4
        W: e4 d5 exd5 Nf6 d4 Bg4 f3 Bf5 c4 e6 dxe6 Nc6 Be3 Qe7
        W: e4 d6 d4 Nf6 Nc3 g6 Be3 Bg7 Qd2 c6 f3 b5 g4 h6 h4 Nbd7 Nh3 Nb6 b3 a5 Bd3
        W: e4 d6 d4 Nf6 Nc3 g6 Be3 c6 Qd2 b5 f3 Nbd7 Nh3 e5 Nf2 Bg7 g4 h6 h4
        W: e4 d6 d4 Nf6 Nc3 e5 Nf3 Nbd7 Bc4 Be7 O-O O-O Re1 c6 a4 b6 Ba2 a6 Ng5
        W: e4 g6 d4 Bg7 Nc3 d6 Be3 a6 Qd2 Nd7 f4 b5 Nf3 Bb7 Bd3 c5 dxc5 dxc5 e5
        W: e4 g6 d4 Bg7 Nc3 c6 Be3 d5 Qd2 dxe4 Nxe4 Nd7 h4 Ngf6 Nxf6+ Nxf6
        W: e4 Nf6 e5 Nd5 d4 d6 Nf3 g6 Bc4 Nb6 Bb3 Bg7 Qe2 Nc6 O-O O-O h3 a5 a4 dxe5 dxe5 Nd4 Nxd4 Qxd4 Re1
        W: e4 Nf6 e5 Nd5 d4 d6 Nf3 Bg4 Be2 e6 O-O Be7 h3 Bh5 c4 Nb6 exd6 cxd6 Nc3
        W: e4 Nf6 e5 Nd5 d4 d6 Nf3 dxe5 Nxe5 g6 Bc4 c6 O-O Bg7 Re1 O-O Bb3
        W: e4 Nc6 Nf3 d6 d4 Nf6 Nc3 Bg4 Be3 e6 h3 Bh5 Bd3 d5 e5 Ne4
        W: e4 b6 d4 Bb7 Bd3 e6 Nf3 c5 c3 Nf6 Qe2 Be7 O-O Nc6 a3 d5 e5 Nd7
        W: e4 a6 d4 b5 Bd3 Bb7 Nf3 e6 O-O Nf6 Qe2 c5 dxc5 Bxc5 a4
        W: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 b5 Bb3 Na5 O-O d6 d4 exd4 Nxd4
        W: e4 e5 Nf3 Nc6 Bb5 Bb4 c3 Ba5 O-O Nge7 d4 exd4 cxd4 d5
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 Qd2 O-O O-O-O Nbd7 g4 b5 g5 b4 Ne2 Ne8 f4 a5 f5 a4 Nbd4 exd4 Nxd4 b3 Kb1 bxc2+ Nxc2 Bb3
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nf3 Be7 Bc4 O-O O-O Be6 Bb3 Nc6 Bg5 Nd7
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nde2 Be7 g3 Nbd7 a4 b6 Bg2 Bb7 h3 O-O O-O
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qb6 Qd2 Qxb2 Rb1 Qa3 f5 Nc6 fxe6 fxe6 Nxc6 bxc6 e5 dxe5 Bxf6 gxf6 Ne4 Qxa2
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qb6 Nb3 Be7 Qf3 Nbd7 O-O-O Qc7
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 Qf3 Qc7 O-O-O Nbd7 Qg3 b5 Bd3 Be7
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7 O-O O-O Be3 Be6 Qd2 Nbd7 a4 Rc8 a5 Qc7
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 b5 O-O Be7 Qf3 Qc7 Qg3 Nc6 Nxc6 Qxc6 Re1 Bb7
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 h3 e5 Nde2 h5 g3 Be6 Bg2 b5 a4 b4 Nd5 Nxd5 exd5 Bf5
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f3 e5 Nb3 Be6 Be3 Nbd7 Qd2 b5 a4 b4 Nd5 Bxd5 exd5 Nb6
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 e5 Nf3 Nbd7 a4 Be7 Bd3 O-O O-O exf4 Bxf4 Nc5
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 g3 e5 Nde2 Be7 Bg2 O-O O-O b5 h3 Bb7
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 a4 Nc6 Be2 e5 Nb3 Be7 O-O O-O
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Qf3 e5 Nb3 Be6 Be3 Nbd7
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 f3 e5 Bb5+ Nbd7 Nf5 d5 exd5 a6 Bxd7+ Bxd7 O-O Bc5+ Kh1 O-O
        B: e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Bd3 Nc6 Nxc6 bxc6 O-O e5 c4 Be7 Nc3 O-O
        B: e4 c5 Nf3 d6 d4 cxd4 Qxd4 Nc6 Bb5 Bd7 Bxc6 Bxc6 Nc3 Nf6 Bg5 e6 O-O-O Be7 Rhe1 O-O Kb1 Qa5
        B: e4 c5 Nf3 d6 d4 Nf6 dxc5 Qa5+ c3 Qxc5 Nbd2 g6 Bd3 Bg7 O-O O-O
        B: e4 c5 Nf3 d6 d4 Nf6 Nc3 cxd4 Nxd4 a6
        B: e4 c5 Nf3 d6 Bb5+ Nd7 d4 Nf6 Nc3 cxd4 Qxd4 e5 Qd3 h6 Be3 Be7 O-O-O O-O g4 a6
        B: e4 c5 Nf3 d6 Bb5+ Nd7 O-O Nf6 Re1 a6 Bf1 b6 d4 cxd4 Nxd4 Bb7 Nc3 e6
        B: e4 c5 Nf3 d6 c3 Nf6 Be2 Nc6 d4 cxd4 cxd4 Nxe4 d5 Qa5+ Nc3 Nxc3 bxc3 Ne5 Nxe5 Qxc3+ Bd2 Qxe5
        B: e4 c5 Nf3 d6 c3 Nf6 h3 Nc6 d4 cxd4 cxd4 Nxe4 d5 Qa5+ Nbd2 Nxd2 Bxd2 Qb6
        B: e4 c5 Nf3 d6 Nc3 Nf6 e5 dxe5 Nxe5 Nbd7 Nxd7 Bxd7 Bc4 e6 O-O Be7
        B: e4 c5 Nf3 d6 Nc3 Nf6 d4 cxd4 Nxd4 a6
        B: e4 c5 Nf3 d6 g3 Nc6 Bg2 g6 O-O Bg7 c3 Nf6 d4 cxd4 cxd4 O-O
        B: e4 c5 Nf3 d6 b4 cxb4 d4 Nf6 Bd3 g6 O-O Bg7 a3
        B: e4 c5 Nc3 d6 Nge2 Nf6 g3 Nc6 Bg2 g6 d4 cxd4 Nxd4 Nxd4 Qxd4 Bg7 O-O O-O
        B: e4 c5 Nc3 d6 f4 Nc6 Nf3 g6 Bb5 Bd7 O-O Bg7 d3 Nf6 Qe1 O-O
        B: e4 c5 Nc3 d6 g3 Nc6 Bg2 g6 d3 Bg7 f4 e6 Nf3 Nge7 O-O O-O
        B: e4 c5 Nc3 d6 Nf3 Nf6 d4 cxd4 Nxd4 a6
        B: e4 c5 c3 d5 exd5 Qxd5 d4 Nf6 Nf3 Bg4 Be2 e6 O-O Nc6 Be3 cxd4 cxd4 Be7 Nc3 Qa5 h3 Bh5
        B: e4 c5 c3 d5 exd5 Qxd5 d4 Nc6 Nf3 Bg4 dxc5 Qxd1+ Kxd1 e5 b4 O-O-O+ Nbd2
        B: e4 c5 d4 cxd4 c3 dxc3 Nxc3 Nc6 Nf3 d6 Bc4 e6 O-O Nf6 Qe2 Be7 Rd1 e5 Be3 O-O
        B: e4 c5 d4 cxd4 Nf3 d6 Nxd4 Nf6 Nc3 a6
        B: e4 c5 d4 cxd4 Qxd4 Nc6 Qd1 d6 Nf3 Nf6 Nc3 g6
        B: e4 c5 f4 d6 Nf3 Nc6 Bb5 Bd7 O-O g6 d3 Bg7 c3 a6 Bxc6 Bxc6
        B: e4 c5 f4 Nc6 Nf3 g6 Bb5 Bg7 Bxc6 dxc6 d3 b6 O-O Nh6
        B: e4 c5 b4 cxb4 a3 d5 exd5 Qxd5 Nf3 e5 axb4 Bxb4 c3 Be7 d4 exd4
        B: e4 c5 d3 Nc6 g3 g6 Bg2 Bg7 f4 d6 Nf3 e6 O-O Nge7 c3 O-O
        B: e4 c5 Bc4 e6 Nc3 a6 a4 Nc6 d3 g6 f4 Bg7 Nf3 Nge7
        B: e4 c5 Ne2 d6 g3 Nc6 Bg2 g6 O-O Bg7 c3 e5
        B: e4 c5 Qh5 d6 Bc4 e6 Nc3 Nf6 Qe2 Nc6
        B: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3+ Qxc3 b6 Bg5 Bb7 f3 h6 Bh4 d5 e3 Nbd7 cxd5 Nxd5 Bxd8 Nxc3 Bh4 Nd5 Bf2 c5
        B: d4 Nf6 c4 e6 Nc3 Bb4 Qc2 c5 dxc5 O-O a3 Bxc5 Nf3 Nc6 Bg5 Nd4 Nxd4 Bxd4 e3 Qa5 exd4 Qxg5
        B: d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Bd3 Nc6 Nf3 Bxc3+ bxc3 d6 e4 e5 d5 Ne7 h3 Ng6 Nh2 O-O f4 exf4 Bxf4 Nxf4
        B: d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nc6 a3 Bxc3 bxc3 dxc4 Bxc4 Qc7 Bd3 e5 Qc2 Re8
        B: d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Ne2 d5 a3 Be7 cxd5 exd5 g3 c5
        B: d4 Nf6 c4 e6 Nc3 Bb4 f3 d5 a3 Bxc3+ bxc3 c5 cxd5 Nxd5 dxc5 f5 e4 fxe4 Qc2 Qh4+ g3 Qf6
        B: d4 Nf6 c4 e6 Nc3 Bb4 Bg5 h6 Bh4 c5 d5 d6 e3 Bxc3+ bxc3 e5 Qc2 Nbd7 Bd3 Qe7 Ne2 g5 Bg3 Nh5
        B: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 c5 g3 cxd4 Nxd4 O-O Bg2 d5 cxd5 Nxd5 Qb3 Qa5 Bd2 Nc6
        B: d4 Nf6 c4 e6 Nc3 Bb4 Nf3 b6 Bg5 Bb7 e3 h6 Bh4 g5 Bg3 Ne4 Qc2 Bxc3+ bxc3 d6 Bd3 f5
        B: d4 Nf6 c4 e6 Nc3 Bb4 Qb3 c5 dxc5 Nc6 Nf3 Ne4 Bd2 Nxd2 Nxd2 O-O e3 Bxc5
        B: d4 Nf6 c4 e6 Nc3 Bb4 a3 Bxc3+ bxc3 c5 f3 d5 e3 O-O cxd5 Nxd5 dxc5 f5
        B: d4 Nf6 c4 e6 Nc3 Bb4 g3 c5 d5 Ne4 Qd3 Qa5 Qxe4 Bxc3+ Bd2
        B: d4 Nf6 c4 e6 Nf3 c5 d5 exd5 cxd5 d6 Nc3 g6 e4 Bg7 Be2 O-O O-O Re8 Nd2 Na6 f3 Nc7 a4 b6 Nc4 Ba6
        B: d4 Nf6 c4 e6 Nf3 c5 d5 exd5 cxd5 d6 Nc3 g6 Bf4 Bg7 Qa4+ Bd7 Qb3 b5
        B: d4 Nf6 c4 e6 Nf3 c5 d5 exd5 cxd5 d6 Nc3 g6 g3 Bg7 Bg2 O-O O-O Re8 Nd2 a6 a4 Nbd7 h3 Rb8
        B: d4 Nf6 c4 e6 Nf3 c5 e3 d5 Nc3 Nc6 a3 a6 dxc5 Bxc5 b4 Ba7 Bb2 O-O
        B: d4 Nf6 c4 e6 Nf3 c5 g3 cxd4 Nxd4 d5 Bg2 e5 Nf3 d4 O-O Nc6 e3 Bc5
        B: d4 Nf6 c4 e6 Nf3 c5 dxc5 Bxc5 Nc3 O-O e3 d5 a3 Nc6 b4 Bd6
        B: d4 Nf6 c4 e6 Nf3 c5 Nc3 cxd4 Nxd4 Bb4 g3 O-O Bg2 d5 cxd5 Nxd5 Qb3 Qa5
        B: d4 Nf6 c4 e6 g3 c5 d5 exd5 cxd5 b5 Bg2 d6 Nf3 g6 b4 Bg7 O-O O-O a4 bxa4
        B: d4 Nf6 c4 e6 g3 c5 Nf3 cxd4 Nxd4 d5 Bg2 e5 Nf3 d4 O-O Nc6
        B: d4 Nf6 c4 e6 e3 c5 Nf3 d5 Nc3 Nc6 a3 a6 dxc5 Bxc5 b4 Ba7
        B: d4 Nf6 c4 e6 a3 c5 d5 b5 dxe6 fxe6 cxb5 d5 e3 Bd6
        B: d4 Nf6 Nf3 e6 c4 c5 d5 exd5 cxd5 d6 Nc3 g6
        B: d4 Nf6 Nf3 e6 Bg5 c5 e3 Qb6 Nbd2 Qxb2 Bd3 Qc3 O-O d5 e4 cxd4 Re1
        B: d4 Nf6 Nf3 e6 Bg5 c5 e4 cxd4 e5 h6 Bh4 g5 exf6 gxh4 Nxd4 Qxf6
        B: d4 Nf6 Nf3 e6 Bf4 c5 e3 Qb6 Nc3 Qxb2 Nb5 Na6 a3 d5 Rb1 Qxa3 Ra1 Qb4+ Nd2 Ne4
        B: d4 Nf6 Nf3 e6 Bf4 c5 c3 cxd4 cxd4 Qb6 Qb3 d5 e3 Nc6 Nc3 Bd7
        B: d4 Nf6 Nf3 e6 g3 c5 Bg2 cxd4 Nxd4 d5 O-O e5 Nf3 Nc6 c4 d4
        B: d4 Nf6 Nf3 e6 e3 c5 Bd3 d5 b3 Nc6 O-O Bd6 Bb2 O-O
        B: d4 Nf6 Nf3 e6 c3 c5 Bg5 h6 Bh4 Qb6 Qb3 Qxb3 axb3
        B: d4 Nf6 Bg5 e6 e4 h6 Bxf6 Qxf6 Nc3 d6 Qd2 g5 O-O-O Nc6 f4 Bd7 Nf3 O-O-O
        B: d4 Nf6 Bg5 e6 Nd2 c5 e3 cxd4 exd4 Qb6 Ngf3 Qxb2 Bd3 d5
        B: d4 Nf6 Bg5 c5 d5 Qb6 Nc3 Qxb2 Bd2 Qb6 e4 d6 f4 g6 Nf3 Bg7 Bd3 O-O
        B: d4 Nf6 Bf4 c5 e3 Qb6 Nc3 Qxb2 Nb5 Na6 a3 d5 Rb1 Qxa3 Ra1 Qb4+ c3 Qa5
        B: d4 Nf6 Bf4 c5 d5 Qb6 Nc3 Qxb2 Bd2 Qb6 e4 d6 f4 g6
        B: d4 Nf6 Nc3 d5 Bg5 c5 Bxf6 gxf6 e4 dxe4 d5 f5 Bb5+ Bd7 Bxd7+ Qxd7 Qh5 Qd6
        B: d4 Nf6 Nc3 d5 Bf4 a6 e3 e6 Nf3 c5
        B: d4 Nf6 e3 e6 c4 c5 d5 exd5 cxd5 d6 Nc3 g6
        B: d4 Nf6 g3 e6 Bg2 c5 d5 exd5 c4 d6 cxd5 g6
        B: d4 Nf6 c3 e6 Bg5 c5 e3 Qb6 Qb3 Qxb3 axb3 Nc6
        B: d4 Nf6 Nd2 e6 e4 c5 e5 Nd5 Ngf3 cxd4 Nxd4 Nc6
        B: c4 Nf6 Nc3 e6 e4 c5 e5 Ng8 Nf3 Nc6 d4 cxd4 Nxd4 Nxe5 Ndb5 a6 Nd6+ Bxd6 Qxd6 f6 Be3 Ne7 Bb6 Nf5 Qc5 Qe7
        B: c4 Nf6 Nc3 e6 Nf3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 e3 Bb7 Be2 d6 O-O Nbd7 b4 Ne4 Qc2 f5
        B: c4 Nf6 Nc3 e6 Nf3 Bb4 g3 c5 Bg2 Nc6 O-O O-O d3 d5 a3 Bxc3 bxc3 dxc4 dxc4 Qa5
        B: c4 Nf6 Nc3 e6 d4 Bb4 Qc2 O-O a3 Bxc3+ Qxc3 b6 Bg5 Bb7
        B: c4 Nf6 Nc3 e6 g3 c5 Bg2 Nc6 Nf3 d5 cxd5 exd5 d4 Be7 O-O O-O
        B: c4 Nf6 Nc3 e6 e3 c5 Nf3 d5 d4 Nc6 a3 a6
        B: c4 Nf6 g3 e6 Bg2 d5 Nf3 dxc4 Qa4+ Nbd7 Qxc4 a6 Qc2 c5 O-O b6 a4 Bb7 Nc3 Be7 d3 O-O
        B: c4 Nf6 g3 e6 Bg2 d5 Nf3 Be7 O-O O-O b3 c5 Bb2 Nc6 e3 d4 exd4 cxd4
        B: c4 Nf6 Nf3 e6 g3 b6 Bg2 Bb7 O-O c5 Nc3 Be7 d4 cxd4 Qxd4 d6 Rd1 a6 b3 Nbd7 e4 Qb8
        B: c4 Nf6 Nf3 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6
        B: c4 Nf6 Nf3 e6 b3 b6 Bb2 Bb7 g3 Be7 Bg2 O-O O-O c5
        B: c4 Nf6 d4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3+ Qxc3 b6
        B: c4 Nf6 b3 e5 Bb2 Nc6 e3 d5 cxd5 Nxd5 a3 Bd6
        B: c4 Nf6 d3 e5 Nc3 Bb4 Bd2 O-O e4 c6
        B: c4 Nf6 e3 e6 Nc3 d5 d4 Be7 Nf3 O-O
        B: Nf3 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3 Qxc3 b6 e3 Bb7 Be2 d6
        B: Nf3 Nf6 c4 e6 g3 b6 Bg2 Bb7 O-O c5 Nc3 Be7 d4 cxd4 Qxd4 d6
        B: Nf3 Nf6 g3 b6 Bg2 Bb7 O-O e6 d3 d5 Nbd2 Be7 e4 dxe4 Ng5 O-O Ngxe4 Nbd7
        B: Nf3 Nf6 d4 e6 c4 c5 d5 exd5 cxd5 d6 Nc3 g6
        B: Nf3 Nf6 d4 e6 Bg5 c5 e3 Qb6 Nbd2 Qxb2
        B: Nf3 Nf6 b3 e6 Bb2 b6 e3 Bb7 Be2 c5 O-O Nc6 c4 Be7 Nc3 O-O d4 cxd4 exd4 d5
        B: Nf3 Nf6 e3 e6 b3 b6 Bb2 Bb7 Be2 c5
        B: Nf3 Nf6 d3 d5 g3 e5 Nbd2 Nc6 Bg2 Be7
        B: Nf3 Nf6 Nc3 d5 d4 e6 Bg5 Be7 e3 O-O
        B: g3 e5 Bg2 d5 d3 Nf6 Nf3 Nc6 O-O Be7 c4 O-O cxd5 Nxd5 Nc3 Be6
        B: b3 e5 Bb2 Nc6 e3 d5 Bb5 Bd6 Nf3 Qe7 c4 Nf6 cxd5 Nxd5 a3 O-O
        B: f4 d5 Nf3 g6 e3 Bg7 d4 c5 c3 Nc6 Bd3 Bg4 O-O e6
        B: e3 Nf6 d4 e6 c4 c5 d5 exd5 cxd5 d6 Nc3 g6
        B: Nc3 d5 e4 d4 Nce2 e5 Ng3 Be6 c3 c5 Bb5+ Nd7
        B: d3 e5 Nf3 Nc6 g3 d5 Bg2 Nf6 O-O Be7 c4 O-O
        B: c3 e5 d4 exd4 cxd4 d5 Nc3 Nf6 Bg5 Be7
        B: a3 e5 c4 Nf6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3
        B: b4 e5 Bb2 Bxb4 Bxe5 Nf6 c4 O-O Nf3 d5
        B: g4 d5 Bg2 Bxg4 c4 c6 cxd5 cxd5 Qb3 Nc6
        B: h3 e5 e4 Nf6 Nc3 Bb4
        B: Na3 e5 c4 Nf6 Nc2 Nc6
        B: Nh3 d5 g3 e5 f4 Bxh3 Bxh3 exf4 O-O
    """,
    "loki": """
        W: d4 Nf6 Bg5 Ne4 Bf4 c5 f3 Qa5+ c3 Nf6 d5 Qb6 Bc1 e6 e4 exd5 exd5 d6 Ne2 Be7 Nd2
        W: d4 Nf6 Bg5 Ne4 Bf4 c5 f3 Nf6 d5 Qb6 e4 Qxb2 Nd2 Qb4 c4 d6 Ne2
        W: d4 Nf6 Bg5 Ne4 Bf4 d5 e3 c5 Bd3 Nf6 c3 Nc6 Nd2
        W: d4 Nf6 Bg5 Ne4 Bh4 c5 f3 g5 fxe4 gxh4 e3 Bh6 Kf2 cxd4 exd4 Qb6 Nf3 Qxb2 Nbd2
        W: d4 Nf6 Bg5 Ne4 Bh4 d5 e3 c5 f3 Nf6 Nc3 cxd4 exd4 Nc6 Bf2 e6 Nge2
        W: d4 Nf6 Bg5 e6 e4 h6 Bxf6 Qxf6 Nc3 d6 Qd2 g5 O-O-O Nc6 f4 gxf4 Nf3 Bd7 Qxf4 Qxf4+
        W: d4 Nf6 Bg5 e6 e4 h6 Bxf6 Qxf6 Nc3 Bb4 Nge2 d6 a3 Ba5 b4 Bb6
        W: d4 Nf6 Bg5 e6 e4 h6 Bxf6 Qxf6 Nf3 d6 Nc3 g6 Qd2 Bg7 O-O-O
        W: d4 Nf6 Bg5 c5 d5 Qb6 Nc3 Qxb2 Bd2 Qb6 e4 d6 f4 g6 e5 dxe5 fxe5 Nfd7 Nf3
        W: d4 Nf6 Bg5 c5 Bxf6 gxf6 d5 Qb6 Qc1 f5 c3 Bg7 e3 d6 Na3
        W: d4 Nf6 Bg5 d5 Bxf6 exf6 e3 c6 c4 dxc4 Bxc4 Bd6 Nc3 O-O Qc2 f5 Nge2 b5 Bd3 Nd7
        W: d4 Nf6 Bg5 d5 Bxf6 exf6 e3 Bd6 c4 dxc4 Bxc4 O-O Nc3 f5 Nge2 Nd7
        W: d4 Nf6 Bg5 d5 e3 c5 Bxf6 gxf6 dxc5 Nc6 Bb5 e6 c4
        W: d4 Nf6 Bg5 g6 Bxf6 exf6 e3 d5 c4 dxc4 Bxc4 Bd6 Nc3 O-O Nge2 f5
        W: d4 Nf6 Bg5 d6 Nc3 Nbd7 e4 h6 Bh4 g5 Bg3 Nh5 Be2 Nxg3 hxg3 Bg7 Qd2 c6 O-O-O
        W: d4 Nf6 Bg5 b6 Bxf6 exf6 e3 Bb7 Nd2 d5 Bd3 g6 c3 f5 Qf3
        W: d4 Nf6 Bg5 Nc6 Nc3 d5 Qd2 Bf5 f3 e6 e4 dxe4 d5
        W: d4 Nf6 Bg5 h6 Bxf6 exf6 c3 d5 e3 Bd6 Bd3 O-O Ne2
        W: d4 d5 c4 e6 Nf3 Nf6 g3 Be7 Bg2 O-O O-O dxc4 Qc2 a6 a4 Bd7 Qxc4 Bc6 Bg5 a5 Nc3 Na6 Rfd1 Nb4
        W: d4 d5 c4 e6 Nf3 Nf6 g3 Be7 Bg2 O-O O-O c6 Qc2 b6 Rd1 Ba6 b3 Nbd7 Nc3 Rc8 e4 dxe4 Nxe4
        W: d4 d5 c4 e6 Nf3 Nf6 g3 Be7 Bg2 O-O O-O Nbd7 Qc2 c6 Nbd2 b6 e4 Ba6 b3 Rc8 Bb2
        W: d4 d5 c4 e6 Nf3 Nf6 g3 dxc4 Bg2 a6 Ne5 c5 Na3 cxd4 Naxc4 Qc7 Bf4 Qc5 Rc1 Nd5
        W: d4 d5 c4 e6 Nf3 Nf6 g3 dxc4 Bg2 c5 O-O Nc6 Qa4 cxd4 Nxd4 Qxd4 Bxc6+ Bd7 Rd1 Qxd1+ Qxd1 Bxc6
        W: d4 d5 c4 e6 Nf3 Nf6 g3 Bb4+ Bd2 Be7 Bg2 O-O O-O c6 Qc2 b6 Rd1 Ba6 b3 Nbd7 Nc3
        W: d4 d5 c4 e6 Nf3 Nf6 g3 c5 cxd5 exd5 Bg2 Nc6 O-O Be7 Nc3 O-O Bg5 cxd4 Nxd4 h6
        W: d4 d5 c4 e6 Nf3 c6 Qc2 Nf6 g3 dxc4 Qxc4 b5 Qc2 Bb7 Bg2 Nbd7 O-O c5
        W: d4 d5 c4 e6 Nf3 Be7 g3 Nf6 Bg2 O-O O-O dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7
        W: d4 d5 c4 e6 Nf3 c5 cxd5 exd5 g3 Nc6 Bg2 Nf6 O-O Be7 Nc3 O-O
        W: d4 d5 c4 c6 Nf3 Nf6 e3 Bf5 Nc3 e6 Nh4 Bg6 Nxg6 hxg6 Bd3 Nbd7 O-O Bd6 h3 dxc4 Bxc4
        W: d4 d5 c4 c6 Nf3 Nf6 e3 a6 c5 Bg4 Nbd2 Nbd7 b4 e5 dxe5 Nxe5 Bb2 Nxf3+ gxf3 Bh5
        W: d4 d5 c4 c6 Nf3 Nf6 e3 Bg4 h3 Bxf3 Qxf3 e6 Nc3 Nbd7 Bd3 Bb4 O-O
        W: d4 d5 c4 c6 Nf3 Nf6 e3 e6 Nbd2 Nbd7 Bd3 c5 b3 Be7 Bb2 O-O
        W: d4 d5 c4 c6 Nf3 Nf6 e3 g6 Nc3 Bg7 Be2 O-O O-O Bg4 h3 Bxf3 Bxf3
        W: d4 d5 c4 c6 Nf3 dxc4 e3 b5 a4 e6 axb5 cxb5 b3 Bb4+ Bd2 Bxd2+ Nbxd2 a5 bxc4 b4
        W: d4 d5 c4 c6 Nf3 e6 Qc2 Nf6 g3 dxc4 Qxc4 b5 Qc2 Bb7 Bg2
        W: d4 d5 c4 dxc4 e4 Nf6 e5 Nd5 Bxc4 Nb6 Bd3 Nc6 Ne2 Bg4 f3 Be6 Nbc3 Qd7 Be3
        W: d4 d5 c4 dxc4 e4 e5 Nf3 exd4 Bxc4 Nc6 O-O Be6 Bxe6 fxe6 Qb3 Qd7 Qxb7 Rb8 Qa6
        W: d4 d5 c4 dxc4 e4 c5 d5 Nf6 Nc3 b5 e5 b4 exf6 bxc3 bxc3 exf6 Bxc4
        W: d4 d5 c4 dxc4 e4 Nc6 Be3 Nf6 Nc3 e5 d5 Na5 Bxc4 Nxc4 Qa4+ Bd7 Qxc4 c6
        W: d4 d5 c4 dxc4 e4 b5 a4 c6 axb5 cxb5 b3 Bb7 bxc4 Bxe4 cxb5
        W: d4 d5 c4 Nc6 Nc3 dxc4 Nf3 Nf6 e4 Bg4 Be3 e6 Bxc4 Bb4 Qc2 O-O Rd1
        W: d4 d5 c4 e5 dxe5 d4 Nf3 Nc6 g3 Bg4 Bg2 Qd7 O-O O-O-O Nbd2 Bh3 b4
        W: d4 d5 c4 Bf5 cxd5 Bxb1 Rxb1 Qxd5 Nf3 e6 e3 Nf6 Bd3 Nc6 O-O Bd6
        W: d4 d5 c4 c5 cxd5 Nf6 e4 Nxe4 dxc5 Nxc5 Nc3 e6 Nf3 exd5 Qxd5 Qxd5 Nxd5
        W: d4 d5 c4 Nf6 cxd5 Nxd5 Nf3 Bf5 a3 e6 e4 Bxe4 Bg5 Be7 Bxe7 Qxe7
        W: d4 d5 c4 e6 Nc3 Be7 Nf3 Nf6 g3 O-O Bg2 dxc4 Ne5 c5
        W: d4 e6 c4 Nf6 Nf3 d5 g3 Be7 Bg2 O-O O-O dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7 Bd2 Be4 Qc1
        W: d4 e6 c4 Bb4+ Bd2 Qe7 g3 Nc6 Nf3 Bxd2+ Nbxd2 d6 Bg2 e5 d5 Nb8 O-O Nf6 e4
        W: d4 e6 c4 Bb4+ Bd2 a5 Nc3 Nf6 e4 d6 Bd3 e5 d5
        W: d4 e6 c4 b6 a3 Bb7 Nc3 f5 d5 Nf6 g3
        W: d4 e6 c4 f5 g3 Nf6 Bg2 Be7 Nf3 O-O O-O d5 b3 c6 Ba3 Bxa3 Nxa3
        W: d4 e6 c4 c5 d5 exd5 cxd5 d6 Nc3 g6 e4 Bg7 Nf3 Ne7 Bd3 O-O O-O
        W: d4 f5 Bg5 g6 Nc3 Bg7 e4 fxe4 Nxe4 d5 Nc5 b6 Nb3 Nc6
        W: d4 f5 Bg5 Nf6 Bxf6 exf6 e3 d5 c4 c6 Nc3 Be6 Qb3 Qb6 c5 Qxb3 axb3 Nd7 b4
        W: d4 f5 Bg5 h6 Bh4 g5 Bg3 f4 e3 h5 exf4 h4
        W: d4 f5 Bg5 d5 e3 Nf6 Bxf6 exf6 c4 c6 Nc3 Bd6 Qb3 Kf7
        W: d4 c5 d5 e5 e4 d6 Nc3 Be7 Bb5+ Bd7 Bxd7+ Qxd7 a4 Nf6 Nf3 O-O O-O Na6 Nd2 Nc7
        W: d4 c5 d5 Nf6 Nc3 d6 e4 g6 Nf3 Bg7 Bb5+ Nbd7 a4 O-O O-O a6 Be2 Rb8
        W: d4 c5 d5 e6 c4 exd5 cxd5 d6 Nc3 g6 e4 Bg7 Nf3 Ne7 Bd3 O-O
        W: d4 c5 d5 d6 c4 g6 Nc3 Bg7 e4 Nf6 Bd3 O-O Nge2 e6 O-O exd5 cxd5
        W: d4 c5 d5 b5 e4 Bb7 Nd2 Qa5 Ngf3 b4 Bd3 e6 dxe6 fxe6
        W: d4 g6 c4 Bg7 Nc3 d6 e4 Nf6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Nd2 a5 a3 Nd7 Rb1 f5 b4
        W: d4 g6 c4 Bg7 Nc3 c5 d5 Bxc3+ bxc3 f5 h4 Nf6 h5
        W: d4 g6 c4 Bg7 e4 d6 Nc3 c5 d5 Nf6 Be2 O-O Nf3 e6 O-O exd5 cxd5
        W: d4 d6 e4 Nf6 Nc3 g6 Be2 Bg7 h4 h5 Bg5 c6 Qd2 Nbd7 Nh3 b5
        W: d4 d6 e4 Nf6 Nc3 e5 Nf3 Nbd7 Bc4 Be7 O-O O-O a4 c6
        W: d4 d6 e4 g6 c4 Bg7 Nc3 Nc6 Be3 e5 d5 Nce7 g4 f5
        W: d4 Nc6 d5 Ne5 e4 e6 f4 Ng6 dxe6 fxe6 Nf3 Bc5 Bd3
        W: d4 Nc6 Nf3 d5 c4 Bg4 Nc3 e6 cxd5 exd5 Bf4 Bb4 e3 Nge7 Rc1
        W: d4 b6 e4 Bb7 Bd3 e6 Nf3 c5 c3 Nf6 Qe2 Be7 O-O Nc6 a3 d5 e5 Nd7
        W: d4 b5 e4 Bb7 Bd3 a6 Nf3 e6 O-O Nf6 Qe2 c5 dxc5 Bxc5
        W: d4 e5 dxe5 Nc6 Nf3 Qe7 Bf4 Qb4+ Bd2 Qxb2 Nc3 Bb4 Rb1 Qa3 e4
        W: d4 c6 e4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7
        W: d4 a6 e4 b5 Bd3 Bb7 Nf3 e6 O-O c5 dxc5 Bxc5
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3 Bc5 Nb3 Ba7 Qe2 Nc6 Be3 Bxe3 Qxe3 Nge7 c4 b5
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3 Nf6 O-O Qc7 Qe2 d6 c4 g6 Nc3 Bg7 Nf3 Nc6 Bf4 e5
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 c4 Nf6 Nc3 Bb4 Bd3 Nc6 Nxc6 dxc6 e5 Qa5 exf6 Bxc3+ bxc3 Qxc3+ Bd2 Qxd3 fxg7 Rg8
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 c4 Qc7 Nc3 Nf6 Be2 Bb4 Bd2 Nc6 Nxc6 dxc6
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Nc3 Qc7 Bd3 Nc6 Nxc6 bxc6 O-O d5 Qf3 Nf6 Bf4 Qb7 Qg3 Bd6
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Nc3 Qc7 g3 Bb4 Bd2 Nf6 Bg2 Nc6 Nxc6 dxc6 O-O e5
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Nc3 Qc7 Be2 Nf6 O-O Bb4 Qd3 Nc6 Nxc6 dxc6
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Nc3 b5 Bd3 Qb6 Nb3 Qc7 Qe2 Bd6 Bd2 Nc6 O-O-O Nge7
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Be2 Nf6 Bf3 Qc7 O-O Nc6 Nxc6 dxc6
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd2 Nf6 Bd3 Nc6 Nxc6 dxc6 O-O e5
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be3 a6 Qd2 Nf6 O-O-O Bb4 f3 Ne5 Nb3 b5 Kb1 Be7 Qf2 d6 Bb6 Qb8 g4
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be2 a6 O-O Nf6 Be3 Bb4 Na4 Be7 Nxc6 bxc6 Nb6 Rb8 Nxc8 Qxc8
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 g3 a6 Bg2 Nf6 O-O Nxd4 Qxd4 Bc5 Bf4 d6 Qd2 h6 Rad1 e5 Be3 Bg4
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 f4 a6 Nxc6 bxc6 Qf3 d5
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nb5 d6 c4 Nf6 N1c3 a6 Na3 Be7 Be2 O-O O-O b6 Be3 Bb7 Qb3 Na5 Qb4
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nxc6 bxc6 Bd3 d5 O-O Nf6 Nd2 Be7 b3 O-O Bb2 a5
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Be3 Nf6 Nc3 Bb4 Bd3 d5 exd5 Nxd5
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Be2 Nf6 Nc3 Bb4 O-O Bxc3 bxc3 Nxe4
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nf3 Nf6 Nc3 Bb4 Bd3 d5 exd5 exd5 O-O O-O
        B: e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 c4 Nf6 Nc3 Bb4 Nxc6 bxc6 Bd3 e5 O-O O-O
        B: e4 c5 Nf3 e6 d4 cxd4 Qxd4 Nc6 Qd1 Nf6 Nc3 Bb4 Bd3 d5 exd5 exd5 O-O O-O
        B: e4 c5 Nf3 e6 c3 d5 exd5 exd5 d4 Nc6 Bb5 Bd6 dxc5 Bxc5 O-O Nge7 Nbd2 O-O Nb3 Bb6 Re1 Bg4
        B: e4 c5 Nf3 e6 c3 Nf6 e5 Nd5 d4 cxd4 cxd4 d6 Bc4 Nb6 Bb3 dxe5 dxe5 Qxd1+ Bxd1 Nc6
        B: e4 c5 Nf3 e6 d3 Nc6 g3 d5 Nbd2 Nf6 Bg2 Be7 O-O O-O Re1 b5 e5 Nd7 Nf1 a5 h4 b4
        B: e4 c5 Nf3 e6 d3 d5 Nbd2 Nc6 g3 g6 Bg2 Bg7 O-O Nge7 Re1 b6 h4 h6
        B: e4 c5 Nf3 e6 Nc3 Nc6 d4 cxd4 Nxd4 Qc7 g3 a6 Bg2 Nf6 O-O Nxd4 Qxd4 Bc5 Bf4 d6 Qd2 h6
        B: e4 c5 Nf3 e6 Nc3 a6 d4 cxd4 Nxd4 Qc7 Bd3 Nc6 Nxc6 bxc6 O-O d5
        B: e4 c5 Nf3 e6 Nc3 a6 g3 b5 Bg2 Bb7 O-O d6 d4 cxd4 Nxd4 Nf6
        B: e4 c5 Nf3 e6 Nc3 d6 d4 cxd4 Nxd4 Nf6 g4 h6 h4 Nc6 Rg1 h5 gxh5 Nxh5 Bg5 Nf6
        B: e4 c5 Nf3 e6 b3 Nc6 Bb2 d6 d4 cxd4 Nxd4 Nf6 Bd3 a6 O-O Qc7 Nxc6 bxc6 f4 e5
        B: e4 c5 Nf3 e6 g3 Nc6 Bg2 d5 exd5 exd5 O-O Nf6 d4 Be7 Be3 c4
        B: e4 c5 Nf3 e6 Bc4 Nc6 Nc3 a6 a4 Nf6 O-O Qc7 d3 d5 exd5 exd5
        B: e4 c5 Nf3 e6 Be2 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 Bb4
        B: e4 c5 Nf3 e6 Qe2 Nc6 g3 d5 d3 Nf6 Bg2 Be7 O-O O-O
        B: e4 c5 Nf3 e6 c4 Nc6 Nc3 Nf6 d4 cxd4 Nxd4 Bb4
        B: e4 c5 Nc3 e6 Nge2 a6 g3 b5 Bg2 Bb7 d4 cxd4 Nxd4 b4 Na4 Nf6 e5 Bxg2
        B: e4 c5 Nc3 e6 Nf3 Nc6 d4 cxd4 Nxd4 Qc7 g3 a6 Bg2 Nf6 O-O Nxd4 Qxd4 Bc5 Bf4 d6 Qd2
        B: e4 c5 Nc3 e6 g3 d5 exd5 exd5 Bg2 Nf6 d3 d4 Ne4 Nxe4 Bxe4 Bd6
        B: e4 c5 Nc3 e6 f4 d5 Nf3 Nf6 e5 d4 exf6 dxc3 bxc3 Qxf6 d4 cxd4 cxd4 Bb4+ Bd2 Bxd2+ Qxd2
        B: e4 c5 c3 e6 d4 d5 exd5 exd5 Nf3 Nc6 Bb5 Bd6 dxc5 Bxc5 O-O Nge7 Nbd2 O-O Nb3 Bb6
        B: e4 c5 c3 e6 d4 d5 e5 Nc6 Nf3 Qb6 a3 c4 Nbd2 Na5 g3 Bd7 h4 Ne7
        B: e4 c5 d4 cxd4 c3 Nf6 e5 Nd5 Qxd4 e6 Nf3 Nc6 Qe4 d6 Bb5 Bd7 exd6 Bxd6 Bxc6 Bxc6 Qg4
        B: e4 c5 d4 cxd4 c3 dxc3 Nxc3 Nc6 Nf3 e6 Bc4 a6 O-O Nge7 Bg5 f6 Be3 b5 Bb3 Na5
        B: e4 c5 d4 cxd4 Nf3 e6 Nxd4 a6
        B: e4 c5 d4 cxd4 Qxd4 Nc6 Qe3 g6 Bd2 Bg7 Nc3 Nf6 O-O-O O-O
        B: e4 c5 f4 e6 Nf3 d5 Bb5+ Bd7 Bxd7+ Qxd7 d3 Nc6 O-O Nf6 e5 Ng8 c4 Nge7 Nc3 d4
        B: e4 c5 f4 e6 Nf3 d5 e5 Nc6 c3 Nh6 d4 Qb6 Na3 cxd4 cxd4 Bb4+ Kf2 O-O
        B: e4 c5 b4 cxb4 a3 d5 exd5 Qxd5 Nf3 e5 axb4 Bxb4 c3 Be7 d4 Nc6
        B: e4 c5 b3 e6 Bb2 Nc6 Nf3 d6 d4 cxd4 Nxd4 Nf6 Bd3 a6
        B: e4 c5 g3 e6 Bg2 d5 exd5 exd5 d4 Nf6 Nf3 Nc6
        B: e4 c5 Bc4 e6 Nc3 a6 a4 Nc6 d3 g6 f4 Bg7 Nf3 Nge7
        B: e4 c5 Ne2 e6 g3 d5 exd5 exd5 Bg2 Nf6 O-O Nc6 d4
        B: e4 c5 a3 Nc6 b4 cxb4 axb4 Nxb4 d4 e5 dxe5 Nc6 Nf3 Qc7
        B: e4 c5 Qh5 e6 Nf3 Nc6 Bb5 Nge7 O-O g6 Qh4 Bg7
        B: d4 Nf6 c4 c5 d5 b5 cxb5 a6 bxa6 g6 Nc3 Bxa6 e4 Bxf1 Kxf1 d6 Nf3 Bg7 g3 O-O Kg2 Nbd7 Re1 Qb6 h3
        B: d4 Nf6 c4 c5 d5 b5 cxb5 a6 bxa6 Bxa6 Nc3 d6 e4 Bxf1 Kxf1 g6 g3 Bg7 Kg2 O-O Nf3 Nbd7 Re1 Qb6
        B: d4 Nf6 c4 c5 d5 b5 cxb5 a6 b6 d6 Nc3 Nbd7 a4 Qxb6 e4 g6 Nf3 Bg7 Nd2 O-O
        B: d4 Nf6 c4 c5 d5 b5 cxb5 a6 e3 g6 Nc3 Bg7 a4 O-O Ra3 e6 dxe6 fxe6
        B: d4 Nf6 c4 c5 d5 b5 cxb5 a6 f3 e6 e4 exd5 e5 Qe7 Qe2 Ng8 Nc3 Bb7
        B: d4 Nf6 c4 c5 d5 b5 Nf3 g6 cxb5 a6 b6 d6 Nc3 Nbd7 a4 Bg7 e4 O-O Be2 Qxb6
        B: d4 Nf6 c4 c5 d5 b5 Nf3 bxc4 Nc3 d6 e4 g6 Bxc4 Bg7 O-O O-O
        B: d4 Nf6 c4 c5 d5 b5 Nd2 bxc4 e4 c3 bxc3 g6 Ngf3 d6 Bd3 Bg7 O-O O-O
        B: d4 Nf6 c4 c5 d5 b5 a4 b4 Nf3 e6 g3 exd5 cxd5 d6 Bg2 g6 Nbd2 Bg7
        B: d4 Nf6 c4 c5 d5 b5 Qc2 bxc4 e4 e6 Bxc4 exd5 exd5 d6 Nc3 Be7
        B: d4 Nf6 c4 c5 d5 b5 Nc3 b4 Nb5 d6 e4 g6 f4 Bg7 Nf3 O-O
        B: d4 Nf6 c4 c5 d5 b5 e4 Nxe4 cxb5 a6 Nc3 Nxc3 bxc3 g6
        B: d4 Nf6 c4 c5 d5 b5 Bg5 Ne4 Bf4 Qa5+ Nd2 g5 Be3 Nxd2 Qxd2 Qxd2+ Kxd2 bxc4
        B: d4 Nf6 c4 c5 Nf3 cxd4 Nxd4 e5 Nb5 d5 cxd5 Bc5 N5c3 O-O e3 e4 Be2 Qe7
        B: d4 Nf6 c4 c5 Nf3 cxd4 Nxd4 e6 g3 Qb6 Nb3 Ne4 Bg2 Bb4+ Bd2 Nxd2 N1xd2 Nc6
        B: d4 Nf6 c4 c5 e3 g6 Nc3 Bg7 Nf3 O-O Be2 cxd4 exd4 d5 O-O Nc6 c5 Ne4
        B: d4 Nf6 c4 c5 dxc5 e6 Nc3 Bxc5 Nf3 b6 e3 Bb7 Be2 O-O O-O d5
        B: d4 Nf6 Nf3 c5 d5 b5 Bg5 Ne4 Bf4 Bb7 Qd3 Qa5+ Nbd2 Nxd2 Bxd2 Qb6 e4 g6
        B: d4 Nf6 Nf3 c5 d5 b5 c4 Bb7 cxb5 a6 bxa6 Qa5+ Nc3 Bxa6 e4 Bxf1 Kxf1 d6
        B: d4 Nf6 Nf3 c5 d5 b5 g3 Bb7 Bg2 b4 c4 e6 O-O exd5 cxd5 Bxd5
        B: d4 Nf6 Nf3 c5 d5 b5 e4 Nxe4 Bd3 Nf6 O-O d6 c4 e6
        B: d4 Nf6 Nf3 c5 c4 cxd4 Nxd4 e5 Nb5 d5 cxd5 Bc5 N5c3 O-O
        B: d4 Nf6 Nf3 c5 e3 g6 c4 Bg7 Nc3 O-O Be2 cxd4 exd4 d5 O-O Nc6
        B: d4 Nf6 Nf3 c5 c3 cxd4 cxd4 d5 Nc3 Nc6 Bf4 Bf5 e3 e6 Qb3 Bb4
        B: d4 Nf6 Nf3 c5 g3 cxd4 Nxd4 d5 Bg2 e5 Nb3 Be6 O-O Nc6
        B: d4 Nf6 Nf3 c5 Bf4 cxd4 Nxd4 Qb6 Nb3 Nc6 e3 d5 Nc3 e5
        B: d4 Nf6 Nf3 c5 Bg5 Ne4 Bf4 Qb6 Nbd2 Qxb2 e3 Qb4 c3 Nxc3 Qb3 Qxb3 axb3
        B: d4 Nf6 Nf3 c5 dxc5 e6 b4 a5 c3 b6 cxb6 axb4
        B: d4 Nf6 Bg5 c5 d5 Qb6 Nc3 Qxb2 Bd2 Qb6 e4 d6 f4 g6 Nf3 Bg7 Bd3 O-O O-O
        B: d4 Nf6 Bg5 c5 Bxf6 gxf6 d5 Qb6 Qc1 f5 c3 Bg7 e3 d6 Na3
        B: d4 Nf6 Bg5 c5 e3 Qb6 Nd2 Qxb2 Bxf6 gxf6 Ngf3 cxd4 exd4 d5
        B: d4 Nf6 Bg5 c5 c3 Qb6 Qb3 cxd4 Qxb6 axb6 cxd4 Nc6 e3 d5
        B: d4 Nf6 Bg5 c5 Nc3 cxd4 Qxd4 Nc6 Qh4 e6 O-O-O Be7 e4 Qa5
        B: d4 Nf6 Bf4 c5 d5 Qb6 Nc3 Qxb2 Bd2 Qb6 e4 d6 f4 g6 Nf3 Bg7 Bd3 O-O
        B: d4 Nf6 Bf4 c5 e3 Qb6 Nc3 Qxb2 Nb5 Na6 a3 d5 Rb1 Qxa3 Ra1 Qb4+ c3 Qa5
        B: d4 Nf6 Bf4 c5 c3 cxd4 cxd4 Qb6 Qb3 Nc6 e3 d5 Nc3 e6
        B: d4 Nf6 Bf4 c5 dxc5 e6 Nc3 Bxc5 e4 Qb6 Qd2 Nc6
        B: d4 Nf6 Nc3 d5 Bg5 c5 Bxf6 gxf6 e4 dxe4 d5 f5 Bb5+ Bd7 Bxd7+ Qxd7 Qh5 Qd6
        B: d4 Nf6 Nc3 d5 Bf4 c5 e4 dxe4 d5 a6 a4 e6
        B: d4 Nf6 Nc3 d5 Nf3 c5 e3 Nc6 Bb5 e6
        B: d4 Nf6 g3 c5 d5 b5 Bg2 Bb7 c4 bxc4 Nc3 Qb6 Nf3 e6
        B: d4 Nf6 e3 c5 c3 b6 Nd2 Bb7 Ngf3 g6 Bd3 Bg7 O-O O-O
        B: d4 Nf6 e3 c5 Nf3 b6 Bd3 Bb7 O-O e6 c4 cxd4 exd4 d5
        B: d4 Nf6 c3 c5 Bg5 Qb6 Qb3 cxd4 Qxb6 axb6 cxd4 Nc6 e3 d5
        B: d4 Nf6 Nd2 c5 e3 cxd4 exd4 d5 Ngf3 Nc6 c3 Bf5
        B: d4 Nf6 f3 c5 d5 d6 e4 g6 c4 Bg7 Nc3 O-O
        B: d4 Nf6 e4 Nxe4 Bd3 Nf6 Bg5 d5 Nd2 c5
        B: c4 c5 Nc3 Nf6 g3 d5 cxd5 Nxd5 Bg2 Nc7 Nf3 Nc6 O-O e5 d3 Be7 Nd2 Be6 Nc4 f6 f4
        B: c4 c5 Nc3 Nf6 Nf3 b6 e4 Nc6 d4 cxd4 Nxd4 Bb7 f3 e6 Be3 Be7 Be2 O-O O-O Qb8
        B: c4 c5 Nc3 Nf6 e4 Nc6 Nf3 e6 d4 cxd4 Nxd4 Bb4 Nxc6 bxc6 Bd3 e5
        B: c4 c5 Nc3 Nf6 d4 cxd4 Qxd4 Nc6 Qd2 g6 b3 Bg7 Bb2 O-O
        B: c4 c5 Nc3 Nf6 e3 e6 Nf3 b6 e4 Bb7 d3 d6 g3 Be7
        B: c4 c5 Nf3 Nf6 Nc3 b6 e4 Nc6 d4 cxd4 Nxd4 Bb7 f3 e6
        B: c4 c5 Nf3 Nf6 g3 b6 Bg2 Bb7 O-O e6 Nc3 Be7 d4 cxd4 Qxd4 d6
        B: c4 c5 Nf3 Nf6 d4 cxd4 Nxd4 e5 Nb5 d5 cxd5 Bc5 N5c3 O-O
        B: c4 c5 Nf3 Nf6 b3 b6 Bb2 Bb7 g3 e6 Bg2 Be7 O-O O-O
        B: c4 c5 g3 g6 Bg2 Bg7 Nc3 Nc6 e3 e6 Nge2 Nge7 d4 cxd4 Nxd4 d5 cxd5 Nxd5
        B: c4 c5 g3 Nf6 Bg2 b6 Nc3 Bb7 e4 Nc6 Nge2 e6 O-O Qb8
        B: c4 c5 b3 Nf6 Bb2 b6 g3 Bb7 Bg2 g6 Bxf6 exf6 Nc3 f5
        B: c4 c5 e4 Nc6 Nc3 g6 g3 Bg7 Bg2 e6 Nge2 Nge7 O-O O-O d3 d6
        B: c4 c5 d4 cxd4 Nf3 Nc6 Nxd4 e6 g3 Qb6 Nb3 Ne5 e4 Bb4+ Bd2 Nxc4
        B: c4 c5 e3 Nf6 Nc3 e6 Nf3 d5 d4 Nc6 a3 a6
        B: c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 Nf3 e5 O-O Nge7 d3 O-O a3 d6 Rb1 a5
        B: Nf3 c5 c4 Nf6 Nc3 b6 e4 Nc6 d4 cxd4 Nxd4 Bb7 f3 e6
        B: Nf3 c5 g3 b6 Bg2 Bb7 O-O g6 d3 Bg7 e4 d6 c3 Nf6 Nbd2 O-O
        B: Nf3 c5 e4 e6 d4 cxd4 Nxd4 a6 Bd3 Bc5 Nb3 Ba7 Qe2 Nc6 Be3 Bxe3 Qxe3 Nge7
        B: Nf3 c5 b3 Nf6 Bb2 b6 e3 Bb7 Be2 e6 O-O Be7 c4 O-O Nc3 d5
        B: Nf3 c5 d4 cxd4 Nxd4 Nf6 Nc3 d5 Bg5 e6 e4 dxe4
        B: Nf3 c5 e3 Nf6 b3 b6 Bb2 Bb7 c4 e6 Be2 Be7
        B: Nf3 c5 d3 Nc6 e4 e6 g3 d5 Nbd2 Nf6 Bg2 Be7
        B: Nf3 c5 c3 Nf6 d4 cxd4 cxd4 d5 Nc3 Nc6 Bf4 Bf5
        B: g3 c5 Bg2 Nc6 e4 g6 Ne2 Bg7 O-O e6 c3 Nge7 d4 cxd4 cxd4 d5
        B: g3 c5 Bg2 Nc6 c4 g6 Nc3 Bg7 Nf3 e6 O-O Nge7 d3 O-O
        B: b3 e5 Bb2 Nc6 e3 Nf6 Bb5 Bd6 Na3 a6 Bxc6 dxc6 Nc4 Qe7 d3 O-O
        B: b3 e5 Bb2 Nc6 c4 Nf6 e3 d5 cxd5 Nxd5 a3 Bd6
        B: f4 c5 Nf3 Nc6 e4 g6 Bb5 Bg7 Bxc6 dxc6 d3 Qc7 O-O Nh6
        B: f4 c5 Nf3 Nc6 e3 g6 Be2 Bg7 O-O d6 c3 Nf6
        B: e3 c5 d4 cxd4 exd4 d5 Nf3 Nc6 Bb5 Bg4 O-O e6
        B: e3 c5 Nf3 Nc6 d4 cxd4 exd4 d5 c3 Bg4
        B: Nc3 c5 e4 e6 Nf3 Nc6 d4 cxd4 Nxd4 Qc7 g3 a6 Bg2 Nf6
        B: Nc3 c5 g3 g6 Bg2 Bg7 d3 Nc6 e4 d6 f4 e6
        B: d3 c5 e4 Nc6 Nf3 g6 g3 Bg7 Bg2 e6 O-O Nge7 c3 O-O
        B: c3 c5 d4 cxd4 cxd4 d5 Nc3 Nc6 Bf4 Bf5
        B: a3 c5 e4 Nc6 Nf3 e6 d4 cxd4 Nxd4 Qc7
        B: b4 c5 bxc5 e5 d4 exd4 Qxd4 Nc6 Qd1 Bxc5
        B: g4 c5 Bg2 Nc6 c4 g6 Nc3 Bg7
        B: h3 c5 e4 Nc6 Nf3 e6 d4 cxd4 Nxd4 a6
        B: Na3 c5 c4 Nc6 Nc2 g6 g3 Bg7
        B: Nh3 c5 g3 Nc6 Bg2 g6 O-O Bg7
    """,
}
