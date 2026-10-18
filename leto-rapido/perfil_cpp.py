"""Root searches only, for a native profiler: claudio_fast and python-chess, no torch."""
import sys, time
sys.path.insert(0, "fast")
import chess, claudio_fast
fens = ["r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 2 9",
        "r2q1rk1/1b2bppp/p2ppn2/1p6/3NP3/1BN1B3/PPP2PPP/R2Q1RK1 w - - 0 12",
        "2r2rk1/pp1bqppp/2n1pn2/3p4/2PP4/2N1PN2/PP1QBPPP/2R2RK1 w - - 0 14"]
values = [0.0, 1.0, 2.13, 2.73, 3.98, 9.06]
def board(b):
    return claudio_fast.Board(b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
                              b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK], b.turn,
                              b.castling_rights, -1 if b.ep_square is None else b.ep_square, b.halfmove_clock, b.fullmove_number)
claudio_fast.Board.root_prunings(True)
boards = [board(chess.Board(f)) for f in fens]
t = time.perf_counter(); n = 0
while time.perf_counter() - t < float(sys.argv[1] if len(sys.argv) > 1 else 20):
    for f in boards:
        codes = list(f.legal_moves())[:10]
        f.root_search(codes, [0.0] * len(codes), [0.0] * len(codes), 1.76, -1e9, values, 7, 5, 559745)
        n += 1
print(n, "root searches", round((time.perf_counter() - t) / n * 1000, 2), "ms each")
