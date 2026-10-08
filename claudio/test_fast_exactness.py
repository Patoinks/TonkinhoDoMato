"""claudio_fast must answer exactly what the original G16 C++ answered: the same legal moves in the same
order, the same endings, encodings, exchange values, deep searches (values, main lines, statuses) and
forced mates - node counts included, so a node limit stops every search at the same point.

results/fast-reference.json.gz holds those answers, recorded once with the original module (claudio-g16.pt's
C++, commit d5e6fda) on positions from real G16 games, random games, special and illegal boards:

    python -m pytest claudio/test_fast_exactness.py -q
    python claudio/test_fast_exactness.py --record --build DIR    (DIR holds the module to record with)
"""
import gzip
import hashlib
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "results/fast-reference.json.gz"
VALUES = [[0.0, 1.0, 3.0, 3.0, 5.0, 9.0],
          [0.0, 1.0, 1.5246, 1.6487, 5.4765, 7.5214],     # G16's own values: opening, middlegame, ending
          [0.0, 1.0, 1.6429, 2.18, 3.6509, 4.9686],
          [0.0, 1.0, 5.593, 6.9578, 11.0142, 22.05],
          [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]                  # every piece alike: ties in every ordering
SPECIAL = [
    "r3k2r/pppq1ppp/2n2n2/3pp3/1b1PP1b1/2NB1N2/PPPQ1PPP/R3K2R w KQkq - 4 9",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "8/P7/8/8/8/8/6k1/K7 w - - 0 1",
    "8/8/8/8/8/8/1p4k1/K7 b - - 0 1",                  # illegal: White in check, Black to move
    "k7/8/1K6/8/8/8/8/1Q6 w - - 0 1",
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    "4k3/8/8/2KPp3/8/8/8/4r3 w - e6 0 1",
    "8/8/8/1k1Pp2Q/8/8/8/4K3 w - e6 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 30",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "3r2k1/1b3ppp/8/8/8/6q1/5PP1/5RK1 w - - 0 1",
    "8/k3K3/8/8/8/8/1R6/8 w - - 0 1",
    "4k3/8/8/8/2p5/3Q4/8/4K3 w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "1r2k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1",
    "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1",
    "K7/8/8/3Pp3/8/8/8/k6b w - e6 0 1",
    "2r5/3P4/8/8/8/8/k7/4K3 w - - 0 1",
    "6k1/6pp/8/8/8/8/1Q6/K7 w - - 0 1",                # back-rank mates
    "6rk/6pp/8/6N1/8/8/8/K7 w - - 0 1",                 # smothered mate
    "4k3/4q3/8/8/8/8/4R3/4K3 w - - 0 1",                # pin on the file
    "4k3/8/8/b7/8/8/3N4/4K3 w - - 0 1",                 # pin on the diagonal
    "8/8/8/8/8/8/8/K6k w - - 0 1",
    "8/8/8/8/8/8/8/8 w - - 0 1",                        # no kings at all
    "QQQQQQQQ/QQQQQQQQ/8/8/8/8/k7/K7 w - - 0 1",        # more than 16 pieces for one side
    "4k3/8/8/8/8/8/3q4/4K3 w - - 0 1",                  # in check by a queen next to the king
    "4k3/8/8/8/1b6/8/3P4/R3K2R w KQ - 0 1",
    "r3k3/1P6/8/8/8/8/8/4K3 w q - 0 1",                 # promotion by capture, castling rook attacked
    "8/8/8/3r4/3KK3/8/8/7k w - - 0 1",                  # two white kings (python-chess can hold them)
    "8/2q5/2q5/K2k4/3K4/8/8/8 w - - 0 1",
    "8/8/1Q6/2K5/1k6/8/8/4K3 b - - 0 1",
    "4k3/8/8/PP6/8/8/8/4K3 w - a6 0 1",                 # en passant square with no pawn to take
    "k5b1/8/1p6/2p1nP2/2K5/7r/8/1r6 w - e6 0 1",        # en passant that answers a double check
    "1q6/5Q2/7Q/2K3Q1/1pk3r1/4R3/8/8 b - c3 0 1",
]


def _native(board):
    from claudio_agent import native_board
    return native_board(board)


def _norm(x):
    if isinstance(x, float):
        return "nan" if math.isnan(x) else x
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if hasattr(x, "tobytes"):
        return hashlib.sha1(x.tobytes()).hexdigest()
    return x


def _call(fast, name, args):
    try:
        return ["ok", _norm(getattr(fast, name)(*args))]
    except Exception as error:      # noqa: BLE001 - the exception itself is part of the answer
        return ["error", type(error).__name__, str(error)]


def _nodes(probe):
    """Exact node count of a search: the smallest node limit with which it is not aborted (NaN)."""
    low, high = 0, 1
    while math.isnan(probe(high)):
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        low, high = (middle, high) if math.isnan(probe(middle)) else (low, middle)
    return high


def _positions(rng):
    import chess
    games = []
    for log in sorted((HERE / "results/matches").glob("g16-vs-g15-100-bo*.jsonl")):
        games += [json.loads(line)["moves"].split() for line in log.read_text().splitlines() if line.strip()]
    cases = []
    for _ in range(160):                                           # real G16 games, history included
        moves = rng.choice(games)
        cases.append((chess.STARTING_FEN, moves[:rng.randrange(0, len(moves) + 1)]))
    for _ in range(60):                                            # random games, captures and checks often
        board = chess.Board()
        for _ in range(rng.randrange(0, 160)):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            sharp = [m for m in legal if board.is_capture(m) or m.promotion or board.gives_check(m)]
            board.push(rng.choice(sharp) if sharp and rng.random() < 0.4 else rng.choice(legal))
        cases.append((chess.STARTING_FEN, [m.uci() for m in board.move_stack]))
    cases += [(fen, []) for fen in SPECIAL]
    for _ in range(25):                                            # random boards, legal or not
        board = chess.Board(None)
        squares = rng.sample(range(64), rng.randrange(3, 14))
        for k, square in enumerate(squares):
            kind = chess.KING if k < 2 and rng.random() < 0.9 else rng.choice([1, 1, 2, 3, 4, 5])
            if kind == chess.PAWN and chess.square_rank(square) in (0, 7):
                kind = chess.KNIGHT
            board.set_piece_at(square, chess.Piece(kind, k % 2 == 0 if k < 2 else rng.random() < 0.5))
        board.turn = rng.random() < 0.5
        cases.append((board.fen(), []))
    return cases


def _calls(board, fast, rng):
    """The calls of one case: [name, args]. rng chooses their parameters."""
    import chess
    codes = list(fast.legal_moves())
    out = [["fen", []], ["legal_moves", []], ["terminal_strict", []], ["terminal", []], ["pieces", []]]
    if not codes:
        return out
    me = board.turn
    values = rng.choice(VALUES)
    some = rng.sample(codes, min(len(codes), rng.randrange(1, 10)))
    out += [["encode_here", [me]], ["encode_after", [codes, me]], ["checks_after", [codes]],
            ["exchange_after_moves", [codes, values, 5, 200_000]],
            ["exchange_after_moves", [codes, values, 5, rng.randrange(1, 3000)]],
            ["exchange_gain", [values, rng.randrange(0, 8), rng.choice([200_000, rng.randrange(1, 400)])]],
            ["deep_after_moves", [some, values, 2, 5, 2_000_000, True]],
            ["deep_after_moves", [some, values, rng.randrange(1, 4), 5, rng.randrange(1, 40_000), rng.random() < 0.6]],
            ["deep_after_moves", [some, values, 3, 5, rng.randrange(2000, 100_000), False]],
            ["root_search", [some, [rng.uniform(-0.5, 1.0) for _ in some], [rng.choice([0.0, 1.0, 3.0]) for _ in some],
                             rng.choice([0.2, 0.7, 1.8069]), rng.choice([-math.inf, rng.uniform(-0.5, 1.0)]), values,
                             rng.randrange(1, 4), 5, rng.choice([2_000_000, rng.randrange(1, 60_000)])]],
            ["forced_gain", [values, rng.randrange(0, 4), 5, rng.choice([500, 50_000, 2_000_000])]]]
    out += [["deep_after_moves", [some[:3], values, rng.choice([4, 5]), 5, 300_000, True]],
            ["root_search", [some, [rng.uniform(-0.5, 1.0) for _ in some], [0.0 for _ in some], 1.8069, -math.inf,
                             values, rng.choice([4, 5]), 5, 400_000]]]
    for attacker in (True, False):
        out += [["forced_mate", [attacker, 3, 20_000, 1]],
                ["forced_mate", [attacker, rng.randrange(1, 5), rng.randrange(1, 40_000), rng.randrange(0, 3)]]]
        if chess.popcount(board.occupied) <= 8:                    # G16's endgame: mates in 6, 400 000 nodes
            out.append(["forced_mate", [attacker, 6, 400_000, 1]])
    return out


def record(build):
    sys.path.insert(0, str(HERE))
    import paths  # noqa: F401
    sys.path.insert(0, build)
    import chess
    import claudio_fast
    assert Path(claudio_fast.__file__).parent == Path(build).resolve(), claudio_fast.__file__
    rng = random.Random(2026)
    cases = []
    for root, moves in _positions(rng):
        board = chess.Board(root)
        for uci in moves:
            board.push_uci(uci)
        fast = _native(board)
        calls = [[name, args, _call(fast, name, args)] for name, args in _calls(board, fast, rng)]
        codes = list(fast.legal_moves())
        ends = []
        for code in codes:
            fast.push(code)
            ends.append(fast.terminal_strict())
            fast.pop()
        nodes = []
        if codes:
            values, plies = rng.choice(VALUES), rng.randrange(1, 5)
            nodes.append(["forced_gain", [values, plies, 5],
                          _nodes(lambda limit: fast.forced_gain(values, plies, 5, limit))])
            code = rng.choice(codes)
            nodes.append(["deep_after_moves", [[code], values, plies, 5],
                          _nodes(lambda limit: fast.deep_after_moves([code], values, plies, 5, limit, True)[0][0])])
        cases.append({"root": root, "moves": moves, "calls": calls, "ends": ends, "nodes": nodes})
    with gzip.open(REFERENCE, "wt", encoding="utf-8") as out:
        json.dump({"module": str(build), "cases": cases}, out)
    print(f"{len(cases)} cases, {sum(len(c['calls']) for c in cases)} calls -> {REFERENCE}")


def test_fast_answers_exactly_what_the_original_answered():
    import pytest
    from tonkai import fast_rules
    if not fast_rules.AVAILABLE:
        pytest.skip("tonkai_fast is not built")
    import chess
    data = json.loads(gzip.decompress(REFERENCE.read_bytes()).decode("utf-8"))
    differences = []
    for case in data["cases"]:
        board = chess.Board(case["root"])
        for uci in case["moves"]:
            board.push_uci(uci)
        fast = _native(board)
        for name, args, expected in case["calls"]:
            got = _call(fast, name, args)
            if got != expected:
                differences.append((case["root"], len(case["moves"]), name, str(args)[:80], str(got)[:120],
                                    str(expected)[:120]))
        codes = list(fast.legal_moves())
        if hasattr(fast, "terminal_after_moves") and codes and list(fast.terminal_after_moves(codes)) != case["ends"]:
            differences.append((case["root"], len(case["moves"]), "terminal_after_moves", "", "", ""))
        for name, args, count in case["nodes"]:
            if name == "forced_gain":
                probe = lambda limit, a=args: fast.forced_gain(a[0], a[1], a[2], limit)   # noqa: E731
            else:
                probe = lambda limit, a=args: fast.deep_after_moves(a[0], a[1], a[2], a[3], limit, True)[0][0]  # noqa: E731
            if math.isnan(probe(count)) or (count > 1 and not math.isnan(probe(count - 1))):
                differences.append((case["root"], len(case["moves"]), "nodes " + name, str(args)[:80], "", count))
    assert not differences, f"{len(differences)} differences, first: {differences[:3]}"


sys.path.insert(0, str(HERE))
import paths  # noqa: E402,F401

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--build", required=True, help="folder with the claudio_fast module to record with")
    args = parser.parse_args()
    if args.record:
        record(args.build)
