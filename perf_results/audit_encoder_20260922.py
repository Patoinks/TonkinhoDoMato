"""Read-only encoder audit: byte equivalence, then optional microbenchmark.

This file never patches tonkai, loads/saves checkpoints, or writes results.
Run from the repository root; --benchmark adds paired timing blocks after checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import numpy as np

from export_web import GOLDEN_FENS
from tonkai import encoding as E


def encode_after_moves_batched(board, moves, self_color, out=None):
    """Keep python-chess push/pop and exact extras; batch only bit expansion."""
    n = len(moves)
    if out is None or out.shape[0] < n:
        out = np.empty((max(n, 1), E.INPUT_SIZE), dtype=np.float32)
    result = out[:n]
    if n == 0:
        return result
    result[:] = 0.0
    masks = np.empty((n, E.N_PLANES), dtype=np.uint64)
    opp = not self_color
    for i, move in enumerate(moves):
        board.push(move)
        try:
            j = 0
            for color in (self_color, opp):
                for piece_type in E.PIECE_ORDER:
                    masks[i, j] = board.pieces_mask(piece_type, color)
                    j += 1
            result[i, E.OFF_STM] = 1.0 if board.turn == self_color else 0.0
            result[i, E.OFF_CASTLE] = float(board.has_kingside_castling_rights(self_color))
            result[i, E.OFF_CASTLE + 1] = float(board.has_queenside_castling_rights(self_color))
            result[i, E.OFF_CASTLE + 2] = float(board.has_kingside_castling_rights(opp))
            result[i, E.OFF_CASTLE + 3] = float(board.has_queenside_castling_rights(opp))
            if board.ep_square is not None:
                result[i, E.OFF_EP + chess.square_file(board.ep_square)] = 1.0
            # Exactly the existing Python float division, then float32 assignment.
            result[i, E.OFF_HALFMOVE] = min(board.halfmove_clock, 100) / 100.0
        finally:
            board.pop()
    bits = np.unpackbits(masks.view(np.uint8), axis=1, bitorder="little")
    if self_color == chess.BLACK:
        # Vertical board flip: reverse the eight ranks inside each piece plane.
        bits = bits.reshape(n, E.N_PLANES, 8, 8)[:, :, ::-1, :].reshape(n, E.PLANE_SIZE)
    result[:, :E.PLANE_SIZE] = bits
    return result


SPECIAL_FENS = [
    ("white_promotions", "1r5k/P7/8/8/8/8/8/7K w - - 0 1"),
    ("black_promotions", "7k/8/8/8/8/8/p7/1R5K b - - 0 1"),
    ("white_ep_castles", "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 7 20"),
    ("black_ep_castles", "r3k2r/8/8/8/3Pp3/8/8/R3K2R b KQkq d3 7 20"),
    ("checkmate", "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"),
    ("stalemate", "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"),
]


def board_signature(board):
    return (
        board.fen(en_passant="fen", promoted=True),
        tuple(board.move_stack),
        tuple(id(state) for state in board._stack),
        board.chess960,
    )


def verify(random_plies):
    digest = hashlib.sha256()
    counts = {"comparisons": 0, "candidate_rows": 0, "random_positions": 0}
    reuse = np.empty((256, E.INPUT_SIZE), dtype=np.float32)
    strided = np.empty((256, E.INPUT_SIZE * 2), dtype=np.float32)[:, ::2]

    def check(board, label, out=None):
        moves = list(board.legal_moves)
        before = board_signature(board)
        for color in (chess.WHITE, chess.BLACK):
            expected = E.encode_after_moves(board, moves, color)
            assert board_signature(board) == before, (label, "reference changed board")
            actual = encode_after_moves_batched(board, moves, color, out)
            assert actual.dtype == np.float32 and actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes(), (label, color)
            assert E.pack_many(actual).tobytes() == E.pack_many(expected).tobytes()
            assert board_signature(board) == before, (label, "prototype changed board")
            assert list(board.legal_moves) == moves, (label, "move order changed")
            digest.update(actual.tobytes())
            counts["comparisons"] += 1
            counts["candidate_rows"] += len(moves)

    corpus = []
    for label, fen, _ in GOLDEN_FENS:
        for halfmove in [*range(101), 125, 200]:
            board = chess.Board(fen)
            board.halfmove_clock = halfmove
            check(board, (label, halfmove), reuse)
        corpus.append(chess.Board(fen))
    for label, fen in SPECIAL_FENS:
        board = chess.Board(fen)
        for buffer in (None, reuse, strided, np.empty((1, E.INPUT_SIZE), dtype=np.float32)):
            check(board, label, buffer)
        corpus.append(board)

    rng = np.random.default_rng(20260922)
    board = chess.Board()
    for ply in range(random_plies):
        check(board, ("random", ply), reuse)
        counts["random_positions"] += 1
        if ply % 10 == 0:
            corpus.append(board.copy())
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            board = chess.Board()
        else:
            board.push(moves[int(rng.integers(len(moves)))])
    counts["float_and_packed_byte_identical"] = True
    counts["board_and_history_restored"] = True
    counts["sha256"] = digest.hexdigest()
    return counts, corpus


def benchmark(corpus, repeats, loops):
    cases = [(board, list(board.legal_moves), color)
             for board in corpus for color in (chess.WHITE, chess.BLACK)]
    buf = np.empty((256, E.INPUT_SIZE), dtype=np.float32)
    functions = {"reference": E.encode_after_moves, "batched": encode_after_moves_batched}
    timings = {name: [] for name in functions}
    for fn in functions.values():
        for board, moves, color in cases:
            fn(board, moves, color, buf)
    for block in range(repeats):
        # Alternate order so one implementation does not always run first.
        names = ("reference", "batched") if block % 2 == 0 else ("batched", "reference")
        for name in names:
            fn = functions[name]
            start = time.perf_counter()
            for _ in range(loops):
                for board, moves, color in cases:
                    fn(board, moves, color, buf)
            timings[name].append((time.perf_counter() - start) * 1000.0 / (loops * len(cases)))
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "cases": len(cases),
        "candidate_rows_per_pass": sum(len(moves) for _, moves, _ in cases),
        "repeats": repeats,
        "loops_per_block": loops,
        "milliseconds_per_candidate_batch": medians,
        "blocks_ms": timings,
        "encoder_speedup": medians["reference"] / medians["batched"],
        "scope": "encoder only, model/learner/opponent unchanged; not games/s",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--random-plies", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--loops", type=int, default=5)
    args = parser.parse_args()
    if args.random_plies < 0 or args.repeats < 1 or args.loops < 1:
        parser.error("random-plies must be nonnegative; repeats and loops must be positive")
    checks, corpus = verify(args.random_plies)
    output = {"equivalence": checks}
    if args.benchmark:
        output["benchmark"] = benchmark(corpus, args.repeats, args.loops)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
