"""A/B microbenchmark of G16's strict terminal checks, same real positions.

Run after building claudio_fast:
    python claudio/bench_terminal_batch.py --positions 128 --repeats 4

This measures the terminal-check hot path only, not full G16 decision time,
and does NOT establish playing-strength equivalence by itself.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: F401, E402
import chess  # noqa: E402
import chess.pgn  # noqa: E402

from claudio_agent import native_board  # noqa: E402


def positions_from_games(limit):
    pgn = Path(__file__).resolve().parent / "results/matches/g16-vs-g15-100-bo30.pgn"
    selected = []
    with pgn.open("r", encoding="utf-8") as stream:
        while len(selected) < limit:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                # Include opening, middlegame and final, without timing replay.
                if ply >= 4 and ply % 4 == 0 and not board.is_game_over():
                    fast = native_board(board)
                    candidates = list(fast.legal_moves())
                    if len(candidates) > 1:
                        selected.append((fast, candidates))
                        if len(selected) == limit:
                            break
                board.push(move)
    if not selected:
        raise RuntimeError(f"No positions found in {pgn}")
    if not hasattr(selected[0][0], "terminal_after_moves"):
        raise RuntimeError("Rebuild claudio_fast from the optimized source first")
    return selected


def original(fast, codes):
    result = []
    for code in codes:
        fast.push(code)
        try:
            result.append(fast.terminal_strict())
        finally:
            fast.pop()
    return result


def batched(fast, codes):
    return fast.terminal_after_moves(codes)


def measure(rows, operation):
    start = time.perf_counter()
    for fast, codes in rows:
        operation(fast, codes)
    return (time.perf_counter() - start) / len(rows) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if args.positions < 1 or args.repeats < 1:
        parser.error("positions and repeats must be positive")
    rows = positions_from_games(args.positions)
    for fast, codes in rows:
        before = fast.fen()
        a = original(fast, codes)
        b = batched(fast, codes)
        if a != b or fast.fen() != before:
            raise AssertionError("Native batch diverges or mutates the board")
    # A/B/B/A, keeping the same positions and move order in each run.
    old, new = [], []
    for _ in range(args.repeats):
        old.append(measure(rows, original))
        new.append(measure(rows, batched))
        new.append(measure(rows, batched))
        old.append(measure(rows, original))
    a, b = statistics.median(old), statistics.median(new)
    print(f"positions={len(rows)} repeats={args.repeats} exact_statuses=True")
    print(f"original={a:.4f} ms/position batched={b:.4f} ms/position")
    print(f"speedup={(a / b):.3f}x; time_saved={(1 - b / a) * 100:.2f}% on this hot path")
    print("Full G16 latency and playing quality still require separate validation.")


if __name__ == "__main__":
    main()
