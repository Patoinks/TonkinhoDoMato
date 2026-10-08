"""Real Claudio G16 checkpoint: exact move equivalence and ABBA timing.

Tests the current native-batch path against the *unchanged original Python
push/terminal_strict/pop implementation* on saved BO30 positions. Neither
checkpoint nor playing parameters are edited. The two agents see identical
network weights, GRU history and seeded RNG streams.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: F401,E402
import chess  # noqa: E402
import chess.pgn  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from claudio_agent import Claudio, ClaudioConfig  # noqa: E402
from claudio_load import load_claudio  # noqa: E402


class LegacyTerminalBoard:
    """Force Claudio to use its previous per-candidate terminal checks."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        if name == "terminal_after_moves":
            raise AttributeError(name)
        return getattr(self.original, name)


def real_positions(limit):
    pgn = Path(__file__).resolve().parent / "results/matches/g16-vs-g15-100-bo30.pgn"
    all_rows = []
    with pgn.open("r", encoding="utf-8") as stream:
        game_id = 0
        while len(all_rows) < limit:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            if game.errors:
                raise RuntimeError(f"Invalid PGN game {game_id}: {game.errors}")
            board = game.board()
            selected_this_game = 0
            for ply, move in enumerate(game.mainline_moves()):
                if (ply >= 4 and ply % 3 == 0 and not board.is_game_over()
                        and selected_this_game < 16):
                    all_rows.append((game_id, board.copy(stack=True)))
                    selected_this_game += 1
                    if len(all_rows) == limit:
                        break
                board.push(move)
            game_id += 1
    if len(all_rows) < limit:
        raise RuntimeError(f"Only {len(all_rows)} positions available; requested {limit}")
    return all_rows


def evaluate(rows, loaded, legacy):
    net, values, config, king = loaded
    players = {}
    previous_game = None
    result = []
    duration = []
    for game_id, board in rows:
        if previous_game != game_id:
            players = {
                side: Claudio(net, values, ClaudioConfig.with_settings([], base=config),
                              rng=np.random.default_rng(321 + side), king_model=king)
                for side in (chess.WHITE, chess.BLACK)
            }
            previous_game = game_id
        player = players[board.turn]
        # Advance the GRU only once per real position; synchronize outside
        # of the measured choose() call identically for the two implementations.
        player._catch_up(board)
        if legacy and not isinstance(player.fast, LegacyTerminalBoard):
            player.fast = LegacyTerminalBoard(player.fast)
        begin = time.perf_counter()
        action = player.choose(board)
        duration.append((time.perf_counter() - begin) * 1000.0)
        if action not in board.legal_moves:
            raise RuntimeError(f"Illegal G16 move: {action} at {board.fen()}")
        result.append(action.uci())
    return result, duration


def metrics(times):
    times = sorted(times)
    n = len(times)
    # nearest-rank P95; avoids an interpolated value not actually observed.
    return {
        "mean_ms": sum(times) / n,
        "median_ms": statistics.median(times),
        "p95_ms": times[min(n - 1, (95 * n + 99) // 100 - 1)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.positions < 1 or args.repeats < 1:
        parser.error("positions and repeats must be > 0")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ckpt = Path(__file__).resolve().parent / "claudio-g16.pt"
    net, values, config, _, king = load_claudio(ckpt, with_config=True, with_king=True)
    if not config.get("critical_depth") or not config.get("endgame_pieces"):
        raise RuntimeError("Expected G16 checkpoint with critical and endgame modules")
    loaded = (net, values, config, king)
    rows = real_positions(args.positions)
    t = defaultdict(list)
    original_moves = None
    optim_moves = None
    for block in range(args.repeats):
        for label in ("original", "batch", "batch", "original"):
            moves, ms = evaluate(rows, loaded, legacy=(label == "original"))
            t[label].extend(ms)
            if label == "original":
                if original_moves is None:
                    original_moves = moves
                elif original_moves != moves:
                    raise AssertionError("Baseline G16 decisions are non-deterministic")
            else:
                if optim_moves is None:
                    optim_moves = moves
                elif optim_moves != moves:
                    raise AssertionError("Optimized G16 decisions are non-deterministic")
            mismatch = [(i, a, b) for i, (a, b) in enumerate(zip(original_moves, moves))
                        if a != b]
            if mismatch:
                raise AssertionError(f"G16 moves changed ({len(mismatch)}): {mismatch[:8]}")
            print(f"block={block + 1} path={label:8s} positions={len(rows)} "
                  f"mean={statistics.mean(ms):.3f}ms", flush=True)
    a, b = metrics(t["original"]), metrics(t["batch"])
    print(f"EXACT G16 move equivalence: {len(rows)} positions x {2 * args.repeats} "
          f"runs per path, 0 changed moves")
    print(f"BASELINE:  mean={a['mean_ms']:.3f}ms "
          f"median={a['median_ms']:.3f}ms p95={a['p95_ms']:.3f}ms")
    print(f"OPTIMIZED: mean={b['mean_ms']:.3f}ms "
          f"median={b['median_ms']:.3f}ms p95={b['p95_ms']:.3f}ms")
    print(f"Observed sample speed change: {(a['mean_ms'] / b['mean_ms'] - 1) * 100:+.2f}%")
    print("Zero mismatches in this sample do not prove a <=2% Elo/winrate loss universally.")


if __name__ == "__main__":
    main()
