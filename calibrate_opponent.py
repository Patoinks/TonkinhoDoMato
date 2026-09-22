"""Measure TinyOpponent against itself, and against a random mover.

Target is roughly 1300 Elo. This does not reach an absolute rating - that needs
an external reference engine - but it gives the two things V0 actually needs:

  * relative Elo between settings, so strength can be dialled
  * a sanity floor: the configured opponent must crush a random mover

    python calibrate_opponent.py
    python calibrate_opponent.py --games 200 --settings 1,2,3
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import time
import json
from pathlib import Path
import shutil

import chess

import tiny_opponent
from tonkai import chess_env as env

MAX_PLIES = 300


def elo_diff(score: float) -> float:
    score = min(max(score, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / score - 1.0)


def play(white, black, max_plies: int = MAX_PLIES) -> float:
    """1.0 white wins, 0.5 draw, 0.0 black wins."""
    board = chess.Board()
    while True:
        if board.is_checkmate():
            return 0.0 if board.turn == chess.WHITE else 1.0
        if (not any(board.generate_legal_moves())
                or board.is_insufficient_material()
                or board.halfmove_clock >= 100
                or (board.halfmove_clock >= 4 and board.is_repetition(3))
                or len(board.move_stack) >= max_plies):
            return 0.5
        mover = white if board.turn == chess.WHITE else black
        env.apply_opponent_move(board, mover(board))


def engine_mover(eng):
    return lambda board: eng.choose_move(board.fen())


def random_mover(rng):
    def f(board):
        return rng.choice(list(board.legal_moves)).uci()
    return f


def make(depth: int, nodes: int, randomness: float, seed: int, quiesce: bool):
    return tiny_opponent.TinyOpponent(
        depth=depth, max_nodes=nodes, randomness=randomness,
        seed=seed, quiescence=quiesce,
    )


def match(a, b, games: int, label_a: str, label_b: str) -> float:
    """Score for `a`, colours alternated."""
    total = 0.0
    for g in range(games):
        if g % 2 == 0:
            total += play(engine_mover(a), engine_mover(b))
        else:
            total += 1.0 - play(engine_mover(b), engine_mover(a))
    score = total / games
    d = elo_diff(score)
    print(f"  {label_a:28s} vs {label_b:28s}  "
          f"score {score:.3f}  ({d:+.0f} Elo)")
    return score


def anchor_match(args):
    """Optional measurement of TinyOpponent only. Never imported by training."""
    executable = shutil.which(args.anchor)
    if executable is None and Path(args.anchor).is_file():
        executable = str(Path(args.anchor).resolve())
    if executable is None:
        print(f"Anchor skipped: reference not found: {args.anchor}")
        return None
    if args.anchor_elo is None:
        raise ValueError("--anchor-elo must state the reference's known rating at these settings")
    import chess.engine

    tiny = make(int(args.settings.split(',')[0]), args.nodes, args.randomness, args.seed, True)
    scores = []
    with chess.engine.SimpleEngine.popen_uci(executable) as reference:
        reference.configure(json.loads(args.anchor_options))
        for g in range(args.games):
            # A fresh token tells the reference this is a new game.
            token = object()

            def reference_move(board):
                return reference.play(board, chess.engine.Limit(time=args.anchor_time), game=token).move.uci()

            score = (play(engine_mover(tiny), reference_move) if g % 2 == 0
                     else 1 - play(reference_move, engine_mover(tiny)))
            scores.append(score)
    score = sum(scores) / len(scores)
    wins, draws, losses = scores.count(1.0), scores.count(0.5), scores.count(0.0)
    # Wilson interval for fractional game scores: an approximate sampling range,
    # not a claim that the supplied reference rating is universally transferable.
    n, z = len(scores), 1.96
    centre = (score + z*z/(2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(score*(1-score)/n + z*z/(4*n*n)) / (1 + z*z/n)
    estimate = args.anchor_elo + elo_diff(score)
    print(f"Anchor: {executable}, supplied Elo {args.anchor_elo}, {args.anchor_time}s/move")
    print(f"Options: {args.anchor_options}; seed {args.seed}; TinyOpponent depth "
          f"{args.settings.split(',')[0]}, nodes {args.nodes}, randomness {args.randomness}")
    print(f"W/D/L {wins}/{draws}/{losses}; score {score:.3f}; conditional Elo estimate {estimate:.0f}")
    print(f"Approximate 95% sampling range: {args.anchor_elo + elo_diff(centre-radius):.0f} to "
          f"{args.anchor_elo + elo_diff(centre+radius):.0f}")
    if score in (0, 1):
        print("All games had the same result: the displayed estimate is clipped, not a finite measured rating.")
    print("Reference rating must be valid for these exact options and time controls. "
          "This measures TinyOpponent only; no reference output enters training.")
    return estimate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--settings", type=str, default="1,2,3",
                    help="comma separated depths to compare")
    ap.add_argument("--nodes", type=int, default=1000)
    ap.add_argument("--randomness", type=float, default=0.35)
    ap.add_argument("--speed-only", action="store_true")
    ap.add_argument("--anchor", help="optional rated UCI reference executable; runs only the anchor match")
    ap.add_argument("--anchor-elo", type=float, help="known reference rating at the configured conditions")
    ap.add_argument("--anchor-time", type=float, default=0.05, help="reference seconds per move")
    ap.add_argument("--anchor-options", default="{}", help="JSON UCI settings for the rated reference")
    args = ap.parse_args()
    if args.games <= 0 or args.anchor_time <= 0:
        ap.error("--games and --anchor-time must be positive")
    if args.anchor:
        try:
            anchor_match(args)
        except (ValueError, OSError) as error:
            ap.error(str(error))
        return 0

    depths = [int(s) for s in args.settings.split(",") if s.strip()]
    rng = random.Random(args.seed)

    # ------------------------------------------------------------- speed
    print("Speed (single move from the opening position, warm object):")
    for d in depths:
        eng = make(d, args.nodes, args.randomness, args.seed, True)
        board = chess.Board()
        t0 = time.perf_counter()
        n = 3000
        fen = board.fen()
        for _ in range(n):
            eng.choose_move(fen)
        dt = time.perf_counter() - t0
        print(f"  depth {d}  nodes<={args.nodes}  "
              f"{n / dt:8.0f} moves/s   {1e6 * dt / n:6.1f} us/move")
    if args.speed_only:
        return 0

    # --------------------------------------------------------- vs random
    print(f"\nSanity floor - {args.games} games vs a uniform random mover:")
    for d in depths:
        eng = make(d, args.nodes, args.randomness, args.seed, True)
        rmover = random_mover(rng)
        total = 0.0
        for g in range(args.games):
            if g % 2 == 0:
                total += play(engine_mover(eng), rmover)
            else:
                total += 1.0 - play(rmover, engine_mover(eng))
        score = total / args.games
        print(f"  depth {d:<2d} score {score:.3f}  ({elo_diff(score):+.0f} Elo)")

    # ---------------------------------------------- relative ladder
    print(f"\nRelative strength - {args.games} games per pair:")
    base = make(depths[0], args.nodes, args.randomness, args.seed, True)
    for d in depths[1:]:
        other = make(d, args.nodes, args.randomness, args.seed + 1, True)
        match(other, base, args.games, f"depth {d}", f"depth {depths[0]}")

    print(f"\nEffect of randomness (depth {depths[0]}, "
          f"{args.games} games vs randomness=0):")
    sharp = make(depths[0], args.nodes, 0.0, args.seed, True)
    for r in (0.15, 0.35, 0.7):
        noisy = make(depths[0], args.nodes, r, args.seed + 2, True)
        match(noisy, sharp, args.games, f"randomness {r}", "randomness 0.0")

    print(f"\nEffect of quiescence (depth {depths[0]}):")
    noq = make(depths[0], args.nodes, args.randomness, args.seed + 3, False)
    match(noq, base, args.games, "quiescence off", "quiescence on")

    print(f"\nEffect of the node cap (depth {max(depths)}):")
    for n in (100, 300, 1000):
        capped = make(max(depths), n, args.randomness, args.seed + 4, True)
        match(capped, base, args.games, f"depth {max(depths)} nodes {n}",
              f"depth {depths[0]} nodes {args.nodes}")

    print("\nNote: these are relative numbers. Anchoring to ~1300 Elo needs an\n"
          "external reference (a rated engine or a Lichess bot account).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
