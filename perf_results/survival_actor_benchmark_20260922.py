"""Actor-only benchmark; run on an otherwise idle machine for comparison.

Example: python perf_results/survival_actor_benchmark_20260922.py --output new.json
This creates fresh networks, plays real games, and never trains or stops a process.
Stockfish's internal randomness is not seedable, so lengths/timings will vary.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tonkai.survival_config import SurvivalConfig
from tonkai.survival_pool import SurvivalActorPool
from tonkai.temporal_model import SurvivalNet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ROOT / "engines/stockfish-11/stockfish-11-win/Windows/stockfish_20011801_x64.exe")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=100)
    parser.add_argument("--net-seed", type=int, default=20260922)
    parser.add_argument("--game-seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional new JSON report; the original measurement is not overwritten by default")
    args = parser.parse_args(argv)
    if args.games < 1 or args.max_plies < 1 or any(not 1 <= count <= 8 for count in args.workers):
        parser.error("games/max-plies must be positive, and workers must be between 1 and 8")
    torch.set_num_threads(1)
    rows = []
    metadata = None
    for workers in args.workers:
        cfg = SurvivalConfig(max_plies=args.max_plies, num_workers=workers,
                             win_pgn_path="", opponent_elo=1600, opponent_move_time=0.02)
        torch.manual_seed(args.net_seed)
        net = SurvivalNet(cfg).cpu().eval()
        state = net.state_dict()
        checksum = hashlib.sha256(b"".join(t.detach().numpy().tobytes() for t in state.values())).hexdigest()
        games = []
        pool = SurvivalActorPool(cfg, args.engine.resolve(), workers)
        metadata = pool.worker_info[0]["engine"]
        started = time.perf_counter()
        next_game = 0
        seeds = {}
        try:
            for worker_id in pool.idle_ids:
                if next_game == args.games:
                    break
                seeds[worker_id] = args.game_seed + next_game
                pool.submit(worker_id, state, "external", None, "Stockfish", seeds[worker_id])
                next_game += 1
            while len(games) < args.games:
                worker_id, episode, board, self_color, stats = pool.receive(timeout=120)
                games.append(dict(worker=worker_id, seed=seeds[worker_id],
                                  self_color="white" if self_color else "black", **stats))
                if next_game < args.games:
                    seeds[worker_id] = args.game_seed + next_game
                    pool.submit(worker_id, state, "external", None, "Stockfish", seeds[worker_id])
                    next_game += 1
            elapsed = time.perf_counter() - started
        finally:
            closed = time.perf_counter()
            pool.close()
            close_seconds = time.perf_counter() - closed
        plies = sum(game["plies"] for game in games)
        row = dict(workers=workers, games=args.games, max_plies=args.max_plies,
                   startup_seconds=pool.startup_seconds, actor_wall_seconds=elapsed,
                   games_per_second=args.games / elapsed, plies=plies,
                   plies_per_second=plies / elapsed, close_seconds=close_seconds,
                   initial_model_sha256=checksum, details=games)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  cpu_count=os.cpu_count(), torch_version=torch.__version__, torch_threads_parent=1,
                  engine=metadata, learner_updates=0, net_seed=args.net_seed,
                  first_game_seed=args.game_seed, conditions=rows,
                  note="Fresh untrained seeded networks; Stockfish randomness is not seedable. "
                       "Actor-only timings exclude startup and shutdown; no full training throughput claim.")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
