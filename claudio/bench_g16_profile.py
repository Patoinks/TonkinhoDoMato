"""Profile real G16 decisions by phase; no network weights or settings change.

    python claudio/bench_g16_profile.py --positions 128
Prints the total cost of every timed region (including 0 for unused stages).
This reports the cost *inside choose*, not checkout, loading or PGN replay.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401
import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench_g16_decisions import real_positions  # noqa: E402
from claudio_agent import Claudio, ClaudioConfig  # noqa: E402
from claudio_load import load_claudio  # noqa: E402


def profile(rows):
    ckpt = Path(__file__).resolve().parent / "claudio-g16.pt"
    net, values, cfg, _, king = load_claudio(ckpt, with_config=True, with_king=True)
    by_stage = defaultdict(float)
    by_group = defaultdict(lambda: {"count": 0, "ms": 0.0})
    slowest = []
    original_game = None
    count = 0
    for game_id, board in rows:
        if original_game != game_id:
            players = {side: Claudio(net, values, ClaudioConfig.with_settings([], base=cfg),
                                     rng=np.random.default_rng(321 + int(side)), king_model=king)
                       for side in (chess.WHITE, chess.BLACK)}
            original_game = game_id
        player = players[board.turn]
        player.clock = {}
        old_critical = getattr(player, "critical_moves", 0)
        start = time.perf_counter()
        move = player.choose(board)
        spent = (time.perf_counter() - start) * 1000
        if move not in board.legal_moves:
            raise AssertionError(f"Illegal move: {move}, {board.fen()}")
        count += 1
        measured = sum(player.clock.values()) * 1000
        for name, elapsed in player.clock.items():
            by_stage[name] += elapsed * 1000
        by_stage["untimed"] += max(0.0, spent - measured)
        by_stage["TOTAL"] += spent
        if chess.popcount(board.occupied) <= cfg["endgame_pieces"]:
            group = "endgame"
        elif getattr(player, "critical_moves", 0) > old_critical:
            group = "critical"
        else:
            group = "normal"
        by_group[group]["count"] += 1
        by_group[group]["ms"] += spent
        slowest.append((spent, group, board.fen(), move.uci(),
                        dict(sorted(player.clock.items(), key=lambda x: -x[1])[:3])))
    print(f"PROFILE decisions={count} checkpoint=claudio-g16.pt", flush=True)
    print("SECTION  ms_decision  percent_total", flush=True)
    for name, ms in sorted(by_stage.items(), key=lambda item: -item[1]):
        if name == "TOTAL":
            continue
        print(f"{name:17s} {ms / count:10.3f}  {100 * ms / by_stage['TOTAL']:9.2f}%", flush=True)
    print(f"TOTAL             {by_stage['TOTAL'] / count:10.3f}     100.00%", flush=True)
    for name, row in sorted(by_group.items()):
        print(f"GROUP {name} decisions={row['count']} avg_ms={row['ms']/row['count']:.3f}", flush=True)
    for spent, group, fen, move, stages in sorted(slowest, reverse=True)[:6]:
        print(f"SLOW {group} {spent:.2f}ms {move} {fen} "
              f"top={[(k, round(1000 * v, 1)) for k, v in stages.items()]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=128)
    args = parser.parse_args()
    if args.positions < 1:
        parser.error("positions must be >0")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    profile(real_positions(args.positions))
