"""Match two sets of decision settings on the SAME network (e.g. v4.1 vs v4).

Colours alternate; every game is its own parallel task; nothing is trained.

    python scripts/match_settings.py --set memory_weight=0.22 --set temperature=0.05 \\
        --name "Tonkinho do Mato v4.1" --games 500 --workers 6 --label v4.1-vs-v4-500
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from evolve_tonkinho import GENES, _final_game, _init, base_genes  # noqa: E402
from match_checkpoints import _historical_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "checkpoints/cognitive-mix-20260923/latest.pt")
    parser.add_argument("--set", action="append", default=[], help="setting=value for the new side")
    parser.add_argument("--name", default="new settings")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    data = torch.load(args.base, map_location="cpu", weights_only=False)
    base = base_genes(_historical_config(data))
    del data
    new = dict(base)
    for item in args.set:
        name, _, value = item.partition("=")
        if name not in GENES:
            parser.error(f"unknown setting {name}")
        new[name] = int(value) if GENES[name][0] == "int" else float(value)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(str(args.base),)) as pool:
        futures = [pool.submit(_final_game, new, base, game, [args.seed, 7000]) for game in range(args.games)]
        scores = []
        for future in futures:
            scores.append(future.result())
            if len(scores) % 100 == 0:
                print(f"{len(scores)}/{args.games} jogos", flush=True)
    mean = sum(scores) / len(scores)
    se = math.sqrt(sum((x - mean) ** 2 for x in scores) / max(1, len(scores) - 1) / len(scores))
    white = scores[0::2]
    black = scores[1::2]
    report = {"current": args.name, "rival": "Tonkinho do Mato v4 (base settings)", "base_file": str(args.base),
              "changed": {k: [base[k], new[k]] for k in new if new[k] != base[k]}, "games": len(scores),
              "win": scores.count(1.0), "draw": scores.count(0.5), "loss": scores.count(0.0), "score": mean,
              "score_95": [max(0, mean - 1.96 * se), min(1, mean + 1.96 * se)],
              "elo_difference": None if mean in (0, 1) else -400 * math.log10(1 / mean - 1),
              "as_white": sum(white) / len(white), "as_black": sum(black) / len(black), "seed": args.seed,
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = ROOT / "history/matches" / f"{args.label}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
