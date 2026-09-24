"""Group stage: every champion of an evolution (and its base) against every
other, N games per pair (colours alternate), same network, nothing trained.
Owner's scoring: win 1, draw 0.4, loss 0.

    python scripts/group_stage.py --evolution history/evolution/<run>/evolution.json \\
        --base checkpoints/obj-max-continuo/latest.pt --games 10 --label group-timdgei
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
import json
from multiprocessing import get_context
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from evolve_tonkinho import _final_game, _init  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evolution", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="the network all players share")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--draw", type=float, default=0.4, help="points for a draw")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--set", action="append", default=[], help="setting=value for every player (e.g. tactic_search=exact)")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    log = json.loads(args.evolution.read_text(encoding="utf-8"))
    base_name = log.get("start", {}).get("player") or next(
        (p["id"] for g in log["generations"] for p in g["population"] if p.get("parent") is None), "base")
    players = {base_name: dict(log["base_genes"])}
    for generation in log["generations"]:
        champion = next(p for p in generation["population"] if p["id"] == generation["champion"])
        players.setdefault(champion["id"], dict(champion["genes"]))
    extra = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        extra[key] = raw
    for genes in players.values():
        genes.update(extra)
    table = {name: {"played": 0, "win": 0, "draw": 0, "loss": 0, "stalled": 0, "points": 0.0} for name in players}
    pairs, jobs = list(combinations(players, 2)), []
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(str(args.base),)) as pool:
        for index, (a, b) in enumerate(pairs):
            for game in range(args.games):
                jobs.append((a, b, pool.submit(_final_game, players[a], players[b], game, [args.seed, index])))
        results = []
        for a, b, future in jobs:
            score = future.result()
            results.append({"white_first": a, "other": b, "score_first": score})
            for name, mine in ((a, score), (b, score if score < 0 else 1 - score)):
                row = table[name]
                row["played"] += 1
                if mine < 0:
                    row["stalled"] += 1
                elif mine == 1:
                    row["win"] += 1
                    row["points"] += 1
                elif mine == 0:
                    row["loss"] += 1
                else:
                    row["draw"] += 1
                    row["points"] += args.draw
    ranking = sorted(table.items(), key=lambda item: -item[1]["points"])
    report = {"evolution": str(args.evolution), "base": str(args.base), "games_per_pair": args.games,
              "draw_points": args.draw, "settings_for_all": extra, "players": players,
              "table": dict(ranking), "games": results,
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = ROOT / "history" / f"{args.label}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for place, (name, row) in enumerate(ranking, start=1):
        print(f"{place}. {name:10s} {row['points']:5.1f} pts  V/E/D {row['win']}/{row['draw']}/{row['loss']}", flush=True)


if __name__ == "__main__":
    main()
