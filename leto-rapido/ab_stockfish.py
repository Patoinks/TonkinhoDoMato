"""Configurations against Stockfish, their games interleaved in one pool: the machine's load (which moves
Stockfish's strength at a few ms per move) falls on all of them alike.

    python leto-rapido/ab_stockfish.py --games 500 m039:safety_weight=4.739,... other:...
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mega_torneio as M  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="name:setting,setting (on top of the fast Leto)")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--elo", type=int, default=2250)
    parser.add_argument("--movetime", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--pin", action="store_true", help="each worker on its own core (see mega_torneio._pin)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    configs = {}
    for spec in args.configs:
        name, _, sets = spec.partition(":")
        configs[name] = [s for s in sets.split(",") if s]
    tasks = [(0, name, sets, g, args.seed, 300) for g in range(args.games) for name, sets in configs.items()]
    rows = []
    context = get_context("spawn")
    initargs = (args.elo, args.movetime) + ((context.Value("i", 0), args.workers) if args.pin else ())
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=M._init, initargs=initargs) as pool:
        for future in as_completed([pool.submit(M.play, t) for t in tasks]):
            rows.append(future.result())
    print("| configuração | jogos | score | ±95% | V/E/D | s/jogo | ms/lance | erros |")
    print("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    for name in configs:
        rs = [r for r in rows if r["id"] == name]
        s = sum(r["score"] for r in rs) / len(rs)
        w = sum(r["score"] == 1 for r in rs); d = sum(r["score"] == 0.5 for r in rs)
        print(f"| {name} | {len(rs)} | {s:.3f} | {1.96 * math.sqrt(s * (1 - s) / len(rs)):.3f} | {w}/{d}/{len(rs) - w - d} | "
              f"{sum(r['seconds'] for r in rs) / len(rs):.2f} | {sum(r.get('leto_ms', 0) for r in rs) / len(rs):.1f} | "
              f"{sum(r['end'] == 'error' for r in rs)} |")
    if args.out:
        args.out.write_text(json.dumps({"configs": configs, "games": rows}), encoding="utf-8")


if __name__ == "__main__":
    main()
