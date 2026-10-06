"""A checkpoint (or white.pt+black.pt) on one colour against an opponent, from the normal
start, no exploration - exactly the training's fixed evaluation, for any checkpoint.

    python claudio/eval_colour.py --checkpoint claudio/claudio-grande-g0.pt --colour white --opponent fairy1450
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

import train_grande as tg  # noqa: E402


def _init(checkpoint, colour):
    import torch
    from claudio_load import load_claudio
    path = checkpoint.split("+")[0 if colour == "white" or "+" not in checkpoint else 1]
    net, values, config, _, _ = load_claudio(path, with_config=True, with_king=True)
    arch = torch.load(path, map_location="cpu", weights_only=False)["architecture"]
    out = ROOT / "claudio/results/eval-colour-tmp"
    out.mkdir(parents=True, exist_ok=True)
    weights = out / f"actor-{abs(hash(path)) % 10 ** 8}.pt"
    if not weights.exists():
        torch.save(net.state_dict(), weights)
    tg._init_worker(str(out), arch, values, config, path, None, "victory", 0.002)
    tg._w["version_file"] = weights


def play(index, seed, stage, colour, max_plies):
    import torch
    if tg._w["version"] != "fixed":
        tg._w["net"].load_state_dict(torch.load(tg._w["version_file"], map_location="cpu"))
        tg._w["version"] = "fixed"
    return tg.play(index, "fixed", seed, stage, 0, max_plies, 0, 0.0, colour, False)["result"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--colour", choices=("white", "black"), required=True)
    parser.add_argument("--opponent", default="fairy1450")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1007)
    parser.add_argument("--max-plies", type=int, default=300)
    args = parser.parse_args()
    stage = tg.parse_stage(args.opponent)
    pool = ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                               initargs=(args.checkpoint, args.colour))
    n = args.games
    results = list(pool.map(play, range(n), [args.seed] * n, [stage] * n, [args.colour] * n, [args.max_plies] * n))
    w, d, l = results.count("win"), results.count("draw"), results.count("loss")
    print(json.dumps({"checkpoint": args.checkpoint, "colour": args.colour, "opponent": stage[2],
                      "win_rate": w / n, "wins": w, "draws": d, "losses": l, "truncated": results.count("truncated")}),
          flush=True)
    for process in list(getattr(pool, "_processes", {}).values()):
        process.kill()                        # the engines' threads would keep the workers alive
    pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
