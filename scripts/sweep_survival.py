"""Survival-weight sweep: short trainings from one start, each evaluated alike.

For every survival share (10%..90%) train N games against TinyOpponent with the
shaped objective (the rest of the value = the owner's measure, here simplify +
material one horizon ahead), then play the same fixed games against
TinyOpponent without exploration. Results go to history/sweep-survival.json as
they arrive. Stop between runs: create history/sweep-survival.stop.

    python scripts/sweep_survival.py --games 100
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_duel  # noqa: E402

FLAGS = subprocess.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0


def train(label, games, workers, start, sets):
    genes = [f"{k}={v}" for k, v in train_duel.genes_of("G5-M23").items()] + train_duel.RULES
    genes += [f"syzygy_path={ROOT / 'engines/syzygy'}", "tablebase_adjudicate=true",
              "mate_attack_ks=0.9", "mate_defense_ks=0.9", "break_repetitions=true"] + sets
    cmd = [sys.executable, "train_cognitive.py", "--resume", str(start), "--new-campaign",
           "--checkpoint-dir", f"checkpoints/obj-{label}", "--games", str(games), "--game-cap", str(games),
           "--workers", str(workers), "--checkpoint-every", str(games),
           "--mix-self", "0", "--mix-tiny", "1", "--mix-fairy", "0",
           "--opponent-path", train_duel.ENGINE, "--opponent-elo", "1350", "--opponent-move-time", "0.002",
           "--objective", "shaped", "--kd-weight", "0.75", "--capture-target-weight", "0.05",
           "--mate-search-moves", "3", "--mate-defense-moves", "2", "--threat-explore", "0.1",
           "--mate-labels", "2", "--trade-explore", "0.1", "--exchange-labels", "1", "--batch-search",
           "--stop-file", f"history/obj-{label}.stop"]
    for item in genes:
        cmd += ["--set", item]
    with open(ROOT / f"history/obj-{label}.log", "w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS, check=True)
    last = [line for line in (ROOT / f"history/obj-{label}.log").read_text(encoding="utf-8").splitlines()
            if line.startswith("{")][-1]
    return json.loads(last)


def evaluate(label, games, workers, seed):
    out = subprocess.run([sys.executable, "scripts/match_engine.py", "--checkpoint", f"checkpoints/obj-{label}/latest.pt",
                          "--tiny", "--greedy", "--games", str(games), "--workers", str(workers), "--seed", str(seed),
                          "--label", f"eval-tiny-obj-{label}"],
                         cwd=ROOT, capture_output=True, text=True, creationflags=FLAGS, check=True)
    return json.loads([line for line in out.stdout.splitlines() if line.startswith("{")][-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--start", type=Path, default=ROOT / "checkpoints/tiny2-G5-M23/latest.pt")
    parser.add_argument("--shares", default="10,20,30,40,50,60,70,80,90", help="survival shares, percent")
    parser.add_argument("--measure", default="simplify:1,material:1")
    args = parser.parse_args()
    report_path, stop = ROOT / "history/sweep-survival.json", ROOT / "history/sweep-survival.stop"
    report = {"games": args.games, "measure": args.measure, "start": str(args.start), "runs": []}
    for share in (int(value) for value in args.shares.split(",")):
        if stop.exists():
            break
        label = f"surv{share}"
        sets = [f"victory_measure={args.measure}", "symmetric_piece_values=true", "measure_lookahead=true",
                f"victory_weight={1 - share / 100:.2f}"]
        trained = train(label, args.games, args.workers, args.start, sets)
        result = evaluate(label, args.eval_games, min(5, args.workers), args.seed)
        total = trained["wins"] + trained["draws"] + trained["losses"] + trained["truncated"]
        report["runs"].append({"survival": share, "train_wins": trained["wins"], "train_games": total,
                               "eval": {k: result[k] for k in ("win", "draw", "loss", "unfinished", "score",
                                                               "mean_plies", "how")}})
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"survival {share}%: eval W/D/L {result['win']}/{result['draw']}/{result['loss']}", flush=True)


if __name__ == "__main__":
    main()
