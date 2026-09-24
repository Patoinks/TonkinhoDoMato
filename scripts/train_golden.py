"""Train with the measured "golden" settings of this machine, then evaluate.

Golden = calculation depth 4 half-moves (tactic_depth=3; deeper costs 4x and
decides no better) and ~6 updates per game (learning and playing both busy at
that depth). The rest: the best objective mix found (80% survival, 20%
simplify + material one horizon ahead) and the full rule set.

    python scripts/train_golden.py --start checkpoints/obj-golden-d4-u14/latest.pt --label golden-a --games 50
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_duel  # noqa: E402
import golden_ratio  # noqa: E402

FLAGS = subprocess.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--elo", type=int, default=1800)
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--depth", type=int, default=3, help="tactic_depth: half-moves after our move")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--eval-elo", type=int, help="evaluate at this Elo (default: the training Elo)")
    parser.add_argument("--seed", type=int, default=1818)
    parser.add_argument("--set", action="append", default=[], dest="settings", help="extra setting name=value")
    parser.add_argument("--values-seed", type=Path, help="checkpoint whose stored games seed the piece values record")
    args = parser.parse_args()
    quality = [q for q in golden_ratio.QUALITY if not q.startswith("tactic_depth")] + [f"tactic_depth={args.depth}"]
    genes = [f"{k}={v}" for k, v in train_duel.genes_of("G5-M23").items()] + train_duel.RULES
    genes += [f"syzygy_path={ROOT / 'engines/syzygy'}", "tablebase_adjudicate=true", "break_repetitions=true",
              "victory_measure=simplify:1,material:1", "symmetric_piece_values=true", "measure_lookahead=true",
              "victory_weight=0.2", f"updates_per_game={args.updates}"] + quality + args.settings
    cmd = [sys.executable, "train_cognitive.py", "--resume", str(args.start), "--new-campaign",
           "--checkpoint-dir", f"checkpoints/obj-{args.label}", "--games", str(args.games),
           "--game-cap", str(args.games), "--workers", str(args.workers), "--checkpoint-every", str(args.games),
           "--log-every", "5", "--mix-self", "0", "--mix-tiny", "0", "--mix-fairy", "0",
           "--opponent-path", train_duel.ENGINE, "--opponent-elo", str(args.elo), "--opponent-move-time", "0.002",
           "--objective", "shaped", "--kd-weight", "0.75", "--capture-target-weight", "0.05",
           "--mate-search-moves", "3", "--mate-defense-moves", "2", "--threat-explore", "0.1",
           "--mate-labels", "2", "--trade-explore", "0.1", "--exchange-labels", "1", "--batch-search",
           "--stop-file", f"history/obj-{args.label}.stop"]
    for item in genes:
        cmd += ["--set", item]
    if args.values_seed:
        cmd += ["--values-seed-checkpoint", str(args.values_seed)]
    with open(ROOT / f"history/obj-{args.label}.log", "w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS, check=True)
    checkpoint = ROOT / f"checkpoints/obj-{args.label}/latest.pt"
    eval_elo = args.eval_elo or args.elo
    command = [sys.executable, "scripts/match_engine.py", "--checkpoint", str(checkpoint), "--elo", str(eval_elo),
               "--greedy", "--games", str(args.eval_games), "--workers", "5", "--seed", str(args.seed),
               "--label", f"eval-sf{eval_elo}-{args.label}"]
    for item in quality + args.settings:
        command += ["--set", item]
    out = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, creationflags=FLAGS, check=True)
    result = json.loads([line for line in out.stdout.splitlines() if line.startswith("{")][-1])
    (ROOT / f"history/{args.label}-eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{args.label}: eval vs {eval_elo} W/D/L {result['win']}/{result['draw']}/{result['loss']} "
          f"score {result['score']:.3f}", flush=True)


if __name__ == "__main__":
    main()
