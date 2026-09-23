"""Train every member of a tournament group on its own copy of the network.

Each member (a set of decision settings from an evolution run) starts a NEW
campaign from the same completed checkpoint, in checkpoints/group-<ID>, and
plays --games games against Stockfish with its own settings plus the owner's
game rules. One member at a time, at below-normal priority.

    python scripts/train_group.py --members G5-M23 G8-M12 --games 100 --elo 1350
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = r"engines\stockfish-11\stockfish-11-win\Windows\stockfish_20011801_x64_bmi2.exe"
RULES = ["draw_value=0.4", "strict_repetition=true", "stall_limit=30"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evolution", type=Path, default=ROOT / "history/evolution/20260923T230708Z/evolution.json")
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "checkpoints/cognitive-mix-20260923/latest.pt")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--elo", type=int, default=1350)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    log = json.loads(args.evolution.read_text(encoding="utf-8"))
    genes = {p["id"]: p["genes"] for g in log["generations"] for p in g["population"]}
    for member in args.members:
        settings = [f"{k}={v}" for k, v in genes[member].items()] + RULES
        command = [sys.executable, "train_cognitive.py", "--resume", str(args.source), "--new-campaign",
                   "--checkpoint-dir", f"checkpoints/group-{member}", "--games", str(args.games),
                   "--game-cap", str(args.games), "--workers", str(args.workers),
                   "--opponent-path", ENGINE, "--opponent-elo", str(args.elo), "--opponent-move-time", "0.002",
                   "--mix-self", "0", "--mix-tiny", "0", "--mix-fairy", "0",
                   "--resign", "--resign-min-non-loss", "0", "--capture-target-weight", "0.05",
                   "--objective", "shaped", "--victory-weight", "0.5", "--kd-weight", "0.75",
                   "--mate-search-moves", "3", "--mate-defense-moves", "2",
                   "--threat-explore", "0.1", "--mate-labels", "2", "--trade-explore", "0.1",
                   "--exchange-labels", "1", "--batch-search"]
        for item in settings:
            command += ["--set", item]
        print(f"== {member}: {args.games} jogos contra Stockfish {args.elo}", flush=True)
        result = subprocess.run(command, cwd=ROOT,
                                creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0)
        print(f"== {member}: terminou com código {result.returncode}", flush=True)


if __name__ == "__main__":
    main()
