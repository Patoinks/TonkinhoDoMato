"""Duel training: two tournament players, each learning on its own network copy,
every game against the other's newest snapshot (renewed every 1000 games).

    python scripts/train_duel.py --a G5-M23 --b G5-M13 --games 10000 --workers 3
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = r"engines\stockfish-11\stockfish-11-win\Windows\stockfish_20011801_x64_bmi2.exe"
RULES = ["draw_value=0.4", "strict_repetition=true"]
EVOLUTIONS = ["history/evolution/20260923T230708Z/evolution.json", "history/evolution/20260924T000026Z/evolution.json"]


def genes_of(player):
    for path in EVOLUTIONS:
        log = json.loads((ROOT / path).read_text(encoding="utf-8"))
        for generation in log["generations"]:
            for member in generation["population"]:
                if member["id"] == player and (path == EVOLUTIONS[-1] or player != "G5-M13"):
                    return member["genes"]
    raise KeyError(player)


def command(me, rival, games, workers, source, every):
    settings = [f"{k}={v}" for k, v in genes_of(me).items()] + RULES
    rival_settings = [f"{k}={v}" for k, v in genes_of(rival).items()] + RULES
    cmd = [sys.executable, "train_cognitive.py", "--resume", str(source), "--new-campaign",
           "--checkpoint-dir", f"checkpoints/duel-{me}", "--games", str(games), "--game-cap", str(games),
           "--workers", str(workers), "--checkpoint-every", str(every), "--self-snapshot-every", str(every),
           "--self-games-per-snapshot", "0", "--rival-dir", f"checkpoints/duel-{rival}/snapshots",
           "--rival-name", f"Tonkinho {rival}",
           "--opponent-path", ENGINE, "--opponent-elo", "1350", "--opponent-move-time", "0.002",
           "--resign", "--resign-min-non-loss", "0", "--capture-target-weight", "0.05",
           "--objective", "shaped", "--victory-weight", "0.5", "--kd-weight", "0.75",
           "--mate-search-moves", "3", "--mate-defense-moves", "2", "--threat-explore", "0.1",
           "--mate-labels", "2", "--trade-explore", "0.1", "--exchange-labels", "1", "--batch-search"]
    for item in settings:
        cmd += ["--set", item]
    for item in rival_settings:
        cmd += ["--rival-set", item]
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="G5-M23")
    parser.add_argument("--b", default="G5-M13")
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--every", type=int, default=1000)
    parser.add_argument("--source", type=Path, default=ROOT / "checkpoints/cognitive-mix-20260923/latest.pt")
    args = parser.parse_args()
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0
    runs = []
    for me, rival in ((args.a, args.b), (args.b, args.a)):
        log = open(ROOT / f"history/duel-{me}.log", "w", encoding="utf-8")
        runs.append((me, subprocess.Popen(command(me, rival, args.games, args.workers, args.source, args.every),
                                          cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)))
        print(f"{me} contra {rival}: PID {runs[-1][1].pid}", flush=True)
    for me, process in runs:
        print(f"{me} terminou com código {process.wait()}", flush=True)


if __name__ == "__main__":
    main()
