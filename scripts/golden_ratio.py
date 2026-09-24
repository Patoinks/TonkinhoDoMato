"""Updates per game at equal training time ("golden ratio" of this machine).

The running reference campaign is stopped at --stop-at games; its training
time T becomes the budget. Every other updates-per-game setting then trains
from the same start for T seconds (graceful stop file), and every result is
evaluated alike against Stockfish. Results: history/golden-ratio.json.

    python scripts/golden_ratio.py --reference-history history/cognitive-...jsonl \\
        --reference-checkpoint checkpoints/obj-sf1800-max/latest.pt --updates 24 8
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_duel  # noqa: E402

FLAGS = subprocess.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0
QUALITY = ["tactic_candidates=4", "tactic_depth=4", "avoid_draws_ahead=3", "endgame_pieces=10",
           "mate_attack_ks=0", "mate_defense_ks=0"]


def records(path, kind):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("kind") == kind:
            out.append(row)
    return out


def seconds(row):
    return datetime.fromisoformat(row["timestamp"]).timestamp()


def stop_reference(history, stop_file, stop_at):
    while True:
        metrics = records(history, "metrics")
        if metrics and metrics[-1]["games"] >= stop_at:
            break
        time.sleep(5)
    Path(stop_file).touch()
    while not records(history, "metrics") or not records(history, "metrics")[-1].get("final"):
        time.sleep(5)
    final = records(history, "metrics")[-1]
    return seconds(final) - seconds(records(history, "run")[0]), final


def train(updates, budget, start, workers, elo):
    label = f"golden-u{updates}"
    genes = [f"{k}={v}" for k, v in train_duel.genes_of("G5-M23").items()] + train_duel.RULES
    genes += [f"syzygy_path={ROOT / 'engines/syzygy'}", "tablebase_adjudicate=true", "break_repetitions=true",
              "victory_measure=simplify:1,material:1", "symmetric_piece_values=true", "measure_lookahead=true",
              "victory_weight=0.2", f"updates_per_game={updates}"] + QUALITY
    stop = ROOT / f"history/obj-{label}.stop"
    stop.unlink(missing_ok=True)
    cmd = [sys.executable, "train_cognitive.py", "--resume", str(start), "--new-campaign",
           "--checkpoint-dir", f"checkpoints/obj-{label}", "--games", "20000", "--game-cap", "20000",
           "--workers", str(workers), "--checkpoint-every", "100000", "--log-every", "5",
           "--mix-self", "0", "--mix-tiny", "0", "--mix-fairy", "0",
           "--opponent-path", train_duel.ENGINE, "--opponent-elo", str(elo), "--opponent-move-time", "0.002",
           "--objective", "shaped", "--kd-weight", "0.75", "--capture-target-weight", "0.05",
           "--mate-search-moves", "3", "--mate-defense-moves", "2", "--threat-explore", "0.1",
           "--mate-labels", "2", "--trade-explore", "0.1", "--exchange-labels", "1", "--batch-search",
           "--stop-file", str(stop)]
    for item in genes:
        cmd += ["--set", item]
    with open(ROOT / f"history/obj-{label}.log", "w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS)
        started = time.monotonic()
        while process.poll() is None and time.monotonic() - started < budget:
            time.sleep(5)
        stop.touch()
        process.wait()
    last = [json.loads(line) for line in (ROOT / f"history/obj-{label}.log").read_text(encoding="utf-8").splitlines()
            if line.startswith("{")][-1]
    return ROOT / f"checkpoints/obj-{label}/latest.pt", last


def evaluate(checkpoint, label, games, workers, elo, seed):
    cmd = [sys.executable, "scripts/match_engine.py", "--checkpoint", str(checkpoint), "--elo", str(elo), "--greedy",
           "--games", str(games), "--workers", str(workers), "--seed", str(seed), "--label", label]
    for item in QUALITY:
        cmd += ["--set", item]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, creationflags=FLAGS, check=True)
    return json.loads([line for line in out.stdout.splitlines() if line.startswith("{")][-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-history", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-stop", type=Path, required=True)
    parser.add_argument("--reference-updates", type=int, default=128)
    parser.add_argument("--stop-at", type=int, default=100)
    parser.add_argument("--start", type=Path, default=ROOT / "checkpoints/obj-sf1800/latest.pt")
    parser.add_argument("--updates", type=int, nargs="+", default=[24, 8])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--elo", type=int, default=1800)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1818)
    args = parser.parse_args()
    report_path = ROOT / "history/golden-ratio.json"
    budget, final = stop_reference(args.reference_history, args.reference_stop, args.stop_at)
    report = {"budget_seconds": round(budget), "elo": args.elo, "start": str(args.start), "runs": []}
    trained = [(args.reference_updates, args.reference_checkpoint, final)]
    for updates in args.updates:
        checkpoint, last = train(updates, budget, args.start, args.workers, args.elo)
        trained.append((updates, checkpoint, last))
    for updates, checkpoint, last in trained:
        result = evaluate(checkpoint, f"eval-sf{args.elo}-golden-u{updates}", args.eval_games, 5, args.elo, args.seed)
        report["runs"].append({"updates_per_game": updates, "train_games": last["games"],
                               "train_updates": last["execution"]["learner_updates"],
                               "eval": {k: result[k] for k in ("win", "draw", "loss", "unfinished", "score",
                                                               "mean_plies", "how")}})
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{updates} updates/jogo: {last['games']} jogos de treino, eval "
              f"{result['win']}/{result['draw']}/{result['loss']} score {result['score']:.3f}", flush=True)


if __name__ == "__main__":
    main()
