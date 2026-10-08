"""How fast is G16 with one claudio_fast build against another - same decisions required.

G16 decides at every ply of recorded games (its memory follows the real game, as in play), with build A
and with build B, alternately A B B A ... in separate processes on the same CPU core, one thread.
Every decision must be the same move with both builds; the report gives each build's time per decision,
per game and games per second on one core (both sides of a game are G16 deciding).

A build is a folder holding a compiled claudio_fast module, or git:REV - the claudio_fast.cpp of that
commit, compiled into the system's temporary folder (e.g. git:d5e6fda, the original G16 C++).

    python claudio/bench_velocidade.py --a git:d5e6fda --b claudio/fast --games 0:6 --rounds 2
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRATCH = Path(tempfile.gettempdir()) / "claudio-builds"


def build_from_git(rev):
    """claudio_fast.cpp of commit rev, compiled with the current setup.py (in the temporary folder)."""
    folder = SCRATCH / rev
    if any(folder.glob("claudio_fast*.so")) or any(folder.glob("claudio_fast*.pyd")):
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    source = subprocess.run(["git", "show", f"{rev}:claudio/fast/claudio_fast.cpp"], cwd=ROOT, check=True,
                            capture_output=True).stdout
    (folder / "claudio_fast.cpp").write_bytes(source)
    shutil.copy(HERE / "fast/setup.py", folder / "setup.py")
    subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace", "--force"], cwd=folder, check=True,
                   capture_output=True)
    return folder


def worker(build, log, games, checkpoint, out):
    """One process: G16 decides at every ply of the games with this build."""
    sys.path.insert(0, str(HERE))
    import paths  # noqa: F401
    sys.path.insert(0, str(build))
    import torch
    torch.set_num_threads(1)
    import chess
    import numpy as np
    import claudio_fast
    from claudio_agent import Claudio, ClaudioConfig
    from claudio_load import load_claudio
    if Path(claudio_fast.__file__).parent.resolve() != Path(build).resolve():
        raise SystemExit(f"claudio_fast came from {claudio_fast.__file__}, not {build}")
    net, values, config, _, king = load_claudio(checkpoint, with_config=True, with_king=True)
    cfg = ClaudioConfig.with_settings([], base=config)
    rows = [json.loads(line) for line in Path(log).read_text().splitlines() if line.strip()]
    low, high = map(int, games.split(":"))
    result = []
    for row in rows[low:high]:
        players = [Claudio(net, values, cfg, rng=np.random.default_rng(0), king_model=king) for _ in range(2)]
        board, decisions = chess.Board(), []
        for uci in row["moves"].split():
            start = time.perf_counter()
            move = players[0 if board.turn == chess.WHITE else 1].choose(board)
            decisions.append([move.uci(), time.perf_counter() - start])
            board.push(chess.Move.from_uci(uci))
        result.append({"id": row["id"], "decisions": decisions})
    Path(out).write_text(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", help="folder with a claudio_fast build, or git:REV")
    parser.add_argument("--b", help="folder with a claudio_fast build, or git:REV")
    parser.add_argument("--log", default=str(HERE / "results/matches/g16-vs-g15-100-bo30.jsonl"))
    parser.add_argument("--games", default="0:6", help="first:last game of the log")
    parser.add_argument("--rounds", type=int, default=2, help="ABBA rounds (each build runs twice per round)")
    parser.add_argument("--checkpoint", default=str(HERE / "claudio-g16.pt"))
    parser.add_argument("--core", type=int, default=0, help="CPU core for every run (Linux)")
    parser.add_argument("--out", default="", help="write the report here (JSON)")
    parser.add_argument("--worker", nargs=2, metavar=("BUILD", "OUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {args.core})
        return worker(args.worker[0], args.log, args.games, args.checkpoint, args.worker[1])
    if not args.a or not args.b:
        parser.error("--a and --b are required")

    builds = {}
    for name, spec in (("A", args.a), ("B", args.b)):
        builds[name] = build_from_git(spec[4:]) if spec.startswith("git:") else Path(spec).resolve()
    runs = {"A": [], "B": []}
    order = []
    for r in range(args.rounds):
        order += ["A", "B", "B", "A"] if r % 2 == 0 else ["B", "A", "A", "B"]
    scratch = SCRATCH
    scratch.mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(order):
        out = scratch / f"bench-{os.getpid()}-{k}.json"
        subprocess.run([sys.executable, __file__, "--worker", str(builds[name]), str(out), "--log", args.log,
                        "--games", args.games, "--checkpoint", args.checkpoint, "--core", str(args.core)], check=True)
        runs[name].append(json.loads(out.read_text()))
        out.unlink()
        spent = sum(d[1] for game in runs[name][-1] for d in game["decisions"])
        print(f"  run {k + 1}/{len(order)} build {name}: {spent:.1f} s", flush=True)

    reference = [[d[0] for d in game["decisions"]] for game in runs["A"][0]]
    different = sum(m != r for run in runs["A"] + runs["B"] for game, ref in zip(run, reference)
                    for (m, _), r in zip(game["decisions"], ref))
    decisions = sum(len(game) for game in reference)
    report = {"a": str(builds["A"]), "b": str(builds["B"]), "games": args.games, "log": args.log,
              "decisions": decisions, "different_moves": different, "rounds": args.rounds, "builds": {}}
    for name in ("A", "B"):
        totals = [sum(d[1] for game in run for d in game["decisions"]) for run in runs[name]]
        mean = sum(totals) / len(totals)
        report["builds"][name] = {"seconds_per_run": totals, "ms_per_decision": 1000 * mean / decisions,
                                  "seconds_per_game": mean / len(reference),
                                  "games_per_second_one_core": len(reference) / mean}
    a, b = report["builds"]["A"], report["builds"]["B"]
    report["speedup_b_over_a"] = a["ms_per_decision"] / b["ms_per_decision"]
    print(f"{decisions} decisions, {len(reference)} games, different moves: {different}")
    for name, row in report["builds"].items():
        print(f"  {name}: {row['ms_per_decision']:.1f} ms/decision, {row['seconds_per_game']:.1f} s/game, "
              f"{row['games_per_second_one_core']:.3f} games/s per core")
    print(f"  B is {report['speedup_b_over_a']:.2f}x faster than A")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
