"""Torneio de mutações com prova de tempo: 1024 mutants of the champion (fast Leto + m039) against Stockfish.

Before playing, each mutant decides on the same recorded positions as the base, in the same worker, base and
mutant alternated (base, mutant, mutant, base): a mutant slower than the base dies there. The rest play bo3;
the best 256 bo10, 64 bo30, 16 bo100, 2 bo500. Each stage is fresh games, ranked by its own score (ties by the
earlier stages). The base plays every stage as the reference. Mutations may now change the thinking time too.

    python leto-rapido/torneio_tempo.py --workers 11
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mega_torneio as M  # noqa: E402

HERE = Path(__file__).resolve().parent
M039 = ["safety_weight=4.739", "opening_weight=0.08524", "draw_value=0.5594", "simplify_margin=0.411",
        "ahead_draws=False", "value_scaling=False", "reply_weight=0.2936"]
STAGES = [(1024, 3), (256, 10), (64, 30), (16, 100), (2, 500)]
POSITIONS = HERE.parent / "mega-torneio/m039-vs-leto-rapido.json"

# Judgement (as in mega_torneio) and now thinking time as well: the time trial decides who may be slower.
SPACE = dict(M.SPACE)
SPACE.update({
    "tactic_depth": ("int", 4, 7), "tactic_top": ("int", 4, 14), "tactic_nodes": ("int", 100_000, 1_500_000),
    "tactic_rounds": ("int", 1, 4), "scan_depth": ("int", 0, 4), "exchange_depth": ("int", 3, 7),
    "reply_top": ("int", 0, 4), "mate_defence": ("int", 1, 3), "mate_attack": ("int", 1, 3),
    "scan_keep": ("int", 0, 30), "check_extension": ("int", 0, 1), "quiesce_promotions": ("bool",),
})


def entrants(seed, count):
    import numpy as np
    from dataclasses import asdict
    from claudio_agent import ClaudioConfig
    from claudio_load import load_claudio
    _, values, config, _ = load_claudio(str(M.CHECKPOINT), with_config=True)
    base = asdict(ClaudioConfig.with_settings(M.RAPIDO + M039, base=config))
    M.SPACE = SPACE                          # mega_torneio.mutate draws from this space
    rng = np.random.default_rng([seed, 1024])
    return [{"id": f"t{i:04d}", "settings": M.mutate(rng, base, values)} for i in range(count)]


def trial(task):
    """(id, settings, games): seconds per decision of base and mutant on the same positions, alternated."""
    import chess
    import numpy as np
    from claudio_agent import Claudio, ClaudioConfig
    ident, settings, games = task
    st = M._state

    def run(sets):
        cfg = ClaudioConfig.with_settings(M.RAPIDO + M039 + list(sets), base=st["config"])
        total, n = 0.0, 0
        for moves in games:
            players = {c: Claudio(st["net"], st["values"], cfg, rng=np.random.default_rng(1), king_model=st["king"])
                       for c in (chess.WHITE, chess.BLACK)}
            for p in players.values():
                p._owner = 1
            board = chess.Board()
            for i, u in enumerate(moves):
                if i >= 6 and not board.is_game_over():
                    t = time.perf_counter()
                    players[board.turn].choose(board)
                    total += time.perf_counter() - t
                    n += 1
                board.push_uci(u)
        return total, n

    try:
        b1, n = run([])
        m1, _ = run(settings)
        m2, _ = run(settings)
        b2, _ = run([])
        return {"id": ident, "ratio": (m1 + m2) / (b1 + b2), "base_ms": 500 * (b1 + b2) / n, "ms": 500 * (m1 + m2) / n}
    except Exception as error:
        return {"id": ident, "ratio": math.inf, "error": f"{type(error).__name__}: {error}"[:300]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--elo", type=int, default=2250)
    parser.add_argument("--movetime", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=4096)
    parser.add_argument("--entrants", type=int, default=1024)
    parser.add_argument("--stages", default=",".join(f"{n}x{g}" for n, g in STAGES[1:]),
                        help="after the first bo, e.g. 256x10,64x30")
    parser.add_argument("--first-games", type=int, default=STAGES[0][1])
    parser.add_argument("--trial-games", type=int, default=1, help="recorded games whose positions time each mutant")
    parser.add_argument("--dir", type=Path, default=HERE.parent / "torneio-tempo")
    args = parser.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    pool_of = {e["id"]: e for e in entrants(args.seed, args.entrants)}
    pool_of["base"] = {"id": "base", "settings": []}
    (args.dir / "entrants.json").write_text(json.dumps(list(pool_of.values()), indent=1), encoding="utf-8")
    games = [g["moves"].split() for g in json.loads(POSITIONS.read_text(encoding="utf-8"))["games"][:args.trial_games]]
    stages = [(args.entrants, args.first_games)] + [tuple(int(x) for x in s.split("x")) for s in args.stages.split(",") if s]
    report = [f"# Torneio de mutações com prova de tempo vs Stockfish {args.elo} ({args.movetime * 1000:g} ms)\n"]
    log = args.dir / "games.jsonl"
    trials_file = args.dir / "tempos.jsonl"
    done, trials = {}, {}
    for path, store, key in ((log, done, None), (trials_file, trials, "id")):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if key:
                        store[r["id"]] = r
                    elif r["end"] != "error":
                        store[(r["stage"], r["id"], r["game"])] = r
    context = get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=M._init,
                             initargs=(args.elo, args.movetime)) as pool:
        # time trial
        todo = [(i, e["settings"], games) for i, e in pool_of.items() if i != "base" and i not in trials]
        print(f"prova de tempo: {len(todo)} mutantes", flush=True)
        started = time.perf_counter()
        with trials_file.open("a", encoding="utf-8") as out:
            for k, future in enumerate(as_completed([pool.submit(trial, t) for t in todo]), 1):
                r = future.result()
                trials[r["id"]] = r
                out.write(json.dumps(r) + "\n")
                out.flush()
                if k % 128 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)}, {k / (time.perf_counter() - started):.1f} mutantes/s", flush=True)
        alive = [i for i in pool_of if i != "base" and trials[i]["ratio"] <= 1.0]
        dead = [i for i in pool_of if i != "base" and trials[i]["ratio"] > 1.0]
        ratios = sorted(trials[i]["ratio"] for i in pool_of if i != "base" and math.isfinite(trials[i]["ratio"]))
        report.append(f"\n## Prova de tempo\n\n{len(alive)} vivos, {len(dead)} mortos (mais lentos que a base"
                      f"{', ' + str(sum('error' in trials[i] for i in dead)) + ' com erro' if any('error' in trials[i] for i in dead) else ''}). "
                      f"Tempo relativo à base: mínimo {ratios[0]:.2f}, mediana {ratios[len(ratios) // 2]:.2f}, máximo {ratios[-1]:.2f}.\n")
        (args.dir / "RESULTADO.md").write_text("".join(report), encoding="utf-8")
        print(report[-1], flush=True)
        timed = [trials[i]["base_ms"] for i in trials if "base_ms" in trials[i]]
        base_ms = sorted(timed)[len(timed) // 2] if timed else 0.0
        history = {i: [] for i in pool_of}
        alive.sort(key=lambda i: trials[i]["ratio"])          # before any game: the fastest first
        for stage, (count, n_games) in enumerate(stages):
            alive = alive[:count]
            players = alive + ["base"]
            tasks = [(stage, i, M039 + pool_of[i]["settings"], g, args.seed + 1000 * stage, 300)
                     for g in range(n_games) for i in players if (stage, i, g) not in done]
            print(f"fase {stage + 1}: {len(alive)} x bo{n_games} ({len(tasks)} jogos)", flush=True)
            started, finished = time.perf_counter(), 0
            with log.open("a", encoding="utf-8") as out:
                for future in as_completed([pool.submit(M.play, t) for t in tasks]):
                    r = future.result()
                    if r["end"] != "error":
                        done[(r["stage"], r["id"], r["game"])] = r
                    out.write(json.dumps(r) + "\n")
                    out.flush()
                    finished += 1
                    if finished % 500 == 0 or finished == len(tasks):
                        rate = finished / (time.perf_counter() - started)
                        print(f"  {finished}/{len(tasks)}, {rate:.1f} jogos/s, faltam {(len(tasks) - finished) / rate / 60:.1f} min", flush=True)
            rows = {i: [done[(stage, i, g)] for g in range(n_games) if (stage, i, g) in done] for i in players}
            for i in players:
                history[i].append((sum(r["score"] for r in rows[i]) / max(1, len(rows[i])), len(rows[i])))

            def total(i):
                n = sum(g for _, g in history[i])
                return sum(s * g for s, g in history[i]) / max(1, n)
            alive.sort(key=lambda i: (history[i][-1][0], total(i), -trials[i]["ratio"]), reverse=True)
            shown = alive[:stages[stage + 1][0] if stage + 1 < len(stages) else count]
            all_rows = [r for i in players for r in rows[i]]
            report.append(f"\n## Fase {stage + 1}: {len(alive)} x bo{n_games}\n\n{len(all_rows)} jogos, "
                          f"{sum(r['seconds'] for r in all_rows) / max(1, len(all_rows)):.2f} s/jogo\n\n"
                          "| # | id | score | ±95% | V/E/D | tempo vs base | ms/lance (prova) | mutações |\n"
                          "| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |\n")
            for rank, i in enumerate(shown + ["base"], 1):
                rs, s = rows[i], history[i][-1][0]
                w = sum(r["score"] == 1 for r in rs); d = sum(r["score"] == 0.5 for r in rs)
                ci = 1.96 * math.sqrt(max(s * (1 - s), 0.05) / max(1, len(rs)))
                tr = trials.get(i, {"ratio": 1.0, "ms": base_ms})
                report.append(f"| {rank if i != 'base' else '-'} | {i} | {s:.3f} | {ci:.3f} | {w}/{d}/{len(rs) - w - d} | "
                              f"{tr['ratio']:.2f} | {tr.get('ms', 0):.2f} | {' '.join(pool_of[i]['settings']) or '(base: --rapido + m039)'} |\n")
            (args.dir / "RESULTADO.md").write_text("".join(report), encoding="utf-8")
            print("".join(report[-1:])[:3000], flush=True)


if __name__ == "__main__":
    main()
