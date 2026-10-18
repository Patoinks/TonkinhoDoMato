"""Torneio de gerações: mutations biased towards what has worked, away from what has not.

Every generation: mutants of the parents (the champion and the best of the last generation), each with 2-5
mutations drawn by how much each setting and direction (up or down from the champion t0467) has helped so far
- a ridge regression of every scored mutant (this run's games and the 1024-mutant tournament's first stage),
one intercept per generation since Stockfish's strength moves with the machine's load. Then: the time trial
(slower than t0467 dies), bo10 for all, the best 16 bo50, the best 4 bo200 with the champion in the same pool.
A mutant that beats the champion there by `--margin` is the new champion.

    python leto-rapido/evolucao.py --generations 8 --workers 11
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mega_torneio as M  # noqa: E402
import torneio_tempo as TT  # noqa: E402

HERE = Path(__file__).resolve().parent
SPACE = TT.SPACE
DIRS = ("+", "-")


def load_base():
    from claudio_agent import ClaudioConfig
    from claudio_load import load_claudio
    _, values, config, _ = load_claudio(str(M.CHECKPOINT), with_config=True)
    return asdict(ClaudioConfig.with_settings(M.RAPIDO, base=config)), values


def parse(raw):
    if raw in ("True", "False"):
        return raw == "True"
    try:
        return float(raw)
    except ValueError:
        return raw


def current(name, sets, base, values):
    """The value a setting has for these settings (on top of the champion)."""
    if name == "piece_values":
        row = sets.get("phase_values")
        return float(row.split(";")[0].split(",")[4]) if row else values[5]
    if name == "critical":
        return float(sets.get("critical_depth", base["critical_depth"])) > 0
    if name == "endgame":
        return float(sets.get("endgame_pieces", base["endgame_pieces"])) > 0
    return parse(sets[name]) if name in sets else base[name]


def features(sets, base, values):
    """(name, direction) for every setting that differs from the champion."""
    out = []
    for name in SPACE:
        v, b = current(name, sets, base, values), current(name, {}, base, values)
        if isinstance(v, str) or isinstance(b, str):
            if v != b:
                out.append((name, "+"))
        elif float(v) != float(b):
            out.append((name, "+" if float(v) > float(b) else "-"))
    return out


def mutate_one(rng, name, d, sets, base, values):
    """Move one setting in direction d from its current value; False when it cannot move that way."""
    spec, v = SPACE[name], current(name, sets, base, values)
    kind = spec[0]
    if kind == "float":
        lo, hi = spec[1:]
        v = float(v)
        if (d == "+" and v >= hi) or (d == "-" and v <= lo):
            return False
        if rng.random() < 0.5 or v == 0:
            new = rng.uniform(v, hi) if d == "+" else rng.uniform(lo, v)
        else:
            f = 4.0 ** rng.uniform(0, 1)
            new = min(hi, v * f) if d == "+" else max(lo, v / f)
        sets[name] = f"{new:.4g}"
    elif kind == "int":
        lo, hi = spec[1:]
        v = int(v)
        if (d == "+" and v >= hi) or (d == "-" and v <= lo):
            return False
        if hi - lo > 100:
            f = 4.0 ** rng.uniform(0.05, 1)
            new = int(min(hi, v * f)) if d == "+" else int(max(lo, v / f))
        else:
            new = int(rng.integers(v + 1, hi + 1)) if d == "+" else int(rng.integers(lo, v))
        if new == v:
            return False
        sets[name] = str(new)
    elif kind == "bool":
        want = d == "+"
        if bool(v) == want:
            return False
        sets[name] = str(want)
    elif kind == "choice":
        if d == "-":
            return False
        sets[name] = str(rng.choice([x for x in spec[1] if x != v]))
    elif kind == "values":
        row = sets.get("phase_values")
        q = [float(x) for x in row.split(";")[0].split(",")] if row else [1.0] + [values[t] for t in (2, 3, 4, 5)]
        f = 1.25 ** rng.uniform(0.2, 1)
        q = [1.0] + [x * f if d == "+" else x / f for x in q[1:]]
        sets["phase_values"] = ";".join([",".join(f"{x:.3f}" for x in q)] * 3)
    elif kind == "critical":
        if d == "+":
            sets.update(critical_depth=str(int(rng.integers(6, 9))), critical_gap=f"{rng.uniform(0.005, 0.1):.4g}",
                        critical_spread=f"{rng.uniform(0.0, 0.05):.4g}")
        else:
            if not v:
                return False
            sets["critical_depth"] = "0"
    elif kind == "endgame":
        if d == "+":
            sets.update(endgame_pieces=str(int(rng.integers(5, 13))), endgame_depth=str(int(rng.integers(0, 9))),
                        endgame_mate=str(int(rng.integers(0, 4))))
        else:
            if not v:
                return False
            sets["endgame_pieces"] = "0"
    return True


def estimate(samples, lam=3.0):
    """Ridge regression: score ~ generation intercept + sum of (setting, direction) effects."""
    import numpy as np
    cols = sorted({f for s in samples for f in s["features"]})
    gens = sorted({s["gen"] for s in samples})
    index = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(samples), len(cols) + len(gens)))
    y = np.array([s["score"] for s in samples])
    w = np.array([s["games"] for s in samples], dtype=float)
    for r, s in enumerate(samples):
        for f in s["features"]:
            X[r, index[f]] = 1.0
        X[r, len(cols) + gens.index(s["gen"])] = 1.0
    penalty = np.diag([lam] * len(cols) + [1e-6] * len(gens))
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + penalty, X.T @ (w * y))
    return {c: float(beta[index[c]]) for c in cols}


def survival(timed):
    """(name, direction) -> share of the mutants with it that passed the time trial (Laplace: 1 of 2 to start)."""
    seen, alive = {}, {}
    for features_, ok in timed:
        for f in features_:
            k = (f[:-1], f[-1])
            seen[k] = seen.get(k, 0) + 1
            alive[k] = alive.get(k, 0) + ok
    return {k: (alive[k] + 1) / (seen[k] + 2) for k in seen}


def sampler(effects, temperature, alive_share=None):
    """(name, direction) -> probability: exp(effect / temperature) times how often it survives the time trial,
    never below a fifth of the average."""
    import numpy as np
    keys = [(n, d) for n in SPACE for d in DIRS]
    alive_share = alive_share or {}
    raw = np.array([math.exp(effects.get(k, 0.0) / temperature) * alive_share.get(k, 0.5) for k in keys])
    p = raw / raw.sum()
    p = np.maximum(p, 0.2 / len(keys))
    return keys, p / p.sum()


def prior_samples(values):
    """The 1024-mutant tournament's first stage (bo3): generation -1, each mutant's directions relative to that
    tournament's own base (the fast Leto of then + m039)."""
    from claudio_agent import ClaudioConfig
    from claudio_load import load_claudio
    folder = HERE.parent / "torneio-tempo"
    if not (folder / "games.jsonl").exists():
        return [], []
    _, _, config, _ = load_claudio(str(M.CHECKPOINT), with_config=True)
    then = ["syzygy=", "root_prunings=True", "tactic_depth=6", "scan_prunings=True", "fast_net=True", "pawn_v4=True",
            "pawn_weight=0.5", "shared_table=True", "reply_top=0", "fast_core=True", "delta_pruning=True", "scan_keep=20",
            "safety_weight=4.739", "opening_weight=0.08524", "draw_value=0.5594", "simplify_margin=0.411",
            "ahead_draws=False", "value_scaling=False", "reply_weight=0.2936"]
    old = asdict(ClaudioConfig.with_settings(then, base=config))
    ents = {e["id"]: e["settings"] for e in json.loads((folder / "entrants.json").read_text(encoding="utf-8"))}
    scores = {}
    for line in (folder / "games.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line) if line.strip() else None
        if r and r["stage"] == 0 and r["end"] != "error" and r["id"] in ents and r["id"] != "base":
            scores.setdefault(r["id"], []).append(r["score"])
    feats = {i: [f"{n}{d}" for n, d in features(dict(x.split("=", 1) for x in s), old, values)] for i, s in ents.items()}
    timed = []
    if (folder / "tempos.jsonl").exists():
        for line in (folder / "tempos.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line) if line.strip() else None
            if r and r["id"] in feats:
                timed.append((feats[r["id"]], r["ratio"] <= 1.0))
    return [{"gen": -1, "features": feats[i], "score": sum(s) / len(s), "games": len(s)} for i, s in scores.items()], timed


def run_games(pool, entrants, n_games, gen, stage, seed, log, done):
    tasks = [(100 * gen + stage, i, [f"{k}={v}" for k, v in sets.items()], g, seed + 7919 * gen + 101 * stage, 300)
             for g in range(n_games) for i, sets in entrants.items() if (100 * gen + stage, i, g) not in done]
    with log.open("a", encoding="utf-8") as out:
        for future in as_completed([pool.submit(M.play, t) for t in tasks]):
            r = future.result()
            if r["end"] != "error":
                done[(r["stage"], r["id"], r["game"])] = r
            out.write(json.dumps(r) + "\n")
            out.flush()
    rows = {i: [done[(100 * gen + stage, i, g)] for g in range(n_games) if (100 * gen + stage, i, g) in done]
            for i in entrants}
    return {i: (sum(r["score"] for r in rs) / max(1, len(rs)), len(rs)) for i, rs in rows.items()}


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--elo", type=int, default=2250)
    parser.add_argument("--movetime", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--temperature", type=float, default=0.02, help="score units: lower = more biased")
    parser.add_argument("--margin", type=float, default=0.03, help="score above the champion to replace it")
    parser.add_argument("--stages", default="10,16x50,4x400", help="bo for all, best Nxbo, final Nxbo with the champion")
    parser.add_argument("--dir", type=Path, default=HERE.parent / "evolucao")
    args = parser.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    first, mid, final = args.stages.split(",")
    first = int(first)
    mid_n, mid_g = (int(x) for x in mid.split("x"))
    fin_n, fin_g = (int(x) for x in final.split("x"))
    base, values = load_base()
    rng = None
    recorded = json.loads(TT.POSITIONS.read_text(encoding="utf-8"))["games"]
    # three games of positions (~450): one game let slow mutants through (a check extension seldom fires there)
    trial_games = [g["moves"].split() for g in recorded[:3]]
    long_games = [g["moves"].split() for g in recorded[3:6]]
    samples, timed = prior_samples(values)
    state_file = args.dir / "estado.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {
        "generation": 0, "champion": {}, "champion_id": "t0467", "parents": [{}], "history": []}
    samples += state.get("samples", [])
    run_timed = [tuple(x) for x in state.get("timed", [])]
    done = {}
    log = args.dir / "games.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["end"] != "error" and r["stage"] < 100 * state["generation"]:
                    done[(r["stage"], r["id"], r["game"])] = r
    report = [(args.dir / "RESULTADO.md").read_text(encoding="utf-8")] if (args.dir / "RESULTADO.md").exists() else [
        f"# Torneio de gerações vs Stockfish {args.elo} ({args.movetime * 1000:g} ms), partida do t0467\n"]
    context = get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=M._init,
                             initargs=(args.elo, args.movetime)) as pool:
        while state["generation"] < args.generations:
            gen = state["generation"]
            rng = np.random.default_rng([args.seed, gen])     # the same mutants if a generation is run again
            started = time.perf_counter()
            effects = {tuple([k[:-1], k[-1]]): v for k, v in estimate(samples).items()} if samples else {}
            keys, p = sampler(effects, args.temperature, survival(timed + run_timed))
            # mutants of the parents
            mutants = {}
            while len(mutants) < args.population:
                parent = state["parents"][int(rng.integers(len(state["parents"])))]
                sets = dict(parent)
                made = 0
                for _ in range(40):
                    if made >= int(rng.integers(2, 6)):
                        break
                    name, d = keys[int(rng.choice(len(keys), p=p))]
                    made += mutate_one(rng, name, d, sets, base, values)
                if made:
                    mutants[f"g{gen}m{len(mutants):03d}"] = sets
            # time trial: slower than the champion t0467 dies
            trials = {}
            for future in as_completed([pool.submit(TT.trial, (i, [f"{k}={v}" for k, v in s.items()], trial_games))
                                        for i, s in mutants.items()]):
                r = future.result()
                trials[r["id"]] = r
            alive = {i: s for i, s in mutants.items() if trials[i]["ratio"] <= 1.0}
            run_timed += [([f"{n}{d}" for n, d in features(s, base, values)], trials[i]["ratio"] <= 1.0) for i, s in mutants.items()]
            champion = {"campeao": dict(state["champion"])}
            # bo for everyone, the champion as reference
            s1 = run_games(pool, {**alive, **champion}, first, gen, 1, args.seed, log, done)
            for i in alive:
                samples.append({"gen": gen, "features": [f"{n}{d}" for n, d in features(alive[i], base, values)],
                                "score": s1[i][0], "games": s1[i][1]})
            ranked = sorted(alive, key=lambda i: s1[i][0], reverse=True)
            best = {i: alive[i] for i in ranked[:mid_n]}
            s2 = run_games(pool, {**best, **champion}, mid_g, gen, 2, args.seed, log, done)
            ranked2 = sorted(best, key=lambda i: s2[i][0], reverse=True)
            finalists = {i: alive[i] for i in ranked2[:fin_n]}
            s3 = run_games(pool, {**finalists, **champion}, fin_g, gen, 3, args.seed, log, done)
            # crowned: the best finalist that beats the champion by the margin and is still not slower on three
            # games of positions (the first trial is one game: a borderline mutant can pass it by noise)
            top, replaced = None, False
            for i in sorted(finalists, key=lambda i: s3[i][0], reverse=True):
                if s3[i][0] < s3["campeao"][0] + args.margin:
                    break
                check = pool.submit(TT.trial, (i, [f"{k}={v}" for k, v in finalists[i].items()], long_games)).result()
                trials[i]["ratio_check"] = check["ratio"]
                if check["ratio"] <= 1.0:
                    top, replaced = i, True
                    break
            if top is None and finalists:
                top = max(finalists, key=lambda i: s3[i][0])
            if replaced:
                state["champion"], state["champion_id"] = dict(finalists[top]), top
            state["parents"] = [dict(state["champion"])] + [dict(finalists[i]) for i in sorted(
                finalists, key=lambda i: s3[i][0], reverse=True) if i != state["champion_id"]][:3]
            effects = {tuple([k[:-1], k[-1]]): v for k, v in estimate(samples).items()}
            good = sorted(effects.items(), key=lambda kv: -kv[1])[:6]
            bad = sorted(effects.items(), key=lambda kv: kv[1])[:6]
            fmt = lambda kv: f"{kv[0][0]}{'↑' if kv[0][1] == '+' else '↓'} {kv[1]:+.3f}"
            report.append(
                f"\n## Geração {gen + 1}\n\n{len(mutants)} mutantes, {len(alive)} passaram a prova de tempo; "
                f"{(time.perf_counter() - started) / 60:.1f} min.\n\n"
                "| finalista | bo%d | ±95%% | tempo vs t0467 | mutações (sobre o t0467) |\n| --- | ---: | ---: | ---: | --- |\n" % fin_g
                + "".join(f"| {i} | {s3[i][0]:.3f} | {1.96 * math.sqrt(max(s3[i][0] * (1 - s3[i][0]), .05) / max(1, s3[i][1])):.3f} | "
                          f"{trials[i]['ratio']:.2f} | {' '.join(f'{k}={v}' for k, v in finalists[i].items())} |\n"
                          for i in sorted(finalists, key=lambda i: s3[i][0], reverse=True))
                + f"| campeão ({'antigo' if replaced else state['champion_id']}) | {s3['campeao'][0]:.3f} | | 1.00 | "
                  f"{' '.join(f'{k}={v}' for k, v in champion['campeao'].items()) or '(t0467)'} |\n\n"
                + (f"**Novo campeão: {top}** (+{s3[top][0] - s3['campeao'][0]:.3f}).\n\n" if replaced else "Campeão mantém-se.\n\n")
                + "Tendências agora — boas: " + ", ".join(map(fmt, good)) + "; más: " + ", ".join(map(fmt, bad)) + ".\n")
            state["generation"] = gen + 1
            state["samples"] = [s for s in samples if s["gen"] >= 0]
            state["timed"] = run_timed
            state["history"].append({"generation": gen + 1, "champion": state["champion_id"], "replaced": replaced,
                                     "finalists": {i: s3[i] for i in finalists}, "champion_score": s3["campeao"]})
            state_file.write_text(json.dumps(state, indent=1), encoding="utf-8")
            (args.dir / "RESULTADO.md").write_text("".join(report), encoding="utf-8")
            print(report[-1], flush=True)


if __name__ == "__main__":
    main()
