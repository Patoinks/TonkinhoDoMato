"""Super mega ultra torneio de mutações: G4 Leto mutants against Stockfish 11 at a fixed Elo.

256 heavy mutants of the fast Leto play bo20 against Stockfish; the best
64 play bo50, the best 32 bo100, the best 2 bo500. Every stage is fresh games: a stage ranks by its own
score (ties by the earlier stages), so a lucky first round does not carry.

Each mutant changes 5-10 judgement settings by a lot (piece values included), none of its thinking time. Results go to --dir as one JSON line per
game; a restarted run skips the games already there.

    python leto-rapido/mega_torneio.py --workers 11
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # one BLAS thread per worker (rede_rapida)
import math
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parent / "releases/g4-leto/models/g4t2-leto-v3.pt"
STOCKFISH = HERE.parent / "engines-local/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe"
RAPIDO = ["syzygy=", "root_prunings=True", "tactic_depth=6", "scan_prunings=True", "fast_net=True",
          "pawn_v4=True", "pawn_weight=0.5", "shared_table=True", "reply_top=0", "fast_core=True",
          "delta_pruning=True", "scan_keep=12",
          # champion t0467 (torneio_tempo.py, 2026-10-18): m039 with its five mutations
          "opening_weight=0.08524", "draw_value=0.5594", "simplify_margin=0.411", "ahead_draws=False",
          "value_scaling=True", "reply_weight=0.2404", "safety_weight=3.416", "fifty_horizon=28",
          # champion g2m077 (evolucao.py, generation 3, kept in 4): t0467 with two mutations
          "ahead_material=0.2605", "pawn_weight=0.1891"]
STAGES = [(256, 20), (64, 50), (32, 100), (2, 500)]   # (entrants, games each)

# name: ("float", low, high) | ("int", low, high) | ("bool",) | ("values",)
# Only what changes how Leto judges, never how much it calculates: depths, node budgets, reply and scan
# counts, extensions and the critical/endgame modes stay the fast Leto's, so every mutant thinks as long.
SPACE = {
    "safety_weight": ("float", 0.1, 6.0), "reply_weight": ("float", 0.0, 1.0), "king_weight": ("float", 0.0, 0.5),
    "ahead_value": ("float", 0.5, 0.9), "ahead_material": ("float", 0.0, 1.0), "simplify_weight": ("float", 0.0, 0.5),
    "simplify_margin": ("float", 0.0, 3.0), "draw_value": ("float", 0.2, 0.6), "pawn_weight": ("float", 0.0, 2.0),
    "pawn_convert": ("float", 0.0, 1.0), "opening_weight": ("float", 0.0, 0.6), "fifty_horizon": ("int", 0, 30),
    "ahead_draws": ("bool",), "repeat_as_draw": ("bool",), "value_scaling": ("bool",), "piece_values": ("values",),
}


def mutate(rng, base, values):
    """5-10 settings, each moved far: half the time anywhere in its range, else x/÷ up to 4 from Leto's."""
    out = []
    for name in rng.choice(sorted(SPACE), int(rng.integers(5, 11)), replace=False):
        kind = SPACE[name][0]
        if kind == "float":
            lo, hi = SPACE[name][1:]
            v = base[name]
            if rng.random() < 0.5 or v == 0:
                new = rng.uniform(lo, hi)
            else:
                new = v * 4.0 ** rng.uniform(-1, 1)
            out.append(f"{name}={min(hi, max(lo, new)):.4g}")
        elif kind == "int":
            lo, hi = SPACE[name][1:]
            v = base[name]
            if hi - lo > 100:
                new = int(min(hi, max(lo, v * 4.0 ** rng.uniform(-1, 1))))
            else:
                options = [x for x in range(lo, hi + 1) if x != v] or [v]
                new = int(rng.choice(options))
            out.append(f"{name}={new}")
        elif kind == "choice":
            out.append(f"{name}={rng.choice([x for x in SPACE[name][1] if x != base[name]])}")
        elif kind == "bool":
            out.append(f"{name}={not base[name]}")
        elif kind == "values":
            row = [1.0] + [values[t] * 1.5 ** rng.uniform(-1, 1) for t in (2, 3, 4, 5)]
            out.append("phase_values=" + ";".join([",".join(f"{x:.3f}" for x in row)] * 3))
        elif kind == "critical":
            out += [f"critical_depth={int(rng.integers(6, 10))}", f"critical_gap={rng.uniform(0.005, 0.1):.4g}",
                    f"critical_spread={rng.uniform(0.0, 0.05):.4g}"]
        elif kind == "endgame":
            out += [f"endgame_pieces={int(rng.integers(5, 15))}", f"endgame_depth={int(rng.integers(0, 11))}",
                    f"endgame_mate={int(rng.integers(0, 5))}"]
    return out


def entrants(seed, count):
    import numpy as np
    from claudio_load import load_claudio
    from claudio_agent import ClaudioConfig
    from dataclasses import asdict
    _, values, config, _ = load_claudio(str(CHECKPOINT), with_config=True)
    base = asdict(ClaudioConfig.with_settings(RAPIDO, base=config))
    rng = np.random.default_rng([seed, 4242])
    out = []
    while len(out) < count:
        out.append({"id": f"m{len(out):03d}", "settings": mutate(rng, base, values)})
    return out


_state = {}


def _pin(counter, workers):
    """This worker (and the Stockfish it starts) on one logical CPU: up to half the CPUs, one per physical core
    (Windows numbers a core's two threads 2k and 2k+1), otherwise one per logical CPU."""
    import ctypes
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    cpus = os.cpu_count() or 1
    cpu = (2 * index) % cpus if workers <= cpus // 2 else index % cpus
    kernel = ctypes.windll.kernel32
    kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.c_size_t(1 << cpu))


def _init(elo, movetime, counter=None, workers=0):
    if counter is not None:
        _pin(counter, workers)
    import torch
    torch.set_num_threads(1)
    from claudio_load import load_claudio
    from vs_stockfish import UCI
    net, values, config, _, king = load_claudio(str(CHECKPOINT), with_config=True, with_king=True)
    _state.update(net=net, values=values, config=config, king=king, engine=UCI(STOCKFISH, elo), movetime=movetime,
                  elo=elo)


def play(task):
    """One game: (stage, entrant id, settings, game index, seed)."""
    import chess
    import numpy as np
    from claudio_agent import Claudio, ClaudioConfig
    from fim_rapido import Judge
    stage, ident, settings, game, seed, max_plies = task
    for attempt in range(3):                  # Stockfish's pipe broke: a new engine, the game again
        row = _game(stage, ident, settings, game, seed, max_plies)
        if row["end"] != "error" or not row["error"].startswith(("OSError", "RuntimeError: engine closed", "BrokenPipe")):
            return row
        from vs_stockfish import UCI
        _state["engine"] = UCI(STOCKFISH, _state["elo"])
    return row


def _game(stage, ident, settings, game, seed, max_plies):
    import chess
    import numpy as np
    from claudio_agent import Claudio, ClaudioConfig
    from fim_rapido import Judge
    row = {"stage": stage, "id": ident, "game": game, "leto_white": game % 2 == 0}
    started = time.perf_counter()
    board = chess.Board()
    try:
        deus = [s.split("=", 1)[1] for s in settings if s.startswith("deus=")]
        pecado = [s.split("=", 1)[1] for s in settings if s.startswith("pecado=")]
        janela = [float(s.split("=", 1)[1]) for s in settings if s.startswith("janela=")]
        tweaks = [s[4:].split("=", 1) for s in settings if s.startswith("sin.")]   # e.g. sin.plan_plies=16
        settings = [s for s in settings if not s.startswith(("deus=", "janela=", "pecado=", "sin."))]
        if pecado:                             # a sin's own engine settings come first; explicit ones override
            from pecados import SINS
            settings = SINS[pecado[0]].settings + settings
        cfg = ClaudioConfig.with_settings(RAPIDO + list(settings), base=_state["config"])
        leto = Claudio(_state["net"], _state["values"], cfg, rng=np.random.default_rng([seed, stage, game]),
                       king_model=_state["king"])
        leto._owner = 1                        # one search table per worker, not one per game
        if deus:                               # a Pantheon personality (janela=: its style window; 0: openings only)
            from personalidades import make
            leto.personality = make(deus[0], janela[0] if janela else None)
        if pecado:                             # one of the Seven Deadly Sins (pecados.py)
            from pecados import make as make_sin
            leto.personality = make_sin(pecado[0], janela[0] if janela else None)
            for key, value in tweaks:
                setattr(leto.personality, key, float(value) if "." in value else int(value))
        judge = Judge(board)
        think = 0.0
        while judge.outcome(board) is None and len(board.move_stack) < max_plies:
            if (board.turn == chess.WHITE) == row["leto_white"]:
                t = time.perf_counter()
                move = leto.choose(board)
                think += time.perf_counter() - t
                if move not in board.legal_moves:
                    raise ValueError(f"illegal move {move} in {board.fen()}")
            else:
                move = _state["engine"].play(board, _state["movetime"])
            judge.push(board, move)
        outcome = judge.outcome(board)
        if outcome is None or outcome.winner is None:
            score = 0.5
        else:
            score = 1.0 if outcome.winner == (chess.WHITE if row["leto_white"] else chess.BLACK) else 0.0
        row.update(score=score, end=outcome.termination.name if outcome else "max_plies",
                   leto_ms=1000 * think / max(1, (len(board.move_stack) + (not row["leto_white"])) // 2))
    except Exception as error:                  # a broken mutant loses the game
        row.update(score=0.0, end="error", error=f"{type(error).__name__}: {error}"[:300])
    row.update(plies=len(board.move_stack), seconds=time.perf_counter() - started,
               moves=" ".join(m.uci() for m in board.move_stack))
    return row


def interval(score, n):
    return 1.96 * math.sqrt(max(score * (1 - score), 0.05) / max(1, n))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--elo", type=int, default=2250)
    parser.add_argument("--movetime", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--stages", default=",".join(f"{n}x{g}" for n, g in STAGES), help="entrants x games, e.g. 256x20,64x50")
    parser.add_argument("--dir", type=Path, default=HERE.parent / "mega-torneio")
    args = parser.parse_args()
    stages = [tuple(int(x) for x in part.split("x")) for part in args.stages.split(",")]
    args.dir.mkdir(parents=True, exist_ok=True)
    pool_of = {e["id"]: e for e in entrants(args.seed, stages[0][0])}
    (args.dir / "entrants.json").write_text(json.dumps(list(pool_of.values()), indent=1), encoding="utf-8")
    log = args.dir / "games.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["end"] != "error":        # an error game is played again (e.g. a worker stopped mid-run)
                    done[(r["stage"], r["id"], r["game"])] = r
    alive = list(pool_of)
    history = {i: [] for i in pool_of}           # every stage's (score, games)
    report = [f"# Mega torneio de mutações vs Stockfish {args.elo} ({args.movetime * 1000:g} ms)\n"]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(args.elo, args.movetime)) as pool:
        for stage, (count, games) in enumerate(stages):
            alive = alive[:count]
            players = list(alive)
            tasks = [(stage, i, pool_of[i]["settings"], g, args.seed + 1000 * stage, args.max_plies)
                     for g in range(games) for i in players if (stage, i, g) not in done]
            started, finished, total = time.perf_counter(), 0, len(tasks)
            print(f"stage {stage + 1}: {len(alive)} entrants x bo{games} ({total} games to play)", flush=True)
            with log.open("a", encoding="utf-8") as out:
                for future in as_completed([pool.submit(play, t) for t in tasks]):
                    r = future.result()
                    done[(r["stage"], r["id"], r["game"])] = r
                    out.write(json.dumps(r) + "\n")
                    out.flush()
                    finished += 1
                    if finished % 200 == 0 or finished == total:
                        rate = finished / (time.perf_counter() - started)
                        print(f"  {finished}/{total} games, {rate:.2f} games/s, eta {(total - finished) / rate / 60:.1f} min",
                              flush=True)
            rows = {i: [done[(stage, i, g)] for g in range(games)] for i in players}
            for i in players:
                history[i].append((sum(r["score"] for r in rows[i]) / games, games))
            def total_score(i):
                n = sum(g for _, g in history[i])
                return sum(s * g for s, g in history[i]) / n
            alive.sort(key=lambda i: (history[i][-1][0], total_score(i)), reverse=True)
            all_rows = [r for i in players for r in rows[i]]
            report.append(f"\n## Fase {stage + 1}: {len(alive)} x bo{games}\n\n"
                          f"{len(all_rows)} jogos, {sum(r['seconds'] for r in all_rows) / len(all_rows):.2f} s/jogo, "
                          f"Leto {sum(r.get('leto_ms', 0) for r in all_rows) / len(all_rows):.1f} ms/lance, "
                          f"erros {sum(r['end'] == 'error' for r in all_rows)}\n\n"
                          "| # | id | score | ±95% | V/E/D | s/jogo | ms/lance | mutações |\n"
                          "| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |\n")
            shown = alive[:max(stages[stage + 1][0] if stage + 1 < len(stages) else count, 2)]
            for rank, i in enumerate(alive, 1):
                if i not in shown:
                    continue
                rs = rows[i]
                s = history[i][-1][0]
                w = sum(r["score"] == 1 for r in rs); d = sum(r["score"] == 0.5 for r in rs)
                report.append(f"| {rank} | {i} | {s:.3f} | {interval(s, games):.3f} | "
                              f"{w}/{d}/{len(rs) - w - d} | {sum(r['seconds'] for r in rs) / len(rs):.2f} | "
                              f"{sum(r.get('leto_ms', 0) for r in rs) / len(rs):.1f} | "
                              f"{' '.join(pool_of[i]['settings'])} |\n")
            (args.dir / "RESULTADO.md").write_text("".join(report), encoding="utf-8")
            print("".join(report[-1:]), flush=True)


if __name__ == "__main__":
    main()
