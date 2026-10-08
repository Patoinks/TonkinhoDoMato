"""Group stage between Claudio players that may have different networks (e.g. a trained
champion against the tournament's champions): every pair best of --games, colours alternating,
each pair of games from its own opening (--random-plies random legal moves, seeded). Ends by the
rules or at --max-plies. Winrate = wins / games (a draw counts 0, the owner's rule).
Measurement only - nothing is trained.

Players: --player name=checkpoint.pt[@setting=value,setting=value]; --champions adds the champions
of a mutation tournament (results/torneio-<label>.json: its checkpoint, each champion's settings,
and its group-stage settings). --set applies to every player.

    python claudio/grupos_finais.py --label pos-treino --champions claudio/results/torneio-p3.json \\
        --player treinado=claudio/train-campeao/claudio-grande-best.pt@tactic_depth=5
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

import arena  # noqa: E402

_loaded = {}


def _init():
    import psutil_free_priority  # noqa: F401
    import torch
    torch.set_num_threads(1)


def _claudio(path, settings, seed):
    """A Claudio from a checkpoint (each loaded once per worker) with settings over its own."""
    import numpy as np
    from claudio_agent import Claudio, ClaudioConfig
    from claudio_load import load_claudio
    if path not in _loaded:
        net, values, config, _, king = load_claudio(path, with_config=True, with_king=True)
        _loaded[path] = (net, values, config, king)
    net, values, config, king = _loaded[path]
    cfg = ClaudioConfig.with_settings([f"{k}={v}" for k, v in settings.items()], base=config)
    return Claudio(net, values, cfg, rng=np.random.default_rng(seed), king_model=king)


def play(a, b, spec_a, spec_b, a_white, seed, opening, random_plies, max_plies):
    import chess
    players = {a: _claudio(*spec_a, [seed, opening, 1]), b: _claudio(*spec_b, [seed, opening, 2])}
    board = arena.opening(seed, opening, random_plies)
    spent = {a: 0.0, b: 0.0}
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        who = a if (board.turn == chess.WHITE) == a_white else b
        start = time.perf_counter()
        move = players[who].choose(board)
        spent[who] += time.perf_counter() - start
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    winner = None if outcome is None or outcome.winner is None else (a if (outcome.winner == chess.WHITE) == a_white else b)
    return {"a": a, "b": b, "white": a if a_white else b, "winner": winner,
            "end": outcome.termination.name if outcome else "max_plies", "plies": len(board.move_stack),
            "time": spent, "moves": " ".join(m.uci() for m in board.move_stack)}


def _setting(text):
    key, _, raw = text.partition("=")
    return key, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--player", action="append", default=[], help="name=checkpoint.pt[@setting=value,...]")
    parser.add_argument("--champions", default="", help="a mutation tournament's results JSON: add its champions")
    parser.add_argument("--set", action="append", default=[], help="setting=value for every player")
    parser.add_argument("--games", type=int, default=10, help="games per pair (best of)")
    parser.add_argument("--random-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    players = {}
    if args.champions:
        report = json.loads(Path(args.champions).read_text(encoding="utf-8"))
        group = dict(_setting(item) for item in report.get("group_set", []))
        for name, changes in report["champions"].items():
            players[name] = (report["checkpoint"], {**{k: str(v) for k, v in changes.items()}, **group})
    for item in args.player:
        name, _, spec = item.partition("=")
        path, _, settings = spec.partition("@")
        players[name] = (path, dict(_setting(s) for s in settings.split(",") if s))
    common = dict(_setting(item) for item in args.set)
    players = {n: (p, {**s, **common}) for n, (p, s) in players.items()}
    if len(players) < 2:
        raise SystemExit("at least two players")

    jobs = []
    for p, (a, b) in enumerate(combinations(players, 2)):
        for g in range(args.games):
            jobs.append((a, b, players[a], players[b], g % 2 == 0, args.seed, p * 100 + g // 2,
                         args.random_plies, args.max_plies))
    print(f"{len(players)} jogadores, {len(jobs)} jogos (cada par melhor de {args.games})", flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init) as pool:
        games = list(pool.map(play, *zip(*jobs), chunksize=1))
    table = {n: {"jogos": 0, "V": 0, "E": 0, "D": 0, "s": 0.0} for n in players}
    pairs = {}
    for game in games:
        a, b, w = game["a"], game["b"], game["winner"]
        for n in (a, b):
            row = table[n]
            row["jogos"] += 1
            row["V" if w == n else "E" if w is None else "D"] += 1
            row["s"] += game["time"][n]
        pair = pairs.setdefault(f"{a} vs {b}", {a: 0, b: 0, "empates": 0})
        pair[w if w else "empates"] += 1
    for row in table.values():
        row["winrate"] = row["V"] / max(1, row["jogos"])
        row["s_por_jogo"] = row["s"] / max(1, row["jogos"])
    out = ROOT / "claudio/results" / f"grupos-{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "players": {n: {"checkpoint": p, "settings": s}
                                                               for n, (p, s) in players.items()},
                               "games_per_pair": args.games, "table": table, "pairs": pairs, "list": games},
                              indent=1), encoding="utf-8")
    for name, row in sorted(table.items(), key=lambda kv: (-kv[1]["V"], kv[1]["D"])):
        print(f"  {name:12s} WR {100 * row['winrate']:5.1f}%  V/E/D {row['V']}/{row['E']}/{row['D']}  "
              f"{row['s_por_jogo']:.1f} s/jogo", flush=True)


if __name__ == "__main__":
    main()
