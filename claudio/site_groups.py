"""Group stage of site-player variants (Tonkclaudinho Filhodegei V2 as the browser
plays it, one setting changed per player): every pair plays --games games,
random openings, colours swapped per opening. Winrate = wins / games.

    python claudio/site_groups.py --games 20 --label kd-groups
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from itertools import combinations
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

import arena  # noqa: E402
import site_mutation as site  # noqa: E402

# table share / network share of the piece values (kd_weight = table share)
PLAYERS = {"site 17/83": 0.17304461792996828, "inversa 83/17": 0.82695538207, "50/50": 0.5,
           "100/0": 1.0, "0/100": 0.0}
_g = {}


def _init():
    site._init([])
    for name, weight in PLAYERS.items():
        cfg = copy.copy(site._s["base"])
        cfg.kd_weight = weight
        cfg.validate()
        _g[name] = cfg


def play(a, b, game, seed, random_plies, max_plies):
    import chess
    players = {a: site.SitePlayer(_g[a]), b: site.SitePlayer(_g[b])}
    a_white = game % 2 == 0
    board = arena.opening(seed, game // 2, random_plies)
    spent = {a: 0.0, b: 0.0}
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        who = a if (board.turn == chess.WHITE) == a_white else b
        start = time.perf_counter()
        move = players[who].choose(board)
        spent[who] += time.perf_counter() - start
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    winner = None if outcome is None or outcome.winner is None else (
        a if (outcome.winner == chess.WHITE) == a_white else b)
    return {"a": a, "b": b, "game": game, "white": a if a_white else b, "winner": winner,
            "end": outcome.termination.name if outcome else "max_plies", "plies": len(board.move_stack),
            "time": spent}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--random-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    jobs = [(a, b, g) for a, b in combinations(PLAYERS, 2) for g in range(args.games)]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init) as pool:
        games = list(pool.map(play, *zip(*[(a, b, g, args.seed, args.random_plies, args.max_plies)
                                             for a, b, g in jobs])))
    table = {name: {"played": 0, "win": 0, "draw": 0, "loss": 0, "seconds": 0.0} for name in PLAYERS}
    for g in games:
        for name in (g["a"], g["b"]):
            row = table[name]
            row["played"] += 1
            row["seconds"] += g["time"][name]
            row["win" if g["winner"] == name else "draw" if g["winner"] is None else "loss"] += 1
    ranking = sorted(table.items(), key=lambda item: (-item[1]["win"], item[1]["loss"]))
    report = {"label": args.label, "players": PLAYERS, "games_per_pair": args.games, "table": dict(ranking),
              "list": games}
    (ROOT / "claudio/results" / f"site-groups-{args.label}.json").write_text(json.dumps(report, indent=1),
                                                                              encoding="utf-8")
    for place, (name, row) in enumerate(ranking, start=1):
        print(f"{place}. {name:14s} WR {100 * row['win'] / row['played']:5.1f}%  V/E/D {row['win']}/{row['draw']}/"
              f"{row['loss']}  {row['seconds'] / row['played']:.2f} s/jogo", flush=True)


if __name__ == "__main__":
    main()
