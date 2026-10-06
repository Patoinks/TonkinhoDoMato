"""Group stage: Claudio checkpoints and Timdgei, every pair --games games, from the
normal starting position (no invented openings: the players decide). Colours
alternate. A Claudio picks among its own best moves with a little temperature
for the first --explore-plies plies (its own choice, so games differ), then its
best move; Timdgei plays as in its tournaments. Ends by the rules or at
--max-plies. Winrate = wins / games. Measurement only.

    python claudio/groups.py --games 20 --label grupos
"""
import argparse
from collections import Counter
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

PLAYERS = {
    "Este (treino 1550)": ROOT / "claudio/train-1550/claudio-grande-last.pt",
    "Grande g0": ROOT / "claudio/claudio-grande-g0.pt",
    "Claudio": ROOT / "claudio/claudio.pt",
    "Timdgei": None,
}
_g = {}


def _init():
    arena._init("", [], [])                      # Timdgei (tournament settings), frozen code
    from claudio_load import load_players
    for name, path in PLAYERS.items():
        if path is not None:
            _g[name] = load_players(path)                  # one checkpoint, or "white.pt+black.pt"


def _player(name, seed, game, explore_plies, temperature):
    import numpy as np
    from dataclasses import replace
    from tonkai.cognitive_agent import CognitiveAgent
    from tonkai.opponents import CognitiveOpponent
    from claudio_agent import Claudio
    if name == "Timdgei":
        evo = arena._state["evo"]
        cfg = evo._config(arena._state["genes"])
        agent = CognitiveOpponent(CognitiveAgent(evo._state["net"], cfg, evo._state["knowledge"],
                                                 kd_values=evo._state["kd"]), cfg, np.random.default_rng([seed, game, 9]))
        return lambda board: agent.choose(board)
    from claudio_load import make_player
    player = make_player(_g[name], np.random.default_rng([seed, game, sum(map(ord, name))]))
    parts = getattr(player, "players", (player,))
    greedy = {id(p): p.cfg for p in parts}
    exploring = {id(p): replace(p.cfg, temperature=temperature) for p in parts}

    def move(board):
        early = temperature > 0 and len(board.move_stack) < explore_plies
        for p in parts:
            p.cfg = exploring[id(p)] if early else greedy[id(p)]
        return player.choose(board)
    return move


def play(a, b, game, seed, explore_plies, temperature, max_plies):
    import chess
    a_white = game % 2 == 0
    players = {a: _player(a, seed, game, explore_plies, temperature),
               b: _player(b, seed, game, explore_plies, temperature)}
    board = chess.Board()
    spent = {a: 0.0, b: 0.0}
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        who = a if (board.turn == chess.WHITE) == a_white else b
        start = time.perf_counter()
        move = players[who](board)
        spent[who] += time.perf_counter() - start
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    winner = None if outcome is None or outcome.winner is None else (a if (outcome.winner == chess.WHITE) == a_white else b)
    return {"a": a, "b": b, "game": game, "white": a if a_white else b, "winner": winner,
            "end": outcome.termination.name if outcome else "max_plies", "plies": len(board.move_stack),
            "time": spent, "moves": " ".join(m.uci() for m in board.move_stack)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--explore-plies", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    jobs = [(a, b, g) for a, b in combinations(PLAYERS, 2) for g in range(args.games)]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init) as pool:
        games = list(pool.map(play, *zip(*[(a, b, g, args.seed, args.explore_plies, args.temperature, args.max_plies)
                                             for a, b, g in jobs])))
    table = {n: Counter() for n in PLAYERS}
    for g in games:
        for n in (g["a"], g["b"]):
            table[n]["jogos"] += 1
            table[n]["V" if g["winner"] == n else "E" if g["winner"] is None else "D"] += 1
            table[n]["s"] += g["time"][n]
    pairs = {}
    for g in games:
        pairs.setdefault(f"{g['a']} vs {g['b']}", Counter())[g["winner"] or "empate"] += 1
    distinct = {k: len({g["moves"] for g in games if f"{g['a']} vs {g['b']}" == k}) for k in pairs}
    report = {"label": args.label, "games_per_pair": args.games, "table": {n: dict(c) for n, c in table.items()},
              "pairs": {k: dict(v) for k, v in pairs.items()}, "distinct_games": distinct, "list": games}
    (ROOT / "claudio/results" / f"groups-{args.label}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for n, c in sorted(table.items(), key=lambda kv: -kv[1]["V"]):
        print(f"{n:20s} WR {100 * c['V'] / c['jogos']:5.1f}%  V/E/D {c['V']}/{c['E']}/{c['D']}  {c['s'] / c['jogos']:.2f} s/jogo")
    for k, v in pairs.items():
        print(f"  {k}: {dict(v)}  (jogos diferentes: {distinct[k]})")


if __name__ == "__main__":
    main()
