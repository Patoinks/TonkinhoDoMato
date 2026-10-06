"""Timdgei vs Timdgei with some settings changed on one side (the 'mutant'):
which of Timdgei's modules are worth their time? Same arena rules (random
openings, colours swapped per opening, no adjudication). Measurement only.

    python claudio/ablate_timdgei.py --games 200 --label no-memory --set memory_weight=0
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

import arena  # noqa: E402

_s = {}


def _parse(raw):
    for kind in (int, float):
        try:
            return kind(raw)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(raw.lower(), raw)


def _init(settings):
    arena._init("", [], [])
    genes = dict(arena._state["genes"])
    for item in settings:
        key, _, raw = item.partition("=")
        genes[key] = _parse(raw)
    _s["mutant"] = genes


def play(game, seed, max_plies, random_plies):
    import chess
    import numpy as np
    from tonkai.cognitive_agent import CognitiveAgent
    from tonkai.opponents import CognitiveOpponent
    evo = arena._state["evo"]
    rng = np.random.default_rng([seed, game])

    def side(genes, extra):
        cfg = evo._config(genes)
        return CognitiveOpponent(CognitiveAgent(evo._state["net"], cfg, evo._state["knowledge"],
                                                kd_values=evo._state["kd"]), cfg, np.random.default_rng([seed, game, extra]))
    mutant, base = side(_s["mutant"], 1), side(arena._state["genes"], 2)
    mutant_white = game % 2 == 0
    board = arena.opening(seed, game // 2, random_plies)
    spent = {"mutant": 0.0, "base": 0.0}
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        who = "mutant" if (board.turn == chess.WHITE) == mutant_white else "base"
        start = time.perf_counter()
        move = (mutant if who == "mutant" else base).choose(board)
        spent[who] += time.perf_counter() - start
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    winner = None if outcome is None else outcome.winner
    result = "draw" if winner is None else "win" if winner == (chess.WHITE if mutant_white else chess.BLACK) else "loss"
    return {"result": result, "plies": len(board.move_stack), "time": spent}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(args.set,)) as pool:
        games = list(pool.map(play, range(args.games), [args.seed] * args.games, [300] * args.games,
                              [4] * args.games))
    n = len(games)
    w = sum(g["result"] == "win" for g in games)
    l = sum(g["result"] == "loss" for g in games)
    report = {"label": args.label, "settings": args.set, "games": n, "win": w, "loss": l, "draw": n - w - l,
              "mutant_s_per_game": sum(g["time"]["mutant"] for g in games) / n,
              "base_s_per_game": sum(g["time"]["base"] for g in games) / n}
    (ROOT / "claudio/results" / f"ablate-{args.label}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{args.label}: mutant WR {100 * w / n:.1f}% vs base {100 * l / n:.1f}% (V/E/D {w}/{n - w - l}/{l}), "
          f"mutant {report['mutant_s_per_game']:.2f} s/jogo, base {report['base_s_per_game']:.2f} s/jogo", flush=True)


if __name__ == "__main__":
    main()
