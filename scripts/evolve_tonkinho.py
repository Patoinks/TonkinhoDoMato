"""Evolution of TonkAI's decision settings (the owner's tournament).

Every mutant uses the SAME network (the v4 base); only its decision settings
(search, weights of its own measures, temperature, mate calculation...) are
mutated at random. A generation is a knock-out tournament: random pairs play a
best-of-3 (colours alternate, 3rd game random colour, a draw is half a point),
the winner goes through and a tie kills both, until one is left. That champion
enters the next generation unchanged together with its mutated children
(generation 1: the base and its mutants). The champion of the last generation
is Tonkão JR, who then plays the unmutated base.

Nothing is trained and nothing is written into a campaign.

    python scripts/evolve_tonkinho.py --population 24 --generations 10 --final-games 200 --workers 4
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import copy
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# gene: (kind, low, high). Only decision-time settings; the network is shared.
GENES = {
    "temperature": ("float", 0.02, 0.20),
    "memory_weight": ("float", 0.0, 0.30),
    "information_bonus": ("float", 0.0, 0.06),
    "planning_candidates": ("int", 1, 6),
    "planning_replies": ("int", 1, 6),
    "planning_nodes": ("int", 24, 256),
    "planning_uncertainty": ("float", 0.02, 0.30),
    "planning_gap": ("float", 0.0, 0.20),
    "robust_weight": ("float", 0.0, 1.0),
    "skill_bonus": ("float", 0.0, 0.06),
    "piece_safety_weight": ("float", 0.0, 0.20),
    "piece_exchange_tolerance": ("float", 0.1, 1.0),
    "king_safety_weight": ("float", 0.0, 0.15),
    "kd_weight": ("float", 0.0, 1.0),
    "mate_search_moves": ("int", 0, 3),
    "mate_defense_moves": ("int", 0, 2),
    "avoid_repetition_margin": ("float", 0.0, 4.0),
    "exchange_depth": ("int", 0, 8),
}

_state = {}


def sub(seed, *extra):
    """A flat integer seed list: seed (an int or a flat list) followed by `extra`."""
    return [*(seed if isinstance(seed, (list, tuple)) else [seed]), *extra]


def base_genes(cfg):
    return {name: getattr(cfg, name) for name in GENES}


def mutate(genes, rng, rate=0.5, strength=0.25):
    """Each gene changes with probability `rate`: floats by a random step of
    `strength` x its range, integers by one step; always kept in range."""
    child = dict(genes)
    for name, (kind, low, high) in GENES.items():
        if rng.random() >= rate:
            continue
        if kind == "int":
            child[name] = int(min(high, max(low, child[name] + rng.choice([-1, 1]) *
                                             max(1, round(strength * (high - low) * rng.random())))))
        else:
            child[name] = float(min(high, max(low, child[name] + rng.normal(0, strength * (high - low)))))
    return child


def _init(base):
    import torch
    torch.set_num_threads(1)
    from match_checkpoints import _load
    name, net, knowledge, cfg, kd = _load(base)
    _state.update(name=name, net=net, knowledge=knowledge, cfg=cfg, kd=kd)


def _config(genes):
    cfg = copy.copy(_state["cfg"])
    for name, value in genes.items():
        setattr(cfg, name, value)
    cfg.validate()
    return cfg


def _game(genes_white, genes_black, seed):
    """1, .5 or 0 for the white side."""
    import chess
    from tonkai.cognitive_agent import CognitiveAgent
    from tonkai.opponents import CognitiveOpponent
    from tonkai.survival_agent import play_episode

    s = _state
    white, black = _config(genes_white), _config(genes_black)
    agent = CognitiveAgent(s["net"], white, s["knowledge"], kd_values=s["kd"])
    rival = CognitiveOpponent(CognitiveAgent(s["net"], black, s["knowledge"], kd_values=s["kd"]), black,
                              np.random.default_rng(sub(seed, 1)))
    _, board, _, _ = play_episode(agent, rival, white, np.random.default_rng(sub(seed, 0)), self_color=chess.WHITE)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def _final_game(genes_current, genes_base, game, seed):
    """Points of the current side in one final game; colours alternate."""
    if game % 2 == 0:
        return _game(genes_current, genes_base, sub(seed, game))
    return 1 - _game(genes_base, genes_current, sub(seed, game))


def tournament(pool, population, rng, seed):
    """Random pairs, best-of-3; returns (survivors, rows). A tie kills both; an
    odd member out survives untouched. Every game runs in parallel: games 1 and
    2 of all pairs at once, then the third games that can still matter."""
    order = list(rng.permutation(len(population)))
    pairs = [(population[order[i]], population[order[i + 1]]) for i in range(0, len(order) - 1, 2)]
    bye = [population[order[-1]]] if len(order) % 2 else []
    points = [[0.0, 0.0] for _ in pairs]
    games = [0 for _ in pairs]

    def play(i, game, a_white):
        a, b = pairs[i]
        white, black = (a, b) if a_white else (b, a)
        return pool.submit(_game, white["genes"], black["genes"], sub(seed, i, game)), i, a_white

    def collect(futures):
        for future, i, a_white in futures:
            score = future.result()
            points[i][0 if a_white else 1] += score
            points[i][1 if a_white else 0] += 1 - score
            games[i] += 1

    collect([play(i, game, game == 0) for i in range(len(pairs)) for game in (0, 1)])
    collect([play(i, 2, bool(rng.random() < .5)) for i in range(len(pairs))
             if abs(points[i][0] - points[i][1]) <= 1])       # the third game can still change the result
    survivors, rows = list(bye), []
    for (a, b), (pa, pb), n in zip(pairs, points, games):
        winner = a if pa > pb else b if pb > pa else None
        if winner is not None:
            survivors.append(winner)
            winner["wins"] = winner.get("wins", 0) + 1
        rows.append({"a": a["id"], "b": b["id"], "points": [pa, pb], "games": n,
                     "survivor": winner["id"] if winner else None})
    return survivors, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "checkpoints/cognitive-mix-20260923/latest.pt")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--final-games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out", type=Path, default=ROOT / "history/evolution")
    parser.add_argument("--resume", type=Path, help="an evolution folder: continue after its last generation")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.resume or args.out / stamp
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from tonkai.cognitive_config import CognitiveConfig
    from match_checkpoints import _historical_config
    data = torch.load(args.base, map_location="cpu", weights_only=False)
    base = {"id": "base", "genes": base_genes(_historical_config(data)), "parent": None}
    del data
    log = {"base_file": str(args.base), "base_genes": base["genes"], "generations": []}
    champion, first = base, 1
    if args.resume:
        log = json.loads((out / "evolution.json").read_text(encoding="utf-8"))
        base = {"id": "base", "genes": log["base_genes"], "parent": None}
        last = log["generations"][-1]
        champion = next(p for p in last["population"] if p["id"] == last["champion"])
        first = last["generation"] + 1
        rng = np.random.default_rng([args.seed, first])        # a fresh stream for the continued run

    def save():
        (out / "evolution.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    def knockout(pool, population, generation):
        """Rounds of best-of-3 until one is left; a round where everyone ties is replayed."""
        rounds, round_number = [], 0
        while len(population) > 1:
            round_number += 1
            survivors, rows = tournament(pool, population, rng, [args.seed, generation, round_number])
            rounds.append({"round": round_number, "players": len(population), "matches": rows,
                           "survivors": [s["id"] for s in survivors]})
            population = survivors or population
        return population[0], rounds

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(str(args.base),)) as pool:
        for generation in range(first, args.generations + 1):
            population = [dict(champion)] + [
                {"id": f"G{generation}-M{i + 1}", "parent": champion["id"],
                 "genes": mutate(champion["genes"], rng, rate=.7 if generation == 1 else .5,
                                 strength=.35 if generation == 1 else .25)}
                for i in range(args.population - 1)]
            champion, rounds = knockout(pool, population, generation)
            log["generations"].append({"generation": generation, "population": population, "rounds": rounds,
                                       "champion": champion["id"]})
            save()
            print(f"geração {generation}: campeão {champion['id']} "
                  f"({sum(r['survivor'] is None for rd in rounds for r in rd['matches'])} empates mataram os dois)",
                  flush=True)
        print(f"Tonkão JR = {champion['id']}", flush=True)

        futures = [pool.submit(_final_game, champion["genes"], base["genes"], game, [args.seed, 5000])
                   for game in range(args.final_games)]
        scores = [future.result() for future in futures]

    mean = sum(scores) / len(scores)
    se = math.sqrt(sum((x - mean) ** 2 for x in scores) / max(1, len(scores) - 1) / len(scores))
    log["tonkao_jr"] = {"id": champion["id"], "genes": champion["genes"],
                        "changed": {k: [base["genes"][k], v] for k, v in champion["genes"].items()
                                    if v != base["genes"][k]}}
    log["final"] = {"games": len(scores), "wins": scores.count(1.0), "draws": scores.count(0.5),
                    "losses": scores.count(0.0), "score": mean,
                    "score_95": [max(0, mean - 1.96 * se), min(1, mean + 1.96 * se)]}
    save()
    print(json.dumps(log["final"], ensure_ascii=False), flush=True)
    print(f"relatório: {out / 'evolution.json'}", flush=True)


if __name__ == "__main__":
    main()
