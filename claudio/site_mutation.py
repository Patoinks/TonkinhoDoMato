"""The site player (Tonkclaudinho Filhodegei V2 exactly as the browser plays it:
v4 engine, site settings, no exploration) against a mutant of itself with some
settings changed. Random openings, colours swapped per opening. Measurement only.

    python claudio/site_mutation.py --games 50 --label kd-inverse --set kd_weight=0.82696
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402  (frozen base code first)

import arena  # noqa: E402

SITE = ROOT / "claudio/site/tonkclaudinho-filhodegei-v2.pt"
SITE_SETTINGS = {"temperature": 0.1379082335254832, "memory_weight": 0.18063696348922267,
                 "information_bonus": 0.011713458709919158, "planning_candidates": 1, "planning_replies": 2,
                 "planning_nodes": 71, "planning_uncertainty": 0.07345297296692446, "planning_gap": 0.2,
                 "robust_weight": 0.4513849993253386, "skill_bonus": 0.0, "piece_safety_weight": 0.2,
                 "piece_exchange_tolerance": 0.21289753701988345, "king_safety_weight": 0.04679613775293547,
                 "kd_weight": 0.17304461792996828, "mate_search_moves": 3, "mate_defense_moves": 2,
                 "avoid_repetition_margin": 1.2958544327908663, "exchange_depth": 5}
_s = {}


def _parse(raw):
    for kind in (int, float):
        try:
            return kind(raw)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(raw.lower(), raw)


def _init(settings):
    import psutil_free_priority  # noqa: F401
    import torch
    torch.set_num_threads(1)
    from tonkai.cognitive_config import CognitiveConfig
    from tonkai.cognitive_trainer import CognitiveKnowledge
    from tonkai.temporal_model import SurvivalNet
    state = torch.load(SITE, map_location="cpu", weights_only=False)
    base = CognitiveConfig.from_dict(state["config"])
    base.device = "cpu"
    for key, value in SITE_SETTINGS.items():
        setattr(base, key, value)
    base.threat_explore = base.trade_explore = 0.0
    base.validate()
    mutant = copy.copy(base)
    for item in settings:
        key, _, raw = item.partition("=")
        setattr(mutant, key, _parse(raw))
    mutant.validate()
    net = SurvivalNet(base).eval().requires_grad_(False)
    net.load_state_dict(state["model"])
    knowledge = CognitiveKnowledge(base, net.encoder)
    knowledge.load_state_dict(state["knowledge"])
    kd = {int(k): float(v) for k, v in state["kd_values_override"].items()}
    _s.update(net=net, knowledge=knowledge, kd=kd, base=base, mutant=mutant)


class SitePlayer:
    """The browser's way: every real position observed in order, then the best move (no exploration)."""

    def __init__(self, cfg):
        from tonkai.cognitive_agent import CognitiveAgent
        self.agent = CognitiveAgent(_s["net"], cfg, _s["knowledge"], kd_values=_s["kd"])
        self.board = None

    def choose(self, board):
        import numpy as np
        colour = board.turn
        if self.board is None:
            self.agent.reset()
            self.board = board.root()
            self.agent.observe(self.board, colour)
        for move in board.move_stack[len(self.board.move_stack):]:
            self.board.push(move)
            self.agent.observe(self.board, colour)
        return self.agent.choose(self.board, colour, np.random.default_rng(0), explore=False, temperature=0).move


def play(game, seed, random_plies, max_plies):
    import chess
    mutant, base = SitePlayer(_s["mutant"]), SitePlayer(_s["base"])
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
    return {"game": game, "mutant_white": mutant_white, "result": result, "plies": len(board.move_stack),
            "end": outcome.termination.name if outcome else "max_plies", "time": spent,
            "moves": " ".join(m.uci() for m in board.move_stack)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--random-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--set", action="append", default=[], help="mutant setting name=value")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    n = args.games
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(args.set,)) as pool:
        games = list(pool.map(play, range(n), [args.seed] * n, [args.random_plies] * n, [args.max_plies] * n))
    w = sum(g["result"] == "win" for g in games)
    l = sum(g["result"] == "loss" for g in games)
    report = {"label": args.label, "mutant_settings": args.set, "games": n, "win": w, "draw": n - w - l, "loss": l,
              "mutant_s_per_game": sum(g["time"]["mutant"] for g in games) / n,
              "base_s_per_game": sum(g["time"]["base"] for g in games) / n, "list": games}
    (ROOT / "claudio/results" / f"site-mutation-{args.label}.json").write_text(json.dumps(report, indent=1),
                                                                                encoding="utf-8")
    print(f"{args.label}: mutante WR {100 * w / n:.0f}% vs site {100 * l / n:.0f}% (V/E/D {w}/{n - w - l}/{l}), "
          f"mutante {report['mutant_s_per_game']:.2f} s/jogo, site {report['base_s_per_game']:.2f} s/jogo", flush=True)


if __name__ == "__main__":
    main()
