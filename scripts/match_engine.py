"""TonkAI (a checkpoint + decision settings) against a UCI engine, in parallel.

Colours alternate; nothing is trained. Optional tablebase adjudication ends a
game at <= 5 pieces with the exact result.

    python scripts/match_engine.py --checkpoint checkpoints/duel-G5-M23/latest.pt --player G5-M23 \\
        --elo 1800 --games 50 --syzygy engines/syzygy --label g5m23-vs-sf1800
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from datetime import datetime, timezone
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
ENGINE = ROOT / "engines/stockfish-11/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe"
_state = {}


def _init(checkpoint, settings, elo, move_time, engine=str(ENGINE), tiny=False, greedy=False):
    import torch
    torch.set_num_threads(1)
    from match_checkpoints import _load
    from tonkai.uci_opponent import UCIOpponent
    name, net, knowledge, cfg, kd = _load(checkpoint)
    cfg = copy(cfg)
    for key, value in settings.items():
        setattr(cfg, key, value)
    cfg.threat_explore = cfg.trade_explore = 0.0          # training-only exploration
    cfg.validate()
    # The engine starts on the first game, not while every worker starts at once,
    # and is closed at exit (its thread would otherwise keep the worker alive).
    import atexit
    atexit.register(lambda: _state.get("engine") and _state["engine"].close())
    _state.update(net=net, knowledge=knowledge, cfg=cfg, kd=kd, tiny=tiny, greedy=greedy,
                  engine=None, make_engine=lambda: UCIOpponent(engine, elo, move_time))


def _game(index, seed):
    import chess
    import numpy as np
    from tonkai.cognitive_agent import CognitiveAgent
    from tonkai.survival_agent import play_episode
    s = _state
    if s["tiny"]:
        from tonkai.opponents import TinyOpponentAdapter
        opponent = TinyOpponentAdapter(seed * 100_003 + index)      # seeded: the same games for every model
    else:
        if s["engine"] is None:
            s["engine"] = s["make_engine"]()
        opponent = s["engine"]
    colour = chess.WHITE if index % 2 == 0 else chess.BLACK
    agent = CognitiveAgent(s["net"], s["cfg"], s["knowledge"], kd_values=s["kd"])
    _, board, colour, stats = play_episode(agent, opponent, s["cfg"], np.random.default_rng([seed, index]),
                                           self_color=colour, opponent_name=opponent.description,
                                           explore=not s["greedy"], log_king_safety=False)
    def out(result, how):
        return result, how, len(board.move_stack), {"game": index, "tonkai": "white" if colour else "black",
                                                    "moves": [move.uci() for move in board.move_stack]}
    if stats.get("stalled"):
        return out("loss", "retardamento")
    if stats.get("tablebase"):
        result, how = stats["result"], "tabela de finais"
    else:
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return out("unfinished", "limite de jogadas")
        result, how = outcome.result(), outcome.termination.name.lower()
    if result == "1/2-1/2":
        return out("draw", how)
    return out("win" if (result == "1-0") == (colour == chess.WHITE) else "loss", how)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--player", help="tournament player whose settings to use (scripts/train_duel.py genes)")
    parser.add_argument("--set", action="append", default=[], dest="settings")
    parser.add_argument("--elo", type=int, default=1800)
    parser.add_argument("--engine", type=Path, default=ENGINE, help="UCI engine (Stockfish 11 by default)")
    parser.add_argument("--tiny", action="store_true", help="play TinyOpponent (seeded per game) instead of a UCI engine")
    parser.add_argument("--greedy", action="store_true", help="no exploration: always the best-valued move")
    parser.add_argument("--move-time", type=float, default=0.002)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--syzygy", type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    from tonkai.cognitive_config import CognitiveConfig
    settings = {}
    if args.player:
        import train_duel
        settings.update(train_duel.genes_of(args.player))
    defaults = CognitiveConfig()
    for item in args.settings:
        key, _, raw = item.partition("=")
        kind = type(getattr(defaults, key)) if getattr(defaults, key) is not None else float
        settings[key] = (raw.lower() in ("1", "true")) if kind is bool else kind(raw)
    if args.syzygy:
        settings.update(syzygy_path=str(args.syzygy.resolve()), tablebase_adjudicate=True)
    pool = ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                               initargs=(str(args.checkpoint), settings, args.elo, args.move_time, str(args.engine),
                                         args.tiny, args.greedy))
    rows = list(pool.map(_game, range(args.games), [args.seed] * args.games))
    counts = {k: sum(r[0] == k for r in rows) for k in ("win", "draw", "loss", "unfinished")}
    done = [r for r in rows if r[0] != "unfinished"]
    score = sum(1 if r[0] == "win" else .5 if r[0] == "draw" else 0 for r in done) / max(1, len(done))
    how = {}
    for result, reason, *_ in rows:
        how[f"{result}: {reason}"] = how.get(f"{result}: {reason}", 0) + 1
    report = {"checkpoint": str(args.checkpoint), "player": args.player,
              "opponent": "TinyOpponent" if args.tiny else str(args.engine.name), "greedy": args.greedy,
              "engine_elo": None if args.tiny else args.elo,
              "tablebase": bool(args.syzygy), **counts, "score": score, "how": how,
              "mean_plies": sum(r[2] for r in rows) / len(rows),
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (ROOT / "history/matches" / f"{args.label}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    games = [{**record, "result": result, "how": how} for result, how, _, record in rows]
    (ROOT / "history/matches" / f"{args.label}.games.json").write_text(json.dumps(games), encoding="utf-8")
    for process in list(pool._processes.values()):   # the report is saved: never wait on engine threads
        process.terminate()
    pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
