"""Claudio or Timdgei against Stockfish at a nominal Elo (UCI_LimitStrength),
same openings for both players (random plies, colours swapped per opening),
no adjudication. Measurement only.

    python claudio/vs_engine.py --player claudio --elo 1800 --games 100 --label claudio-sf1800
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FAIRY, ROOT, STOCKFISH  # noqa: E402

import arena  # noqa: E402

ENGINES = {"sf": STOCKFISH, "fairy": FAIRY}
_w = {}


def _close_worker_engine():
    engine = _w.get("engine")
    _w["engine"] = None
    if engine is not None:
        engine.close()


def _init(elo, move_time, checkpoint, settings=(), engine="sf"):
    arena._init(str(checkpoint), list(settings), [])
    _w.update(elo=elo, move_time=move_time, engine=None, path=ENGINES[engine])


def play(game, seed, player, random_plies, max_plies):
    import chess
    import numpy as np
    from tonkai.cognitive_agent import CognitiveAgent
    from tonkai.opponents import CognitiveOpponent
    from tonkai.uci_opponent import UCIOpponent
    from claudio_agent import Claudio

    if _w["engine"] is None:
        _w["engine"] = UCIOpponent(_w["path"], _w["elo"], _w["move_time"])
    engine = _w["engine"]
    engine.reset()
    s, evo = arena._state, arena._state["evo"]
    rng = np.random.default_rng([seed, game])
    if player == "claudio":
        if "loaded" in s:
            from claudio_load import make_player
            me = make_player(s["loaded"], np.random.default_rng([seed, game, 1]))
        else:
            net, values, king = s["claudio"]
            me = Claudio(net, values, s["cfg"], rng=np.random.default_rng([seed, game, 1]), king_model=king)
    else:
        cfg = evo._config(s["genes"])
        me = CognitiveOpponent(CognitiveAgent(evo._state["net"], cfg, evo._state["knowledge"],
                                              kd_values=evo._state["kd"]), cfg, rng)
    white = game % 2 == 0
    board = arena.opening(seed, game // 2, random_plies)
    first = len(board.move_stack)
    spent, decisions = 0.0, 0
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        if (board.turn == chess.WHITE) == white:
            start = time.perf_counter()
            move = me.choose(board)
            spent += time.perf_counter() - start
            decisions += 1
        else:
            move = engine.choose(board, rng)
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    winner = None if outcome is None else outcome.winner
    result = "draw" if winner is None else "win" if winner == (chess.WHITE if white else chess.BLACK) else "loss"
    return {"game": game, "white": white, "result": result, "plies": len(board.move_stack) - first,
            "end": outcome.termination.name if outcome else "max_plies", "seconds": spent, "decisions": decisions,
            "moves": " ".join(m.uci() for m in board.move_stack)}


def play_batch(games, seed, player, random_plies, max_plies):
    """Run one complete worker batch, then close its persistent UCI engine."""
    try:
        return [play(game, seed, player, random_plies, max_plies) for game in games]
    finally:
        _close_worker_engine()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", choices=["claudio", "timdgei"], required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "claudio/claudio.pt",
                        help="Claudio checkpoint used when --player claudio")
    parser.add_argument("--set", action="append", default=[], help="Claudio setting name=value (over the checkpoint's)")
    parser.add_argument("--elo", type=int, default=1800)
    parser.add_argument("--engine", choices=("sf", "fairy"), default="sf")
    parser.add_argument("--move-time", type=float, default=0.002)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1800)
    parser.add_argument("--random-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    n = args.games
    batches = [list(range(worker, n, args.workers)) for worker in range(min(args.workers, n))]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(args.elo, args.move_time, args.checkpoint, args.set, args.engine)) as pool:
        parts = list(pool.map(play_batch, batches, [args.seed] * len(batches), [args.player] * len(batches),
                              [args.random_plies] * len(batches), [args.max_plies] * len(batches)))
    games = sorted((game for part in parts for game in part), key=lambda game: game["game"])
    w = sum(g["result"] == "win" for g in games)
    l = sum(g["result"] == "loss" for g in games)
    seconds, decisions = sum(g["seconds"] for g in games), sum(g["decisions"] for g in games)
    report = {"label": args.label, "player": args.player, "checkpoint": str(args.checkpoint), "settings": args.set,
              "elo": args.elo, "games": n, "win": w, "draw": n - w - l,
              "loss": l, "winrate": w / n, "s_per_game": seconds / n, "ms_per_move": 1000 * seconds / max(1, decisions),
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "list": games}
    (ROOT / "claudio/results" / f"engine-{args.label}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{args.label}: WR {100 * w / n:.1f}% (V/E/D {w}/{n - w - l}/{l}), {report['s_per_game']:.2f} s/jogo, "
          f"{report['ms_per_move']:.2f} ms/lance", flush=True)


if __name__ == "__main__":
    main()
