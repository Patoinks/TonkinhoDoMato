"""Claudio vs Timdgei (tournament settings): result AND thinking time per game.

Measurement only - nothing is trained. Each side's thinking time is the wall
time of its own decisions, in the same process and the same game, so both
sides share whatever load the machine has. Colours alternate. The game ends by
the rules (checkmate, stalemate, threefold, fifty moves, insufficient material)
or at --max-plies (a draw); no tablebase adjudication.

Both players are nearly deterministic, so every game starts after
--random-plies uniformly random legal moves (seeded); each opening is played
twice, colours swapped (games 2k and 2k+1). Only measurement uses them.

    python claudio/arena.py --games 200 --workers 6 --label v0 --set tactic_rounds=4
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402  (frozen base code first)

EVOLUTION = ROOT / "history/evolution/20260924T190310Z/evolution.json"
TIMDGEI = ROOT / "checkpoints/obj-max-continuo/latest.pt"
_state = {}


def _init(claudio_path, settings, rival_settings):
    import psutil_free_priority  # noqa: F401  (below-normal priority, see module)
    import torch
    torch.set_num_threads(1)
    import evolve_tonkinho as evo
    from claudio_agent import ClaudioConfig
    evo._init(str(TIMDGEI))
    genes = dict(json.loads(EVOLUTION.read_text(encoding="utf-8"))["base_genes"], tactic_search="exact")
    for item in rival_settings:
        key, _, raw = item.partition("=")
        genes[key] = raw
    _state.update(evo=evo, genes=genes, cfg=ClaudioConfig.with_settings(settings))
    if claudio_path:
        from claudio_load import load_claudio
        from claudio_load import load_players
        _state["loaded"] = load_players(claudio_path, settings)          # one checkpoint, or white+black
        net, values, cfg, king = _state["loaded"][0]
        _state["claudio"] = (net, values, king)
        _state["cfg"] = cfg
    else:
        from tonkai.king_safety_model import load_cached
        _state["claudio"] = (evo._state["net"], evo._state["kd"], load_cached(evo._state["cfg"].king_safety_model))


def opening(seed, index, plies):
    import chess
    import numpy as np
    rng = np.random.default_rng([seed, index, 7])
    board = chess.Board()
    while True:
        board.reset()
        for _ in range(plies):
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
        if not board.is_game_over(claim_draw=True):
            return board


def play(game, seed, max_plies, random_plies=0):
    import chess
    import numpy as np
    from tonkai.cognitive_agent import CognitiveAgent
    from tonkai.opponents import CognitiveOpponent
    from claudio_agent import Claudio

    s, evo = _state, _state["evo"]
    rng = np.random.default_rng([seed, game])
    if "loaded" in s:
        from claudio_load import make_player
        claudio = make_player(s["loaded"], np.random.default_rng([seed, game, 1]))
    else:
        net, values, king = s["claudio"]
        claudio = Claudio(net, values, s["cfg"], rng=np.random.default_rng([seed, game, 1]), king_model=king)
    rcfg = evo._config(s["genes"])
    timdgei = CognitiveOpponent(CognitiveAgent(evo._state["net"], rcfg, evo._state["knowledge"],
                                               kd_values=evo._state["kd"]), rcfg, rng)
    claudio_white = game % 2 == 0
    board = opening(seed, game // 2, random_plies) if random_plies else chess.Board()
    first = len(board.move_stack)
    spent = {"claudio": 0.0, "timdgei": 0.0}
    decisions = {"claudio": 0, "timdgei": 0}
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        side = "claudio" if (board.turn == chess.WHITE) == claudio_white else "timdgei"
        start = time.perf_counter()
        move = claudio.choose(board) if side == "claudio" else timdgei.choose(board)
        spent[side] += time.perf_counter() - start
        decisions[side] += 1
        if move not in board.legal_moves:
            raise ValueError(f"{side} played an illegal move {move} in {board.fen()}")
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        score = 0.5
    else:
        score = 1.0 if outcome.winner == (chess.WHITE if claudio_white else chess.BLACK) else 0.0
    return {"game": game, "claudio_white": claudio_white, "score": score, "plies": len(board.move_stack) - first,
            "end": outcome.termination.name if outcome else "max_plies", "time": spent, "decisions": decisions,
            "moves": " ".join(m.uci() for m in board.move_stack)}


def summary(games, draw=0.4):
    n = len(games)
    win = sum(g["score"] == 1 for g in games)
    loss = sum(g["score"] == 0 for g in games)
    drawn = n - win - loss
    score = (win + 0.5 * drawn) / n
    rows = {}
    for side in ("claudio", "timdgei"):
        seconds = sum(g["time"][side] for g in games)
        count = sum(g["decisions"][side] for g in games)
        rows[side] = {"s_per_game": seconds / n, "ms_per_decision": 1000 * seconds / max(1, count)}
    ci = 1.96 * math.sqrt(max(score * (1 - score), 1e-9) / n)
    return {"games": n, "win": win, "draw": drawn, "loss": loss, "score": score, "score_95": [score - ci, score + ci],
            "points_draw_0_4": (win + draw * drawn) / n,
            "white": _colour(games, True), "black": _colour(games, False),
            "plies_per_game": sum(g["plies"] for g in games) / n, "time": rows,
            "speedup": rows["timdgei"]["s_per_game"] / max(1e-9, rows["claudio"]["s_per_game"])}


def _colour(games, white):
    part = [g for g in games if g["claudio_white"] == white]
    return {"W/D/L": [sum(g["score"] == 1 for g in part), sum(g["score"] == .5 for g in part),
                      sum(g["score"] == 0 for g in part)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--random-plies", type=int, default=4, help="random opening plies (0 = start position)")
    parser.add_argument("--claudio", type=Path, help="Claudio checkpoint (default: Timdgei's network and values)")
    parser.add_argument("--set", action="append", default=[], help="Claudio setting name=value")
    parser.add_argument("--rival-set", action="append", default=[], help="Timdgei setting name=value")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    out = ROOT / "claudio/results" / f"arena-{args.label}.json"
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(str(args.claudio) if args.claudio else "", args.set, args.rival_set)) as pool:
        games = list(pool.map(play, range(args.games), [args.seed] * args.games, [args.max_plies] * args.games,
                              [args.random_plies] * args.games))
    result = {"label": args.label, "settings": args.set, "random_plies": args.random_plies, "rival_settings": args.rival_set,
              "claudio": str(args.claudio or ""), "wall_seconds": time.perf_counter() - start,
              "summary": summary(games), "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "games": games}
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    s = result["summary"]
    print(f"{args.label}: score {s['score']:.3f} (W/D/L {s['win']}/{s['draw']}/{s['loss']}), "
          f"Claudio {s['time']['claudio']['ms_per_decision']:.2f} ms/lance {s['time']['claudio']['s_per_game']:.2f} s/jogo, "
          f"Timdgei {s['time']['timdgei']['ms_per_decision']:.2f} ms/lance {s['time']['timdgei']['s_per_game']:.2f} s/jogo, "
          f"x{s['speedup']:.1f} mais rapido, {s['plies_per_game']:.0f} meios-lances/jogo", flush=True)


if __name__ == "__main__":
    main()
