"""G4 Leto mutation tournament: the checkpoint's Leto against a mutant of itself.

Each opening (random plies from a seed) is played twice, colours swapped. The
result file records every move, so two runs of different code can be compared
move for move, and the seconds per game, the measure being optimized.

    python leto-rapido/torneio.py --games 20 --set reply_top=5 --out r.json
    python leto-rapido/torneio.py --games 20 --workers 6 --set ...
"""
import argparse
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # one BLAS thread per worker (rede_rapida)
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401  (frozen base code first)

CHECKPOINT = Path(__file__).resolve().parents[1] / "releases/g4-leto/models/g4t2-leto-v3.pt"
_state = {}
# The fast Leto: same strength as the original (0.487 over 192 games), ~6.7x faster per game.
RAPIDO = ["root_prunings=True", "tactic_depth=7", "scan_prunings=True", "fast_net=True", "pawn_v4=True", "pawn_weight=0.5",
          "shared_table=True", "reply_top=0"]   # one table per decision; no reply module (ab_stockfish.py, 500 games)


def _init(checkpoint, base_settings, mutant_settings, profile, deus=None, window=None):
    import torch
    torch.set_num_threads(1)
    from claudio_load import load_players
    common = ["syzygy="]   # the parity reference ran without tablebases (cfg.syzygy = "")
    _state.update(base=load_players(checkpoint, common + list(base_settings)),
                  mutant=load_players(checkpoint, common + list(mutant_settings)),
                  profile=profile, deus=deus, window=window)


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


def play(game, seed, max_plies, random_plies):
    import chess
    import numpy as np
    from claudio_load import make_player
    base = make_player(_state["base"], np.random.default_rng([seed, game, 1]))
    mutant = make_player(_state["mutant"], np.random.default_rng([seed, game, 2]))
    if _state["profile"]:
        base.clock, mutant.clock = {}, {}
    if _state.get("deus"):
        from personalidades import make
        mutant.personality = make(_state["deus"], _state.get("window"))
    mutant_white = game % 2 == 0
    board = opening(seed, game // 2, random_plies)
    first = len(board.move_stack)
    from fim_rapido import Judge
    judge = Judge(board)   # = board.outcome(claim_draw=True), without replaying the game each ply
    started = time.perf_counter()
    while judge.outcome(board) is None and len(board.move_stack) < max_plies:
        mutant_turn = (board.turn == chess.WHITE) == mutant_white
        move = (mutant if mutant_turn else base).choose(board)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {move} in {board.fen()}")
        judge.push(board, move)
    seconds = time.perf_counter() - started
    outcome = judge.outcome(board)
    if outcome is None or outcome.winner is None:
        score = 0.5
    else:
        score = 1.0 if outcome.winner == (chess.WHITE if mutant_white else chess.BLACK) else 0.0
    row = {"game": game, "mutant_white": mutant_white, "score": score, "plies": len(board.move_stack) - first,
           "end": outcome.termination.name if outcome else "max_plies", "seconds": seconds,
           "moves": " ".join(m.uci() for m in board.move_stack)}
    if _state.get("deus"):
        row["style"] = dict(mutant.personality.stats)
    if _state["profile"]:
        clock = {}
        for player in (base, mutant):
            for key, value in (player.clock or {}).items():
                clock[key] = clock.get(key, 0.0) + value
        row["clock"] = clock
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--random-plies", type=int, default=6)
    parser.add_argument("--set", action="append", default=[], help="mutant setting name=value")
    parser.add_argument("--base-set", action="append", default=[], help="baseline setting name=value")
    parser.add_argument("--profile", action="store_true", help="record seconds per decision section")
    parser.add_argument("--deus", help="the mutant plays as this Pantheon personality (personalidades.py)")
    parser.add_argument("--window", type=float, help="personality strength: score window (default 0.06)")
    parser.add_argument("--rapido", action="store_true", help="the fast Leto (RAPIDO settings) for every player")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.rapido:
        args.set = RAPIDO + list(args.set)
        args.base_set = RAPIDO + list(args.base_set)
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(str(args.checkpoint), args.base_set, args.set, args.profile, args.deus, args.window)) as pool:
        games = list(pool.map(play, range(args.games), [args.seed] * args.games,
                              [args.max_plies] * args.games, [args.random_plies] * args.games))
    wall = time.perf_counter() - started
    seconds = sum(g["seconds"] for g in games)
    score = sum(g["score"] for g in games) / len(games)
    summary = {"games": len(games), "wall_seconds": wall, "game_seconds": seconds,
               "seconds_per_game": seconds / len(games), "plies_per_game": sum(g["plies"] for g in games) / len(games),
               "mutant_score": score, "set": args.set, "base_set": args.base_set, "workers": args.workers,
               "seed": args.seed, "deus": args.deus}
    if args.deus:
        totals = {k: sum(g["style"][k] for g in games) for k in ("decisions", "changed", "book")}
        totals["changed_rate"] = round(totals["changed"] / max(1, totals["decisions"]), 3)
        summary["style"] = totals
    if args.profile:
        clock = {}
        for g in games:
            for key, value in g.get("clock", {}).items():
                clock[key] = clock.get(key, 0.0) + value
        summary["clock_share"] = {k: round(v / seconds, 4) for k, v in sorted(clock.items(), key=lambda kv: -kv[1])}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "games": games}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
