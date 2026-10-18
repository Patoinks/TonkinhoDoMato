"""G4 Leto (neutral or a Pantheon personality) against Stockfish 11 at a nominal Elo.

Stockfish plays with UCI_LimitStrength at --elo and --movetime seconds to think per move; Leto
plays with its checkpoint budgets. Games start from the initial position (the personality's
openings apply), colours alternate. Score is Leto's: win 1, draw 0.5, loss 0.

    python leto-rapido/vs_stockfish.py --deus zeus --games 12 --workers 6
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parent / "releases/g4-leto/models/g4t2-leto-v3.pt"
STOCKFISH = HERE.parent / "engines-local/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe"
_state = {}
# The fast Leto: same strength as the original (0.487 over 192 games), ~6.7x faster per game.
RAPIDO = ["root_prunings=True", "tactic_depth=7", "scan_prunings=True", "fast_net=True", "pawn_v4=True", "pawn_weight=0.5"]


class UCI:
    """A minimal blocking UCI client: no threads, no asyncio (python-chess's engine hung in pool workers)."""

    def __init__(self, path, elo):
        import subprocess
        self.proc = subprocess.Popen([str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("uci"); self.wait("uciok")
        for name, value in (("Threads", 1), ("UCI_LimitStrength", "true"), ("UCI_Elo", elo)):
            self.send(f"setoption name {name} value {value}")
        self.send("isready"); self.wait("readyok")

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def wait(self, token):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine closed")
            if line.startswith(token):
                return line

    def play(self, board, movetime):
        import chess
        moves = " ".join(m.uci() for m in board.move_stack)
        self.send(f"position startpos moves {moves}" if moves else "position startpos")
        self.send(f"go movetime {max(1, int(movetime * 1000))}")
        return chess.Move.from_uci(self.wait("bestmove").split()[1])


def _init(deus, elo, movetime, settings=()):
    import torch
    torch.set_num_threads(1)
    from claudio_load import load_players
    engine = UCI(STOCKFISH, elo)
    _state.update(players=load_players(CHECKPOINT, ["syzygy="] + list(settings)), deus=deus, engine=engine, movetime=movetime)


def play(game, seed, max_plies):
    import chess
    import numpy as np
    from claudio_load import make_player
    leto = make_player(_state["players"], np.random.default_rng([seed, game, 1]))
    if _state["deus"]:
        from personalidades import make
        leto.personality = make(_state["deus"])
    leto_white = game % 2 == 0
    board = chess.Board()
    started = time.perf_counter()
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        if (board.turn == chess.WHITE) == leto_white:
            move = leto.choose(board)
        else:
            move = _state["engine"].play(board, _state["movetime"])
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        score = 0.5
    else:
        score = 1.0 if outcome.winner == (chess.WHITE if leto_white else chess.BLACK) else 0.0
    return {"game": game, "leto_white": leto_white, "score": score, "plies": len(board.move_stack),
            "end": outcome.termination.name if outcome else "max_plies", "seconds": time.perf_counter() - started,
            "moves": " ".join(m.uci() for m in board.move_stack)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deus", help="personality name; omit for neutral Leto")
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--elo", type=int, default=1800)
    parser.add_argument("--movetime", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--set", action="append", default=[], help="Leto setting name=value")
    parser.add_argument("--rapido", action="store_true", help="the fast Leto (RAPIDO settings) for every player")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.rapido:
        args.set = RAPIDO + list(args.set)
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(args.deus, args.elo, args.movetime, args.set)) as pool:
        games = list(pool.map(play, range(args.games), [args.seed] * args.games, [args.max_plies] * args.games))
    wins = sum(g["score"] == 1 for g in games)
    draws = sum(g["score"] == 0.5 for g in games)
    summary = {"who": args.deus or "leto", "set": args.set, "elo": args.elo, "movetime": args.movetime, "games": len(games),
               "wins": wins, "draws": draws, "losses": len(games) - wins - draws,
               "score": sum(g["score"] for g in games) / len(games),
               "white_score": sum(g["score"] for g in games if g["leto_white"]) / max(1, sum(g["leto_white"] for g in games)),
               "black_score": sum(g["score"] for g in games if not g["leto_white"]) / max(1, sum(not g["leto_white"] for g in games))}
    print(json.dumps(summary))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "games": games}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
