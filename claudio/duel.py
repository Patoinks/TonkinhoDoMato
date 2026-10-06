"""A neutral 1v1 between two UCI engines, each in its own process.

Each engine's time is measured from outside: from sending "go" to receiving
"bestmove" (the arbiter's own bookkeeping is not counted). Colours alternate;
with --random-plies each opening is played twice, colours swapped. The game
ends by the rules (checkmate, stalemate, threefold, fifty moves, insufficient
material) or at --max-plies (a draw). An illegal move loses the game.
Winrate = wins / games; a draw is nobody's win.

    python claudio/duel.py --a "python claudio/uci.py" --a-name Claudio \
                           --b "<Gepeto's UCI command>" --b-name Gepeto --games 20
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import chess
import numpy as np

HERE = Path(__file__).resolve().parent


class Engine:
    def __init__(self, command, name, cwd):
        self.name = name
        self.proc = subprocess.Popen(shlex.split(command, posix=False), cwd=cwd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("uci")
        self.wait("uciok")
        self.ready()

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def wait(self, token):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.name} closed its output")
            if line.split() and line.split()[0] == token:
                return line.split()

    def ready(self):
        self.send("isready")
        self.wait("readyok")

    def new_game(self):
        self.send("ucinewgame")
        self.ready()

    def move(self, board):
        root = board.root()
        start = "startpos" if root.fen() == chess.STARTING_FEN else f"fen {root.fen()}"
        moves = " ".join(m.uci() for m in board.move_stack)
        self.send(f"position {start}" + (f" moves {moves}" if moves else ""))
        self.send("go")
        began = time.perf_counter()
        words = self.wait("bestmove")
        return words[1], time.perf_counter() - began

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def opening(seed, index, plies):
    rng = np.random.default_rng([seed, index, 7])
    board = chess.Board()
    while True:
        board.reset()
        for _ in range(plies):
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
        if not board.is_game_over(claim_draw=True):
            return chess.Board(board.fen())      # the opening is the game's starting position


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="UCI command of engine A")
    parser.add_argument("--a-name", default="A")
    parser.add_argument("--b", required=True, help="UCI command of engine B")
    parser.add_argument("--b-name", default="B")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--random-plies", type=int, default=0, help="random opening plies (0 = normal start)")
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cwd", default=str(HERE.parent), help="working directory of both engines")
    parser.add_argument("--a-cwd", help="working directory of engine A (default --cwd)")
    parser.add_argument("--b-cwd", help="working directory of engine B (default --cwd)")
    parser.add_argument("--label", default="duel")
    args = parser.parse_args()
    engines = {"a": Engine(args.a, args.a_name, args.a_cwd or args.cwd),
               "b": Engine(args.b, args.b_name, args.b_cwd or args.cwd)}
    names = {"a": args.a_name, "b": args.b_name}
    games = []
    try:
        for game in range(args.games):
            a_white = game % 2 == 0
            board = opening(args.seed, game // 2, args.random_plies) if args.random_plies else chess.Board()
            start_fen = board.fen()
            for engine in engines.values():
                engine.new_game()
            spent, moves_made, forfeit = {"a": 0.0, "b": 0.0}, {"a": 0, "b": 0}, None
            while not board.is_game_over(claim_draw=True) and len(board.move_stack) < args.max_plies:
                side = "a" if (board.turn == chess.WHITE) == a_white else "b"
                uci, seconds = engines[side].move(board)
                spent[side] += seconds
                moves_made[side] += 1
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    move = None
                if move is None or move not in board.legal_moves:
                    forfeit = side
                    break
                board.push(move)
            outcome = board.outcome(claim_draw=True)
            if forfeit:
                winner = "b" if forfeit == "a" else "a"
                end = f"illegal move by {names[forfeit]}"
            elif outcome is None or outcome.winner is None:
                winner, end = None, outcome.termination.name if outcome else "max_plies"
            else:
                winner = "a" if (outcome.winner == chess.WHITE) == a_white else "b"
                end = outcome.termination.name
            games.append({"game": game + 1, "white": names["a" if a_white else "b"], "start": start_fen,
                          "winner": names[winner] if winner else None, "end": end, "plies": len(board.move_stack),
                          "seconds": {names[k]: v for k, v in spent.items()},
                          "moves": {names[k]: v for k, v in moves_made.items()},
                          "uci": " ".join(m.uci() for m in board.move_stack)})
            print(f"game {game + 1}: white {games[-1]['white']}, winner {games[-1]['winner'] or 'draw'} ({end}, "
                  f"{len(board.move_stack)} plies), {names['a']} {spent['a']:.2f} s, {names['b']} {spent['b']:.2f} s",
                  flush=True)
    finally:
        for engine in engines.values():
            engine.close()
    n = len(games)
    table = {}
    for key in ("a", "b"):
        name = names[key]
        wins = sum(g["winner"] == name for g in games)
        seconds = sum(g["seconds"][name] for g in games)
        moves = sum(g["moves"][name] for g in games)
        table[name] = {"wins": wins, "winrate": wins / n, "s_per_game": seconds / n,
                       "ms_per_move": 1000 * seconds / max(1, moves)}
    draws = sum(g["winner"] is None for g in games)
    a, b = table[names["a"]], table[names["b"]]
    report = {"label": args.label, "games": n, "draws": draws, "table": table,
              "random_plies": args.random_plies, "max_plies": args.max_plies,
              f"{names['a']}_faster_by": 1 - a["s_per_game"] / b["s_per_game"] if b["s_per_game"] else None,
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "list": games}
    out = HERE / "results" / f"duel-{args.label}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n{n} jogos, {draws} empates")
    for name, row in table.items():
        print(f"{name:10s} WR {100 * row['winrate']:5.1f}%  ({row['wins']} vitorias)  "
              f"{row['s_per_game']:.3f} s/jogo  {row['ms_per_move']:.2f} ms/lance")
    faster = 1 - a["s_per_game"] / b["s_per_game"] if b["s_per_game"] else 0
    print(f"{names['a']} {'mais rapido' if faster > 0 else 'mais lento'} {100 * abs(faster):.0f}% por jogo  -> {out}")


if __name__ == "__main__":
    main()
