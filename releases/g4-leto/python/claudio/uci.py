"""Claudio as a UCI engine: any arbiter (cutechess, python-chess, a duel script)
can play it in its own process and time it from outside.

    python claudio/uci.py [--checkpoint claudio/claudio.pt]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401  (frozen base code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "claudio.pt")
    args = parser.parse_args()
    import chess
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from claudio_agent import Claudio, ClaudioConfig
    from claudio_load import load_claudio

    from claudio_load import load_players, make_player
    loaded = load_players(args.checkpoint)                 # one checkpoint, or "white.pt+black.pt"
    player = make_player(loaded, np.random.default_rng())
    board, pending = chess.Board(), ["startpos"]
    for line in sys.stdin:
        words = line.split()
        if not words:
            continue
        command = words[0]
        if command == "uci":
            print("id name Claudio", flush=True)
            print("id author Claude (Anthropic) for Martim", flush=True)
            print("uciok", flush=True)
        elif command == "isready":
            print("readyok", flush=True)
        elif command == "ucinewgame":
            player.reset()
            board = chess.Board()
        elif command == "position":
            pending = words[1:]         # read after "go": the arbiter's clock counts all the work
        elif command == "go":
            board = _position(pending, board)
            move = player.choose(board)
            print(f"bestmove {move.uci()}", flush=True)
        elif command == "quit":
            break


def _position(words, previous):
    """The position after "go"; only the new moves are played when it continues the previous one."""
    import chess
    if words[0] == "startpos":
        root, rest = chess.STARTING_FEN, words[1:]
    else:
        end = words.index("moves") if "moves" in words else len(words)
        root, rest = " ".join(words[1:end]), words[end:]
    moves = rest[1:] if rest and rest[0] == "moves" else []
    done = [m.uci() for m in previous.move_stack]
    if previous.root().fen() == chess.Board(root).fen() and moves[:len(done)] == done:
        board = previous
    else:
        board, done = chess.Board(root), []
    for uci in moves[len(done):]:
        board.push_uci(uci)
    return board


if __name__ == "__main__":
    main()
