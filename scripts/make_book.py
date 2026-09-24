"""Build an opening book by deep engine analysis (the owner's choice: not organic).

From the root line, the book side plays the engine's best move; for the other
side the `--replies` best answers are followed, `--plies` half-moves deep.
Output: {"name", "side", "positions": {epd: [{"move", "eval"}]}}.

    python scripts/make_book.py --name "Portuguese Opening" --line e2e4 e7e5 f1b5 \\
        --side white --plies 12 --out books/portuguese.json
"""
import argparse
from collections import deque
import json
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engines/stockfish-11/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe"


def key(board):
    return " ".join(board.fen().split(" ")[:4])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--line", nargs="+", required=True, help="the opening's defining moves (UCI)")
    parser.add_argument("--side", choices=("white", "black"), required=True)
    parser.add_argument("--plies", type=int, default=12, help="half-moves after the defining line")
    parser.add_argument("--replies", type=int, default=4, help="opponent answers followed per position")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    side = chess.WHITE if args.side == "white" else chess.BLACK
    engine = chess.engine.SimpleEngine.popen_uci(str(ENGINE))
    engine.configure({"Threads": 4, "Hash": 256})
    limit = chess.engine.Limit(depth=args.depth)
    positions = {}
    board = chess.Board()
    for uci in args.line:                                     # the defining line itself
        move = chess.Move.from_uci(uci)
        if board.turn == side:
            positions[key(board)] = [{"move": uci, "eval": None}]
        board.push(move)
    queue = deque([(board, 0)])
    seen = set()
    while queue:
        board, depth = queue.popleft()
        if depth >= args.plies or board.is_game_over() or key(board) in seen:
            continue
        seen.add(key(board))
        if board.turn == side:
            info = engine.analyse(board, limit)
            move = info["pv"][0]
            positions[key(board)] = [{"move": move.uci(), "eval": info["score"].pov(side).score(mate_score=10_000)}]
            child = board.copy(stack=False)
            child.push(move)
            queue.append((child, depth + 1))
        else:
            for info in engine.analyse(board, limit, multipv=args.replies):
                child = board.copy(stack=False)
                child.push(info["pv"][0])
                queue.append((child, depth + 1))
    engine.quit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"name": args.name, "side": args.side, "line": args.line,
                                    "depth": args.depth, "positions": positions}, indent=1), encoding="utf-8")
    print(f"{args.name}: {len(positions)} book positions -> {args.out}")


if __name__ == "__main__":
    main()
