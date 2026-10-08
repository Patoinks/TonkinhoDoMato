"""Claudio's opening experience: its own win rate with each opening move, White and Black apart.

It plays --games games against itself from the normal starting position, choosing among its own
best moves with a little temperature for the first --explore-plies half-moves (its own choices, so
the openings differ), then its best move. For every position of the first --plies half-moves, each
move gets (games, wins of the side that played it) - a draw counts 0 (the owner's rule). The book
(claudio/experiencia.json) is what the setting book=... reads; the report shows the win rate of
White's first moves, Black's answers and the most played lines, for White and for Black.

    python claudio/experiencia.py --checkpoint claudio/claudio-g16.pt --games 3000 --set tactic_depth=3
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

_e = {}


def _init(checkpoint, settings):
    import psutil_free_priority  # noqa: F401
    import torch
    torch.set_num_threads(1)
    from claudio_load import load_players
    _e["loaded"] = load_players(checkpoint, settings)


def play(game, seed, explore_plies, temperature, max_plies):
    import chess
    import numpy as np
    from dataclasses import replace
    from claudio_load import make_player
    players = [make_player(_e["loaded"], np.random.default_rng([seed, game, side])) for side in (0, 1)]
    greedy = [p.cfg for p in players]
    exploring = [replace(p.cfg, temperature=temperature, book="", book_weight=0.0) for p in players]
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        side = 0 if board.turn == chess.WHITE else 1
        players[side].cfg = exploring[side] if len(board.move_stack) < explore_plies else greedy[side]
        board.push(players[side].choose(board))
    outcome = board.outcome(claim_draw=True)
    return {"game": game, "moves": " ".join(m.uci() for m in board.move_stack),
            "winner": None if outcome is None or outcome.winner is None else ("white" if outcome.winner else "black"),
            "end": outcome.termination.name if outcome else "max_plies"}


def key_of(board):
    return board.fen().rsplit(" ", 2)[0]


def build(games, plies):
    """{position key: {uci: [games, wins of the mover]}} and per-line stats."""
    import chess
    book = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    lines = defaultdict(lambda: [0, 0, 0])                 # san line -> games, white wins, black wins
    for row in games:
        board, san = chess.Board(), []
        for ply, uci in enumerate(row["moves"].split()[:plies]):
            move = chess.Move.from_uci(uci)
            mover = "white" if board.turn == chess.WHITE else "black"
            cell = book[key_of(board)][uci]
            cell[0] += 1
            cell[1] += int(row["winner"] == mover)
            san.append(board.san(move))
            board.push(move)
            if ply < 6:
                line = lines[" ".join(san)]
                line[0] += 1
                line[1] += int(row["winner"] == "white")
                line[2] += int(row["winner"] == "black")
    return book, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(ROOT / "claudio/claudio-g16.pt"))
    parser.add_argument("--set", action="append", default=[], help="setting=value over the checkpoint's")
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--explore-plies", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--plies", type=int, default=16, help="half-moves of each game kept in the book")
    parser.add_argument("--min-games", type=int, default=2, help="book entries with fewer games are left out")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--out", default=str(ROOT / "claudio/experiencia.json"))
    args = parser.parse_args()

    settings = list(args.set) + ["book=", "book_weight=0"]          # it learns its experience, not from it
    started, games = time.perf_counter(), []
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(args.checkpoint, settings)) as pool:
        jobs = [(g, args.seed, args.explore_plies, args.temperature, args.max_plies) for g in range(args.games)]
        for row in pool.map(play, *zip(*jobs), chunksize=1):
            games.append(row)
            if len(games) % 100 == 0 or len(games) == args.games:
                spent = time.perf_counter() - started
                print(f"  {len(games)}/{args.games} jogos, {spent / 60:.0f} min, "
                      f"faltam ~{spent / len(games) * (args.games - len(games)) / 60:.0f} min", flush=True)
    book, lines = build(games, args.plies)
    positions = {k: {u: c for u, c in moves.items() if c[0] >= args.min_games} for k, moves in book.items()}
    positions = {k: v for k, v in positions.items() if v}
    Path(args.out).write_text(json.dumps({"checkpoint": args.checkpoint, "settings": args.set, "games": len(games),
                                          "plies": args.plies, "positions": positions}), encoding="utf-8")
    white = sum(g["winner"] == "white" for g in games)
    black = sum(g["winner"] == "black" for g in games)
    report = {"games": len(games), "white_wins": white, "black_wins": black, "draws": len(games) - white - black,
              "lines": {line: {"games": n, "white_wr": w / n, "black_wr": b / n}
                        for line, (n, w, b) in sorted(lines.items(), key=lambda kv: -kv[1][0])},
              "list": games}
    (ROOT / "claudio/results/experiencia-relatorio.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{len(games)} jogos: brancas ganham {100 * white / len(games):.1f}%, pretas {100 * black / len(games):.1f}%, "
          f"empates {100 * (len(games) - white - black) / len(games):.1f}%")
    print("De brancas, primeiro lance (winrate das brancas):")
    for line, (n, w, b) in sorted(((l, v) for l, v in lines.items() if " " not in l), key=lambda kv: -kv[1][0]):
        print(f"  1.{line:6s} {n:5d} jogos  WR brancas {100 * w / n:5.1f}%  (pretas {100 * b / n:5.1f}%)")
    print("De pretas, resposta (winrate das pretas), nas aberturas com mais jogos:")
    for line, (n, w, b) in sorted(((l, v) for l, v in lines.items() if l.count(" ") == 1), key=lambda kv: -kv[1][0])[:12]:
        first, reply = line.split()
        print(f"  1.{first} {reply:6s} {n:5d} jogos  WR pretas {100 * b / n:5.1f}%  (brancas {100 * w / n:5.1f}%)")
    print(f"livro: {len(positions)} posicoes -> {args.out}")


if __name__ == "__main__":
    main()
