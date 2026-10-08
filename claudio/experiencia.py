"""Claudio's opening experience: how good each opening move turned out in its own games, White
and Black apart.

It plays --games games against itself from the normal starting position, choosing among its own
best moves with a little temperature for the first --explore-plies half-moves (its own choices, so
the openings differ), then its best move. A move is GOOD in a game (the owner's rule) when, --later
half-moves after it (6 moves), the position is advantageous or level for the side that played it -
its own network judges it at least as well from that side as from the other - AND that side won
the game. For every position of the first --plies half-moves, each move gets (games, good games).
The book (claudio/experiencia.json) is what the setting book=... reads; the report shows White's
first moves and Black's answers with their good rate and win rate (a draw counts 0).

    python claudio/experiencia.py --checkpoint claudio/claudio-g16.pt --games 1500 --set tactic_depth=3
    python claudio/experiencia.py --rebuild      (the book again from the games already played)
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


def judgements(checkpoint, games):
    """For each game: (white's view, black's view) - its own network's value of every position of the
    game (index = plies played), each side's view with its memory of the whole game so far."""
    import chess
    import numpy as np
    import torch
    from claudio_agent import native_board
    from claudio_load import load_claudio
    net = load_claudio(checkpoint)[0]
    out = []
    with torch.inference_mode():
        for row in games:
            fast = native_board(chess.Board())
            rows = {True: [np.asarray(fast.encode_here(True), dtype=np.float32)],
                    False: [np.asarray(fast.encode_here(False), dtype=np.float32)]}
            for uci in row["moves"].split():
                fast.push(fast_code(uci))
                for side in (True, False):
                    rows[side].append(np.asarray(fast.encode_here(side), dtype=np.float32))
            views = []
            for side in (True, False):
                latent = net.encode(torch.from_numpy(np.stack(rows[side])))
                memory, contexts = net.initial_memory(1), []
                for t in range(len(latent)):
                    memory = net.memory_cell(latent[t:t + 1], memory)
                    contexts.append(net.context_from_latent(latent[t:t + 1], memory))
                views.append(net.survival(torch.cat(contexts))[0].mean(-1).numpy())
            out.append(views)
    return out


def fast_code(uci):
    import chess
    from tonkai import fast_rules
    return fast_rules.code(chess.Move.from_uci(uci))


def build(games, plies, views, later):
    """{position key: {uci: [games, good games]}} and per-line stats (games, good and wins for each side)."""
    import chess
    book = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    lines = defaultdict(lambda: {"games": 0, "white_good": 0, "black_good": 0, "white_wins": 0, "black_wins": 0})
    for row, (white_view, black_view) in zip(games, views):
        board, san, last = chess.Board(), [], len(white_view) - 1
        good = {}
        for ply, uci in enumerate(row["moves"].split()[:plies]):
            move = chess.Move.from_uci(uci)
            mover = "white" if board.turn == chess.WHITE else "black"
            then = min(ply + later, last)
            mine, theirs = (white_view, black_view) if mover == "white" else (black_view, white_view)
            good[ply] = row["winner"] == mover and mine[then] >= theirs[then]
            cell = book[key_of(board)][uci]
            cell[0] += 1
            cell[1] += int(good[ply])
            san.append(board.san(move))
            board.push(move)
            if ply < 2:
                line = lines[" ".join(san)]
                line["games"] += 1
                line[f"{mover}_good"] += int(good[ply])
                line["white_wins"] += int(row["winner"] == "white")
                line["black_wins"] += int(row["winner"] == "black")
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
    parser.add_argument("--later", type=int, default=12, help="half-moves after a move its position is judged")
    parser.add_argument("--rebuild", action="store_true", help="the book again from results/experiencia-relatorio.json")
    parser.add_argument("--out", default=str(ROOT / "claudio/experiencia.json"))
    args = parser.parse_args()

    report_path = ROOT / "claudio/results/experiencia-relatorio.json"
    if args.rebuild:
        games = json.loads(report_path.read_text(encoding="utf-8"))["list"]
    else:
        settings = list(args.set) + ["book=", "book_weight=0"]      # it learns its experience, not from it
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
    views = judgements(args.checkpoint, games)
    book, lines = build(games, args.plies, views, args.later)
    positions = {k: {u: c for u, c in moves.items() if c[0] >= args.min_games} for k, moves in book.items()}
    positions = {k: v for k, v in positions.items() if v}
    Path(args.out).write_text(json.dumps({"checkpoint": args.checkpoint, "settings": args.set, "games": len(games),
                                          "plies": args.plies, "later": args.later, "rule": "good = level or better "
                                          "after `later` half-moves (own network) and won", "positions": positions}),
                              encoding="utf-8")
    white = sum(g["winner"] == "white" for g in games)
    black = sum(g["winner"] == "black" for g in games)
    report = {"games": len(games), "white_wins": white, "black_wins": black, "draws": len(games) - white - black,
              "later": args.later, "lines": dict(sorted(lines.items(), key=lambda kv: -kv[1]["games"])),
              "list": games}
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{len(games)} jogos: brancas ganham {100 * white / len(games):.1f}%, pretas {100 * black / len(games):.1f}%, "
          f"empates {100 * (len(games) - white - black) / len(games):.1f}%")
    print(f"Bom = posicao igual ou melhor {args.later} meios-lances depois (avaliacao dele) E vitoria no fim")
    print("De brancas, primeiro lance:")
    for line, row in sorted(((l, v) for l, v in lines.items() if " " not in l), key=lambda kv: -kv[1]["games"]):
        n = row["games"]
        print(f"  1.{line:6s} {n:5d} jogos  Bom {100 * row['white_good'] / n:5.1f}%  WR brancas "
              f"{100 * row['white_wins'] / n:5.1f}%  WR pretas {100 * row['black_wins'] / n:5.1f}%")
    print("De pretas, resposta (nas aberturas com mais jogos):")
    for line, row in sorted(((l, v) for l, v in lines.items() if l.count(" ") == 1), key=lambda kv: -kv[1]["games"])[:12]:
        n, (first, reply) = row["games"], line.split()
        print(f"  1.{first} {reply:6s} {n:5d} jogos  Bom {100 * row['black_good'] / n:5.1f}%  WR pretas "
              f"{100 * row['black_wins'] / n:5.1f}%  WR brancas {100 * row['white_wins'] / n:5.1f}%")
    print(f"livro: {len(positions)} posicoes -> {args.out}")


if __name__ == "__main__":
    main()
