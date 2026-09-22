"""Play a game against TonkAI, and watch it think.

    python play.py                    # you are white
    python play.py --black
    python play.py --checkpoint checkpoints/latest.pt

Type moves as SAN (e4, Nf3, O-O, exd5, e8=Q) or UCI (e2e4, e7e8q).
Commands: help  eval  pieces  predict  fen  board  undo  resign  quit
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

import tonkai
from tonkai import chess_env as env
from tonkai.config import Config, add_config_args, config_from_args
from tonkai.encoding import PIECE_NAMES

UNICODE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖",
    "Q": "♕", "K": "♔", "p": "♟", "n": "♞",
    "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

BANNER = """
  TonkAI - no piece values, no centipawns, no search.
  It picks moves by P(win)/P(draw)/P(loss), and it is trying to guess
  your next move before you play it.
"""

HELP = """
  <move>    SAN (e4, Nf3, O-O, exd5, e8=Q) or UCI (e2e4, e7e8q)
  eval      TonkAI's ranking of every legal move in this position
  pieces    TonkAI's contextual value for each piece on the board
  predict   what TonkAI thinks you are about to play
  hint      show its most expected reply (a prediction, not strength advice)
  moves     show the move list
  pgn [file] print PGN, or export it to a file
  fen       print the FEN
  board     redraw the board
  undo      take back your last move (and TonkAI's reply)
  resign    give up
  quit      leave
"""


def render(board: chess.Board, flip: bool, unicode: bool, last: chess.Move | None):
    ranks = range(7, -1, -1) if not flip else range(8)
    files = range(8) if not flip else range(7, -1, -1)
    hot = {last.from_square, last.to_square} if last else set()
    out = []
    for r in ranks:
        row = [f" {r + 1} "]
        for f in files:
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            if piece is None:
                ch = "."
            else:
                ch = UNICODE[piece.symbol()] if unicode else piece.symbol()
            row.append(f"({ch})" if sq in hot else f" {ch} ")
        out.append("".join(row))
    letters = "abcdefgh" if not flip else "hgfedcba"
    out.append("   " + "".join(f" {c} " for c in letters))
    return "\n".join(out)


def parse_move(board: chess.Board, text: str):
    for parser in (board.parse_san, chess.Move.from_uci):
        try:
            mv = parser(text)
        except ValueError:
            continue
        if mv in board.legal_moves:
            return mv
    return None


def show_eval(agent, board, self_color, top):
    mc = agent.evaluate_moves(board, self_color)
    if mc is None:
        print("  no legal moves")
        return
    order = (-mc.utilities).argsort()
    print(f"\n  {'Move':7s} {'Win':>6s} {'Draw':>6s} {'Loss':>6s} {'Utility':>9s}")
    for i in order[:top]:
        w, d, l = mc.probs[i]
        print(f"  {board.san(mc.moves[i]):7s} {w:6.3f} {d:6.3f} {l:6.3f} "
              f"{mc.utilities[i]:+9.5f}")
    spread = float(mc.utilities.max() - mc.utilities.min())
    print(f"  spread between best and worst move: {spread:.6f}")


def show_pieces(agent, board, self_color):
    entries = agent.piece_values(board, self_color)
    entries.sort(key=lambda e: -e[3])
    print("\n  TonkAI's own contextual values (not chess piece values)")
    for label, want in (("its pieces", True), ("your pieces", False)):
        print(f"    {label}")
        rows = [e for e in entries if e[2] == want][:8]
        for sq, piece, _s, val in rows:
            print(f"      {PIECE_NAMES[piece.piece_type - 1]:7s} "
                  f"{chess.square_name(sq):3s} {val:+.4f}")


def show_predict(agent, board, self_color, top=5):
    """What TonkAI expects the human to play from this position."""
    moves, probs, _ = agent.predict_enemy(board, self_color)
    order = (-probs).argsort()
    print("\n  TonkAI expects you to play:")
    for i in order[:top]:
        bar = "#" * max(1, int(round(probs[i] * 30)))
        print(f"    {board.san(moves[i]):8s} {probs[i]:6.1%}  {bar}")
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    print(f"    entropy {ent:.2f} over {len(moves)} replies "
          f"(uniform would be {math.log(len(moves)):.2f})")
    return moves, probs


def game_pgn(board, human, result="*"):
    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "Human" if human == chess.WHITE else "TonkAI"
    game.headers["Black"] = "Human" if human == chess.BLACK else "TonkAI"
    game.headers["Result"] = result
    return game


def export_pgn(board, human, path, result="*"):
    Path(path).write_text(str(game_pgn(board, human, result)) + "\n", encoding="utf-8")


def undo_human_turn(board, human, observations):
    """Undo through the last human move, including an optional TonkAI reply."""
    if not board.move_stack:
        return False
    # A lone opening move by TonkAI (human black) cannot be taken back here.
    if len(board.move_stack) == 1 and human == chess.BLACK:
        return False
    board.pop()
    while board.move_stack and board.turn != human:
        board.pop()
    if observations:
        observations.pop()
    return True


def print_summary(board, observations, outcome):
    print(f"\n  {outcome}. {len(board.move_stack)} plies played.")
    hits = sum(hit for _, hit in observations)
    print(f"  TonkAI saw {hits} of your {len(observations)} moves coming.")
    if observations:
        mean = sum(s for s, _ in observations) / len(observations)
        print(f"  Mean surprise: {mean:.2f} (lower = it saw you coming)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Play against TonkAI")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    ap.add_argument("--black", action="store_true", help="you play black")
    ap.add_argument("--unicode", action="store_true", help="unicode pieces")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--quiet", action="store_true",
                    help="do not show predictions automatically")
    ap.add_argument("--pgn", help="export PGN on exit, including unfinished games")
    add_config_args(ap)
    args = ap.parse_args()

    cfg = Config()
    if os.path.exists(args.checkpoint):
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(payload.get("config", {}))
    cfg.device = "cpu"          # single positions: CPU beats GPU here
    if getattr(args, "temperature", None) is None:
        cfg.temperature = 0.0   # play its best, not a sample
    cfg = config_from_args(args, cfg)

    net, agent, opponent, worker, trainer, metrics = tonkai.build(cfg, "cpu")
    if os.path.exists(args.checkpoint):
        trainer.load(args.checkpoint, load_optimizer=False)
        trained = f"{trainer.games} games trained"
    else:
        trained = "UNTRAINED (no checkpoint found)"
    net.eval()

    human = chess.BLACK if args.black else chess.WHITE
    tonk = not human
    board = chess.Board()
    rng = np.random.default_rng()

    print(BANNER)
    print(f"  checkpoint: {args.checkpoint} - {trained}")
    print(f"  you are {'BLACK' if human == chess.BLACK else 'WHITE'}, "
          f"TonkAI is {'WHITE' if tonk == chess.WHITE else 'BLACK'}")
    print("  type 'help' for commands\n")

    last = None
    pending = None      # (moves, probs) TonkAI's guess about your next move
    observations = []  # (surprise, top1 hit), one per retained human move
    pgn_result = "*"
    outcome = "Game unfinished"

    while True:
        result, reason = env.terminal(board, tonk, cfg.max_plies)
        if result is not None:
            print("\n" + render(board, human == chess.BLACK, args.unicode, last))
            outcome = {0: "TonkAI wins", 1: "Draw", 2: "You win"}[result]
            pgn_result = ("1/2-1/2" if result == 1 else
                          "1-0" if (tonk == chess.WHITE) == (result == 0) else "0-1")
            print(f"\n  *** {outcome} - {reason} ***")
            break

        if board.turn == human:
            print("\n" + render(board, human == chess.BLACK, args.unicode, last))
            if pending is None:
                if args.quiet:
                    moves, probs, _ = agent.predict_enemy(board, tonk)
                    pending = moves, probs
                else:
                    pending = show_predict(agent, board, tonk)

            try:
                text = input("\n  your move> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  bye")
                break
            if not text:
                continue
            low = text.lower()

            if low in ("quit", "exit", "q"):
                print("  bye")
                break
            if low == "resign":
                outcome = "TonkAI wins by resignation"
                pgn_result = "1-0" if tonk == chess.WHITE else "0-1"
                print("  you resign - TonkAI wins")
                break
            if low == "help":
                print(HELP)
                continue
            if low == "fen":
                print(f"  {board.fen()}")
                continue
            if low == "board":
                continue
            if low == "moves":
                print(game_pgn(board, human).mainline())
                continue
            if low == "pgn" or low.startswith("pgn "):
                path = text[3:].strip().strip('"')
                if path:
                    try:
                        export_pgn(board, human, path)
                        print(f"  PGN written to {path}")
                    except OSError as error:
                        print(f"  Could not export PGN: {error}")
                else:
                    print(game_pgn(board, human))
                continue
            if low == "hint":
                moves, probs = pending
                i = int(np.argmax(probs))
                print(f"  Prediction hint: {board.san(moves[i])} ({probs[i]:.1%}). "
                      "This is its most expected reply, not a strength evaluation.")
                continue
            if low == "eval":
                show_eval(agent, board, tonk, args.top)
                continue
            if low == "pieces":
                show_pieces(agent, board, tonk)
                continue
            if low == "predict":
                pending = show_predict(agent, board, tonk)
                continue
            if low == "undo":
                if undo_human_turn(board, human, observations):
                    last = board.move_stack[-1] if board.move_stack else None
                    pending = None
                    print("  took back one move each")
                else:
                    print("  nothing to undo")
                continue

            mv = parse_move(board, text)
            if mv is None:
                sample = ", ".join(board.san(m) for m in list(board.legal_moves)[:8])
                print(f"  '{text}' is not legal here. Try: {sample} ...")
                continue

            # score TonkAI's guess before playing
            if pending is not None:
                moves, probs = pending
                idx = moves.index(mv)
                p = float(probs[idx])
                rank = int((-probs).argsort().tolist().index(idx)) + 1
                s = -math.log(max(p, 1e-12))
                observations.append((s, rank == 1))
                verdict = ("called it" if rank == 1 else
                           f"had it {rank}{'nd' if rank == 2 else 'rd' if rank == 3 else 'th'}")
                print(f"  TonkAI gave {board.san(mv)} {p:.1%} - {verdict} "
                      f"(surprise {s:.2f})")
            pending = None
            last = mv
            board.push(mv)
            continue

        # ---------------------------------------------------------- TonkAI
        mc = agent.evaluate_moves(board, tonk)
        u = mc.utilities
        idx = int(np.argmax(u)) if cfg.temperature <= 0 else int(
            rng.choice(len(u), p=np.exp((u - u.max()) / cfg.temperature)
                       / np.exp((u - u.max()) / cfg.temperature).sum())
        )
        mv = mc.moves[idx]
        w, d, l = mc.probs[idx]
        san = board.san(mv)
        print(f"\n  TonkAI plays {san}"
              f"   (win {w:.3f}  draw {d:.3f}  loss {l:.3f}  U {u[idx]:+.5f})")
        last = mv
        board.push(mv)

    print_summary(board, observations, outcome)
    if args.pgn:
        try:
            export_pgn(board, human, args.pgn, pgn_result)
            print(f"  PGN written to {args.pgn}")
        except OSError as error:
            print(f"  Could not export PGN: {error}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
