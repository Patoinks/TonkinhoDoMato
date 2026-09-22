"""Look inside TonkAI for one position.

    python analyze.py
    python analyze.py --fen "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    python analyze.py --fen "..." --black --top 10
"""

from __future__ import annotations

import argparse
import math
import os

import chess
import numpy as np
import torch

import tonkai
from tonkai.config import Config, add_config_args, config_from_args
from tonkai.encoding import PIECE_NAMES

START = chess.STARTING_FEN


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze a FEN with TonkAI")
    ap.add_argument("--fen", type=str, default=START)
    ap.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    ap.add_argument("--top", type=int, default=12, help="moves to show")
    ap.add_argument("--black", action="store_true",
                    help="analyze as if TonkAI were black (default: side to move)")
    ap.add_argument("--white", action="store_true")
    add_config_args(ap)
    args = ap.parse_args()

    cfg = Config()
    if os.path.exists(args.checkpoint):
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(payload.get("config", {}))
    cfg = config_from_args(args, cfg)

    device = cfg.resolved_device()
    net, agent, opponent, worker, trainer, metrics = tonkai.build(cfg, device)
    if os.path.exists(args.checkpoint):
        trainer.load(args.checkpoint, load_optimizer=False)
        print(f"checkpoint: {args.checkpoint}  ({trainer.games} games trained)")
    else:
        print("no checkpoint - showing an untrained network")
    net.eval()

    board = chess.Board(args.fen)
    self_color = board.turn
    if args.white:
        self_color = chess.WHITE
    if args.black:
        self_color = chess.BLACK

    print()
    print(board)
    print(f"\nFEN: {board.fen()}")
    print(f"TonkAI is {'WHITE' if self_color == chess.WHITE else 'BLACK'}; "
          f"{'WHITE' if board.turn == chess.WHITE else 'BLACK'} to move")

    p_now, u_now = agent.value_of(board, self_color)
    print(f"\nThis position:  win {p_now[0]:.3f}  draw {p_now[1]:.3f}  "
          f"loss {p_now[2]:.3f}   utility {u_now:+.4f}")

    # ---------------------------------------------------------- move values
    if board.turn == self_color:
        mc = agent.evaluate_moves(board, self_color)
        if mc is None:
            print("\nno legal moves")
        else:
            order = (-mc.utilities).argsort()
            print(f"\nLegal moves by TonkAI utility  "
                  f"(U = win + {cfg.draw_weight}*draw - loss)")
            print(f"  {'Move':8s} {'SAN':10s} {'Win':>6s} {'Draw':>6s} "
                  f"{'Loss':>6s} {'Utility':>9s}")
            for rank, i in enumerate(order[: args.top]):
                mv = mc.moves[i]
                san = board.san(mv)
                w, d, l = mc.probs[i]
                mark = "  <-- best" if rank == 0 else ""
                print(f"  {mv.uci():8s} {san:10s} {w:6.3f} {d:6.3f} "
                      f"{l:6.3f} {mc.utilities[i]:+9.4f}{mark}")
            if len(order) > args.top:
                print(f"  ... {len(order) - args.top} more")
    else:
        # -------------------------------------------------- predict head
        moves, probs, _ = agent.predict_enemy(board, self_color)
        order = (-probs).argsort()
        print("\nPredict: what TonkAI thinks the opponent will play")
        print(f"  {'Move':8s} {'SAN':10s} {'P':>8s}")
        for i in order[: args.top]:
            print(f"  {moves[i].uci():8s} {board.san(moves[i]):10s} "
                  f"{probs[i]:8.4f}")
        ent = float(-(probs * np.log(probs + 1e-12)).sum())
        print(f"  entropy {ent:.3f}  (uniform would be "
              f"{math.log(len(moves)):.3f})")

    # ------------------------------------------------- contextual piece values
    print("\nContextual piece values (TonkAI's own estimates, not chess values)")
    entries = agent.piece_values(board, self_color)
    entries.sort(key=lambda e: -e[3])
    own = [e for e in entries if e[2]]
    opp = [e for e in entries if not e[2]]
    for label, group in (("SELF", own), ("OPPONENT", opp)):
        print(f"  {label}")
        for sq, piece, _is_self, val in group:
            print(f"    {PIECE_NAMES[piece.piece_type - 1]:7s} "
                  f"{chess.square_name(sq):3s}  {val:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
