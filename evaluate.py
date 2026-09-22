"""Play games with learning switched off.

    python evaluate.py --games 200
    python evaluate.py --checkpoint checkpoints/latest.pt --games 200 --temperature 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import chess
import torch

import tonkai
from tonkai.config import Config, add_config_args, config_from_args
from tonkai.win_history import LongestWinRecorder


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate TonkAI (no learning)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    ap.add_argument("--white", action="store_true", help="always play white")
    ap.add_argument("--black", action="store_true", help="always play black")
    add_config_args(ap)
    args = ap.parse_args()

    cfg = Config()
    if os.path.exists(args.checkpoint):
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(payload.get("config", {}))
    # evaluation is argmax unless the caller asks otherwise
    if getattr(args, "temperature", None) is None:
        cfg.temperature = 0.0
    cfg = config_from_args(args, cfg)

    device = cfg.resolved_device()
    net, agent, opponent, worker, trainer, metrics = tonkai.build(cfg, device)

    if os.path.exists(args.checkpoint):
        trainer.load(args.checkpoint, load_optimizer=False)
        print(f"loaded {args.checkpoint}  ({trainer.games} games trained)")
    else:
        print(f"no checkpoint at {args.checkpoint} - evaluating an untrained net")

    net.eval()
    color = None
    if args.white:
        color = chess.WHITE
    elif args.black:
        color = chess.BLACK

    print(f"device: {device}   temperature: {cfg.temperature}   "
          f"opponent: depth={cfg.opponent_depth} nodes={cfg.opponent_nodes} "
          f"randomness={cfg.opponent_randomness}")
    print("-" * 64, flush=True)

    t0 = time.perf_counter()
    win_recorder = LongestWinRecorder(cfg.win_pgn_path)
    for i in range(args.games):
        record = worker.play_game(self_color=color, collect=False)
        try:
            if win_recorder.consider_record(record, "TinyOpponent"):
                print(f"[longest external win] {cfg.win_pgn_path} ({record.plies} plies)", flush=True)
        except (OSError, ValueError) as exc:
            print(f"[win recording failed] {exc}", file=sys.stderr, flush=True)
        if (i + 1) % max(args.games // 5, 1) == 0:
            w, d, l = metrics.wdl_percent()
            print(f"  {i + 1:5d} games   W/D/L {w:5.1f} / {d:5.1f} / {l:5.1f}",
                  flush=True)
    elapsed = time.perf_counter() - t0

    w, d, l = metrics.wdl_percent()
    score = (metrics.wins + 0.5 * metrics.draws) / max(metrics.games, 1)
    print("=" * 64)
    print(f"Games:        {metrics.games}")
    print(f"W/D/L:        {w:.1f} / {d:.1f} / {l:.1f}   (score {score:.3f})")
    print(f"Avg length:   {metrics.game_length.mean:.1f} plies")
    print(f"Predict Top1: {100 * metrics.top1.mean:.1f}%")
    print(f"Predict Top3: {100 * metrics.top3.mean:.1f}%")
    print(f"p(true move): {metrics.true_move_prob.mean:.4f}")
    print(f"Surprise:     {metrics.surprise.mean:.3f}   "
          f"Entropy: {metrics.entropy.mean:.3f}")
    print(f"Avg legal:    {metrics.legal_moves.mean:.1f} (TonkAI)   "
          f"{metrics.enemy_moves.mean:.1f} (opponent)")
    print(f"Terminations: {dict(metrics.termination)}")
    print(f"Speed:        {metrics.games / max(elapsed, 1e-9):.1f} games/s   "
          f"{metrics.tonkai_positions / max(elapsed, 1e-9):.0f} positions/s")
    print("time spent:")
    print(metrics.timing_breakdown())
    print(f"bottleneck:   {metrics.bottleneck()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
