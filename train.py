"""Train TonkAI.

    python train.py --games 2000
    python train.py --resume checkpoints/latest.pt --games 10000
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import torch

import tonkai
from tonkai.config import Config, add_config_args, config_from_args

_stop = False


def _handle_sigint(signum, frame):
    global _stop
    if _stop:
        sys.exit(1)
    _stop = True
    print("\n[stopping after this game - Ctrl+C again to force]", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train TonkAI")
    ap.add_argument("--games", type=int, default=1000, help="games to play")
    ap.add_argument("--resume", type=str, default=None, help="checkpoint to resume")
    ap.add_argument("--no-train", action="store_true", help="self-play only")
    ap.add_argument("--dump-theory", type=str, default=None)
    ap.add_argument("--dump-sacrifices", type=str, default=None)
    add_config_args(ap)
    args = ap.parse_args()

    cfg = Config()
    if args.resume and os.path.exists(args.resume):
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(payload.get("config", {}))
        print(f"resuming from {args.resume}")
    cfg = config_from_args(args, cfg)

    device = cfg.resolved_device()
    net, agent, opponent, worker, trainer, metrics = tonkai.build(cfg, device)

    if args.resume and os.path.exists(args.resume):
        trainer.load(args.resume)
        print(f"  {trainer.games} games / {trainer.steps} steps already done")

    print(f"device: {device}   seed: {cfg.seed}")
    print(f"opponent: {opponent}")
    print(f"TonkAI parameters: {net.n_parameters():,}")
    print(net.parameter_report())
    print(f"playing {args.games} games "
          f"({'no training' if args.no_train else f'{cfg.updates_per_game} updates/game'})")
    print("-" * 64, flush=True)

    signal.signal(signal.SIGINT, _handle_sigint)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    t0 = time.perf_counter()
    played = 0
    for _ in range(args.games):
        if _stop:
            break
        record = worker.play_game()
        trainer.add_game(record)
        if not args.no_train:
            trainer.train(cfg.updates_per_game)
        played += 1

        if metrics.games % cfg.log_every == 0:
            print(metrics.report(
                f"Replay: {len(trainer.replay):,} value / "
                f"{len(trainer.predict_replay):,} predict "
                f"(truncated {trainer.predict_replay.truncated})"
            ))
            print("-" * 64, flush=True)

        if cfg.checkpoint_every and metrics.games % cfg.checkpoint_every == 0:
            path = os.path.join(cfg.checkpoint_dir, "latest.pt")
            trainer.save(path)
            print(f"[checkpoint] {path} @ {metrics.games} games", flush=True)

    elapsed = time.perf_counter() - t0
    path = os.path.join(cfg.checkpoint_dir, "latest.pt")
    trainer.save(path)

    print("=" * 64)
    print(metrics.report())
    print("-" * 64)
    print(f"{played} games in {elapsed:.1f}s   "
          f"{played / max(elapsed, 1e-9):.1f} games/s   "
          f"{metrics.tonkai_positions / max(elapsed, 1e-9):.0f} positions/s")
    print("time spent:")
    print(metrics.timing_breakdown())
    print(f"bottleneck: {metrics.bottleneck()}")
    print(f"terminations: {dict(metrics.termination)}")
    print(f"checkpoint: {path}")

    if args.dump_theory:
        n = trainer.theory.dump_json(args.dump_theory)
        print(f"theory events written: {n} -> {args.dump_theory}")
    if args.dump_sacrifices:
        n = trainer.sacrifice_log.dump_json(args.dump_sacrifices)
        print(f"sacrifices written: {n} -> {args.dump_sacrifices}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
