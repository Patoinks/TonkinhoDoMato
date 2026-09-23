"""Train the v4 Emergent King Safety model (tonkai/king_safety_training.py).

Only checkmate grounds it: exact forced-mate search plus adversarial
attack/defence search bootstrapped from the model itself. Positions come from
real TonkAI games; attack/defence episodes add the positions attacks lead to.

    python train_king_safety.py --minutes 60 --workers 8
    python train_king_safety.py --resume --minutes 60        # continue

Writes checkpoints/king_safety/latest.pt (KingSafetyModel) and a JSONL log of
every iteration in history/king_safety-<time>.jsonl. A STOP file next to the
model stops it after the current iteration.
"""
import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import time

from tonkai.king_safety_model import KingSafetyModel
from tonkai.king_safety_training import KSConfig, KingSafetyTrainer, fens_from_checkpoint, fens_from_pgn

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, default=ROOT / "checkpoints/cognitive-mix-20260923/latest.pt",
                        help="a V3 checkpoint whose replay provides real game positions")
    parser.add_argument("--pgn", nargs="*", default=[str(ROOT / "benchmarks/*.pgn")],
                        help="PGN files (globs) with more real games")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints/king_safety/latest.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--minutes", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--history-dir", type=Path, default=ROOT / "history")
    for field in fields(KSConfig):
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(field.default), default=None)
    args = parser.parse_args(argv)
    if args.minutes is None and args.iterations is None:
        parser.error("give --minutes or --iterations")

    model, cfg = None, KSConfig()
    if args.resume:
        model = KingSafetyModel.load(args.out)
        cfg = KSConfig(**model.meta.get("config", {}))
    for field in fields(KSConfig):
        value = getattr(args, field.name)
        if value is not None:
            if args.resume and field.name == "gamma" and value != cfg.gamma:
                parser.error("gamma defines what KS means; it cannot change on resume")
            setattr(cfg, field.name, value)

    roots = []
    if args.positions and args.positions.exists():
        roots += fens_from_checkpoint(args.positions)
    for pattern in args.pgn:
        roots += fens_from_pgn(sorted(glob.glob(pattern)))
    roots = list(dict.fromkeys(roots))
    if not roots:
        parser.error("no positions found")
    history = list(model.meta.get("history", [])) if model else []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.history_dir.mkdir(parents=True, exist_ok=True)
    log = (args.history_dir / f"king_safety-{stamp}.jsonl").open("a", encoding="utf-8", buffering=1)
    stop = args.out.parent / "STOP"
    trainer = KingSafetyTrainer(cfg, roots, model, workers=args.workers)
    if model is not None:                      # keep counting where the saved model stopped
        trainer.iteration = int(model.meta.get("iterations", 0))
        trainer.samples = int(model.meta.get("samples", 0))
    print(f"King Safety: {len(roots)} real positions, gamma {cfg.gamma}, exact mates up to "
          f"{cfg.mate_moves} moves, {args.workers} workers", flush=True)
    started, first_iteration = time.monotonic(), trainer.iteration
    try:
        while True:
            metrics = trainer.iterate()
            metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
            history.append(metrics)
            log.write(json.dumps(metrics, allow_nan=False) + "\n")
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}), flush=True)
            if trainer.iteration % args.save_every == 0:
                trainer.save(args.out, history)
            if args.iterations and trainer.iteration - first_iteration >= args.iterations:
                break
            if args.minutes and time.monotonic() - started >= args.minutes * 60:
                break
            if stop.exists():
                stop.unlink()
                break
    finally:
        trainer.save(args.out, history)
        trainer.close()
        log.close()
    print(f"saved {args.out} ({trainer.iteration} iterations, {trainer.samples} samples)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
