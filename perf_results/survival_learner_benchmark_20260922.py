"""Isolated V2 CPU/CUDA learner timing; never writes a training checkpoint.

Run only after the legacy trainer and the actor benchmark have finished.
The source model/optimizer are from the isolated smoke checkpoint. A fixed
legal random corpus gives the 48-state learner longer episodes than the smoke
run's 24-ply truncations. Each device starts from identical replay/RNG state.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chess
import numpy as np
import torch

from tonkai.encoding import encode, pack_many
from tonkai.sequence_replay import SequenceEpisode, SequenceReplay
from tonkai.survival_config import SurvivalConfig
from tonkai.survival_trainer import SurvivalTrainer


def assert_legacy_finished(pid):
    if platform.system() != "Windows" or pid is None:
        return
    check = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"if (Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue) {{ exit 3 }}"],
        capture_output=True, text=True)
    if check.returncode == 3:
        raise RuntimeError(f"Legacy PID {pid} is still running; no benchmark started")
    if check.returncode:
        raise RuntimeError(f"Cannot verify legacy process: {check.stderr}")


def corpus(seed=20260922, count=12, max_plies=80):
    rng = np.random.default_rng(seed)
    result = []
    for number in range(count):
        board = chess.Board()
        owner = bool(number % 2)
        positions = [encode(board, owner)]
        while len(board.move_stack) < max_plies and not board.is_game_over(claim_draw=True):
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
            positions.append(encode(board, owner))
        outcome = board.outcome(claim_draw=True)
        result.append(SequenceEpisode(
            packed=pack_many(np.stack(positions)),
            lost=bool(outcome and outcome.winner is not None and outcome.winner != owner),
            terminated=outcome is not None,
            opponent="benchmark-legal-random",
            surprises=np.zeros(len(positions), dtype=np.float32),
            actions=tuple(move.uci() for move in board.move_stack)))
    return result


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def run_device(device, cfg, state, warmup, measured):
    local_cfg = deepcopy(cfg)
    local_cfg.device = device
    trainer = SurvivalTrainer(local_cfg)
    trainer.load_state_dict(deepcopy(state))
    batches = []
    original_sample = trainer.replay.sample

    def sample(*args, **kwargs):
        batch = original_sample(*args, **kwargs)
        batches.append(batch["sample_keys"])
        return batch

    trainer.replay.sample = sample
    sync(device)
    warmup_metrics = []
    for _ in range(warmup):
        warmup_metrics.append(trainer.update())
    sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    records = []
    for step in range(measured):
        sync(device)
        started = time.perf_counter()
        metrics = trainer.update()
        sync(device)
        elapsed = time.perf_counter() - started
        if metrics is None or not all(np.isfinite(value) for value in metrics.values()):
            raise AssertionError(f"Missing or non-finite metrics on {device}: {metrics}")
        records.append({"step": step, "seconds": elapsed, **metrics})
    durations = [item["seconds"] for item in records]
    positions = sum(item["learned_positions"] for item in records)
    output = {
        "device": device, "threads": torch.get_num_threads(),
        "warmup_updates": warmup, "measured_updates": measured,
        "mean_update_seconds": statistics.mean(durations),
        "median_update_seconds": statistics.median(durations),
        "min_update_seconds": min(durations), "max_update_seconds": max(durations),
        "updates_per_second": measured / sum(durations),
        "learned_positions_per_second": positions / sum(durations),
        "learner_seconds_per_game_at_two_updates": 2 * statistics.mean(durations),
        "learner_only_games_per_second_at_two_updates": measured / sum(durations) / 2,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "all_losses_and_gradients_finite": all(
            value is not None and all(np.isfinite(x) for x in value.values())
            for value in warmup_metrics + records),
        "sample_keys": batches,
        "warmup_metrics": warmup_metrics, "records": records,
    }
    del trainer
    if device == "cuda":
        torch.cuda.empty_cache()
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=ROOT / "perf_results/survival-smoke/checkpoints/latest.pt")
    ap.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    ap.add_argument("--legacy-pid", type=int, default=30676)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--measure", type=int, default=10)
    args = ap.parse_args()
    assert_legacy_finished(args.legacy_pid)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("This comparison requires CUDA")
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    cfg = SurvivalConfig.from_dict(checkpoint["config"])
    cfg.device = "cpu"
    cfg.sequence_length = 48
    cfg.burn_in = 16
    cfg.sequence_batch_size = 8
    cfg.updates_per_game = 2
    cfg.min_replay = 256
    episodes = corpus()
    replay = SequenceReplay(cfg.replay_capacity, seed=cfg.seed + 1)
    for episode in episodes:
        replay.add(episode)
    checkpoint["replay"] = replay.state_dict()
    checkpoint["config"] = cfg.to_dict()
    # CUDA context initialization occurs before either measured run.
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "source_games": checkpoint["games"], "source_steps": checkpoint["steps"],
        "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "config": cfg.to_dict(),
        "corpus": {
            "policy": "uniform legal random moves for both sides; independent diagnostic data",
            "seed": 20260922, "episodes": len(episodes), "positions": len(replay),
            "plies": [len(episode.actions) for episode in episodes],
            "terminated": [episode.terminated for episode in episodes],
            "packed_sha256": hashlib.sha256(b"".join(e.packed.tobytes() for e in episodes)).hexdigest(),
        },
        "method": "Identical initial model, optimizer, replay and RNG; 3 warmup + 10 timed full updates per device; torch threads=1; CUDA synchronized around every timed update. Replay priorities evolve normally, so device numeric differences may later change samples.",
        "limitations": "Learner timing only, without actors or checkpoint I/O. Random legal episodes are a diagnostic workload, not evidence of playing strength. CPU/CUDA floating-point results are not claimed bit-identical.",
        "source_file_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("tonkai/survival_trainer.py", "tonkai/temporal_model.py", "tonkai/sequence_replay.py")
        },
    }
    output["results"] = [run_device(device, cfg, checkpoint, args.warmup, args.measure)
                         for device in ("cpu", "cuda")]
    cpu, cuda = output["results"]
    output["comparison"] = {
        "cpu_speedup_over_cuda_mean": cuda["mean_update_seconds"] / cpu["mean_update_seconds"],
        "sample_keys_equal": cpu["sample_keys"] == cuda["sample_keys"],
        "learned_positions_equal": [r["learned_positions"] for r in cpu["records"]] == [r["learned_positions"] for r in cuda["records"]],
        "maximum_absolute_loss_difference": max(abs(c["loss"] - g["loss"]) for c, g in zip(cpu["records"], cuda["records"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "comparison": output["comparison"],
                      "runs": [{key: value for key, value in run.items()
                                if key not in ("records", "sample_keys", "warmup_metrics")}
                               for run in output["results"]]}, indent=2))


if __name__ == "__main__":
    main()
