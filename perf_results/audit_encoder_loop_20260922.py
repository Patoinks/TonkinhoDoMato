"""Paired whole-loop encoder audit; monkeypatch lives only in this process."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import tonkai
import tonkai.agent as agent_module
from tonkai.config import Config
from audit_encoder_20260922 import encode_after_moves_batched


def update_game_hash(digest, record):
    digest.update(record.packed.tobytes())
    digest.update(json.dumps([record.result, record.reason, record.plies]).encode())
    for packed, true_index in record.predict_samples:
        digest.update(packed.tobytes())
        digest.update(int(true_index).to_bytes(4, "little"))


def parameter_hash(net):
    digest = hashlib.sha256()
    for key, tensor in net.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--checkpoint", default="checkpoints/deployed_5020games.pt")
    parser.add_argument("--out", default="perf_results/audit_encoder_loop_20260922.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This audit requires the GPU learner for the matched comparison")
    cfg = Config(device="cuda", inference_device="cpu", temperature=0.1,
                 samples_per_game=2048, updates_per_game=1, num_workers=1)
    state = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)["model"]
    original_encoder = agent_module.encode_after_moves
    net, _, _, worker, trainer, _ = tonkai.build(cfg)
    net.load_state_dict(state)
    trainer.sync_inference()
    corpus = []
    while len(corpus) < 70 or not trainer.ready():
        record = worker.play_game()
        corpus.append(record)
        trainer.add_game(record)
    initial_replay = {"games": len(corpus), "value": len(trainer.replay),
                      "predict": len(trainer.predict_replay)}
    del worker, trainer, net
    output = {"checkpoint": args.checkpoint, "cfg": cfg.to_dict(),
              "corpus": initial_replay, "warmup_games": args.warmup,
              "measured_games": args.games, "order": ["reference", "batched", "batched", "reference"],
              "runs": [], "notes": "Identical frozen start weights and corpus, fresh optimizer, seeded single worker; monkeypatch only in audit process."}
    out_path = ROOT / args.out
    reference = None
    try:
        for name in output["order"]:
            agent_module.encode_after_moves = original_encoder if name == "reference" else encode_after_moves_batched
            net, _, _, worker, trainer, metrics = tonkai.build(cfg)
            net.load_state_dict(state)
            trainer.sync_inference()
            for record in corpus:
                trainer.add_game(record)
            assert trainer.ready()
            for _ in range(args.warmup):
                trainer.add_game(worker.play_game())
                trainer.train(1)
            start_steps = trainer.steps
            before_times = dict(metrics.times)
            before_plies, before_positions = metrics.plies, metrics.tonkai_positions
            before_refresh = getattr(metrics, "inference_refresh_seconds", 0.0)
            digest = hashlib.sha256()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.games):
                record = worker.play_game()
                update_game_hash(digest, record)
                trainer.add_game(record)
                trainer.train(1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            signature = {"games_sha256": digest.hexdigest(), "parameters_sha256": parameter_hash(net),
                         "rng": trainer.rng.bit_generator.state}
            if reference is None:
                reference = signature
            assert signature == reference, (name, "game/weight/RNG equivalence failed", signature, reference)
            assert trainer.steps - start_steps == args.games
            row = {"encoder": name, "wall_s": elapsed, "games_s": args.games / elapsed,
                   "plies": metrics.plies - before_plies,
                   "plies_s": (metrics.plies - before_plies) / elapsed,
                   "candidate_positions": metrics.tonkai_positions - before_positions,
                   "candidate_positions_s": (metrics.tonkai_positions - before_positions) / elapsed,
                   "actual_updates": trainer.steps - start_steps,
                   "times_s": {key: value - before_times.get(key, 0.0) for key, value in metrics.times.items()},
                   "inference_refresh_s": getattr(metrics, "inference_refresh_seconds", 0.0) - before_refresh,
                   "games_sha256": signature["games_sha256"], "parameters_sha256": signature["parameters_sha256"],
                   "games_weights_rng_identical": True}
            output["runs"].append(row)
            out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
            del worker, trainer, net, metrics
        means = {name: sum(row["games_s"] for row in output["runs"] if row["encoder"] == name) / 2
                 for name in ("reference", "batched")}
        output["mean_games_s"] = means
        output["mean_speedup"] = means["batched"] / means["reference"]
        output["all_runs_games_weights_rng_identical"] = True
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps({"mean_games_s": means, "mean_speedup": output["mean_speedup"],
                          "all_runs_games_weights_rng_identical": True}), flush=True)
    finally:
        agent_module.encode_after_moves = original_encoder


if __name__ == "__main__":
    main()
