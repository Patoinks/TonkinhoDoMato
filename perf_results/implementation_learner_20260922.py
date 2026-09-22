"""Learner-only attribution: original, simple cleanup, grouped CUDA readback.

Three A/B/C/C/B/A blocks, fresh identical model/replay/Adam/RNG each run.
No checkpoint writes. The middle variant uses the original train_step with the
current loss helpers: it retains no_grad and removal of the redundant clone,
but reads metrics immediately. This isolates the grouped readback change.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import implementation_20260922 as I


def grouped_package():
    """Load the rejected candidate explicitly, independent of production state."""
    source = I.ROOT / "perf_results" / "implementation_grouped_trainer_20260922.py"
    spec = importlib.util.spec_from_file_location("tonkai._audit_grouped", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def build(cfg, device=None):
        rig = list(I.tonkai.build(cfg, device))
        net, _, _, worker, previous, metrics = rig
        trainer = module.Trainer(net, cfg, previous.device, metrics, worker.rng)
        trainer.inference_net = previous.inference_net
        rig[4] = trainer
        return tuple(rig)

    return types.SimpleNamespace(build=build), source


def main():
    reference = I.reference_package(I.source_snapshot())
    grouped, grouped_source = grouped_package()
    checkpoint = I.ROOT / "checkpoints" / "deployed_5020games.pt"
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state = I.torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    cfg = I.Config(device="cuda", inference_device="cpu", seed=42, temperature=.1,
                   num_workers=1, updates_per_game=1, samples_per_game=2048)
    generator = reference.build(I.copy.deepcopy(cfg), "cuda")
    generator[0].load_state_dict(state)
    generator[4].sync_inference()
    corpus = [generator[3].play_game() for _ in range(70)]
    del generator
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = I.ROOT / "perf_results" / f"implementation_learner_{stamp}_{uuid.uuid4().hex[:8]}.json"
    report = {
        "status": "running", "baseline_ref": I.BASELINE_REF,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checksum,
        "config": cfg.to_dict(), "warmup_updates": 10, "timed_updates": 50,
        "blocks": 3, "order_per_block": ["baseline", "simple", "grouped", "grouped", "simple", "baseline"],
        "simple_definition": "Current loss helpers with original Trainer.train_step; immediate metric readback",
        "grouped_source": str(grouped_source),
        "grouped_source_sha256": hashlib.sha256(grouped_source.read_bytes()).hexdigest(),
        "corpus_sha256": I.digest([I.record_data(r) for r in corpus]),
        "environment": {"torch": I.torch.__version__, "numpy": I.np.__version__,
                        "torch_threads": I.torch.get_num_threads(),
                        "gpu": I.torch.cuda.get_device_name()},
        "production_source_sha256": {str(p.relative_to(I.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted((I.ROOT / "tonkai").glob("*.py"))},
        "runs": [],
    }

    def persist():
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        for block in range(report["blocks"]):
            for label in report["order_per_block"]:
                package = {"baseline": reference, "simple": I.tonkai, "grouped": grouped}[label]
                rig = I.build_rig(package, cfg, corpus, state)
                trainer = rig[4]
                if label == "simple":
                    trainer.train_step = types.MethodType(reference.trainer.Trainer.train_step, trainer)
                for _ in range(report["warmup_updates"]):
                    trainer.train(1)
                warmup_state = I.fingerprint(rig)
                I.synchronize("cuda")
                start = time.perf_counter()
                for _ in range(report["timed_updates"]):
                    trainer.train(1)
                I.synchronize("cuda")
                elapsed = time.perf_counter() - start
                row = {"block": block, "label": label,
                       "wall_seconds": elapsed,
                       "milliseconds_per_update": elapsed * 1000 / report["timed_updates"],
                       "warmup_state": warmup_state, "final_state": I.fingerprint(rig)}
                if report["runs"]:
                    first = report["runs"][0]
                    I.equal_fingerprints(first["warmup_state"], row["warmup_state"], f"warmup {block}/{label}")
                    I.equal_fingerprints(first["final_state"], row["final_state"], f"final {block}/{label}")
                report["runs"].append(row)
                print(f"block {block + 1} {label}: {row['milliseconds_per_update']:.4f} ms/update", flush=True)
                persist()
                del trainer, rig
        report["summary"] = {}
        for label in ("baseline", "simple", "grouped"):
            values = [r["milliseconds_per_update"] for r in report["runs"] if r["label"] == label]
            report["summary"][label] = {"values_ms": values, "mean_ms": float(I.np.mean(values)),
                                       "median_ms": float(I.np.median(values)),
                                       "min_ms": min(values), "max_ms": max(values)}
        for old, new in (("baseline", "simple"), ("simple", "grouped"), ("baseline", "grouped")):
            report["summary"][old + "_to_" + new] = {
                "mean_speedup": report["summary"][old]["mean_ms"] / report["summary"][new]["mean_ms"],
                "median_speedup": report["summary"][old]["median_ms"] / report["summary"][new]["median_ms"]}
        report["all_parameters_optimizer_rng_losses_metrics_bit_identical"] = True
        report["checkpoint_unchanged"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum
        assert report["checkpoint_unchanged"]
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        persist()
        print(f"RESULT {output}", flush=True)


if __name__ == "__main__":
    main()
