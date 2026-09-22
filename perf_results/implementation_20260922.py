"""Frozen-source equivalence and paired throughput checks; never saves checkpoints.

    python perf_results/implementation_20260922.py --mode verify
    python perf_results/implementation_20260922.py --mode bench

The reference is the commit before the September 22 implementation, not a moving
HEAD. Its sources are retained in a hashed JSON artifact. Verification compares
raw tensor/array bytes, including optimizer, RNG and metric window contents.
The timed benchmark runs baseline/current/current/baseline with identical initial
weights, corpus, opponent seed and training schedule. Timed games include replay
ingestion and actual updates, but exclude startup, warm-up and result hashing.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time
import types
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chess
import numpy as np
import torch
import tonkai
from tonkai.config import Config

BASELINE_REF = "e490be536d9d6ede9bbc924025114908507dfae2"
SNAPSHOT = ROOT / "perf_results" / ("implementation_baseline_" + BASELINE_REF[:12] + ".json")
MODULE_ORDER = ["config", "encoding", "model", "metrics", "chess_env",
                "theory_buffer", "replay", "agent", "trainer"]


def source_snapshot():
    if SNAPSHOT.exists():
        payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        assert payload["git_ref"] == BASELINE_REF
    else:
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", BASELINE_REF, "tonkai"],
            cwd=ROOT, text=True).splitlines()
        sources = {name: subprocess.check_output(
            ["git", "show", f"{BASELINE_REF}:{name}"], cwd=ROOT).decode("utf-8")
            for name in names if name.endswith(".py")}
        payload = {"git_ref": BASELINE_REF, "sources": sources,
                   "sha256": {name: hashlib.sha256(value.encode()).hexdigest()
                              for name, value in sources.items()}}
        SNAPSHOT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, source in payload["sources"].items():
        assert hashlib.sha256(source.encode()).hexdigest() == payload["sha256"][name], name
    return payload


def reference_package(snapshot):
    name = "_implementation_reference_tonkai"
    package = types.ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(ROOT / "tonkai")]
    package.__file__ = str(ROOT / "tonkai" / "__init__.py")
    sys.modules[name] = package
    for short in MODULE_ORDER:
        module = types.ModuleType(name + "." + short)
        module.__package__ = name
        module.__file__ = f"git:{BASELINE_REF}:tonkai/{short}.py"
        sys.modules[module.__name__] = module
        setattr(package, short, module)
        source = snapshot["sources"][f"tonkai/{short}.py"]
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    exec(compile(snapshot["sources"]["tonkai/__init__.py"],
                 f"git:{BASELINE_REF}:tonkai/__init__.py", "exec"), package.__dict__)
    return package


def update_hash(h, item):
    if isinstance(item, torch.Tensor):
        item = item.detach().cpu().contiguous().numpy()
    if isinstance(item, np.ndarray):
        update_hash(h, ("array", str(item.dtype), tuple(item.shape)))
        h.update(item.tobytes(order="C"))
    elif isinstance(item, dict):
        h.update(b"dict{")
        for key in sorted(item, key=repr):
            update_hash(h, key)
            update_hash(h, item[key])
        h.update(b"}")
    elif isinstance(item, (list, tuple)):
        h.update(b"sequence[")
        for value in item:
            update_hash(h, value)
        h.update(b"]")
    elif isinstance(item, (float, np.floating)):
        h.update(b"float" + struct.pack("!d", float(item)))
    elif isinstance(item, (int, np.integer, bool)):
        h.update(("int:" + str(int(item)) + ";").encode())
    elif item is None:
        h.update(b"none;")
    elif isinstance(item, str):
        value = item.encode()
        h.update(f"str:{len(value)}:".encode() + value)
    else:
        raise TypeError(type(item))


def digest(item):
    h = hashlib.sha256()
    update_hash(h, item)
    return h.hexdigest()


def metric_data(metrics):
    result = {key: getattr(metrics, key) for key in (
        "games", "wins", "draws", "losses", "plies", "tonkai_positions",
        "opponent_moves", "train_batches", "sacrifices", "theory_events")}
    for key, value in vars(metrics).items():
        if hasattr(value, "values") and hasattr(value, "mean"):
            result[key] = np.asarray(value.values, dtype=np.float64)
    result["piece_value"] = {key: np.asarray(value.values, dtype=np.float64)
                             for key, value in metrics.piece_value.items()}
    result["termination"] = dict(metrics.termination)
    return result


def rng_state(trainer):
    return {"numpy": copy.deepcopy(trainer.rng.bit_generator.state),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_torch_rng(state):
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def fingerprint(rig):
    net, agent, _, _, trainer, metrics = rig
    windows = metric_data(metrics)
    return {"parameters": digest(net.state_dict()),
            "inference_parameters": digest(agent.net.state_dict()),
            "optimizer": digest(trainer.opt.state_dict()),
            "rng": digest(rng_state(trainer)),
            "losses": digest({k: windows[k] for k in ("value_loss", "predict_loss", "piece_loss")}),
            "piece_metrics": digest(windows["piece_value"]),
            "metrics": digest(windows),
            "counters": digest({"steps": trainer.steps, "games": trainer.games}),
            "replay": digest({"value": trainer.replay.pos[:len(trainer.replay)],
                              "result": trainer.replay.result[:len(trainer.replay)],
                              "predict": trainer.predict_replay.cand[:len(trainer.predict_replay)],
                              "count": trainer.predict_replay.count[:len(trainer.predict_replay)],
                              "true_idx": trainer.predict_replay.true_idx[:len(trainer.predict_replay)]})}


def equal_fingerprints(a, b, context):
    wrong = [key for key in a if a[key] != b.get(key)]
    if wrong:
        raise AssertionError(f"{context}: different {wrong}\nreference={a}\ncurrent={b}")


def record_data(record):
    return {"packed": record.packed, "result": record.result,
            "reason": record.reason, "plies": record.plies,
            "predict_samples": record.predict_samples,
            "theory_events": [dataclasses.asdict(v) for v in record.theory_events],
            "sacrifices": [dataclasses.asdict(v) for v in record.sacrifices],
            "tonkai_positions": record.tonkai_positions}


def make_fixed_corpus(reference):
    """Legal positions of varying occupancy, with every result label represented."""
    rng = np.random.default_rng(8157)
    board = chess.Board()
    starts = [chess.STARTING_FEN, "8/8/3k4/8/8/3K4/8/8 w - - 0 1",
              "8/P7/3k4/8/8/3K4/7p/8 w - - 0 1"]
    records = []
    for block in range(144):
        if block % 48 == 0:
            board = chess.Board(starts[block // 48])
        packed, predict = [], []
        for ply in range(16):
            moves = list(board.legal_moves)
            if not moves:
                board = chess.Board()
                moves = list(board.legal_moves)
            color = bool((block + ply) % 2)
            encoded = reference.encoding.encode_after_moves(board, moves, color)
            candidates = reference.encoding.pack_many(encoded)
            selected = int(rng.integers(len(moves)))
            packed.append(candidates[selected].copy())
            predict.append((candidates, selected))
            board.push(moves[selected])
        records.append(reference.agent.GameRecord(
            packed=np.stack(packed), result=block % 3, reason="fixed_corpus",
            plies=len(packed), predict_samples=predict))
    return records


def build_rig(package, cfg, corpus, state=None):
    rig = package.build(copy.deepcopy(cfg), cfg.device)
    if state is not None:
        rig[0].load_state_dict(state)
        rig[4].sync_inference()
    for record in corpus:
        rig[4].add_game(record)
    assert rig[4].ready()
    return rig


def verify(reference, devices, write_progress):
    corpus = make_fixed_corpus(reference)
    result = {"corpus_sha256": digest([record_data(r) for r in corpus]), "cases": []}
    for device in devices:
        if device == "cuda" and not torch.cuda.is_available():
            result["cases"].append({"device": device, "skipped": "CUDA unavailable"})
            continue
        for schedule in ("legacy", "consolidated"):
            for pieces in (0, 1, 2):
                cfg = Config(device=device, inference_device="cpu", seed=741,
                             min_replay=1, replay_capacity=4096, predict_capacity=512,
                             pieces_per_position=pieces, samples_per_game=0,
                             batch_size=17, updates_per_game=4 if schedule == "legacy" else 1)
                rigs = [build_rig(package, cfg, corpus) for package in (reference, tonkai)]
                states = [rng_state(rig[4]) for rig in rigs]
                equal_fingerprints(fingerprint(rigs[0]), fingerprint(rigs[1]), "initial state")
                phases = []
                for batch in (17, 129, 512, 17):
                    checks = []
                    for index, rig in enumerate(rigs):
                        trainer = rig[4]
                        trainer.cfg.batch_size = batch
                        trainer.cfg.samples_per_game = 0 if schedule == "legacy" else batch * 4
                        restore_torch_rng(states[index])
                        trainer.train(trainer.cfg.updates_per_game)
                        states[index] = rng_state(trainer)
                        checks.append(fingerprint(rig))
                    context = f"{device}/{schedule}/pieces={pieces}/batch={batch}"
                    equal_fingerprints(*checks, context)
                    phases.append({"batch_size": batch,
                                   "effective_batch": batch if schedule == "legacy" else min(batch * 4, len(corpus) * 16),
                                   "hashes": checks[0]})
                result["cases"].append({"device": device, "schedule": schedule,
                                        "pieces_per_position": pieces,
                                        "bit_identical": True, "phases": phases})
                print(f"PASS {device} {schedule} pieces={pieces}", flush=True)
                write_progress(result)
    return result


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def bench(reference, args, write_progress):
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("Integrated benchmark requires CUDA learner and CPU inference")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checksum_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"]
    cfg = Config(device=device, inference_device="cpu", seed=args.seed,
                 temperature=.1, num_workers=1,
                 updates_per_game=4 if args.bench_schedule == "legacy" else 1,
                 samples_per_game=0 if args.bench_schedule == "legacy" else 2048)
    generator = reference.build(copy.deepcopy(cfg), device)
    generator[0].load_state_dict(state)
    generator[4].sync_inference()
    corpus = [generator[3].play_game() for _ in range(args.corpus_games)]
    del generator
    result = {"config": cfg.to_dict(), "checkpoint": str(checkpoint),
              "checkpoint_sha256": checksum_before, "checkpoint_games": payload.get("games"),
              "corpus_games": len(corpus),
              "corpus_sha256": digest([record_data(r) for r in corpus]),
              "warmup_games": args.warmup, "timed_games": args.games,
              "order": ["baseline", "current", "current", "baseline"], "runs": []}
    for index, label in enumerate(result["order"]):
        package = reference if label == "baseline" else tonkai
        rig = build_rig(package, cfg, corpus, state)
        _, _, _, worker, trainer, metrics = rig
        warmup_records = []
        for _ in range(args.warmup):
            record = worker.play_game()
            warmup_records.append(record)
            trainer.add_game(record)
            trainer.train(cfg.updates_per_game)
        warmup_state = fingerprint(rig)
        old_steps = trainer.steps
        old_positions, old_plies = metrics.tonkai_positions, metrics.plies
        old_times = dict(metrics.times)
        records = []
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.games):
            record = worker.play_game()
            records.append(record)
            trainer.add_game(record)
            trainer.train(cfg.updates_per_game)
        synchronize(device)
        elapsed = time.perf_counter() - start
        assert trainer.steps - old_steps == args.games * cfg.updates_per_game
        row = {"label": label, "sequence": index, "wall_seconds": elapsed,
               "games_per_second": args.games / elapsed,
               "plies_per_second": (metrics.plies - old_plies) / elapsed,
               "candidate_positions_per_second": (metrics.tonkai_positions - old_positions) / elapsed,
               "plies": metrics.plies - old_plies,
               "candidate_positions": metrics.tonkai_positions - old_positions,
               "updates": trainer.steps - old_steps,
               "times": {k: v - old_times.get(k, 0) for k, v in metrics.times.items()},
               "wdl": [sum(r.result == outcome for r in records) for outcome in (0, 1, 2)],
               "warmup_games_sha256": digest([record_data(r) for r in warmup_records]),
               "games_sha256": digest([record_data(r) for r in records]),
               "warmup_state": warmup_state, "final_state": fingerprint(rig)}
        if result["runs"]:
            first = result["runs"][0]
            equal_fingerprints(first["warmup_state"], row["warmup_state"], f"warmup run {index}")
            equal_fingerprints(first["final_state"], row["final_state"], f"final run {index}")
            assert row["warmup_games_sha256"] == first["warmup_games_sha256"]
            assert row["games_sha256"] == first["games_sha256"]
        result["runs"].append(row)
        print(f"{index + 1}/4 {label}: {row['games_per_second']:.3f} games/s, "
              f"{row['candidate_positions_per_second']:.0f} candidate positions/s", flush=True)
        write_progress(result)
        del rig, trainer, worker, metrics, records, warmup_records
    baseline = [r["games_per_second"] for r in result["runs"] if r["label"] == "baseline"]
    current = [r["games_per_second"] for r in result["runs"] if r["label"] == "current"]
    result["mean_games_per_second"] = {"baseline": float(np.mean(baseline)), "current": float(np.mean(current))}
    result["paired_speedups"] = [current[0] / baseline[0], current[1] / baseline[1]]
    result["mean_speedup"] = float(np.mean(current) / np.mean(baseline))
    result["all_games_parameters_optimizer_rng_metrics_bit_identical"] = True
    result["checkpoint_unchanged"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum_before
    assert result["checkpoint_unchanged"]
    write_progress(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("snapshot", "verify", "bench", "all"), default="verify")
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--checkpoint", default="checkpoints/deployed_5020games.pt")
    parser.add_argument("--bench-schedule", choices=("legacy", "consolidated"), default="consolidated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--corpus-games", type=int, default=70)
    args = parser.parse_args()
    if min(args.games, args.corpus_games) < 1 or args.warmup < 0:
        parser.error("games/corpus-games must be positive and warmup non-negative")
    snapshot = source_snapshot()
    if args.mode == "snapshot":
        print(SNAPSHOT)
        return
    reference = reference_package(snapshot)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = ROOT / "perf_results" / f"implementation_{args.mode}_{stamp}_{uuid.uuid4().hex[:8]}.json"
    report = {"status": "running", "baseline_ref": BASELINE_REF,
              "baseline_snapshot": str(SNAPSHOT),
              "baseline_source_sha256": snapshot["sha256"],
              "production_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted((ROOT / "tonkai").glob("*.py"))},
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "torch_threads": torch.get_num_threads(),
                              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
              "arguments": vars(args)}

    def persist(section, value):
        report[section] = value
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        if args.mode in ("verify", "all"):
            report["verification"] = verify(reference, args.devices.split(","),
                                             lambda value: persist("verification", value))
        if args.mode in ("bench", "all"):
            report["benchmark"] = bench(reference, args, lambda value: persist("benchmark", value))
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"RESULT {output}", flush=True)


if __name__ == "__main__":
    main()
