"""Benchmark a frozen background save overlapping real updates, without an opponent."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.benchmark_cpp_iteration import digest
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveTrainer


def main():
    torch.set_num_threads(1)
    source = ROOT/'checkpoints/cognitive-rodent-100k-20260922/latest.pt'
    blob = source.read_bytes()
    checksum = hashlib.sha256(blob).hexdigest()
    target = ROOT/'perf_results'/('checkpoint-overlap-' + uuid.uuid4().hex + '.pt')
    assert target.resolve().parent == (ROOT/'perf_results').resolve()
    rows = []
    reference = None
    try:
        for mode in ('sync', 'async', 'async', 'sync'):
            state = torch.load(BytesIO(blob), weights_only=False, map_location='cpu')
            cfg = CognitiveConfig.from_dict(state['config'])
            cfg.device = 'cpu'
            trainer = CognitiveTrainer(cfg)
            trainer.load_state_dict(state)
            # Identical warmup, including the serialization path, before timing.
            for _ in range(3):
                trainer.update()
            trainer.save(target)
            expected_snapshot = digest(trainer.state_dict())
            updates = []
            with ThreadPoolExecutor(max_workers=1) as writer:
                start = time.perf_counter()
                if mode == 'sync':
                    trainer.save(target)
                    pending = None
                else:
                    snapshot = deepcopy(trainer.state_dict())
                    pending = writer.submit(torch.save, snapshot, target)
                pause = time.perf_counter() - start
                # Current save frequency: 100 games, 2 updates per game.
                for _ in range(200):
                    updates.append(trainer.update())
                if pending is not None:
                    pending.result()
                seconds = time.perf_counter() - start
            assert digest(torch.load(target, weights_only=False, map_location='cpu')) == expected_snapshot
            final = digest((trainer.state_dict(), updates))
            if reference is None:
                reference = final
            assert final == reference, 'Learning changed during concurrent save'
            row = dict(mode=mode, updates=len(updates), seconds=seconds, blocking_save_ms=pause*1000,
                       final_state_digest=final, frozen_save_exact=True)
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        target.unlink(missing_ok=True)
        target.with_suffix('.pt.tmp').unlink(missing_ok=True)
    means = {mode: sum(r['seconds'] for r in rows if r['mode'] == mode)/2 for mode in ('sync', 'async')}
    report = dict(runs=rows, mean_seconds=means, speedup=means['sync']/means['async'],
                  checkpoint_unchanged=hashlib.sha256(source.read_bytes()).hexdigest() == checksum,
                  checkpoint_sha256=checksum,
                  conditions='CPU learner only, 200 actual updates per save; no actors or opponent. Background prototype uses a full deep copy. Frozen saved state and final learning state checked exactly.')
    (ROOT/'perf_results/checkpoint-overlap-experiment.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}), flush=True)


if __name__ == '__main__':
    main()
