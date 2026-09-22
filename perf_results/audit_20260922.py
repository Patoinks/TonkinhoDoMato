"""Bounded audit only: no checkpoint, production module or old result writes.

Run from repository root: python -B -u perf_results/audit_20260922.py
Reports current integrated training throughput; does not establish strength.
"""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import tonkai
from tonkai.config import Config
from tonkai.parallel import ActorPool


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checkpoint = ROOT / 'checkpoints/deployed_5020games.pt'
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'checkpoints').glob('*.pt')}
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    cfg = Config(device='cuda', inference_device='cpu', temperature=0.1)
    net, _, _, worker, trainer, _ = tonkai.build(cfg)
    net.load_state_dict(state)
    trainer.sync_inference()
    corpus = [worker.play_game() for _ in range(70)]
    print('Fixed replay corpus ready', flush=True)
    result = {
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(0),
                        'parent_torch_threads': torch.get_num_threads()},
        'config': cfg.to_dict(),
        'checkpoint_sha256': hashes,
        'method': {'warmup_games': 15, 'timed_games': 120, 'repeats': 2,
                   'same_initial_weights': True, 'same_70_game_replay': True,
                   'includes_startup_in_timed_window': False,
                   'includes_checkpoint_writes': False, 'curriculum': False,
                   'quality_claim': False},
        'windows': [],
    }
    for repeat, order in enumerate(([1, 2, 4], [4, 2, 1])):
        for workers in order:
            cfg = Config(device='cuda', inference_device='cpu', temperature=0.1,
                         num_workers=workers)
            net, _, _, worker, trainer, metrics = tonkai.build(cfg)
            net.load_state_dict(state)
            trainer.sync_inference()
            for record in corpus:
                trainer.add_game(record)
            pool = ActorPool(cfg, net, metrics) if workers > 1 else None
            try:
                for _ in range(15):
                    trainer.add_game(pool.next_game() if pool else worker.play_game())
                    trainer.train(cfg.updates_per_game)
                    if pool:
                        pool.refresh(net, trainer.games)
                before = metrics.to_dict()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(120):
                    trainer.add_game(pool.next_game() if pool else worker.play_game())
                    trainer.train(cfg.updates_per_game)
                    if pool:
                        pool.refresh(net, trainer.games)
                torch.cuda.synchronize()
                duration = time.perf_counter() - start
                after = metrics.to_dict()
                delta = {key: after[key] - before[key] for key in
                         ('games', 'plies', 'tonkai_positions', 'wins', 'draws', 'losses', 'train_batches')}
                row = {'repeat': repeat + 1, 'workers': workers, 'seconds': duration,
                       'games_s': delta['games'] / duration,
                       'positions_s': delta['tonkai_positions'] / duration,
                       'plies_s': delta['plies'] / duration,
                       'mean_plies': delta['plies'] / delta['games'], **delta,
                       'times': {key: value - before['times'].get(key, 0)
                                 for key, value in after['times'].items()},
                       'actor_times': {key: value - before['actor_times'].get(key, 0)
                                       for key, value in after['actor_times'].items()}}
                result['windows'].append(row)
                print(json.dumps(row), flush=True)
            finally:
                if pool:
                    pool.close()
    result['checkpoints_unchanged'] = all(digest(ROOT / path) == val for path, val in hashes.items())
    target = ROOT / 'perf_results/audit_20260922.json'
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved', target, flush=True)


if __name__ == '__main__':
    main()
