"""Read-only experiments for actor serialization, replay preparation and checkpoint pauses."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chess
import numpy as np
import torch
from tonkai import fast_rules
from tonkai.actor_snapshot import pack_snapshot, unpack_snapshot
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveTrainer


def measure(fn, n=50):
    values = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values) * 1000


def main():
    torch.set_num_threads(1)
    source = ROOT/'checkpoints/cognitive-rodent-100k-20260922/latest.pt'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    state = torch.load(source, map_location='cpu', weights_only=False)
    trainer = CognitiveTrainer(CognitiveConfig.from_dict(state['config']))
    trainer.load_state_dict(state)
    def old_pack():
        weights = {k: t.detach().cpu() for k, t in trainer.net.state_dict().items()}
        knowledge = trainer.knowledge.state_dict(include_episodes=False, copy=False)
        return pickle.dumps((weights, knowledge), protocol=pickle.HIGHEST_PROTOCOL)
    legacy, current = old_pack(), pack_snapshot(trainer.net, trainer.knowledge)
    report = {'legacy_payload_bytes': len(legacy), 'array_payload_bytes': len(current),
              'legacy_pack_ms': measure(old_pack),
              'array_pack_ms': measure(lambda: pack_snapshot(trainer.net, trainer.knowledge)),
              'legacy_unpack_ms': measure(lambda: pickle.loads(legacy)),
              'array_unpack_ms': measure(lambda: unpack_snapshot(current))}
    episodes = list(trainer.replay._episodes.values())
    rng = np.random.default_rng(567)
    tasks = [(int(i), int(rng.integers(len(episodes[int(i)].actions) + 1)))
             for i in rng.integers(len(episodes), size=2500)]
    def prepare(cached):
        cache, hits, outputs = OrderedDict(), 0, []
        start = time.perf_counter()
        for identifier, index in tasks:
            episode = episodes[identifier]
            if cached and identifier in cache:
                root, codes = cache[identifier]
                cache.move_to_end(identifier)
                hits += 1
                native = root.copy()
            else:
                native = fast_rules.board(chess.Board(episode.initial_fen or chess.STARTING_FEN))
                codes = tuple(fast_rules.code(chess.Move.from_uci(a)) for a in
                              (episode.actions if cached else episode.actions[:index]))
                if cached:
                    cache[identifier] = native.copy(), codes
                    if len(cache) > 512:
                        cache.popitem(last=False)
            native.push_many_checked(codes[:index])
            outputs.append(native.fen())
        return time.perf_counter() - start, hits, outputs
    cold = prepare(False)
    cached = prepare(True)
    assert cold[2] == cached[2]
    report['reply_cache_experiment'] = dict(tasks=len(tasks), capacity=512, hits=cached[1],
                                            baseline_seconds=cold[0], cache_seconds=cached[0], exact=True)
    target = ROOT/'perf_results'/('save-experiment-' + uuid.uuid4().hex + '.pt')
    assert target.resolve().parent == (ROOT/'perf_results').resolve() and target.resolve() != source.resolve()
    sync, pause, total = [], [], []
    try:
        trainer.save(target)  # warm disk and serializers
        with ThreadPoolExecutor(max_workers=1) as writer:
            for _ in range(4):
                start = time.perf_counter()
                trainer.save(target)
                sync.append(time.perf_counter() - start)
                start = time.perf_counter()
                # Prototype: an independent, internally consistent snapshot.
                snapshot = deepcopy(trainer.state_dict())
                future = writer.submit(torch.save, snapshot, target)
                pause.append(time.perf_counter() - start)
                future.result()
                total.append(time.perf_counter() - start)
        restored = torch.load(target, weights_only=False, map_location='cpu')
        assert all(torch.equal(v, restored['model'][k]) for k, v in trainer.net.state_dict().items())
        report['checkpoint_experiment'] = dict(sync_pause_ms=statistics.median(sync)*1000,
            async_snapshot_pause_ms=statistics.median(pause)*1000, async_total_ms=statistics.median(total)*1000,
            saved_bytes=target.stat().st_size, model_exact=True)
    finally:
        target.unlink(missing_ok=True)
        target.with_suffix('.pt.tmp').unlink(missing_ok=True)
    report['checkpoint_unchanged'] = hashlib.sha256(source.read_bytes()).hexdigest() == before
    report['checkpoint_sha256'] = before
    (ROOT/'perf_results/data-path-experiments.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
