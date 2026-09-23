"""Controlled eight-process snapshot benchmark, with no chess opponent or learning."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
from multiprocessing import get_context
from pathlib import Path
import pickle
import sys
import time
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tonkai.actor_snapshot import pack_snapshot, unpack_snapshot
from scripts.experimental_shared_snapshot import SnapshotSlots, read_payload
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveTrainer, CognitiveKnowledge
from tonkai.temporal_model import SurvivalNet
from scripts.benchmark_cpp_iteration import digest

_state = {}


def initialize(cfg, barrier):
    torch.set_num_threads(1)
    net = SurvivalNet(cfg).eval()
    _state.update(net=net, knowledge=CognitiveKnowledge(cfg, net.encoder))
    barrier.wait(timeout=90)


def consume(payload):
    weights, knowledge = unpack_snapshot(read_payload(payload))
    _state['net'].load_state_dict(weights)
    _state['knowledge'].load_state_dict(knowledge)
    return digest((_state['net'].state_dict(), _state['knowledge'].state_dict(include_episodes=False)))


def main():
    torch.set_num_threads(1)
    source = ROOT/'checkpoints/cognitive-rodent-100k-20260922/latest.pt'
    blob = source.read_bytes()
    checksum = hashlib.sha256(blob).hexdigest()
    state = torch.load(source, map_location='cpu', weights_only=False)
    trainer = CognitiveTrainer(CognitiveConfig.from_dict(state['config']))
    trainer.load_state_dict(state)
    def legacy():
        weights = {key: value.detach().cpu() for key, value in trainer.net.state_dict().items()}
        knowledge = trainer.knowledge.state_dict(include_episodes=False, copy=False)
        return pickle.dumps((weights, knowledge), protocol=pickle.HIGHEST_PROTOCOL)
    expected = digest((trainer.net.state_dict(), trainer.knowledge.state_dict(include_episodes=False)))
    ctx = get_context('spawn')
    rows = []
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx, initializer=initialize,
                             initargs=(trainer.cfg, ctx.Barrier(8))) as pool:
        warm = [pool.submit(consume, legacy()) for _ in range(8)]
        assert all(f.result(timeout=90) == expected for f in warm)
        for mode in ('legacy', 'arrays', 'shared', 'shared', 'arrays', 'legacy'):
            slots, pending, submitted, completed = SnapshotSlots(8), {}, 0, 0
            start = time.perf_counter()
            pack_seconds = 0.
            try:
                while completed < 256:
                    while len(pending) < 8 and submitted < 256:
                        before = time.perf_counter()
                        payload = legacy() if mode == 'legacy' else pack_snapshot(trainer.net, trainer.knowledge)
                        slot = None
                        if mode == 'shared':
                            slot, payload = slots.publish(payload)
                        pack_seconds += time.perf_counter() - before
                        pending[pool.submit(consume, payload)] = slot
                        submitted += 1
                    ready, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=90)
                    if not ready:
                        raise TimeoutError('Snapshot consumers stalled')
                    for future in ready:
                        slot = pending.pop(future)
                        assert future.result() == expected
                        if slot is not None:
                            slots.release(slot)
                        completed += 1
                seconds = time.perf_counter() - start
            finally:
                # Do not reclaim mappings until every consumer has finished.
                if pending:
                    wait(pending)
                slots.close()
            row = dict(mode=mode, snapshots=completed, seconds=seconds,
                       snapshots_per_second=completed/seconds, pack_seconds=pack_seconds)
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = dict(runs=rows, workers=8, exact_snapshot_state=True, state_digest=expected,
                  checkpoint_sha256=checksum, checkpoint_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==checksum,
                  conditions='No opponent, no learning; same frozen snapshot each job, deserialized and loaded into actor net/knowledge then fully hashed. Process startup excluded.')
    (ROOT/'perf_results/snapshot-transport-controlled.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
