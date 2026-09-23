from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory

import pytest
import torch

from tonkai.actor_snapshot import pack_snapshot, unpack_snapshot
from scripts.experimental_shared_snapshot import SnapshotSlots, read_payload
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.temporal_model import SurvivalNet


def test_shared_slots_are_independent_and_reused_only_after_release():
    slots = SnapshotSlots(2)
    try:
        first, a = slots.publish(b'first snapshot')
        second, b = slots.publish(b'second snapshot')
        with pytest.raises(RuntimeError):
            slots.publish(b'pool full')
        assert read_payload(a) == b'first snapshot'
        assert read_payload(b) == b'second snapshot'
        slots.release(first)
        reused, c = slots.publish(b'new')
        assert reused == first and c[0] == a[0]
        assert read_payload(c) == b'new'
        assert read_payload(b) == b'second snapshot'
        slots.release(reused)
        slots.release(second)
    finally:
        names = [s.name for s in slots.buffers if s is not None]
        slots.close()
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)


def test_shared_snapshot_survives_parent_weight_updates_in_spawned_process():
    cfg = CognitiveConfig(hidden_size=24, latent_size=16, memory_size=12, piece_hidden=8, head_hidden=8)
    net = SurvivalNet(cfg)
    knowledge = CognitiveKnowledge(cfg, net.encoder)
    expected = {key: value.clone() for key, value in net.state_dict().items()}
    payload = pack_snapshot(net, knowledge)
    slots = SnapshotSlots(1)
    try:
        slot, descriptor = slots.publish(payload)
        with torch.no_grad():
            for value in net.parameters():
                value.add_(10)
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn')) as pool:
            received = pool.submit(read_payload, descriptor).result(timeout=45)
        assert received == payload
        weights, state = unpack_snapshot(received)
        assert all(torch.equal(weights[key], value) for key, value in expected.items())
        assert state['memory']['episodes'] == []
        slots.release(slot)
        grown, large = slots.publish(payload * 10)
        assert read_payload(large) == payload * 10
        slots.release(grown)
    finally:
        slots.close()
