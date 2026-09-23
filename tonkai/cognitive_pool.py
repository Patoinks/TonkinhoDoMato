"""Spawn-only V3 actors. This pool is never used by an existing V2 process."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context
import time

import numpy as np
import torch

from .cognitive_agent import CognitiveAgent
from .cognitive_trainer import CognitiveKnowledge
from .resign import ResignTracker
from .survival_agent import play_episode
from .temporal_model import SurvivalNet
from .uci_opponent import UCIOpponent
from .actor_snapshot import pack_snapshot, unpack_snapshot
from .opponents import CognitiveOpponent, TinyOpponentAdapter


_actor_state = {}

# Windows only: the learner is the serial bottleneck, so it can own physical
# core 0 (logical CPUs 0 and 1) while actors and their engines use the rest.
LEARNER_MASK = 0b11
ABOVE_NORMAL_PRIORITY_CLASS = 0x8000


def pin_process(mask=None, above_normal=False):
    """Restrict this process to ``mask`` CPUs and/or raise its priority; False if unsupported."""
    import os
    if os.name != "nt":
        return False
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = ctypes.c_void_p(kernel32.GetCurrentProcess())
    ok = True
    if mask is not None:
        ok &= bool(kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)))
    if above_normal:
        ok &= bool(kernel32.SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS))
    return ok


def actor_mask():
    import os
    return ((1 << (os.cpu_count() or 1)) - 1) & ~LEARNER_MASK


def _initialize(cfg, engine_path, closing_barrier, dedicated=False):
    torch.set_num_threads(1)
    if dedicated:   # before any engine starts: child processes inherit the mask
        pin_process(actor_mask())
    net = SurvivalNet(cfg).cpu().eval().requires_grad_(False)
    knowledge = CognitiveKnowledge(cfg, net.encoder)
    external = UCIOpponent(engine_path, cfg.opponent_elo, cfg.opponent_move_time)
    fairy = UCIOpponent(cfg.fairy_path, cfg.fairy_elo, cfg.opponent_move_time) if cfg.mix_fairy > 0 else None
    _actor_state.update(cfg=cfg, net=net, knowledge=knowledge, external=external, fairy=fairy,
                        closing_barrier=closing_barrier)


def _shutdown_actor():
    # python-chess owns a non-daemon thread. Close it BEFORE interpreter shutdown,
    # not in an atexit/finalizer reached only after Python waits on that thread.
    _actor_state["external"].close()
    if _actor_state.get("fairy") is not None:
        _actor_state["fairy"].close()
    _actor_state["closing_barrier"].wait(timeout=30)
    return True


def _opponent(kind, state, seed, kd_values):
    if kind == "self":
        twin = CognitiveAgent(state["net"], state["cfg"], state["knowledge"], kd_values=kd_values)
        return CognitiveOpponent(twin, state["cfg"], np.random.default_rng([seed, 1]))
    if kind == "tiny":
        return TinyOpponentAdapter(seed)
    if kind == "fairy":
        return state["fairy"]
    if kind == "external":
        return state["external"]
    raise ValueError(f"unknown opponent kind {kind!r}")


def _play(payload, seed, game_number, resign_plan=None, kd_values=None, kind="external"):
    weights, knowledge_state = unpack_snapshot(payload)
    state = _actor_state
    state["net"].load_state_dict(weights)
    state["knowledge"].load_state_dict(knowledge_state)
    agent = CognitiveAgent(state["net"], state["cfg"], state["knowledge"], kd_values=kd_values)
    started = time.perf_counter()
    opponent = _opponent(kind, state, seed, kd_values)
    episode, board, colour, stats = play_episode(agent, opponent, state["cfg"],
        np.random.default_rng(seed), opponent_name=opponent.description,
        resign=ResignTracker.from_plan(resign_plan))
    stats.update(actor_seconds=time.perf_counter() - started, cognitive=dict(agent.diagnostics),
                 game_number=game_number, opponent_kind=kind)
    return episode, board, colour, stats, agent.trace


class CognitivePool:
    def __init__(self, cfg, engine_path, dedicated=False):
        context = get_context("spawn")
        self.barrier = context.Barrier(cfg.num_workers)
        self.executor = ProcessPoolExecutor(max_workers=cfg.num_workers, mp_context=context,
                                            initializer=_initialize, initargs=(cfg, str(engine_path), self.barrier, dedicated))
        self.pending = set()
        self.capacity = cfg.num_workers
        self.closed = False

    def submit(self, net, knowledge, seed, game_number=0, resign_plan=None, kd_values=None, kind="external"):
        if len(self.pending) >= self.capacity:
            raise RuntimeError("Actor pool is full")
        payload = pack_snapshot(net, knowledge)
        future = self.executor.submit(_play, payload, int(seed), game_number, resign_plan, kd_values, kind)
        self.pending.add(future)

    def receive(self, timeout=120):
        if not self.pending:
            raise RuntimeError("No games are in flight")
        ready, _ = wait(self.pending, timeout=timeout, return_when=FIRST_COMPLETED)
        if not ready:
            raise TimeoutError("Cognitive actors did not finish a game in time")
        future = next(iter(ready))
        self.pending.remove(future)
        return future.result()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            # Barrier prevents one worker consuming two shutdown requests before
            # every worker has closed its own persistent engine.
            futures = [self.executor.submit(_shutdown_actor) for _ in range(self.capacity)]
            for future in futures:
                future.result(timeout=120)
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
