"""Spawn-only V3 actors. This pool is never used by an existing V2 process."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context
import pickle
import time

import numpy as np
import torch

from .cognitive_agent import CognitiveAgent
from .cognitive_trainer import CognitiveKnowledge
from .resign import ResignTracker
from .survival_agent import play_episode
from .temporal_model import SurvivalNet
from .uci_opponent import UCIOpponent


_actor_state = {}


def _initialize(cfg, engine_path, closing_barrier):
    torch.set_num_threads(1)
    net = SurvivalNet(cfg).cpu().eval().requires_grad_(False)
    knowledge = CognitiveKnowledge(cfg, net.encoder)
    external = UCIOpponent(engine_path, cfg.opponent_elo, cfg.opponent_move_time)
    _actor_state.update(cfg=cfg, net=net, knowledge=knowledge, external=external, closing_barrier=closing_barrier)


def _shutdown_actor():
    # python-chess owns a non-daemon thread. Close it BEFORE interpreter shutdown,
    # not in an atexit/finalizer reached only after Python waits on that thread.
    _actor_state["external"].close()
    _actor_state["closing_barrier"].wait(timeout=30)
    return True


def _play(payload, seed, game_number, resign_plan=None):
    weights, knowledge_state = pickle.loads(payload)
    state = _actor_state
    state["net"].load_state_dict(weights)
    state["knowledge"].load_state_dict(knowledge_state)
    agent = CognitiveAgent(state["net"], state["cfg"], state["knowledge"])
    started = time.perf_counter()
    episode, board, colour, stats = play_episode(agent, state["external"], state["cfg"],
        np.random.default_rng(seed), opponent_name=state["external"].description,
        resign=ResignTracker.from_plan(resign_plan))
    stats.update(actor_seconds=time.perf_counter() - started, cognitive=dict(agent.diagnostics), game_number=game_number)
    return episode, board, colour, stats, agent.trace


class CognitivePool:
    def __init__(self, cfg, engine_path):
        context = get_context("spawn")
        self.barrier = context.Barrier(cfg.num_workers)
        self.executor = ProcessPoolExecutor(max_workers=cfg.num_workers, mp_context=context,
                                            initializer=_initialize, initargs=(cfg, str(engine_path), self.barrier))
        self.pending = set()
        self.capacity = cfg.num_workers
        self.closed = False

    def submit(self, net, knowledge, seed, game_number=0, resign_plan=None):
        if len(self.pending) >= self.capacity:
            raise RuntimeError("Actor pool is full")
        weights = {key: tensor.detach().cpu().clone() for key, tensor in net.state_dict().items()}
        state = knowledge.state_dict()
        state["memory"]["episodes"] = []  # Protected rehearsal belongs only to learner.
        payload = pickle.dumps((weights, state), protocol=pickle.HIGHEST_PROTOCOL)
        self.pending.add(self.executor.submit(_play, payload, int(seed), game_number, resign_plan))

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
