"""Bounded CPU actors; only the parent owns the learner and its GPU.

Weights are a locked shared snapshot. Finished records travel through a bounded
queue. Metrics from actors are cumulative CPU work, separate from parent wall time.
"""
from __future__ import annotations
import multiprocessing as mp
import queue
import signal
import traceback
import numpy as np
import torch
from .agent import GameWorker, TonkAIAgent
from .model import TonkAINet
from .metrics import Metrics, Window
from . import make_opponent


def _actor(cfg, actor_id, weights, version, lock, stop, results, opponent_settings):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        torch.manual_seed(cfg.seed + actor_id + 1)
        net = TonkAINet(cfg).cpu().eval()
        agent = TonkAIAgent(net, cfg, 'cpu')
        opponent = make_opponent(cfg, cfg.seed + actor_id + 1)
        worker = GameWorker(agent, opponent, cfg, Metrics(),
                            np.random.default_rng(cfg.seed + actor_id + 1))
        local_version = -1
        while not stop.is_set():
            m = Metrics(); worker.metrics = m
            with m.timer('weight_load'):
                with lock:
                    if local_version != version.value:
                        offset = 0
                        with torch.no_grad():
                            for tensor in net.state_dict().values():
                                n = tensor.numel()
                                tensor.copy_(torch.from_numpy(np.frombuffer(weights, dtype=np.float32,
                                    count=n, offset=offset * 4).reshape(tensor.shape)))
                                offset += n
                        local_version = version.value
                    opponent.quiescence = bool(opponent_settings[0])
                    opponent.depth = int(opponent_settings[1])
                    opponent.max_nodes = int(opponent_settings[2])
                    opponent.randomness = opponent_settings[3]
            record = worker.play_game()
            while not stop.is_set():
                try:
                    results.put((record, m, None), timeout=.1)
                    break
                except queue.Full:
                    pass
    except BaseException:
        results.put((None, None, traceback.format_exc()), timeout=2)
    finally:
        results.cancel_join_thread()


class ActorPool:
    def __init__(self, cfg, net, metrics):
        if cfg.num_workers < 2:
            raise ValueError('ActorPool requires at least two workers')
        if cfg.weight_refresh_games < 1:
            raise ValueError('weight_refresh_games must be positive')
        self.cfg, self.metrics = cfg, metrics
        ctx = mp.get_context('spawn')
        self.lock = ctx.Lock(); self.stop = ctx.Event()
        self.weights = ctx.RawArray('f', sum(t.numel() for t in net.state_dict().values()))
        self.version = ctx.RawValue('q', 0)
        self.settings = ctx.RawArray('d', [cfg.opponent_quiescence, cfg.opponent_depth,
                                         cfg.opponent_nodes, cfg.opponent_randomness])
        self.results = ctx.Queue(maxsize=2 * cfg.num_workers)
        self.processes = []
        self.last_refresh = 0
        self.publish(net)
        try:
            with metrics.timer('actor_start'):
                for i in range(cfg.num_workers):
                    p = ctx.Process(target=_actor, args=(cfg, i, self.weights, self.version,
                        self.lock, self.stop, self.results, self.settings), daemon=True)
                    p.start(); self.processes.append(p)
        except BaseException:
            self.close(); raise

    def publish(self, net):
        with self.metrics.timer('weight_publish'):
            snapshot = np.concatenate([t.detach().cpu().numpy().reshape(-1)
                                       for t in net.state_dict().values()])
            with self.lock:
                np.copyto(np.frombuffer(self.weights, dtype=np.float32), snapshot)
                self.version.value += 1

    def refresh(self, net, games):
        if games - self.last_refresh >= self.cfg.weight_refresh_games:
            self.publish(net); self.last_refresh = games

    def set_opponent(self, opponent):
        with self.lock:
            self.settings[:] = [opponent.quiescence, opponent.depth,
                                opponent.max_nodes, opponent.randomness]

    def next_game(self):
        with self.metrics.timer('actor_wait'):
            while True:
                try:
                    record, actor_metrics, error = self.results.get(timeout=.2)
                    if error:
                        raise RuntimeError('CPU actor failed:\n' + error)
                    break
                except InterruptedError:
                    continue
                except queue.Empty:
                    if any(p.exitcode is not None for p in self.processes):
                        raise RuntimeError('CPU actor exited unexpectedly')
        # Actor timings overlap; keep them OUT of parent wall-clock percentages.
        for key, value in vars(actor_metrics).items():
            target = getattr(self.metrics, key, None)
            if isinstance(target, Window):
                target.extend(value.values)
        for key in ('games','wins','draws','losses','plies','tonkai_positions',
                    'opponent_moves','sacrifices','theory_events'):
            setattr(self.metrics, key, getattr(self.metrics, key) + getattr(actor_metrics, key))
        for key, value in actor_metrics.termination.items():
            self.metrics.termination[key] += value
        if not hasattr(self.metrics, 'actor_times'):
            from collections import defaultdict
            self.metrics.actor_times = defaultdict(float)
        for key, value in actor_metrics.times.items():
            self.metrics.actor_times[key] += value
        return record

    def close(self):
        self.stop.set()
        with self.metrics.timer('actor_stop'):
            for p in self.processes:
                p.join(timeout=2)
                if p.is_alive():
                    p.terminate(); p.join(timeout=2)
            self.results.close()
            self.results.cancel_join_thread()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

