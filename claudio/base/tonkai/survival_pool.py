"""Bounded recurrent CPU actors, each owning a persistent external UCI engine.

Only complete episodes cross the process boundary. Network snapshots are copied
by value when submitted; no mutable learner tensors or recurrent states are shared.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import pickle
import signal
import time
import traceback

import numpy as np
import torch

from .survival_agent import NeuralOpponent, SurvivalAgent, play_episode
from .temporal_model import SurvivalNet
from .uci_opponent import UCIOpponent


class SurvivalActorError(RuntimeError):
    """A startup, transport or game failure, with the responsible actor ID."""

    def __init__(self, worker_id: int, message: str):
        self.worker_id = worker_id
        super().__init__(f"Survival actor {worker_id}: {message}")


class _StopRequested(Exception):
    pass


class _InterruptibleAgent(SurvivalAgent):
    def __init__(self, net, cfg, stop):
        self.stop = stop
        super().__init__(net, cfg, "cpu")

    def observe(self, board, self_color):
        # Between real/hypothetical observations, without changing episode data.
        if self.stop.is_set():
            raise _StopRequested()
        return super().observe(board, self_color)


def _send(connection, message):
    # Ordinary pickle avoids multiprocessing's tensor shared-memory reducers.
    # Serialization finishes before returning ownership to a mutating learner.
    connection.send_bytes(pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL))


def _actor(cfg, opponent_path, worker_id, connection, stop):
    external = None
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if stop.is_set():
            return
        net = SurvivalNet(cfg).cpu().requires_grad_(False).eval()
        enemy = SurvivalNet(cfg).cpu().requires_grad_(False).eval()
        agent = _InterruptibleAgent(net, cfg, stop)
        external = UCIOpponent(opponent_path, elo=cfg.opponent_elo,
                               move_time=cfg.opponent_move_time,
                               threads=1, hash_mb=16)
        if stop.is_set():
            return
        _send(connection, ("ready", {
            "pid": os.getpid(), "torch_threads": torch.get_num_threads(),
            "engine_pid": external._active().transport.get_pid(),
            "engine": external.info,
        }))
        while not stop.is_set():
            job = pickle.loads(connection.recv_bytes())
            if job is None or stop.is_set():
                break
            state, kind, opponent_state, opponent_name, seed = job
            net.load_state_dict(state, strict=True)
            rng = np.random.default_rng(seed)
            if kind == "external":
                opponent = external
            else:
                enemy.load_state_dict(opponent_state, strict=True)
                opponent = NeuralOpponent(enemy, cfg, rng)
            started = time.perf_counter()
            episode, board, self_color, stats = play_episode(
                agent, opponent, cfg, rng, opponent_name=opponent_name,
            )
            stats["actor_seconds"] = time.perf_counter() - started
            stats["opponent_kind"] = kind
            _send(connection, ("result", (episode, board, self_color, stats)))
    except _StopRequested:
        pass
    except BaseException:
        if not stop.is_set():
            try:
                _send(connection, ("error", traceback.format_exc()))
            except (EOFError, OSError):
                pass
    finally:
        if external is not None:
            external.close()
        connection.close()


class SurvivalActorPool:
    """One submitted game per actor, with explicit scheduling by the learner.

    ``receive`` returns ``(worker_id, episode, board, self_color, stats)``. The
    board includes its full legal move stack. Timeouts leave games in flight;
    actor errors identify the worker and must be handled by closing the pool.
    """

    STARTUP_TIMEOUT = 45.0

    def __init__(self, cfg, opponent_path, workers: int):
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
            raise ValueError("workers must be an integer between 1 and 8")
        self.workers = workers
        self.processes = []
        self.worker_info = {}
        self._connections = []
        self._busy = set()
        self._closed = False
        self._failure = None
        context = mp.get_context("spawn")
        self._stop = context.Event()
        started = time.perf_counter()
        try:
            for worker_id in range(workers):
                parent, child = context.Pipe(duplex=True)
                self._connections.append(parent)
                process = context.Process(
                    target=_actor,
                    args=(deepcopy(cfg), str(opponent_path), worker_id, child, self._stop),
                    name=f"tonkai-survival-{worker_id}",
                )
                try:
                    process.start()
                    self.processes.append(process)
                finally:
                    child.close()
            deadline = time.monotonic() + self.STARTUP_TIMEOUT
            while len(self.worker_info) < workers:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    pending = sorted(set(range(workers)) - self.worker_info.keys())
                    raise SurvivalActorError(pending[0], "startup timed out")
                try:
                    worker_id, kind, payload = self._message(remaining)
                except TimeoutError as exc:
                    pending = sorted(set(range(workers)) - self.worker_info.keys())
                    raise SurvivalActorError(pending[0], "startup timed out") from exc
                if kind != "ready" or worker_id in self.worker_info:
                    raise SurvivalActorError(worker_id, "invalid startup response")
                self.worker_info[worker_id] = payload
            self.startup_seconds = time.perf_counter() - started
        except BaseException:
            self.close()
            raise

    @property
    def idle_ids(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.workers) if i not in self._busy) if not self._closed else ()

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Survival actor pool is closed")
        if self._failure is not None:
            raise self._failure

    def _error(self, worker_id, message):
        self._failure = SurvivalActorError(worker_id, message)
        raise self._failure

    def _message(self, timeout):
        handles = [*self._connections, *(process.sentinel for process in self.processes)]
        ready = wait(handles, timeout=timeout)
        # Prefer a complete exception/result to the less informative exit code.
        for worker_id, connection in enumerate(self._connections):
            if connection not in ready:
                continue
            try:
                kind, payload = pickle.loads(connection.recv_bytes())
            except (EOFError, OSError) as exc:
                self._error(worker_id, f"connection closed unexpectedly ({exc})")
            if kind == "error":
                self._error(worker_id, payload)
            return worker_id, kind, payload
        for worker_id, process in enumerate(self.processes):
            if process.sentinel in ready or not process.is_alive():
                self._error(worker_id, f"process exited unexpectedly (code {process.exitcode})")
        raise TimeoutError(f"No survival episode received within {timeout:g}s")

    @staticmethod
    def _validate_state(state, name):
        if not isinstance(state, Mapping):
            raise TypeError(f"{name} must be a model state mapping")
        if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu"
               for value in state.values()):
            raise ValueError(f"{name} must contain only CPU tensors")

    def submit(self, worker_id, net_state_cpu, opponent_kind,
               opponent_state_cpu_or_None, opponent_name, seed) -> None:
        self._ensure_open()
        if (isinstance(worker_id, bool) or not isinstance(worker_id, int)
                or not 0 <= worker_id < self.workers):
            raise ValueError("invalid survival actor ID")
        if worker_id in self._busy:
            raise RuntimeError(f"Survival actor {worker_id} already has a game in flight")
        if not self.processes[worker_id].is_alive():
            self._error(worker_id, "process exited before submission")
        if opponent_kind not in ("external", "historical", "current"):
            raise ValueError("opponent_kind must be external, historical or current")
        if not isinstance(opponent_name, str) or not opponent_name:
            raise ValueError("opponent_name must be a nonempty string")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self._validate_state(net_state_cpu, "net_state_cpu")
        if opponent_kind != "external":
            self._validate_state(opponent_state_cpu_or_None, "opponent_state_cpu")
        elif opponent_state_cpu_or_None is not None:
            raise ValueError("external opponents do not accept neural opponent weights")
        try:
            _send(self._connections[worker_id], (
                net_state_cpu, opponent_kind, opponent_state_cpu_or_None,
                opponent_name, int(seed),
            ))
        except (EOFError, OSError) as exc:
            self._error(worker_id, f"could not submit game ({exc})")
        self._busy.add(worker_id)

    def receive(self, timeout=120):
        self._ensure_open()
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative")
        if not self._busy:
            raise RuntimeError("No survival games are in flight")
        worker_id, kind, payload = self._message(float(timeout))
        if kind != "result" or worker_id not in self._busy:
            self._error(worker_id, "unexpected response while collecting episodes")
        self._busy.remove(worker_id)
        return worker_id, *payload

    def close(self, timeout=10.0) -> None:
        """Request cancellation between plies, quit each UCI, then join actors.

        The finite fallback terminates only process objects created by this pool.
        Closing its pipes also prevents a pending result from blocking shutdown.
        """
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        for worker_id, connection in enumerate(self._connections):
            if worker_id not in self._busy:
                try:
                    _send(connection, None)
                except (EOFError, OSError):
                    pass
            connection.close()
        deadline = time.monotonic() + max(0.0, float(timeout))
        for process in self.processes:
            process.join(max(0.0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 2.0
        for process in self.processes:
            process.join(max(0.0, deadline - time.monotonic()))
        self._busy.clear()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
