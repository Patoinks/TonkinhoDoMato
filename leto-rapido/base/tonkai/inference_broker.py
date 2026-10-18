"""Optional cross-game encoder microbatching with immutable weight versions.

The learner owns the service thread. Spawned actors only send observations;
recurrent state, exploration and search stay local. Versions are reference
counted for the lifetime of games, never silently replaced with latest weights.
"""
import copy
import hashlib
import queue
import threading
import time

import numpy as np
import torch


class EncoderClient:
    def __init__(self, requests, responses, slot, version=None):
        self.requests, self.responses, self.slot, self.version = requests, responses, slot, version

    def encode(self, x):
        if x.device.type != "cpu" or x.requires_grad:
            raise ValueError("broker is for CPU actor inference only")
        self.requests.put((self.slot, self.version, x.detach().numpy().copy()))
        ok, payload = self.responses.get(timeout=60)
        if not ok:
            raise RuntimeError(payload)
        return torch.from_numpy(payload)


class InferenceBroker:
    def __init__(self, context, workers, wait_seconds=0.0005):
        self.requests = context.Queue()
        self.responses = [context.Queue() for _ in range(workers)]
        self.slot_counter = context.Value('i', 0)
        self.wait_seconds = wait_seconds
        self.models, self.lock = {}, threading.Lock()
        self.closed = False
        self.stats = dict(requests=0, batches=0, rows=0)
        self.thread = threading.Thread(target=self._serve, name="Tonkinho-inference", daemon=True)
        self.thread.start()

    def register(self, encoder):
        state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        digest = hashlib.sha256()
        for key, value in state.items():
            digest.update(key.encode()); digest.update(value.numpy().tobytes())
        version = digest.hexdigest()
        with self.lock:
            if version not in self.models:
                frozen = copy.deepcopy(encoder).cpu().eval().requires_grad_(False)
                frozen.load_state_dict(state)
                self.models[version] = [frozen, 0]
            self.models[version][1] += 1
        return version

    def release(self, version):
        with self.lock:
            self.models[version][1] -= 1
            if not self.models[version][1]:
                del self.models[version]

    @torch.inference_mode()
    def _serve(self):
        while True:
            request = self.requests.get()
            if request is None:
                return
            pending = [request]
            deadline = time.perf_counter() + self.wait_seconds
            stop = False
            while time.perf_counter() < deadline:
                try:
                    item = self.requests.get(timeout=max(0.000001, deadline - time.perf_counter()))
                except queue.Empty:
                    break
                if item is None:
                    stop = True
                    break
                pending.append(item)
            groups = {}
            for slot, version, x in pending:
                groups.setdefault(version, []).append((slot, x))
            for version, group in groups.items():
                try:
                    with self.lock:
                        model = self.models[version][0]
                    sizes = [len(x) for _, x in group]
                    output = model(torch.from_numpy(np.concatenate([x for _, x in group]))).numpy()
                    offset = 0
                    for (slot, _), size in zip(group, sizes):
                        self.responses[slot].put((True, output[offset:offset + size].copy()))
                        offset += size
                    self.stats['requests'] += len(group)
                    self.stats['batches'] += 1
                    self.stats['rows'] += len(output)
                except Exception as error:
                    for slot, _ in group:
                        self.responses[slot].put((False, str(error)))
            if stop:
                return

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.requests.put(None)
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            raise TimeoutError("inference service did not stop")
        for q in [self.requests, *self.responses]:
            q.close(); q.join_thread()
