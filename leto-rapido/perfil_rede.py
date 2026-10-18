"""Time each layer of the numpy network on the real candidate batches of one fast-Leto game."""
import os; os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, time, collections
import numpy as np
sys.argv = ["torneio.py"]
import torneio
torneio._init(str(torneio.CHECKPOINT), torneio.RAPIDO, torneio.RAPIDO, False)
import claudio_agent as A
batches = []
real_init = A.Claudio.__init__
def init(self, *a, **k):
    real_init(self, *a, **k)
    net = self.net
    if getattr(net, "_grabbed", False): return
    f = net.candidates
    def g(x, m):
        batches.append((x.numpy().copy(), np.asarray(m.numpy() if hasattr(m, "numpy") else m).copy()))
        return f(x, m)
    object.__setattr__(net, "candidates", g); object.__setattr__(net, "_grabbed", True)
    torneio_net[0] = net
torneio_net = [None]
A.Claudio.__init__ = init
torneio.play(0, 2026, 200, 6)
net = torneio_net[0]
print("batches", len(batches), "rows mean", np.mean([len(b[0]) for b in batches]).round(1),
      "nonzero cols mean", np.mean([np.any(b[0] != 0, axis=0).sum() for b in batches]).round(1), "of", batches[0][0].shape[1])
import rede_rapida as R
# rebuild the pieces to time them
enc = None
for cell in net.candidates.__closure__ or []: pass
from claudio_load import load_players
T = collections.Counter()
import torch
raw = load_players(torneio.CHECKPOINT, ["syzygy="])[0][0]
E = R._Encoder(raw.encoder)
print("encoder steps", [(s[0], s[1].shape if len(s) > 1 else "") for s in E.steps])
def run(x):
    first = True
    for i, step in enumerate(E.steps):
        t = time.perf_counter()
        if step[0] == "linear":
            w, b = step[1], step[2]
            if first:
                cols = np.flatnonzero(np.any(x != 0, axis=0)); x = x[:, cols] @ w[cols] + b
            else:
                x = x @ w + b
            first = False
        elif step[0] == "silu": x = R._silu(x)
        else:
            _, w1, b1, w2, b2 = step; x = x + R._silu(x @ w1 + b1) @ w2 + b2
        T[f"{i} {step[0]}"] += time.perf_counter() - t
    return x
for _ in range(5):
    for x, m in batches: run(x.astype(np.float32))
n = 5 * len(batches)
for k, v in sorted(T.items(), key=lambda kv: int(kv[0].split()[0])): print(f"{k:14s} {v/n*1000:.4f} ms")
t = time.perf_counter()
for _ in range(5):
    for x, m in batches: net.candidates.__wrapped__(x, m) if hasattr(net.candidates, "__wrapped__") else None
