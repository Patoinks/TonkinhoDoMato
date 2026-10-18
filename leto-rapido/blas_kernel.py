"""Time the numpy network on a realistic batch with the OpenBLAS kernel given by OPENBLAS_CORETYPE."""
import os; os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, time
import numpy as np, torch
sys.path.insert(0, "."); import paths  # noqa
torch.set_num_threads(1)
import torneio, rede_rapida
from claudio_load import load_claudio
net, *_ = load_claudio(str(torneio.CHECKPOINT))
rede_rapida.acelerar(net)
rng = np.random.default_rng(0)
x = (rng.random((54, 782)) < 0.05).astype(np.float32)
m = np.zeros((1, 128 if True else 0), dtype=np.float32)
mem = net.initial_memory(1)
for _ in range(50): net.candidates(torch.from_numpy(x), mem)
t = time.perf_counter()
for _ in range(2000): net.candidates(torch.from_numpy(x), mem)
cfg = np.show_config(mode="dicts") if "mode" in np.show_config.__code__.co_varnames else {}
print(os.environ.get("OPENBLAS_CORETYPE", "auto"), round((time.perf_counter() - t) / 2000 * 1000, 4), "ms por chamada")
