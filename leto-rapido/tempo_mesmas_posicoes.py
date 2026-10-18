"""Decision time of two configurations on the same positions (replayed games), one process, interleaved."""
import os; os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, sys, time, collections
import chess, numpy as np, torch
sys.path.insert(0, "."); import paths  # noqa
torch.set_num_threads(1)
from claudio_load import load_claudio
from claudio_agent import Claudio, ClaudioConfig
import torneio
M039 = ["safety_weight=4.739", "opening_weight=0.08524", "draw_value=0.5594", "simplify_margin=0.411",
        "ahead_draws=False", "value_scaling=False", "reply_weight=0.2936"]
net, values, config, _, king = load_claudio(str(torneio.CHECKPOINT), with_config=True, with_king=True)
configs = {"leto": ["syzygy="] + torneio.RAPIDO, "m039": ["syzygy="] + torneio.RAPIDO + M039}
games = json.load(open(sys.argv[1], encoding="utf-8"))["games"][:int(sys.argv[2])]
T = collections.defaultdict(list); phase = collections.defaultdict(list)
for g in games:
    moves = [chess.Move.from_uci(u) for u in g["moves"].split()]
    for name in (["leto", "m039"] if g["game"] % 2 else ["m039", "leto"]):
        p = Claudio(net, values, ClaudioConfig.with_settings(configs[name], base=config), rng=np.random.default_rng(1), king_model=king)
        p._owner = 1
        b = chess.Board()
        for i, m in enumerate(moves):
            if i >= 6:
                t = time.perf_counter(); p.choose(b); dt = (time.perf_counter() - t) * 1000
                T[name].append(dt); phase[(name, min(i // 40, 4))].append(dt)
            b.push(m)
import statistics as st
for name in ("leto", "m039"):
    x = sorted(T[name])
    print(f"{name}: {len(x)} decisões, média {st.mean(x):.2f} ms, mediana {x[len(x)//2]:.2f}, p90 {x[int(.9*len(x))]:.2f}, máx {x[-1]:.1f}")
for k in range(5):
    a, b = phase[("leto", k)], phase[("m039", k)]
    if a: print(f"meios-lances {40*k}-{40*k+39}: leto {st.mean(a):.2f} ms  m039 {st.mean(b):.2f} ms  ({len(a)} posições)")
