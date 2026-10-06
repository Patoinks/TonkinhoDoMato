"""Cost of Claudio's calculation settings on real positions (same positions, same memory):
ms per decision and how often the move differs from the current settings."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
import psutil_free_priority  # noqa: E402,F401

import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(1)
from claudio_agent import Claudio, ClaudioConfig  # noqa: E402
from claudio_load import load_claudio  # noqa: E402

CONFIGS = {
    "atual": [],
    "trocas 4": ["exchange_depth=4"],
    "trocas 3": ["exchange_depth=3"],
    "top 3": ["tactic_top=3"],
    "top 2": ["tactic_top=2"],
    "3 meios-lances": ["tactic_depth=2"],
    "nos 200k": ["tactic_nodes=200000"],
}
net, values, config, _, king = load_claudio(ROOT / "claudio/claudio-grande-g0.pt", with_config=True, with_king=True)
games = json.loads((ROOT / "claudio/results/engine-grande-claudio-grande-g0-sf1800.json").read_text())["list"][:int(sys.argv[1])]
chosen = {k: [] for k in CONFIGS}
spent = {k: 0.0 for k in CONFIGS}
for g in games:
    players = {k: Claudio(net, values, ClaudioConfig.with_settings(v, base=config), king_model=king) for k, v in CONFIGS.items()}
    board, me = chess.Board(), chess.WHITE if g["white"] else chess.BLACK
    for ply, u in enumerate(g["moves"].split()):
        if ply >= 4 and board.turn == me:
            for k, p in players.items():
                start = time.perf_counter()
                chosen[k].append(p.choose(board))
                spent[k] += time.perf_counter() - start
        board.push_uci(u)
n = len(chosen["atual"])
print(f"{n} decisoes")
for k in CONFIGS:
    same = sum(a == b for a, b in zip(chosen[k], chosen["atual"]))
    print(f"{k:16s} {1000 * spent[k] / n:6.2f} ms/lance   lance diferente do atual {100 * (1 - same / n):5.1f}%")
