"""Same positions, different Claudio settings: time per decision and how often
the chosen move differs from a thorough reference. Positions: Claudio's turns
in arena games (each game replayed in order, so memory is exact)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arena  # noqa: E402
from paths import ROOT  # noqa: E402

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
CONFIGS = {
    "ref": ["tactic_top=4", "root_pvs=false"],
    "top4_pvs": ["tactic_top=4"],
    "top4_pvs_mate2": ["tactic_top=4", "mate_attack=2"],
    "top6_pvs_mate2": ["tactic_top=6", "mate_attack=2"],
    "top4_pvs_mate2_d2": ["tactic_top=4", "mate_attack=2", "tactic_depth=2"],
    "v0_mate2": ["mate_attack=2"],
}
import psutil_free_priority  # noqa: E402,F401
arena._init("", [], [])
import chess  # noqa: E402
import numpy as np  # noqa: E402
from claudio_agent import Claudio, ClaudioConfig  # noqa: E402

net, values, king = arena._state["claudio"]
games = json.loads((ROOT / "claudio/results/arena-v0.json").read_text(encoding="utf-8"))["games"][:GAMES]
chosen = {name: [] for name in CONFIGS}
spent = {name: 0.0 for name in CONFIGS}
for g in games:
    players = {name: Claudio(net, values, ClaudioConfig.with_settings(s), rng=np.random.default_rng(0),
                             king_model=king) for name, s in CONFIGS.items()}
    board = chess.Board()
    white = g["claudio_white"]
    for ply, uci in enumerate(g["moves"].split()):
        if ply >= 4 and (board.turn == chess.WHITE) == white:
            for name, player in players.items():
                start = time.perf_counter()
                move = player.choose(board)
                spent[name] += time.perf_counter() - start
                chosen[name].append(move)
        board.push_uci(uci)
n = len(chosen["ref"])
print(f"{n} decisions from {len(games)} games")
for name in CONFIGS:
    same = sum(a == b for a, b in zip(chosen[name], chosen["ref"]))
    print(f"{name:14s} {1000 * spent[name] / n:6.2f} ms/decision   differs from ref {100 * (1 - same / n):5.1f}%")
