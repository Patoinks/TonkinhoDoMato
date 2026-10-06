"""Own mates in 3 vs the learned KS of the enemy king: can KS tell when to look?"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
import psutil_free_priority  # noqa: E402,F401

import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tonkai import fast_rules  # noqa: E402
from tonkai.king_safety_model import load_cached  # noqa: E402

cfg = torch.load(ROOT / "checkpoints/obj-max-continuo/latest.pt", map_location="cpu", weights_only=False)["config"]
ks = load_cached(cfg["king_safety_model"])
games = json.loads((ROOT / "claudio/results/arena-v4-all.json").read_text(encoding="utf-8"))["games"][:150]
rows = []
for g in games:
    board = chess.Board()
    white = g["claudio_white"]
    fast = fast_rules.board(board)
    for ply, uci in enumerate(g["moves"].split()):
        if ply >= 4 and (board.turn == chess.WHITE) == white:
            me = board.turn
            m3 = fast.forced_mate(me, 3, 20000, 1)[0]
            m2 = fast.forced_mate(me, 2, 20000, 1)[0]
            w, b = ks.ks_both(board)
            rows.append((m3, m2, b if me else w, chess.popcount(board.occupied)))
        board.push_uci(uci)
        fast.push(fast_rules.code(board.move_stack[-1]))
rows = np.array(rows, dtype=float)
only3 = (rows[:, 0] == 3) & (rows[:, 1] == 0)
print(f"{len(rows)} decisions; mate<=3 found {int((rows[:, 0] > 0).sum())}, of which only at depth 3: {int(only3.sum())}")
for gate in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
    searched = rows[:, 2] < gate
    print(f"KS gate {gate:.2f}: search depth 3 in {100 * searched.mean():5.1f}% of decisions, "
          f"catches {int((only3 & searched).sum())}/{int(only3.sum())} depth-3-only mates")
for pieces in (10, 14, 18, 22):
    searched = rows[:, 3] <= pieces
    print(f"pieces <= {pieces}: {100 * searched.mean():5.1f}% of decisions, catches {int((only3 & searched).sum())}")
