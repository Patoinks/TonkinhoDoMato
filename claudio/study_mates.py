"""How often own-mate searches find something, and what they cost (positions from arena games)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

import chess  # noqa: E402
from tonkai import fast_rules  # noqa: E402

games = json.loads((ROOT / "claudio/results/arena-v0.json").read_text(encoding="utf-8"))["games"][:60]
setups = [(3, 20000), (3, 5000), (2, 20000), (2, 3000), (1, 1000)]
cost = {s: 0.0 for s in setups}
found = {s: 0 for s in setups}
defence = {s: 0.0 for s in [(2, 20000), (1, 1000)]}
positions = 0
for g in games:
    board = chess.Board()
    for uci in g["moves"].split():
        board.push_uci(uci)
        fast = fast_rules.board(board)
        me = board.turn
        positions += 1
        for depth, nodes in setups:
            start = time.perf_counter()
            n = fast.forced_mate(me, depth, nodes, 1)[0]
            cost[(depth, nodes)] += time.perf_counter() - start
            found[(depth, nodes)] += bool(n)
print(f"{positions} positions")
for s in setups:
    print(f"mate_attack depth {s[0]} nodes {s[1]:6d}: {1000 * cost[s] / positions:6.3f} ms/pos, found {found[s]}")
