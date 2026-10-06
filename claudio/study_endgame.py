"""What does Claudio's intuition say in the won endgames it failed to convert?"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
import psutil_free_priority  # noqa: E402,F401
import arena  # noqa: E402

arena._init("", [], [])
import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from claudio_agent import Claudio, ClaudioConfig  # noqa: E402
from tonkai import fast_rules  # noqa: E402

net, values, king = arena._state["claudio"]
val = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}
games = json.loads((ROOT / "claudio/results/arena-v4-all.json").read_text(encoding="utf-8"))["games"]
shown = 0
for g in games:
    if g["end"] != "max_plies":
        continue
    board = chess.Board()
    moves = g["moves"].split()
    for u in moves[:250]:
        board.push_uci(u)
    me = chess.WHITE if g["claudio_white"] else chess.BLACK
    mat = sum(val.get(p.piece_type, 0) * (1 if p.color == me else -1) for p in board.piece_map().values())
    if mat < 5 or chess.popcount(board.occupied) > 6:
        continue
    if board.turn != me:
        board.pop()
    c = Claudio(net, values, ClaudioConfig(), king_model=king)
    c._catch_up(board)
    codes = list(c.fast.legal_moves())
    x = torch.from_numpy(fast_rules.encode_after_moves(c.fast, codes, me))
    with torch.inference_mode():
        ctx, _ = net.candidates(x, c.memory)
        heads, _ = net.survival(ctx)
    v = heads.mean(-1).numpy()
    print(board.fen(), f"material {mat}, values min {v.min():.4f} max {v.max():.4f} spread {v.max() - v.min():.4f}")
    shown += 1
    if shown >= 6:
        break
