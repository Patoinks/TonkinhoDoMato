"""Claudio's view of one position: value, material, final score per move.
    python claudio/debug_position.py "<fen>" [settings...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401
import arena  # noqa: E402

arena._init("", [], [])
import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from claudio_agent import Claudio, ClaudioConfig, move_of  # noqa: E402
from tonkai import fast_rules  # noqa: E402

net, values, king = arena._state["claudio"]
board = chess.Board(sys.argv[1])
cfg = ClaudioConfig.with_settings(sys.argv[2:])
c = Claudio(net, values, cfg, king_model=king)
me = board.turn
c._catch_up(board)
codes = list(c.fast.legal_moves())
x = torch.from_numpy(fast_rules.encode_after_moves(c.fast, codes, me))
with torch.inference_mode():
    ctx, _ = net.candidates(x, c.memory)
    heads, _ = net.survival(ctx)
v = heads.mean(-1).numpy()
gains = np.array([c._gain(board, move_of(k)) for k in codes])
nets = gains - np.asarray(c.fast.exchange_after_moves(codes, c.values, 5, 200000))
order = np.argsort(-(v + 0.2 * np.tanh(nets / 3)))
for i in order[:12]:
    print(f"{move_of(codes[i]).uci():6s} value {v[i]:.4f} net {nets[i]:+.2f} score {v[i] + 0.2 * np.tanh(nets[i] / 3):.4f}")
print("chosen:", Claudio(net, values, cfg, king_model=king).choose(board))
