"""claudio_fast vs claudio_fast_next on real positions: identical results? how much faster?"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
sys.path.insert(0, str(ROOT / "claudio/fast-next"))
import psutil_free_priority  # noqa: E402,F401

import chess  # noqa: E402
import claudio_fast  # noqa: E402
import claudio_fast_next  # noqa: E402
from tonkai import fast_rules  # noqa: E402

VALUES = [0.0, 1.0, 2.13, 2.73, 3.98, 9.06]


def board_of(module, position):
    root = position.root()
    b = module.Board(root.pawns, root.knights, root.bishops, root.rooks, root.queens, root.kings,
                     root.occupied_co[chess.WHITE], root.occupied_co[chess.BLACK], root.turn, root.castling_rights,
                     -1 if root.ep_square is None else root.ep_square, root.halfmove_clock, root.fullmove_number)
    b.push_many([fast_rules.code(m) for m in position.move_stack])
    return b


def same(a, b):
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


games = json.loads((ROOT / "claudio/results/engine-grande-claudio-grande-g0-sf1800.json").read_text())["list"][:int(sys.argv[1])]
checks = mismatches = 0
spent = {"old": 0.0, "new": 0.0}
for g in games:
    board = chess.Board()
    for ply, uci in enumerate(g["moves"].split()):
        board.push_uci(uci)
        if ply < 4 or board.is_game_over():
            continue
        results = {}
        for name, module in (("old", claudio_fast), ("new", claudio_fast_next)):
            fast = board_of(module, board)
            codes = list(fast.legal_moves())
            top = codes[:4]
            start = time.perf_counter()
            out = (fast.exchange_after_moves(codes, VALUES, 5, 200000),
                   fast.deep_after_moves(top, VALUES, 3, 5, 2000000, True),
                   fast.root_search(top, [0.5, 0.49, 0.48, 0.47][:len(top)], [0.0] * len(top), 0.2, 0.3, VALUES,
                                    3, 5, 2000000),
                   fast.forced_mate(board.turn, 3, 20000, 1), fast.forced_mate(not board.turn, 2, 20000, 1))
            spent[name] += time.perf_counter() - start
            results[name] = out
        checks += 1
        if not same(list(results["old"]), list(results["new"])):
            mismatches += 1
print(f"{checks} posicoes: resultados diferentes em {mismatches}; antigo {1000 * spent['old'] / checks:.2f} ms, "
      f"novo {1000 * spent['new'] / checks:.2f} ms  (x{spent['old'] / spent['new']:.2f})")
