"""Wall-clock time per module of Timdgei (tournament settings), ONE process.
Wraps the agent's methods with timers (inclusive times); nothing is saved."""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import evolve_tonkinho as evo  # noqa: E402
from tonkai import cognitive_agent as ca  # noqa: E402
from tonkai import king_safety_model as ksm  # noqa: E402
from tonkai import cognitive_memory as cm  # noqa: E402

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 6
spent, calls = defaultdict(float), defaultdict(int)


def wrap(owner, name, label):
    inner = getattr(owner, name)

    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return inner(*args, **kwargs)
        finally:
            spent[label] += time.perf_counter() - start
            calls[label] += 1
    import inspect
    static = isinstance(inspect.getattr_static(owner, name), staticmethod)
    setattr(owner, name, staticmethod(timed) if static else timed)


A = ca.CognitiveAgent
for name in ["choose", "evaluate", "_tactic_check", "_moves_allowing_mate", "_mating_line", "_robust",
             "_exchange_nets", "_repeated_moves", "position_proxy", "predict_reply", "observe", "keys",
             "_avoid_repetition_when_ahead", "_tablebase_value", "game_over"]:
    wrap(A, name, name)
wrap(ksm.KingSafetyModel, "ks_rows", "king_model.ks_rows")
wrap(ksm.KingSafetyModel, "ks_both", "king_model.ks_both")
for cls in vars(cm).values():
    if isinstance(cls, type) and hasattr(cls, "query"):
        wrap(cls, "query", f"{cls.__name__}.query")
    if isinstance(cls, type) and hasattr(cls, "retrieve"):
        wrap(cls, "retrieve", f"{cls.__name__}.retrieve")

log = json.loads((ROOT / "history/evolution/20260924T190310Z/evolution.json").read_text(encoding="utf-8"))
genes = dict(log["base_genes"], tactic_search="exact")
evo._init(str(ROOT / "checkpoints/obj-max-continuo/latest.pt"))
plies = 0
start = time.perf_counter()
import chess  # noqa: E402
orig_game = evo._game
results = []
for game in range(GAMES):
    results.append(orig_game(genes, genes, [777, game]))
total = time.perf_counter() - start
moves = calls["choose"]
print(f"{GAMES} games in {total:.1f} s = {total / GAMES:.2f} s/game; results {results}")
print(f"decisions (choose) {moves} = {moves / GAMES:.0f} per game (both sides); "
      f"choose {1000 * spent['choose'] / moves:.1f} ms per decision")
for label, seconds in sorted(spent.items(), key=lambda kv: -kv[1]):
    print(f"{label:34s} {seconds:7.2f} s  {100 * seconds / total:5.1f}%  calls {calls[label]:6d}  "
          f"{1000 * seconds / max(1, moves):6.2f} ms per decision")
