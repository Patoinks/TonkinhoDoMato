"""Where Timdgei (tournament settings) spends its thinking time, per module.
Read-only: plays a few Timdgei-vs-Timdgei games in ONE process under cProfile."""
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import evolve_tonkinho as evo  # noqa: E402

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
log = json.loads((ROOT / "history/evolution/20260924T190310Z/evolution.json").read_text(encoding="utf-8"))
genes = dict(log["base_genes"], tactic_search="exact")
evo._init(str(ROOT / "checkpoints/obj-max-continuo/latest.pt"))
profile = cProfile.Profile()
start = time.perf_counter()
profile.enable()
for game in range(GAMES):
    evo._game(genes, genes, [777, game])
profile.disable()
total = time.perf_counter() - start
stats = pstats.Stats(profile)
rows = {}
for (file, line, name), (cc, nc, tt, ct, callers) in stats.stats.items():
    rows[f"{Path(file).name}:{name}"] = (nc, tt, ct)
watch = ["cognitive_agent.py:evaluate", "cognitive_agent.py:choose", "cognitive_agent.py:_tactic_check",
         "cognitive_agent.py:_moves_allowing_mate", "cognitive_agent.py:_mating_line", "cognitive_agent.py:_robust",
         "cognitive_agent.py:_exchange_nets", "cognitive_agent.py:position_proxy", "cognitive_agent.py:predict_reply",
         "cognitive_agent.py:_repeated_moves", "cognitive_agent.py:_avoid_repetition_when_ahead",
         "cognitive_agent.py:_tablebase_value", "cognitive_agent.py:game_over", "cognitive_agent.py:keys",
         "survival_agent.py:observe", "cognitive_memory.py:query", "king_safety_model.py:ks_rows",
         "king_safety_model.py:ks_both", "cognitive_model.py:encode", "cognitive_agent.py:_valuer",
         "cognitive_agent.py:piece_valuer", "cognitive_agent.py:_piece_raw", "execution_fastpaths.py:batched_replies"]
print(f"{GAMES} games, {total:.1f} s total, {total / GAMES:.2f} s/game (profiled, ~1.5-2x slower than real)")
for key in watch:
    for name, (nc, tt, ct) in rows.items():
        if name == key:
            print(f"{key:48s} calls {nc:7d}  cum {ct:7.2f} s  ({100 * ct / total:5.1f}%)")
print("\n--- top 25 own time ---")
stats.sort_stats("tottime").print_stats(25)
