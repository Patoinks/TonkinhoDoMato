"""Which mutations help or hurt: each setting's effect on the first-stage score (all entrants, fresh games) and
on the time trial, split by direction (above or below the base value)."""
import json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "."); import paths  # noqa
from dataclasses import asdict
from claudio_agent import ClaudioConfig
from claudio_load import load_claudio
import mega_torneio as M

ROOT = Path(__file__).resolve().parent.parent
_, values, config, _ = load_claudio(str(M.CHECKPOINT), with_config=True)
M039 = ["safety_weight=4.739", "opening_weight=0.08524", "draw_value=0.5594", "simplify_margin=0.411",
        "ahead_draws=False", "value_scaling=False", "reply_weight=0.2936"]
SPEED = ["syzygy=", "root_prunings=True", "tactic_depth=6", "scan_prunings=True", "fast_net=True", "pawn_v4=True",
         "pawn_weight=0.5", "shared_table=True", "reply_top=0", "fast_core=True", "delta_pruning=True", "scan_keep=20"]
OLD = ["syzygy=", "root_prunings=True", "tactic_depth=7", "scan_prunings=True", "fast_net=True", "pawn_v4=True", "pawn_weight=0.5"]

def analyse(folder, base_sets, stage0_games, trials=None):
    base = asdict(ClaudioConfig.with_settings(base_sets, base=config))
    ents = {e["id"]: dict(s.split("=", 1) for s in e["settings"]) for e in json.load(open(folder / "entrants.json", encoding="utf-8"))}
    rows = [json.loads(l) for l in open(folder / "games.jsonl", encoding="utf-8") if l.strip()]
    score = defaultdict(list)
    for r in rows:
        if r["stage"] == 0 and r["end"] != "error" and r["id"] in ents and ents[r["id"]] is not None:
            score[r["id"]].append(r["score"])
    played = {i: st.mean(v) for i, v in score.items() if v and i != "base" and i != "leto"}
    overall = st.mean(played.values())
    effects = []
    keys = sorted({k for i in ents for k in ents[i]})
    for k in keys:
        name = "piece_values" if k == "phase_values" else k
        groups = {"↑": [], "↓": [], "=": []}
        for i, sets in ents.items():
            if k not in sets or i not in played and not trials: continue
            raw = sets[k]
            if k == "phase_values":
                q = [float(x) for x in raw.split(";")[0].split(",")]
                d = "↑" if q[4] > values[5] else "↓"          # by the queen
            else:
                b = base.get(k)
                try:
                    v = float(raw.lower() == "true") if raw in ("True", "False") else float(raw)
                    bv = float(b) if not isinstance(b, str) else None
                    d = "=" if bv is None else ("↑" if v > bv else "↓" if v < bv else "=")
                except ValueError:
                    d = "="
            groups[d].append(i)
        for d, ids in groups.items():
            s = [played[i] for i in ids if i in played]
            if len(s) < 8 and not (trials and len(ids) >= 8): continue
            dead = None
            if trials:
                tr = [trials[i] for i in ids if i in trials]
                dead = sum(t["ratio"] > 1.0 for t in tr) / max(1, len(tr))
                ratio = st.median(t["ratio"] for t in tr if math.isfinite(t["ratio"])) if tr else float("nan")
            effects.append((name, d, len(s), (st.mean(s) - overall) if s else float("nan"),
                            (st.stdev(s) / math.sqrt(len(s))) if len(s) > 1 else float("nan"), dead, ratio if trials else None))
    return overall, len(played), effects

def show(title, overall, n, effects, trials):
    print(f"\n### {title}: {n} mutantes, score médio {overall:.3f}\n")
    print("| mutação | direção | n | efeito no score | ±1σ |" + (" mortos na prova | tempo mediano |" if trials else ""))
    print("| --- | :---: | ---: | ---: | ---: |" + (" ---: | ---: |" if trials else ""))
    for name, d, n_, e, se, dead, ratio in sorted(effects, key=lambda x: -x[3] if not math.isnan(x[3]) else 9):
        extra = f" {dead:.0%} | {ratio:.2f} |" if trials else ""
        print(f"| {name} | {d} | {n_} | {e:+.3f} | {se:.3f} |" + extra)

tt = ROOT / "torneio-tempo"
trials = {json.loads(l)["id"]: json.loads(l) for l in open(tt / "tempos.jsonl", encoding="utf-8") if l.strip()}
o, n, e = analyse(tt, SPEED + M039, 3, trials)
show("Torneio com prova de tempo (fase 1, bo3; prova de tempo de todos os 1024)", o, n, e, True)
o, n, e = analyse(ROOT / "mega-torneio", OLD, 20)
show("Mega torneio (só julgamento, fase 1, bo20)", o, n, e, False)
