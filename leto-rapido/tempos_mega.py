"""Tabela de tempos do mega torneio (mega-torneio/games.jsonl)."""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
d = Path(__file__).resolve().parent.parent / "mega-torneio"
rows = [json.loads(l) for l in (d / "games.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
ent = {e["id"]: e["settings"] for e in json.loads((d / "entrants.json").read_text(encoding="utf-8"))}
by = defaultdict(list)
for r in rows: by[r["id"]].append(r)
s = sorted(r["seconds"] for r in rows)
q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
print(f"{len(rows)} jogos | média {st.mean(s):.2f} s | mediana {q(.5):.2f} | p90 {q(.9):.2f} | p99 {q(.99):.2f} | máx {s[-1]:.1f} s")
print(f"meios-lances/jogo {st.mean(r['plies'] for r in rows):.0f} | Leto {st.mean(r.get('leto_ms', 0) for r in rows):.1f} ms/lance | erros {sum(r['end'] == 'error' for r in rows)}\n")
ms = sorted(st.mean(r.get("leto_ms", 0) for r in rs) for rs in by.values())
print(f"ms/lance por mutante: mín {ms[0]:.1f} | mediana {ms[len(ms) // 2]:.1f} | máx {ms[-1]:.1f} ({len(ms)} mutantes)\n")
print("| fase | jogos | s/jogo | mediana | p90 | p99 | máx | ms/lance Leto | meios-lances |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for stage in sorted({r["stage"] for r in rows}):
    rs = [r for r in rows if r["stage"] == stage]
    x = sorted(r["seconds"] for r in rs)
    k = lambda p: x[min(len(x) - 1, int(p * len(x)))]
    print(f"| {stage + 1} | {len(rs)} | {st.mean(x):.2f} | {k(.5):.2f} | {k(.9):.2f} | {k(.99):.2f} | {x[-1]:.2f} | "
          f"{st.mean(r.get('leto_ms', 0) for r in rs):.1f} | {st.mean(r['plies'] for r in rs):.0f} |")
