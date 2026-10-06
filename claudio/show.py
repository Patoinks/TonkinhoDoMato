"""One line per arena result: python claudio/show.py [labels...]"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
files = sorted((HERE / "results").glob("arena-*.json"), key=lambda p: p.stat().st_mtime)
wanted = set(sys.argv[1:])
for f in files:
    r = json.loads(f.read_text(encoding="utf-8"))
    if wanted and r["label"] not in wanted:
        continue
    s = r["summary"]
    t = s["time"]
    n = s["games"]
    print(f"{r['label']:18s} WR {100 * s['win'] / n:5.1f}% (Timdgei {100 * s['loss'] / n:5.1f}%)  "
          f"V/E/D {s['win']:3d}/{s['draw']:3d}/{s['loss']:3d}  "
          f"Claudio {t['claudio']['ms_per_decision']:5.2f} ms {t['claudio']['s_per_game']:.2f} s/jogo  "
          f"Timdgei {t['timdgei']['s_per_game']:.2f} s/jogo  x{s['speedup']:.1f}  {s['plies_per_game']:.0f} ml  "
          f"{' '.join(r['settings'])}")
