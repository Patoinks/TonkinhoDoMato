"""Proof metrics of the sins' modules, from an ab_stockfish.py --out file: each sin against neutral Leto."""
import json
import statistics as st
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pecados as S  # noqa: E402

METRICS = {   # sin: (metric, what it should do vs Leto)
    "sloth": ("ms/lance", "↓"), "gluttony": ("materialismo", "↑"), "lust": ("dama cedo", "↑"),
    "wrath": ("ataque ao rei", "↑"), "envy": ("ameaça do adversário", "↓"), "greed": ("material pendurado", "↓"),
    "pride": ("empates", "↓"),
}


def measure(games):
    acc = {k: [] for k in ("materialismo", "dama cedo", "tropismo", "ataque ao rei", "ameaça do adversário",
                           "material pendurado", "empates", "ms/lance")}
    for g in games:
        me_white = g["leto_white"]
        acc["empates"].append(float(g["score"] == 0.5))
        acc["ms/lance"].append(g.get("leto_ms", 0.0))
        b = chess.Board()
        for i, u in enumerate(g["moves"].split()):
            m = chess.Move.from_uci(u)
            mine = (b.turn == chess.WHITE) == me_white
            if mine:
                me = b.turn
                if any(True for _ in b.generate_legal_captures()):
                    acc["materialismo"].append(float(b.is_capture(m)))
                if i < 30:
                    acc["dama cedo"].append(float(b.piece_type_at(m.from_square) == chess.QUEEN))
                after = b.copy(stack=False)
                after.push(m)
                if len(b.piece_map()) >= 16:
                    acc["tropismo"].append(S._tropism(after, me))
                    acc["ataque ao rei"].append(S._attack_units(after, me))
                    acc["ameaça do adversário"].append(S._threat(after, not me))
                    acc["material pendurado"].append(S._hanging(after, me))
            b.push(m)
    return {k: st.mean(v) if v else float("nan") for k, v in acc.items()}


def main():
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    by = {}
    for g in data["games"]:
        by.setdefault(g["id"], []).append(g)
    base = measure(by["leto"])
    cols = list(base)
    print("| jogador | " + " | ".join(cols) + " |\n| --- |" + " ---: |" * len(cols))
    for name in ["leto"] + [n for n in S.SINS if n in by]:
        m = base if name == "leto" else measure(by[name])
        cells = []
        for c in cols:
            v = m[c]
            fmt = f"{v:.0%}" if c in ("materialismo", "dama cedo", "empates") else f"{v:.2f}"
            if name != "leto" and METRICS.get(name, ("",))[0] == c:
                ok = (v > base[c]) if METRICS[name][1] == "↑" else (v < base[c])
                fmt = f"**{fmt}** {'✓' if ok else '✗'}"
            cells.append(fmt)
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
