"""Plot one or more run histories as standalone SVG, with no extra packages.

    python plot_history.py history/*.jsonl --metric top1.mean --out top1.svg
"""
from __future__ import annotations

import argparse
import glob
from html import escape
import json
import math
from pathlib import Path


def load_series(path, metric):
    points = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("kind") != "metrics":
            continue
        value = row
        for key in metric.split("."):
            value = value[key]
        if isinstance(value, (int, float)) and math.isfinite(value):
            points[row["games"]] = float(value)
    return sorted(points.items())


def render(series, metric):
    points = [point for _, values in series for point in values]
    if not points:
        raise ValueError("No metric samples to plot")
    xmax = max(max(x for x, _ in points), 1)
    ymin = min(0, min(y for _, y in points))
    ymax = max(y for _, y in points)
    ymax = ymax if ymax > ymin else ymin + 1
    colors = ["#4fb3c8", "#e8aa53", "#b093ef", "#72bc83", "#f08080"]
    height = 420 + 24 * len(series)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">',
           f'<title>{escape(metric)} by games</title>',
           '<rect width="100%" height="100%" fill="#15201d"/>',
           '<g fill="#edf2ee" font-family="sans-serif" font-size="13">',
           f'<text x="70" y="30" font-size="20">{escape(metric)} by games</text>']
    for tick in range(6):
        x, y = 70 + tick * 150, 350 - tick * 56
        out.extend([f'<path d="M70 {y}H820" stroke="#34453d"/>',
                    f'<text x="12" y="{y + 5}">{ymin + (ymax-ymin)*tick/5:.3g}</text>',
                    f'<text x="{x}" y="375">{xmax*tick/5:.0f}</text>'])
    out.append('<text x="430" y="402">Games (reported cumulative counter)</text>')
    for i, (label, values) in enumerate(series):
        color = colors[i % len(colors)]
        coords = [(70 + x / xmax * 750, 350 - (y-ymin)/(ymax-ymin)*280) for x, y in values]
        poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
        out.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in coords:
            out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
        out.append(f'<text x="70" y="{430+i*24}" fill="{color}">{escape(label)}</text>')
    return "\n".join(out + ['</g></svg>'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--metric", default="surprise.mean")
    parser.add_argument("--out", default="history.svg")
    args = parser.parse_args()
    paths = sorted({p for pattern in args.paths for p in glob.glob(pattern)})
    if not paths:
        parser.error("No history files matched")
    try:
        svg = render([(Path(p).stem, load_series(p, args.metric)) for p in paths], args.metric)
    except (KeyError, ValueError) as error:
        parser.error(f"Cannot plot {args.metric}: {error}")
    Path(args.out).write_text(svg, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
