#!/usr/bin/env python3
"""Render a walk-forward run's equity curves as an SVG (standard library only).

usage: scripts/render_equity.py <run_dir>

Reads <run_dir>/equity.csv (date column, then one column per curve, as written by
`marketrank --walkforward`) and writes <run_dir>/equity.svg: every curve on a log
scale, strategies as solid lines, benchmarks ("bench:*") dashed, with a legend.
"""
import csv
import html
import math
import sys
from pathlib import Path

W, H = 1000, 560
LEFT, RIGHT, TOP, BOTTOM = 70, 230, 40, 50
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
           "#17becf", "#bcbd22", "#ff7f0e", "#7f7f7f", "#393b79", "#637939"]
BENCH = {"bench:buyhold": "#000000", "bench:rebalanced": "#555555", "bench:VOO": "#999999"}


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], {}
    header, body = rows[0], rows[1:]
    dates = [r[0][:10] for r in body]
    series = {}
    for c, name in enumerate(header[1:], start=1):
        pts = []
        for k, r in enumerate(body):
            if c < len(r) and r[c] != "":
                v = float(r[c])
                if v > 0 and math.isfinite(v):
                    pts.append((k, v))
        series[name] = pts
    return dates, series


def render(dates, series):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           'font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{LEFT}" y="22" font-size="15">Walk-forward equity (log scale, start = 1.0)</text>']
    vals = [v for pts in series.values() for _, v in pts]
    if not dates or not vals:
        out.append(f'<text x="{LEFT}" y="{H / 2}">no equity data (too few rebalances after warm-up)</text>')
        out.append("</svg>")
        return "\n".join(out)
    lo, hi = math.log(min(vals)), math.log(max(vals))
    if hi - lo < 1e-9:
        lo, hi = lo - 0.01, hi + 0.01
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM
    n = max(len(dates) - 1, 1)

    def x(k):
        return LEFT + pw * k / n

    def y(v):
        return TOP + ph * (1 - (math.log(v) - lo) / (hi - lo))

    out.append(f'<rect x="{LEFT}" y="{TOP}" width="{pw}" height="{ph}" fill="none" stroke="#cccccc"/>')
    # Y grid at "nice" values.
    span = math.exp(hi) / math.exp(lo)
    steps = [1.01, 1.02, 1.05, 1.1, 1.25, 1.5, 2, 3, 5, 10]
    step = next((s for s in steps if span ** (1 / 6) <= s), 10)
    v = math.exp(lo)
    g = step ** math.floor(math.log(v) / math.log(step))
    while g <= math.exp(hi) * 1.0001:
        if g >= math.exp(lo) * 0.9999:
            yy = y(g)
            out.append(f'<line x1="{LEFT}" x2="{LEFT + pw}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#eeeeee"/>')
            out.append(f'<text x="{LEFT - 6}" y="{yy + 4:.1f}" text-anchor="end">{g:.3g}</text>')
        g *= step
    # X ticks: about 6 dates.
    for j in range(7):
        k = round(j * n / 6)
        if k < len(dates):
            out.append(f'<text x="{x(k):.1f}" y="{TOP + ph + 18}" text-anchor="middle">{dates[k]}</text>')
    # Curves: strategies first, benchmarks on top.
    names = sorted(series, key=lambda s: (s.startswith("bench:"), s != "blend", s))
    legend = []
    ci = 0
    for name in names:
        pts = series[name]
        if not pts:
            continue
        bench = name.startswith("bench:")
        if bench:
            color = BENCH.get(name, "#777777")
        else:
            color = PALETTE[ci % len(PALETTE)]
            ci += 1
        width = 2.4 if name == "blend" else (1.8 if bench else 1.2)
        dash = ' stroke-dasharray="6 4"' if bench else ""
        d = " ".join(f"{'M' if i == 0 else 'L'}{x(k):.1f},{y(v):.1f}" for i, (k, v) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash}/>')
        legend.append((name, color, dash, width, pts[-1][1]))
    lx, ly = LEFT + pw + 16, TOP + 8
    for i, (name, color, dash, width, last) in enumerate(legend):
        yy = ly + 18 * i
        out.append(f'<line x1="{lx}" x2="{lx + 24}" y1="{yy}" y2="{yy}" stroke="{color}" stroke-width="{width}"{dash}/>')
        out.append(f'<text x="{lx + 30}" y="{yy + 4}">{html.escape(name)} ({last:.3f})</text>')
    out.append("</svg>")
    return "\n".join(out)


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    run = Path(argv[1])
    src = run / "equity.csv"
    if not src.exists():
        print(f"render_equity: {src} not found", file=sys.stderr)
        return 1
    dates, series = load(src)
    dst = run / "equity.svg"
    dst.write_text(render(dates, series) + "\n")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
