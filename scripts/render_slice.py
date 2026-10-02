#!/usr/bin/env python3
"""Render a MarketRank graph slice (marketrank --export-slice) as an SVG, styled like the A/B/C example.

Usage: scripts/render_slice.py [IN.json] [OUT.svg]
Defaults: docs/img/marketrank-slice.json -> docs/img/marketrank-slice.svg. Standard library only.
"""
import json
import math
import sys
from html import escape

W, H = 900, 560
GRAPH_CX, GRAPH_CY = 300, 300      # centre of the node ring
RING_RX, RING_RY = 232, 186        # ring radii
R_MAX = 54                         # radius of the node with the largest slice_pi (area ~ slice_pi)
BEND = 0.13                        # curvature: control-point offset as a fraction of the chord
FILLS = ["#fbd5c5", "#cfe3f3", "#d9ecd0", "#efe0f5", "#fdf0c4", "#d7eeee", "#f3d9e4", "#e3e3e3"]
STROKES = ["#c0583a", "#3d7bb0", "#4f8f3c", "#8a5aa6", "#9a7410", "#3b8f8f", "#a84d75", "#777777"]
INK = "#1f2733"
MUTED = "#5b6573"


def money(x):
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            v = x / div
            return f"${v:.2f}{suf}" if v < 10 else f"${v:.1f}{suf}" if v < 100 else f"${v:.0f}{suf}"
    return f"${x:.0f}"


def bezier(p0, c, p1, t):
    a = (1 - t) ** 2
    b = 2 * (1 - t) * t
    d = t * t
    return (a * p0[0] + b * c[0] + d * p1[0], a * p0[1] + b * c[1] + d * p1[1])


def unit(dx, dy):
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def render(d):
    tickers, sectors = d["tickers"], d["sectors"]
    spi, mr = d["slice_pi"], d["mr"]
    k = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}
    pmax = max(spi)
    radius = [max(16.0, R_MAX * math.sqrt(p / pmax)) for p in spi]
    pos = []
    for i in range(k):
        a = -math.pi / 2 + 2 * math.pi * i / k
        pos.append((GRAPH_CX + RING_RX * math.cos(a), GRAPH_CY + RING_RY * math.sin(a)))
    dmax = max(e["dollars"] for e in d["edges"]) if d["edges"] else 1.0

    out = []
    put = out.append
    put(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        'font-family="Helvetica, Arial, sans-serif">')
    put("<defs>" + "".join(
        f'<marker id="arr{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
        f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{STROKES[i % len(STROKES)]}"/>'
        f'</marker>' for i in range(k)) + "</defs>")
    put(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    day = d["t"][:10]
    put(f'<text x="24" y="34" font-size="19" font-weight="bold" fill="{INK}">MarketRank on a real slice of the '
        f'money-flow graph</text>')
    put(f'<text x="24" y="56" font-size="12.5" fill="{MUTED}">{escape(d.get("bar", ""))} bar of {day} · top-π stock and '
        f'its {k - 1} strongest flux partners · {d["n_active"]:,} active stocks · p = {d["p"]:.2f}</text>')

    # Edges: one curved arrow per direction, each bending to its own left, so A->B and B->A separate.
    labels = []
    for e in sorted(d["edges"], key=lambda e: e["dollars"]):
        a, b = idx[e["from"]], idx[e["to"]]
        (x0, y0), (x1, y1) = pos[a], pos[b]
        L = math.hypot(x1 - x0, y1 - y0)
        ux, uy = unit(x1 - x0, y1 - y0)
        nx, ny = uy, -ux  # left normal in screen coordinates (y down)
        c = ((x0 + x1) / 2 + nx * BEND * L, (y0 + y1) / 2 + ny * BEND * L)
        sx, sy = unit(c[0] - x0, c[1] - y0)
        ex, ey = unit(c[0] - x1, c[1] - y1)
        p0 = (x0 + sx * (radius[a] + 2), y0 + sy * (radius[a] + 2))
        p1 = (x1 + ex * (radius[b] + 3), y1 + ey * (radius[b] + 3))
        w = 0.8 + 4.2 * e["dollars"] / dmax
        op = 0.35 + 0.5 * e["dollars"] / dmax
        put(f'<path d="M{p0[0]:.1f},{p0[1]:.1f} Q{c[0]:.1f},{c[1]:.1f} {p1[0]:.1f},{p1[1]:.1f}" fill="none" '
            f'stroke="{STROKES[a % len(STROKES)]}" stroke-opacity="{op:.2f}" stroke-width="{w:.2f}" '
            f'marker-end="url(#arr{a})"/>')
        labels.append((bezier(p0, c, p1, 0.42), money(e["dollars"]), e["dollars"] / dmax, STROKES[a % len(STROKES)]))
    for (lx, ly), text, rel, col in labels:
        weight = "bold" if rel > 0.6 else "normal"
        put(f'<text x="{lx:.1f}" y="{ly + 3.5:.1f}" font-size="10" font-weight="{weight}" text-anchor="middle" '
            f'fill="{col}" stroke="#ffffff" stroke-width="3" stroke-linejoin="round" paint-order="stroke">'
            f'{escape(text)}</text>')

    # Nodes: area ~ slice_pi; ticker and slice_pi inside, sector outside, away from the ring centre.
    for i in range(k):
        x, y = pos[i]
        r = radius[i]
        put(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{FILLS[i % len(FILLS)]}" '
            f'stroke="{STROKES[i % len(STROKES)]}" stroke-width="2"/>')
        put(f'<text x="{x:.1f}" y="{y + 1:.1f}" font-size="{15 if r > 30 else 13}" font-weight="bold" '
            f'text-anchor="middle" fill="{INK}">{escape(tickers[i])}</text>')
        put(f'<text x="{x:.1f}" y="{y + 15:.1f}" font-size="10.5" text-anchor="middle" fill="{INK}">'
            f'{spi[i]:.3f}</text>')
        put(f'<text x="{x:.1f}" y="{y + r + 14:.1f}" font-size="11" font-style="italic" text-anchor="middle" '
            f'fill="{MUTED}" stroke="#ffffff" stroke-width="3" stroke-linejoin="round" paint-order="stroke">'
            f'{escape(sectors[i])}</text>')

    # Rank table: slice_pi ranks and the global MarketRank score pi*N.
    tx, ty = 624, 120
    put(f'<rect x="{tx - 12}" y="{ty - 30}" width="{W - tx + 2}" height="{56 + 26 * k}" rx="8" fill="#f7f8fa" '
        f'stroke="#d5d9e0"/>')
    put(f'<text x="{tx}" y="{ty - 10}" font-size="13" font-weight="bold" fill="{INK}">MarketRank on the slice alone</text>')
    cols = [(tx, "start", "#"), (tx + 22, "start", "ticker"), (tx + 128, "end", "slice π"), (tx + 246, "end", "global π·N")]
    for x, anc, h in cols:
        put(f'<text x="{x}" y="{ty + 14}" font-size="11" font-weight="bold" text-anchor="{anc}" fill="{MUTED}">{h}</text>')
    put(f'<line x1="{tx}" y1="{ty + 21}" x2="{tx + 246}" y2="{ty + 21}" stroke="#c9ced6"/>')
    order = sorted(range(k), key=lambda i: (-spi[i], i))
    for r, i in enumerate(order):
        y = ty + 42 + 26 * r
        vals = [str(r + 1), tickers[i], f"{spi[i]:.4f}", f"{mr[i]:.1f}"]
        for (x, anc, _), v in zip(cols, vals):
            bold = ' font-weight="bold"' if r == 0 else ""
            put(f'<text x="{x}" y="{y}" font-size="12.5" text-anchor="{anc}" fill="{INK}"{bold}>{escape(v)}</text>')
        put(f'<circle cx="{tx + 142}" cy="{y - 4}" r="5" fill="{FILLS[i % len(FILLS)]}" '
            f'stroke="{STROKES[i % len(STROKES)]}" stroke-width="1.5"/>')
    note_y = ty + 56 + 26 * k + 4
    notes = ["slice π: the damped chain re-solved on these",
             f"{k} stocks alone (p = {d['p']:.2f}); circle area ∝ slice π.",
             "global π·N: MarketRank in the full graph",
             "(1 = average active stock).",
             "",
             "Arrows: estimated dollar flux from → to,",
             "accumulated over the data window; width ∝",
             "dollars, colour = sending stock. Estimated",
             "from public price and volume bars, not",
             "observed account-level trades.",
             "",
             "marketrank --export-slice · scripts/render_slice.py"]
    for n, line in enumerate(notes):
        put(f'<text x="{tx - 8}" y="{note_y + 16 * n}" font-size="11" fill="{MUTED}">{escape(line)}</text>')
    put("</svg>")
    return "\n".join(out) + "\n"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "docs/img/marketrank-slice.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "docs/img/marketrank-slice.svg"
    with open(src) as f:
        d = json.load(f)
    with open(dst, "w") as f:
        f.write(render(d))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
