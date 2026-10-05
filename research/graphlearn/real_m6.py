#!/usr/bin/env python3
"""M6: does MarketRank's own chain predict on real 15-minute data? (rule: REAL_M6.md, committed before any real run)

  real_m6.py --panel DIR --out DIR [--workers 3]

Runs wf_m5.py with --variants Bprior,Bshuf,B0E,B0 on the real panel:
  rolling, model seeds 1 and 2   (primary: the deployed regime)
  scratch, model seed 1          (secondary: reported, not gated)
then applies the pre-registered rule (`verdict`) and writes DIR/results.json and DIR/results.md.

Data gate (checked before any training, `data_gate`): at least 400 sessions, and the median number of eligible
nodes per labeled bar at least 800. If the gate fails, nothing is trained.

Primary rule (`verdict`), for EACH rolling seed:
  Bprior - B0E    day-clustered t > 2  AND  positive mean IC difference in >= 2/3 of calendar months
  Bprior - Bshuf  day-clustered t > 2
Pass = both conditions hold for both rolling seeds.
Secondary (reported only): Bprior - B0, Bshuf - B0E, the scratch run, and a top-minus-bottom decile spread of label_6 on
non-overlapping bars (session slots 0, 6, 12, 18), gross and net of 5 bps per side on both legs (20 bps per period).
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import intraday as itd

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = "Bprior,Bshuf,B0E,B0"
RUNS = (("scratch", 1), ("rolling", 1), ("rolling", 2))  # the slow scratch run first (scheduling only)
PRIMARY = (("rolling", 1), ("rolling", 2))
MIN_SESSIONS, MIN_MEDIAN_NODES = 400, 800
COST_PER_PERIOD = 4 * 5e-4  # two legs, in and out, 5 bps per side
SPREAD_SLOTS = (0, 6, 12, 18)


def data_gate(p):
    lm = itd.label_mask(p.session, p.horizon)
    ok = (np.asarray(p.a["elig"]) == 1) & np.isfinite(np.asarray(p.a["ret1"]))
    nodes = ok[lm].sum(axis=1)
    out = {"sessions": int(len(np.unique(p.session))), "median_nodes": float(np.median(nodes)) if len(nodes) else 0.0,
           "tickers": int(p.N), "bars": int(p.T)}
    out["passes"] = out["sessions"] >= MIN_SESSIONS and out["median_nodes"] >= MIN_MEDIAN_NODES
    return out


def verdict(oos_by_run):
    """oos_by_run: {(mode, seed): run.json["oos_ic"]}. Returns (passes, per-run checks)."""
    checks = {}
    for key in PRIMARY:
        s = oos_by_run[key]["Bprior"]
        e, sh = s["minus_B0E"], s["minus_Bshuf"]
        checks[f"{key[0]}_s{key[1]}"] = {
            "Bprior-B0E t": e["t"], "Bprior-B0E months_positive": e["months_positive"],
            "Bprior-Bshuf t": sh["t"],
            "ok": bool(e["t"] > 2 and e["months_positive"] >= 2 / 3 and sh["t"] > 2)}
    return all(c["ok"] for c in checks.values()), checks


def decile_spread(p, csv_path):
    """Mean top-minus-bottom decile label_6 on non-overlapping bars, gross and net, and its day-clustered t."""
    col = {tk: i for i, tk in enumerate(p.tickers)}
    bar = {int(t): i for i, t in enumerate(p.times)}
    slot = np.zeros(p.T, dtype=np.int64)
    for s in np.unique(p.session):
        idx = np.flatnonzero(p.session == s)
        slot[idx] = np.arange(len(idx))
    lm = itd.label_mask(p.session, p.horizon)
    by = {}
    with open(csv_path) as f:
        next(f)
        for line in f:
            t, tk, sc = line.rstrip("\n").split(",")
            d = bar[int(t)]
            if lm[d] and slot[d] in SPREAD_SLOTS:
                by.setdefault(d, ([], []))
                by[d][0].append(col[tk])
                by[d][1].append(float(sc))
    lab = p.a[itd.LABEL]
    per_day = {}
    for d, (idx, sc) in by.items():
        y, s = np.asarray(lab[d, idx], float), np.asarray(sc)
        ok = np.isfinite(y) & np.isfinite(s)
        if ok.sum() < 20:
            continue
        y, s = y[ok], s[ok]
        k = max(1, len(s) // 10)
        o = np.argsort(s, kind="stable")
        per_day.setdefault(int(p.session[d]), []).append(y[o[-k:]].mean() - y[o[:k]].mean())
    days = np.array([np.mean(v) for _, v in sorted(per_day.items())])
    if len(days) < 2:
        return {"gross": float("nan"), "net": float("nan"), "t_gross": float("nan"), "n_days": len(days)}
    g = float(days.mean())
    return {"gross": g, "net": g - COST_PER_PERIOD, "t_gross": float(g / days.std(ddof=1) * np.sqrt(len(days))),
            "n_days": int(len(days))}


def run_one(panel, out, mode, seed, threads):
    d = os.path.join(out, f"{mode}_s{seed}")
    cmd = [sys.executable, os.path.join(HERE, "wf_m5.py"), "--panel", panel, "--out", d, "--store",
           os.path.join(d, "store"), "--variants", VARIANTS, "--mode", mode, "--seed", str(seed), "--threads",
           str(threads), "--csv"]
    with open(os.path.join(out, f"{mode}_s{seed}.log"), "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    with open(os.path.join(d, "run.json")) as f:
        return (mode, seed), json.load(f)


def fmt(s):
    return f"{s['mean']:+.4f} (t={s['t']:+.1f}, months+ {s.get('months_positive', float('nan')):.0%})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4, help="torch threads per run")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    p = itd.load_intraday(a.panel)
    gate = data_gate(p)
    print("data gate:", gate, flush=True)
    res = {"data_gate": gate}
    if not gate["passes"]:
        res["passes"] = None
        res["stopped"] = "data gate failed: nothing trained"
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            runs = dict(ex.map(lambda r: run_one(a.panel, a.out, r[0], r[1], a.threads), RUNS))
        oos = {k: r["oos_ic"] for k, r in runs.items()}
        res["passes"], res["checks"] = verdict(oos)
        res["oos_ic"] = {f"{k[0]}_s{k[1]}": v for k, v in oos.items()}
        res["spread"] = {f"{k[0]}_s{k[1]}": {v: decile_spread(p, os.path.join(a.out, f"{k[0]}_s{k[1]}", f"{v}.csv"))
                                              for v in VARIANTS.split(",")} for k in runs}
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    lines = ["# M6 results", "", f"data gate: `{json.dumps(gate)}`", ""]
    if res.get("checks"):
        lines += [f"**PASS: {res['passes']}**", "", "| run | variant | IC | - B0E | - Bshuf | - B0 |", "|---|---|---|---|---|---|"]
        for run, o in res["oos_ic"].items():
            for v in VARIANTS.split(","):
                s = o[v]
                lines.append(f"| {run} | {v} | {s['mean']:+.4f} (t={s['t']:+.1f}) | "
                             + " | ".join(fmt(s["minus_" + b]) if "minus_" + b in s else "" for b in ("B0E", "Bshuf", "B0"))
                             + " |")
        lines += ["", "Decile spread of label_6 (non-overlapping bars; net of 20 bps per period):", "",
                  "| run | variant | gross | t | net |", "|---|---|---|---|---|"]
        for run, sp in res["spread"].items():
            for v, s in sp.items():
                lines.append(f"| {run} | {v} | {s['gross']:+.5f} | {s['t_gross']:+.1f} | {s['net']:+.5f} |")
    else:
        lines.append(res.get("stopped", ""))
    with open(os.path.join(a.out, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("PASS:", res.get("passes"), flush=True)
    return res


if __name__ == "__main__":
    main()
