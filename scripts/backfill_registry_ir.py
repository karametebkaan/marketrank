#!/usr/bin/env python3
"""One-off backfill of the ir_daily column (9th) of a walk-forward registry (2026-10-02 amendment: gate c3
deflates the daily IR of the excess over buy-and-hold, so the registry must carry each trial's IR).

usage: scripts/backfill_registry_ir.py [WF_OUT]   (default data/walkforward)

For every 8-column row it reads WF_OUT/<run_id>/equity.csv and computes ir_daily of the row's strategy column
against bench:buyhold exactly as mr::performance does: common dates only, the first common return of each curve
measured against its own previous value (1.0 if it starts there), e = r - rb, ir_daily = mean(e) / sample sd(e),
NaN when sd <= 1e-12 * max(1, |mean|). The day count must match the row's T. Rows that already have 9 fields are
left alone, so running it twice changes nothing. The registry is rewritten atomically (temp file + rename) under
the same flock as marketrank (WF_OUT/registry.csv.lock).
"""
import csv
import fcntl
import math
import os
import sys
from pathlib import Path

HEADER = "run_id,strategy,params_hash,cost_bps,top_n,sharpe_daily,T,ann_excess,ir_daily"


def load_equity(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    cols = {name: [r[c] if c < len(r) else "" for r in body] for c, name in enumerate(header) if c > 0}
    return cols


def ir_daily(s_col, b_col):
    e = []
    ps = pb = None  # previous own values
    started = False
    for sv, bv in zip(s_col, b_col):
        s = float(sv) if sv != "" else None
        b = float(bv) if bv != "" else None
        if s is not None and b is not None:
            if not started:
                ps = ps if ps is not None else 1.0
                pb = pb if pb is not None else 1.0
                started = True
            e.append((s / ps - 1.0) - (b / pb - 1.0))
        if s is not None:
            ps = s
        if b is not None:
            pb = b
    T = len(e)
    if T == 0:
        return float("nan"), 0
    m = 0.0
    for v in e:
        m += v
    m /= T
    if T < 2:
        sd = 0.0
    else:
        ss = 0.0
        for v in e:
            ss += (v - m) * (v - m)
        sd = math.sqrt(ss / (T - 1))
    if not (sd > 1e-12 * max(1.0, abs(m))):
        return float("nan"), T
    return m / sd, T


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/walkforward")
    reg = out / "registry.csv"
    with open(out / "registry.csv.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lines = [l.rstrip("\n") for l in reg.read_text().splitlines()]
        rows = [l for l in lines[1:] if l]
        cache, new_rows, filled = {}, [], 0
        for line in rows:
            f = line.split(",")
            if len(f) != 8:
                new_rows.append(line)
                continue
            run, strat, T = f[0], f[1], int(f[6])
            if run not in cache:
                cache[run] = load_equity(out / run / "equity.csv")
            cols = cache[run]
            ir, n = ir_daily(cols[strat], cols["bench:buyhold"])
            if n != T:
                sys.exit(f"{run}/{strat}: {n} common days in equity.csv, registry says {T}")
            new_rows.append(line + "," + format(ir, ".17g"))
            filled += 1
        tmp = out / "registry.csv.tmp"
        with open(tmp, "w") as f:
            f.write(HEADER + "\n")
            for r in new_rows:
                f.write(r + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, reg)
    print(f"backfilled ir_daily for {filled} of {len(rows)} rows in {reg}")


if __name__ == "__main__":
    main()
