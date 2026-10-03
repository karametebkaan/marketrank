#!/usr/bin/env python3
"""M5 power curve on synthetic intraday markets (synth_intraday.py) -- no real data.

  power_m5.py --out DIR --stage tune|power|n1000 [--workers 12]

Each point = one market x one mode (rolling/scratch) x one model seed, run through wf_m5.py exactly as deployed.
The baselines (Bprior, B0E, B0) do not depend on the learned configuration: they run once per
(market, mode, model seed) and every configuration's learned run is paired with them. Reported per point:
  oracle_ic        mean per-bar Spearman of the true predictable part vs label_6 over the predicted bars
  ic               mean OOS per-bar IC of each variant
  diff[base]       learned - base: mean per-bar IC difference and its day-clustered t (sessions are the units)
  recovery         corr(learned logit correction g + u.v, planted delta) on the prior support (planted only)
Results go to DIR/<stage>.jsonl (a rerun skips finished points) and DIR/<stage>.md.

Pre-registered rule (POWER_M5.md, committed before the grid ran) -- `select`:
  detected(point) = learned - Bprior t > 2 AND learned - B0E t > 2 (day-clustered SE).
  passes = detected in >= 10 of the 12 planted tuning markets at oracle IC 0.05 (3 market seeds x rolling/scratch x
  2 model seeds) AND every null is clean: null_a |t| < 2 vs Bprior and vs B0E; null_b |t| < 2 vs Bprior (there the
  prior is predictive, so learned - B0E > 0 is expected and only reported).
  Among passing configurations: the most detections at oracle IC 0.02, then 0.03, then 0.05; ties -> fewer
  learned parameters. Nothing passes -> None (stop and report).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import intraday as itd  # noqa: E402
import prior_model as pm  # noqa: E402
import synth_intraday as si  # noqa: E402
import wf_m5  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TUNE_MARKET_SEEDS = (201, 202, 203)
EVAL_MARKET_SEEDS = (1, 2, 3)
MODEL_SEEDS = (1, 2)
TUNE_ICS = (0.02, 0.03, 0.05)
EVAL_ICS = (0.02, 0.03, 0.05, 0.08)
NULL_B_IC = 0.05
MODES = ("rolling", "scratch")
# candidate configurations of the learned model (everything else = wf_m5.DEFAULTS, shared by all variants)
CONFIGS = {
    "unsigned": {"signed": False},
    "signed": {"signed": True},
}
FLAG = {"signed": "--signed"}


def market_key(m):
    return f"{m['kind']}_n{m['n']}_ic{m['target_ic']:g}_s{m['seed']}"


def make_market(root, m):
    d = os.path.join(root, "markets", market_key(m))
    if not os.path.exists(os.path.join(d, "synth.json")):
        si.make_market(d, n=m["n"], sessions=250, target_ic=m["target_ic"], kind=m["kind"], seed=m["seed"])
    return d


def config_args(cfg):
    out = []
    for k, v in cfg.items():
        if isinstance(v, bool):
            out.append(FLAG[k] if v else "--no-" + FLAG[k][2:])
        else:
            out += ["--" + k.replace("_", "-"), str(v)]
    return out


def n_learned_params(cfg, n=300):
    args = wf_m5.parse_args(["--panel", "x", "--out", "y"] + config_args(cfg))
    m = wf_m5.new_model(args, "learned", [f"T{i}" for i in range(n)])
    return int(sum(p.numel() for p in m.parameters()))


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")


def wf_cmd(pdir, work, variants, mode, seed, extra=()):
    return [sys.executable, os.path.join(HERE, "wf_m5.py"), "--panel", pdir, "--out", os.path.join(work, "out"),
            "--store", os.path.join(work, "store"), "--variants", variants, "--mode", mode, "--seed", str(seed),
            "--threads", "1", "--quiet", *extra]


def correction_recovery(model, pdir, session_pos=-1):
    """Pearson corr of the learned logit correction (g + u.v) with the planted delta over the prior support edges
    that are not sign-flipped, on one session (the last by default)."""
    truth = np.load(os.path.join(pdir, "truth.npz"))
    p = itd.load_intraday(pdir)
    fb = itd.FeatureBuilder(p)
    sid, a, b = p.sessions()[session_pos]
    bt = pm.prepare(fb.session_batch(sid, np.arange(a, a + 1)))
    if len(bt.e_row) == 0:
        return float("nan")
    with torch.no_grad():
        tr, tc = bt.node_col_t[bt.e_row], bt.node_col_t[bt.e_col]
        corr = (model.g(bt.e_feat)[:, 0] + (model.U[tr] * model.V[tc]).sum(1)).numpy()
    s, d = tr.numpy(), tc.numpy()
    keep = ~truth["flip"][s, d]
    x, y = corr[keep], truth["delta"][s, d][keep]
    if x.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def oracle_ic(p, signal, preds):
    bar = {int(t): i for i, t in enumerate(p.times)}
    col = {tk: i for i, tk in enumerate(p.tickers)}
    lm = itd.label_mask(p.session, p.horizon)
    by = {}
    for t, tk in zip(preds["t"], preds["ticker"]):
        by.setdefault(bar[int(t)], []).append(col[tk])
    ics = []
    for d, idx in by.items():
        if not lm[d]:
            continue
        s, y = signal[d, idx].astype(float), np.asarray(p.a[itd.LABEL][d, idx], float)
        ok = np.isfinite(s) & np.isfinite(y)
        if ok.sum() >= 3 and np.ptp(s[ok]) > 0:
            ics.append(pm.spearman(s[ok], y[ok]))
    return float(np.mean(ics)) if ics else 0.0


def run_point(root, stage, cfg_name, cfg, m, mode, seed):
    pdir = make_market(root, m)
    t0 = time.time()
    base = os.path.join(root, "baselines", f"{market_key(m)}__{mode}__s{seed}")
    if not os.path.exists(os.path.join(base, "out", "run.json")):
        _run(wf_cmd(pdir, base, "Bprior,B0E,B0", mode, seed))
    work = os.path.join(root, stage, f"{cfg_name}__{market_key(m)}__{mode}__s{seed}")
    _run(wf_cmd(pdir, work, "learned", mode, seed, ["--fresh"] + config_args(cfg)))
    p = itd.load_intraday(pdir)
    preds = {v: wf_m5.Store(os.path.join(base, "store", v)).all_predictions() for v in ("Bprior", "B0E", "B0")}
    preds["learned"] = wf_m5.Store(os.path.join(work, "store", "learned")).all_predictions()
    ic, _ = wf_m5.ic_summary(p, preds)
    with open(os.path.join(pdir, "synth.json")) as f:
        info = json.load(f)
    model, ck = wf_m5.load_model(os.path.join(work, "store", "learned"))
    hist = ck["history"]
    return {"stage": stage, "config": cfg_name, "params": {**cfg, "n_params": n_learned_params(cfg, m["n"])},
            "market": m, "mode": mode, "seed": seed, "synth": info,
            "oracle_ic": oracle_ic(p, si.load_signal(pdir, p.T, p.N), preds["learned"]),
            "ic": {v: ic[v]["mean"] for v in preds},
            "diff": {b: ic["learned"]["minus_" + b] for b in ("Bprior", "B0E", "B0")},
            "t": {b: ic["learned"]["minus_" + b]["t"] for b in ("Bprior", "B0E", "B0")},
            "Bprior_minus_B0E": ic["Bprior"]["minus_B0E"],
            "recovery": correction_recovery(model, pdir) if m["kind"] == "planted" else None,
            "first_best_epoch": hist[0]["best_epoch"], "seconds": round(time.time() - t0, 1)}


def points(stage, frozen=None):
    if stage == "tune":
        for name, cfg in CONFIGS.items():
            for ms in TUNE_MARKET_SEEDS:
                for mode in MODES:
                    for seed in MODEL_SEEDS:
                        for ic in TUNE_ICS:
                            yield name, cfg, {"kind": "planted", "target_ic": ic, "seed": ms, "n": 300}, mode, seed
                    yield name, cfg, {"kind": "null_a", "target_ic": 0.0, "seed": ms, "n": 300}, mode, MODEL_SEEDS[0]
                    yield name, cfg, {"kind": "null_b", "target_ic": NULL_B_IC, "seed": ms, "n": 300}, mode, \
                        MODEL_SEEDS[0]
    elif stage == "power":
        name, cfg = frozen
        for ms in EVAL_MARKET_SEEDS:
            for mode in MODES:
                for seed in MODEL_SEEDS:
                    for ic in EVAL_ICS:
                        yield name, cfg, {"kind": "planted", "target_ic": ic, "seed": ms, "n": 300}, mode, seed
                    yield name, cfg, {"kind": "null_a", "target_ic": 0.0, "seed": ms, "n": 300}, mode, seed
                    yield name, cfg, {"kind": "null_b", "target_ic": NULL_B_IC, "seed": ms, "n": 300}, mode, seed
    elif stage == "n1000":
        name, cfg = frozen
        for mode in MODES:
            yield name, cfg, {"kind": "planted", "target_ic": 0.05, "seed": 1, "n": 1000}, mode, MODEL_SEEDS[0]
    else:
        raise ValueError(stage)


def point_id(name, m, mode, seed):
    return f"{name}__{market_key(m)}__{mode}__s{seed}"


def load_results(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def detected(r):
    return r["t"]["Bprior"] > 2 and r["t"]["B0E"] > 2


def null_clean(r):
    if r["market"]["kind"] == "null_a":
        return abs(r["t"]["Bprior"]) < 2 and abs(r["t"]["B0E"]) < 2
    return abs(r["t"]["Bprior"]) < 2


def select(results):
    """The pre-registered rule (module doc / POWER_M5.md). Returns (config or None, table)."""
    by = {}
    for r in results:
        by.setdefault(r["config"], []).append(r)
    table = []
    for name, rs in by.items():
        planted = [r for r in rs if r["market"]["kind"] == "planted"]
        nulls = [r for r in rs if r["market"]["kind"] != "planted"]
        det = {ic: sum(detected(r) for r in planted if abs(r["market"]["target_ic"] - ic) < 1e-9) for ic in TUNE_ICS}
        n05 = sum(abs(r["market"]["target_ic"] - 0.05) < 1e-9 for r in planted)
        expected_nulls = len(TUNE_MARKET_SEEDS) * len(MODES) * 2
        complete = n05 == len(TUNE_MARKET_SEEDS) * len(MODES) * len(MODEL_SEEDS) and len(nulls) == expected_nulls
        nulls_ok = all(null_clean(r) for r in nulls)
        table.append({"config": name, "detected": det, "n_planted_05": n05, "n_nulls": len(nulls),
                      "nulls_ok": nulls_ok, "complete": complete,
                      "passes": bool(complete and nulls_ok and det[0.05] >= 10),
                      "n_params": rs[0]["params"].get("n_params", 0)})
    ok = [r for r in table if r["passes"]]
    best = max(ok, key=lambda r: (r["detected"][0.02], r["detected"][0.03], r["detected"][0.05],
                                  -r["n_params"]))["config"] if ok else None
    return best, sorted(table, key=lambda r: (-r["passes"], -r["detected"][0.05]))


def fmt(d):
    return f"{d['mean']:+.4f} ({d['t']:+.1f})"


def write_table(results, path):
    lines = ["| config | kind | N | target IC | mkt seed | mode | model seed | oracle IC | prior-graph IC | "
             "learned IC | Bprior IC | learned-Bprior (t) | learned-B0E (t) | Bprior-B0E (t) | recovery | det |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r["market"]
        rec = f"{r['recovery']:+.2f}" if r.get("recovery") is not None else ""
        pg = r["synth"].get("prior_graph_ic")
        pg = f"{pg:.3f}" if pg is not None and np.isfinite(pg) else ""
        lines.append(f"| {r['config']} | {m['kind']} | {m['n']} | {m['target_ic']:g} | {m['seed']} | {r['mode']} | "
                     f"{r['seed']} | {r['oracle_ic']:.3f} | {pg} | {r['ic']['learned']:+.4f} | "
                     f"{r['ic']['Bprior']:+.4f} | {fmt(r['diff']['Bprior'])} | {fmt(r['diff']['B0E'])} | "
                     f"{fmt(r['Bprior_minus_B0E'])} | {rec} | {'Y' if detected(r) else ''} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("tune", "power", "n1000"), required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--results", default=os.path.join(HERE, "power_m5_results"),
                    help="where the jsonl/md/selection files go (small; committed)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.results, exist_ok=True)
    frozen = None
    if a.stage != "tune":
        with open(os.path.join(a.results, "selection_tune.json")) as f:
            sel = json.load(f)
        if sel["best"] is None:
            raise SystemExit("no configuration passed the pre-registered rule: nothing to freeze")
        frozen = (sel["best"], CONFIGS[sel["best"]])
    res_path = os.path.join(a.results, f"{a.stage}.jsonl")
    done = {point_id(r["config"], r["market"], r["mode"], r["seed"]) for r in load_results(res_path)}
    todo = [pt for pt in points(a.stage, frozen) if point_id(pt[0], pt[2], pt[3], pt[4]) not in done]
    markets = {market_key(pt[2]): pt[2] for pt in todo}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:  # markets first (numpy releases the GIL)
        list(ex.map(lambda m: make_market(a.out, m), markets.values()))
    print(f"{len(todo)} points to run ({len(done)} done)", flush=True)
    # baselines are shared: order so that the points sharing a baseline do not race on it
    order = sorted(todo, key=lambda pt: (pt[0] != list(CONFIGS)[0] if a.stage == "tune" else 0))
    first_cfg = [pt for pt in order if pt[0] == order[0][0]] if order else []
    rest = [pt for pt in order if pt not in first_cfg]

    def job(pt):
        return run_point(a.out, a.stage, *pt)

    for batch in (first_cfg, rest):
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(job, batch):
                with open(res_path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"{point_id(r['config'], r['market'], r['mode'], r['seed'])}: oracle {r['oracle_ic']:.3f} "
                      f"-Bprior {fmt(r['diff']['Bprior'])} -B0E {fmt(r['diff']['B0E'])} {r['seconds']}s",
                      flush=True)
    results = load_results(res_path)
    write_table(results, os.path.join(a.results, f"{a.stage}.md"))
    if a.stage == "tune":
        best, table = select(results)
        with open(os.path.join(a.results, "selection_tune.json"), "w") as f:
            json.dump({"best": best, "table": table}, f, indent=1, default=str)
        print("selected:", best)


if __name__ == "__main__":
    main()
