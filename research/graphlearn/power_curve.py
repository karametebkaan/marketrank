#!/usr/bin/env python3
"""Power curve and hyper-parameter tuning of the learned graph on synthetic markets only.

  power_curve.py --out DIR --stage tune|power [--workers 3] [--threads 2]

Each point = one synthetic market (synth.py: N, T=1500, beta, seed, churn, null) x one setting x one mode, run
through train_wf.py exactly as deployed with the variants learned, B0, B0E. Reported per point:
  oracle_ic   mean per-date Spearman of the TRUE predictable signal vs the label (the ceiling any model can reach)
  learned / B0 / B0E mean OOS IC, learned-B0 and learned-B0E paired IC differences with t
  edge_auc    AUC of the final learned S for true vs false edges (rows that have true edges)
Results are appended to DIR/<stage>.jsonl (a rerun skips finished points), the table to DIR/<stage>.md.

Stages:
  tune   one-at-a-time variations around the starting setting (TUNE_SETTINGS) on the tuning markets
         (beta 0.2/0.3/0.4 and the null market, seeds 101/102 -- disjoint from the reported seeds), rolling mode.
  power  the frozen setting (frozen_settings.json) on beta 0.1..0.8 x seeds 1/2 x rolling/scratch, the null
         market, and one N=1000 point with eligibility churn.
Selection rule (fixed before the grid was run; see POWER.md) is applied by `select_setting`.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model as mdl  # noqa: E402
import panel as pnl  # noqa: E402
import synth  # noqa: E402
import train_wf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
START = {"rank": 8, "topk": 20, "emb_lr": 1e-3, "emb_l2": 1e-3, "l1": 1e-2, "finetune_epochs": 2, "ft_holdout": 13,
         "window": 156, "patience": 10, "epochs": 50, "embargo": 1, "retrain_every": 13}
VARIATIONS = [("emb_lr", 1e-2), ("emb_l2", 1e-4), ("emb_l2", 1e-2), ("rank", 4), ("rank", 16), ("l1", 1e-3),
              ("l1", 1e-1), ("finetune_epochs", 5)]
TUNE_SETTINGS = [("start", dict(START))] + [(f"{k}={v:g}", {**START, k: v}) for k, v in VARIATIONS]
TUNE_BETAS, TUNE_SEEDS = (0.2, 0.3, 0.4), (101, 102)
POWER_BETAS, POWER_SEEDS = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8), (1, 2)
FLAG = {k: "--" + k.replace("_", "-") for k in START}


def market_key(m):
    return (f"n{m['n']}_b{m['beta']:g}_s{m['seed']}" + ("_null" if m.get("null") else "")
            + (f"_churn{m['churn']:g}" if m.get("churn") else ""))


def make_market(root, m):
    d = os.path.join(root, "markets", market_key(m))
    if not os.path.exists(os.path.join(d, "meta.json")):
        info = synth.make_planted_panel(d, n=m["n"], t=1500, beta=m["beta"], seed=m["seed"],
                                        null=m.get("null", False), churn=m.get("churn", 0.0))
        np.savez(os.path.join(d, "truth.npz"), W=info["W"], targets=info["targets"])
    return d


def edge_auc(model, W, targets, tickers):
    S = model.scores(model.rows_for(tickers)).detach().numpy()[targets]
    truth = W[targets] > 0
    if not truth.any():
        return float("nan")
    off = np.ones_like(truth)
    off[np.arange(len(targets)), targets] = False
    pos, neg = S[truth], S[~truth & off]
    r = mdl.avg_ranks(np.r_[pos, neg])
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def oracle_ic(p, signal, preds):
    """Mean per-date Spearman(true signal, label) over the predicted dates' node sets."""
    bar = {int(t): i for i, t in enumerate(p.times)}
    col = {tk: i for i, tk in enumerate(p.tickers)}
    by_t = {}
    for t, tk in zip(preds["t"], preds["ticker"]):
        by_t.setdefault(int(t), []).append(col[tk])
    ics = []
    for t, idx in by_t.items():
        s, y = signal[bar[t], idx], p.a["label_w"][bar[t], idx]
        ok = np.isfinite(s) & np.isfinite(y)
        if ok.sum() >= 3 and np.ptp(s[ok]) > 0:
            ics.append(mdl.spearman_np(s[ok].astype(float), y[ok].astype(float)))
    return float(np.mean(ics)) if ics else 0.0


def run_point(root, stage, name, setting, m, mode, threads):
    pdir = make_market(root, m)
    tag = f"{name}__{market_key(m)}__{mode}"
    work = os.path.join(root, stage, tag)
    cmd = [sys.executable, os.path.join(HERE, "train_wf.py"), "--panel", pdir, "--out", os.path.join(work, "out"),
           "--store", os.path.join(work, "store"), "--variants", "learned,B0,B0E", "--mode", mode, "--fresh",
           "--jobs", "3", "--threads", str(threads), "--quiet"]
    for k, v in setting.items():
        cmd += [FLAG[k], str(v)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{tag} failed:\n{r.stdout}\n{r.stderr}")
    with open(os.path.join(work, "out", "run.json")) as f:
        run = json.load(f)
    p = pnl.load_panel(pdir)
    truth = np.load(os.path.join(pdir, "truth.npz"))
    model, _ = train_wf.load_model_from_store(os.path.join(work, "store", "learned"))
    preds = train_wf.Store(os.path.join(work, "store", "B0")).all_predictions()
    ic = run["oos_ic"]
    hist = run["variants"]["learned"]["history"]
    return {"stage": stage, "setting": name, "params": setting, "market": m, "mode": mode,
            "oracle_ic": oracle_ic(p, synth.load_signal(pdir, p.T, p.N), preds),
            "ic": {v: ic[v]["mean"] for v in ("learned", "B0", "B0E")},
            "minus_B0": ic["learned"]["minus_B0"], "minus_B0E": ic["learned"]["minus_B0E"],
            "B0E_minus_B0": ic["B0E"]["minus_B0"],
            "edge_auc": edge_auc(model, truth["W"], truth["targets"], p.tickers),
            "first_best_epoch": hist[0]["best_epoch"], "n_dates": ic["learned"]["n"],
            "seconds": round(time.time() - t0, 1)}


def points(stage, settings_override=None):
    if stage == "tune":
        for name, s in TUNE_SETTINGS:
            for seed in TUNE_SEEDS:
                for beta in TUNE_BETAS:
                    yield name, s, {"n": 300, "beta": beta, "seed": seed}, "rolling"
                yield name, s, {"n": 300, "beta": 0.8, "seed": seed, "null": True}, "rolling"
    elif stage == "power":
        s = settings_override or train_wf.frozen_settings()
        for mode in ("rolling", "scratch"):
            for seed in POWER_SEEDS:
                for beta in POWER_BETAS:
                    yield "frozen", s, {"n": 300, "beta": beta, "seed": seed}, mode
                yield "frozen", s, {"n": 300, "beta": 0.8, "seed": seed, "null": True}, mode
            yield "frozen", s, {"n": 1000, "beta": 0.4, "seed": 1, "churn": 0.3}, mode
    else:
        raise ValueError(stage)


def point_id(name, m, mode):
    return f"{name}__{market_key(m)}__{mode}"


def load_results(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def select_setting(results):
    """The pre-registered rule (POWER.md). Admissible: on both null markets |t| < 2 for learned-B0 and
    learned-B0E. Score: number of tuning points (beta 0.2/0.3/0.4 x 2 seeds) with learned-B0 t > 2 AND
    learned-B0E t > 2; ties broken by the mean over those points of min(t_B0, t_B0E). Returns (name, table)."""
    by = {}
    for r in results:
        by.setdefault(r["setting"], []).append(r)
    table = []
    for name, rs in by.items():
        null = [r for r in rs if r["market"].get("null")]
        live = [r for r in rs if not r["market"].get("null")]
        admissible = len(null) == len(TUNE_SEEDS) and all(
            abs(r["minus_B0"]["t"]) < 2 and abs(r["minus_B0E"]["t"]) < 2 for r in null)
        score = sum(r["minus_B0"]["t"] > 2 and r["minus_B0E"]["t"] > 2 for r in live)
        tie = float(np.mean([min(r["minus_B0"]["t"], r["minus_B0E"]["t"]) for r in live])) if live else -np.inf
        table.append({"setting": name, "admissible": admissible, "score": score, "tie": tie,
                      "complete": len(live) == len(TUNE_BETAS) * len(TUNE_SEEDS)})
    ok = [r for r in table if r["admissible"] and r["complete"]]
    best = max(ok, key=lambda r: (r["score"], r["tie"]))["setting"] if ok else None
    return best, sorted(table, key=lambda r: (-r["admissible"], -r["score"], -r["tie"]))


def fmt_t(d):
    return f"{d['mean']:+.3f} ({d['t']:+.1f})"


def write_table(results, path):
    lines = ["| setting | N | beta | seed | null/churn | mode | oracle IC | learned IC | learned-B0 (t) | "
             "learned-B0E (t) | B0E-B0 (t) | edge AUC | 1st best epoch |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r["market"]
        tag = "null" if m.get("null") else (f"churn {m['churn']:g}" if m.get("churn") else "")
        lines.append(f"| {r['setting']} | {m['n']} | {m['beta']:g} | {m['seed']} | {tag} | {r['mode']} | "
                     f"{r['oracle_ic']:.3f} | {r['ic']['learned']:+.3f} | {fmt_t(r['minus_B0'])} | "
                     f"{fmt_t(r['minus_B0E'])} | {fmt_t(r['B0E_minus_B0'])} | {r['edge_auc']:.3f} | "
                     f"{r['first_best_epoch']} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("tune", "power"), required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, f"{a.stage}.jsonl")
    done = {point_id(r["setting"], r["market"], r["mode"]) for r in load_results(res_path)}
    todo = [pt for pt in points(a.stage) if point_id(pt[0], pt[2], pt[3]) not in done]
    for pt in todo:  # build the markets up front (sequentially)
        make_market(a.out, pt[2])
    print(f"{len(todo)} points to run ({len(done)} done)", flush=True)

    def job(pt):
        return run_point(a.out, a.stage, pt[0], pt[1], pt[2], pt[3], a.threads)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(job, todo):
            with open(res_path, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{point_id(r['setting'], r['market'], r['mode'])}: oracle {r['oracle_ic']:.3f} "
                  f"-B0 {fmt_t(r['minus_B0'])} -B0E {fmt_t(r['minus_B0E'])} auc {r['edge_auc']:.3f} "
                  f"{r['seconds']}s", flush=True)
    results = load_results(res_path)
    write_table(results, os.path.join(a.out, f"{a.stage}.md"))
    if a.stage == "tune":
        best, table = select_setting(results)
        with open(os.path.join(a.out, "selection.json"), "w") as f:
            json.dump({"best": best, "table": table}, f, indent=1)
        print("selected:", best)


if __name__ == "__main__":
    main()
