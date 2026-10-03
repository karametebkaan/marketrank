#!/usr/bin/env python3
"""Walk-forward trainer for the learned-connectivity model and its baselines.

  train_wf.py --panel DIR --out DIR2 --variants learned,B0,B2 [--b1-edges F] [--retrain-every 13] [--embargo 1]
              [--max-nodes 3000] [--seed 0] [--mode rolling|scratch] [--window 0] [--finetune-epochs 5]
              [--store DIR] [--fresh]

Calendar: the first prediction date is the first rebalance with >= --min-history (156) earlier rebalances. Every
--retrain-every rebalances the model is retrained at the cutoff = the first date of the block, on the rebalance
dates d_k with d_k + 1 + h + h*embargo <= cutoff (label_w = open[t+1+h]/open[t+1]-1 fully realized, plus the
embargo; h = 5 from the panel meta), optionally only
the --window most recent of them; it then predicts the block's dates.

Modes:
  rolling (default; what would run live): the first retrain trains from scratch (early stopping on the last 20%
      of its dates); every later retrain warm-starts from the previous checkpoint (weights, Adam state, embeddings)
      and fine-tunes for --finetune-epochs on all of its window (the last 20% is reported as an in-sample loss).
  scratch: every retrain trains a fresh model with early stopping (for comparison).

Persistence: after each retrain the variant's store (--store/<variant>/) receives checkpoint_<t>.pt,
graph_<t>.parquet (learned only), predictions_<t>.parquet and then latest.json, all written atomically. A rerun
with the same params resumes from latest.json: it only predicts the dates after the stored ones and retrains only
for later blocks. Different params refuse to resume unless --fresh (which discards the variant's store).

Outputs in --out: <variant>.csv (t,ticker,score for every stored prediction), learned_edges_<k>.csv (src,dst,w:
the learned top-k edges at each retrain k), run.json. Prints each variant's out-of-sample mean IC and t-stat.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model as mdl  # noqa: E402
import panel as pnl  # noqa: E402
from store import Store  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_STORE = os.path.join(REPO, "data", "graphlearn")
HORIZON = 5  # default label horizon; the panel's meta label_horizons.label_w wins


# ---------------------------------------------------------------- calendar

def plan_retrains(rebalance, min_history=156, retrain_every=13, embargo=1, horizon=HORIZON):
    """Blocks of predicted rebalance positions with their usable training positions.

    label_w at bar d_k reads opens d_k+1 .. d_k+1+h, so it is realized at bar d_k+1+h; a training date is usable
    for the block whose first prediction date (the cutoff) is d_m iff d_k + 1 + h + embargo*h <= d_m. (One bar
    stricter than "d_k + h + embargo*h <= d_m", so that --embargo 0 is leak-free too.)"""
    reb = np.asarray(rebalance)
    blocks = []
    start = min_history
    while start < len(reb):
        pred = list(range(start, min(start + retrain_every, len(reb))))
        cutoff = reb[start]
        train = [k for k in range(start) if reb[k] + 1 + horizon + embargo * horizon <= cutoff]
        blocks.append({"block": len(blocks), "pred": pred, "train": train})
        start += retrain_every
    return blocks


# ---------------------------------------------------------------- args / params

def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="learned,B0,B2")
    ap.add_argument("--b1-edges", default=None, help="CSV t,src,dst,w (fixed graph for B1; B1 skipped if absent)")
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--embargo", type=int, default=1)
    ap.add_argument("--max-nodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("rolling", "scratch"), default="rolling")
    ap.add_argument("--window", type=int, default=0, help="train on the N most recent usable dates (0 = expanding)")
    ap.add_argument("--finetune-epochs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50, help="max epochs of a from-scratch training")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-history", type=int, default=156)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--emb-lr", type=float, default=1e-2, help="Adam lr of the ticker embeddings")
    ap.add_argument("--lambda-ic", type=float, default=1.0)
    ap.add_argument("--l1", type=float, default=1e-4)
    ap.add_argument("--store", default=DEFAULT_STORE, help="store root; each variant uses <store>/<variant>/")
    ap.add_argument("--fresh", action="store_true", help="discard the variant stores and start over")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=1, help="variants trained in parallel processes")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def params_for(args, variant, horizon=HORIZON):
    """Everything that shapes the model trajectory (not threads/paths/verbosity)."""
    p = {k: getattr(args, k) for k in ("mode", "seed", "rank", "topk", "dropout", "lr", "emb_lr", "weight_decay",
                                        "lambda_ic", "l1", "epochs", "patience", "finetune_epochs", "window",
                                        "retrain_every", "embargo", "min_history", "max_nodes", "val_frac")}
    p["variant"] = variant
    p["horizon"] = horizon
    if variant == "B1":
        p["b1_edges"] = file_digest(args.b1_edges)
    return p


def params_hash(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def git_sha():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def setup_torch(threads):
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)


# ---------------------------------------------------------------- B1 fixed graph

class B1Lookup:
    """As-of lookup of a fixed sparse graph from CSV t,src,dst,w: the latest snapshot with t <= the date.
    Row src aggregates from its dst columns; |w| is row-normalized over the date's node set."""

    def __init__(self, path):
        snaps = {}
        with open(path) as f:
            for r in csv.DictReader(f):
                snaps.setdefault(int(float(r["t"])), []).append((r["src"], r["dst"], abs(float(r["w"]))))
        self.times = sorted(snaps)
        self.snaps = snaps

    def __call__(self, t, tickers):
        i = np.searchsorted(self.times, t, side="right") - 1
        if i < 0:
            return None
        pos = {tk: j for j, tk in enumerate(tickers)}
        e = [(pos[s], pos[d], w) for s, d, w in self.snaps[self.times[i]]
             if s in pos and d in pos and s != d and w > 0]
        if not e:
            return None
        rows = np.array([x[0] for x in e], dtype=np.int64)
        cols = np.array([x[1] for x in e], dtype=np.int64)
        w = np.array([x[2] for x in e], dtype=np.float64)
        w /= np.bincount(rows, weights=w, minlength=len(tickers))[rows]
        return torch.from_numpy(rows), torch.from_numpy(cols), torch.from_numpy(w.astype(np.float32))


# ---------------------------------------------------------------- one variant

class ResumeRefused(RuntimeError):
    pass


def _new_model(args, variant):
    return mdl.GraphNet(variant, rank=args.rank, topk=args.topk, dropout=args.dropout, seed=args.seed)


def restore(args, variant, ck):
    m = _new_model(args, variant)
    m.ensure_tickers(ck["tickers"])
    m.load_state_dict(ck["model_state"])
    opt = mdl.make_optimizer(m, args.lr, args.weight_decay, args.emb_lr)
    opt.load_state_dict(ck["opt_state"])
    m.eval()
    return m, opt


def load_model_from_store(variant_store, args=None):
    """The latest checkpointed model of a variant store (for inspection / tests)."""
    st = Store(variant_store)
    info = st.latest()
    ck = st.load_checkpoint(info["path"])
    p = ck["params"]
    ns = argparse.Namespace(rank=p["rank"], topk=p["topk"], dropout=p["dropout"], seed=p["seed"], lr=p["lr"],
                            weight_decay=p["weight_decay"], emb_lr=p["emb_lr"])
    return restore(ns, p["variant"], ck)[0], ck


def graph_columns(model, b):
    idx, w, _ = model.adjacency(model.rows_for(b.tickers))
    n, k = idx.shape
    idx, w = idx.numpy(), w.detach().numpy()
    return {"t": [b.t] * (n * k),
            "src_ticker": [b.tickers[i] for i in range(n) for _ in range(k)],
            "dst_ticker": [b.tickers[j] for j in idx.reshape(-1)],
            "weight": w.reshape(-1).astype(np.float64).tolist(),
            "rank_in_row": [r + 1 for _ in range(n) for r in range(k)]}


def run_variant(p, variant, args, log=print):
    store = Store(os.path.join(args.store, variant))
    if args.fresh:
        store.clear()
    params = params_for(args, variant, p.horizon)
    ph = params_hash(params)
    latest = store.latest()
    if latest is not None and latest.get("params_hash") != ph:
        raise ResumeRefused(
            f"{store.root}: the stored model was trained with params hash {latest.get('params_hash')} "
            f"({json.dumps(latest.get('params'), sort_keys=True)}); this run has {ph} "
            f"({json.dumps(params, sort_keys=True)}). Refusing to resume -- rerun with --fresh to discard the "
            f"store, or point --store elsewhere.")
    sectors = sorted(set(p.sectors))
    sector_codes = np.array([sectors.index(s) for s in p.sectors])
    b1 = B1Lookup(args.b1_edges) if variant == "B1" else None
    reb = p.rebalance
    cache = {}

    def batch(pos, train):
        key = (pos, train)
        if key not in cache:
            cache[key] = mdl.make_batch(p, int(reb[pos]), args.max_nodes, train, sector_codes, b1)
        return cache[key]

    def predict_rows(positions):
        cols = {"t": [], "ticker": [], "score": []}
        for m in positions:
            b = batch(m, False)
            if len(b.nodes) == 0:
                continue
            model.ensure_tickers(b.tickers, opt)  # predicting a brand-new ticker: fresh embedding row
            s = mdl.predict(model, b)
            cols["t"].extend([b.t] * len(b.nodes))
            cols["ticker"].extend(b.tickers)
            cols["score"].extend(s.astype(np.float64).tolist())
        return cols

    blocks = plan_retrains(reb, args.min_history, args.retrain_every, args.embargo, p.horizon)
    model = opt = None
    history, resume_block, retrained = [], -1, []
    if latest is not None:
        ck = store.load_checkpoint(latest["path"])
        model, opt = restore(args, variant, ck)
        history, resume_block = ck["history"], ck["block"]
        if resume_block >= len(blocks) or int(p.times[reb[blocks[resume_block]["pred"][0]]]) != ck["t"]:
            raise ResumeRefused(f"{store.root}: checkpoint block {resume_block} (cutoff {ck['t']}) does not match "
                                f"this panel's rebalance calendar; rerun with --fresh")
        log(f"[{variant}] resuming from {latest['path']} (block {resume_block}, cutoff {ck['t']})")

    for b in blocks:
        j = b["block"]
        if j < resume_block:
            continue
        cutoff = int(p.times[reb[b["pred"][0]]])
        if j == resume_block:  # predict this block's dates that were not available when it was stored
            prev = store.load_predictions(cutoff) or {"t": [], "ticker": [], "score": []}
            done = set(prev["t"])
            todo = [m for m in b["pred"] if int(p.times[reb[m]]) not in done]
            if todo:
                new = predict_rows(todo)
                for k in prev:
                    prev[k].extend(new[k])
                store.save_predictions(cutoff, prev)
                latest["last_pred_t"] = int(p.times[reb[todo[-1]]])
                store.write_latest(latest)
            continue

        t0 = time.time()
        usable = b["train"][-args.window:] if args.window > 0 else b["train"]
        train = [x for x in (batch(k, True) for k in usable) if len(x.nodes) >= 2]
        if not train:
            raise RuntimeError(f"block {j}: no usable training dates")
        n_val = int(round(args.val_frac * len(train))) if len(train) >= 5 else 0
        val = train[len(train) - n_val:] if n_val else []
        bseed = args.seed * 100_003 + j
        warm = args.mode == "rolling" and model is not None
        if warm:
            model.ensure_tickers(p.tickers, opt)
            torch.manual_seed(bseed)
            hist = mdl.fit(model, opt, train, val, args.finetune_epochs, args.lambda_ic, args.l1,
                           patience=None, seed=bseed)
        else:
            torch.manual_seed(bseed)
            model = _new_model(args, variant)
            model.ensure_tickers(p.tickers)
            opt = mdl.make_optimizer(model, args.lr, args.weight_decay, args.emb_lr)
            torch.manual_seed(bseed)
            hist = mdl.fit(model, opt, train[:len(train) - n_val], val, args.epochs, args.lambda_ic, args.l1,
                           patience=args.patience, restore_best=True, seed=bseed)
        preds = predict_rows(b["pred"])
        entry = {"block": j, "cutoff_t": cutoff, "kind": "finetune" if warm else "scratch",
                 "n_train": len(train) if warm else len(train) - n_val, "n_val": n_val,
                 "val_in_sample": bool(warm), "first_train_t": train[0].t, "last_train_t": train[-1].t,
                 "train_loss": hist["train_loss"], "val_loss": hist["val_loss"],
                 "best_epoch": hist["best_epoch"], "seconds": round(time.time() - t0, 3)}
        history = history + [entry]
        retrained.append(j)
        ck_name = store.save_checkpoint(cutoff, {
            "model_state": model.state_dict(), "opt_state": opt.state_dict(), "tickers": list(model.tickers),
            "params": params, "params_hash": ph, "t": cutoff, "block": j, "history": history,
            "torch": torch.__version__})
        graph_name = None
        if variant == "learned":
            with torch.no_grad():
                graph_name = store.save_graph(cutoff, graph_columns(model, batch(b["pred"][0], False)))
        pred_name = store.save_predictions(cutoff, preds)
        latest = {"t": cutoff, "path": ck_name, "graph": graph_name, "predictions": pred_name, "block": j,
                  "last_pred_t": int(p.times[reb[b["pred"][-1]]]), "git_sha": git_sha(), "params_hash": ph,
                  "params": params, "variant": variant}
        store.write_latest(latest)
        log(f"[{variant}] block {j} cutoff {cutoff} {entry['kind']} n_train={entry['n_train']} "
            f"epochs={len(hist['train_loss'])} train={hist['train_loss'][-1]:.4f} "
            f"val={hist['val_loss'][-1]:.4f} {entry['seconds']:.1f}s")

    return {"variant": variant, "params": params, "params_hash": ph, "store": store.root,
            "resumed_from_block": resume_block if resume_block >= 0 else None, "retrained_blocks": retrained,
            "history": history}


def write_outputs(args, variant):
    store = Store(os.path.join(args.store, variant))
    preds = store.all_predictions()
    path = os.path.join(args.out, f"{variant}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["t", "ticker", "score"])
        for t, tk, s in zip(preds["t"], preds["ticker"], preds["score"]):
            w.writerow([t, tk, f"{s:.9g}"])
    if variant == "learned":
        import pyarrow.parquet as pq
        for k, g in enumerate(store.graph_files()):
            tab = pq.read_table(g).to_pydict()
            with open(os.path.join(args.out, f"learned_edges_{k}.csv"), "w", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(["src", "dst", "w"])
                for s, d, x in zip(tab["src_ticker"], tab["dst_ticker"], tab["weight"]):
                    w.writerow([s, d, f"{x:.6g}"])
    return preds


# ---------------------------------------------------------------- quick IC look

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = mdl.avg_ranks(a[m]), mdl.avg_ranks(b[m])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def daily_ic(p, preds):
    """{t: Spearman(score, label_w)} per predicted date with known labels."""
    bar = {int(t): i for i, t in enumerate(p.times)}
    col = {tk: i for i, tk in enumerate(p.tickers)}
    by_t = {}
    for t, tk, s in zip(preds["t"], preds["ticker"], preds["score"]):
        by_t.setdefault(int(t), ([], []))
        by_t[int(t)][0].append(col[tk])
        by_t[int(t)][1].append(s)
    out = {}
    for t, (idx, s) in sorted(by_t.items()):
        ic = spearman(s, p.a["label_w"][bar[t], idx])
        if np.isfinite(ic):
            out[t] = ic
    return out


def mean_t(x):
    x = np.asarray(list(x), float)
    if len(x) < 2:
        return {"mean": float(np.mean(x)) if len(x) else float("nan"), "t": float("nan"), "n": len(x)}
    sd = x.std(ddof=1)
    return {"mean": float(x.mean()), "t": float(x.mean() / sd * math.sqrt(len(x))) if sd > 0 else float("nan"),
            "n": len(x)}


def ic_summary(p, all_preds):
    ics = {v: daily_ic(p, pr) for v, pr in all_preds.items()}
    out = {v: mean_t(ic.values()) for v, ic in ics.items()}
    if "B0" in ics:
        for v, ic in ics.items():
            if v != "B0":
                common = sorted(set(ic) & set(ics["B0"]))
                out[v]["minus_B0"] = mean_t(ic[t] - ics["B0"][t] for t in common)
    return out, ics


# ---------------------------------------------------------------- main

def _worker(argv, variant):
    args = parse_args(argv)
    setup_torch(args.threads)
    np.random.seed(args.seed)
    p = pnl.load_panel(args.panel)
    return run_variant(p, variant, args, log=(lambda *a: None) if args.quiet else print)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    log = (lambda *a: None) if args.quiet else print
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in mdl.VARIANTS:
            raise SystemExit(f"unknown variant {v}")
    if "B1" in variants and not args.b1_edges:
        log("B1 skipped: no --b1-edges given")
        variants.remove("B1")
    os.makedirs(args.out, exist_ok=True)
    setup_torch(args.threads)
    np.random.seed(args.seed)
    t0 = time.time()
    p = pnl.load_panel(args.panel)
    try:
        if args.jobs > 1 and len(variants) > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:
                results = list(ex.map(_worker, [argv] * len(variants), variants))
        else:
            results = [run_variant(p, v, args, log) for v in variants]
    except ResumeRefused as e:
        raise SystemExit(f"refusing to resume: {e}")
    all_preds = {r["variant"]: write_outputs(args, r["variant"]) for r in results}
    ic, _ = ic_summary(p, all_preds)
    for v in variants:
        s = ic[v]
        extra = ""
        if "minus_B0" in s:
            extra = f"  minus B0: {s['minus_B0']['mean']:+.4f} (t={s['minus_B0']['t']:.2f})"
        log(f"OOS IC {v:8s} mean={s['mean']:+.4f} t={s['t']:.2f} dates={s['n']}{extra}")
    run = {"args": vars(args), "variants": {r["variant"]: r for r in results}, "oos_ic": ic,
           "git_sha": git_sha(), "torch": torch.__version__, "seconds": round(time.time() - t0, 3),
           "panel_meta": {k: p.meta.get(k) for k in ("N", "T")}}
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(run, f, indent=1, sort_keys=True, default=str)
    return run


if __name__ == "__main__":
    main()
