#!/usr/bin/env python3
"""Walk-forward trainer for the learned-connectivity model and its baselines.

  train_wf.py --panel DIR --out DIR2 --variants learned,B0,B0E,B2 [--b1-edges F] [--retrain-every 13]
              [--embargo 1] [--max-nodes 3000] [--seed 0] [--mode rolling|scratch] [--store DIR] [--fresh]

Defaults are the frozen settings chosen on synthetic markets (frozen_settings.json, POWER.md).

Calendar: the first prediction date is the first rebalance with >= --min-history (156) earlier rebalances. Every
--retrain-every rebalances the model is retrained at the cutoff = the first date of the block, on the rebalance
dates d_k with d_k + 1 + h + h*embargo <= cutoff (label_w = open[t+1+h]/open[t+1]-1 fully realized, plus the
embargo; h = 5 from the panel meta), only the --window most recent of them (0 = expanding); it then predicts the
block's dates.

Modes:
  rolling (default; what would run live): the first retrain trains from scratch (early stopping on the last
      --val-frac of its dates). Every later retrain first scores the previous model on the dates that became usable
      since the previous retrain (an honest out-of-sample loss and IC, logged as "oos" in run.json), then
      warm-starts from it (weights, Adam state, embeddings) and fine-tunes for at most --finetune-epochs, with
      early stopping on a held-out tail of the last --ft-holdout usable dates (the pre-fine-tune model counts as
      a candidate, so a fine-tune that does not help the held-out loss is discarded).
  scratch: every retrain trains a fresh model with early stopping (for comparison; also logs "oos").
  Use separate --store and --out roots for scratch and rolling runs (the mode is in the params hash, so a store
  only ever holds one of them).

Persistence: after each retrain the variant's store (--store/<variant>/) receives checkpoint_<t>.pt,
graph_<t>.parquet (learned only), predictions_<t>.parquet and then latest.json, all written atomically. A rerun
with the same params resumes from latest.json: it only predicts the dates after the stored ones and retrains only
for later blocks. The params hash covers every hyper-parameter, the thread count, the source of model.py and
train_wf.py and the panel's descriptive meta (features, eligibility, walk-forward params, label horizon); the
panel's times up to the stored cutoff must also match (the export may grow: N and T are not hashed). Any mismatch
refuses to resume unless --fresh (which discards the variant's store). A different git SHA only warns.

Outputs in --out: <variant>.csv (t,ticker,score: raw predictions for every stored date), learned_edges_<k>.csv
(src,dst,w: the learned top-k edges at each retrain k), run.json. Prints each variant's out-of-sample mean IC and
t-stat, and the paired differences to B0 and B0E.
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

FROZEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_settings.json")


def frozen_settings():
    """The settings frozen on synthetic markets (POWER.md); they are the CLI defaults."""
    with open(FROZEN_PATH) as f:
        return json.load(f)["settings"]


def parse_args(argv):
    fz = frozen_settings()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="learned,B0,B0E,B2")
    ap.add_argument("--b1-edges", default=None, help="CSV t,src,dst,w (fixed graph for B1; B1 skipped if absent)")
    ap.add_argument("--retrain-every", type=int, default=fz["retrain_every"])
    ap.add_argument("--embargo", type=int, default=fz["embargo"])
    ap.add_argument("--max-nodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("rolling", "scratch"), default="rolling")
    ap.add_argument("--window", type=int, default=fz["window"],
                    help="train on the N most recent usable dates (0 = expanding)")
    ap.add_argument("--finetune-epochs", type=int, default=fz["finetune_epochs"])
    ap.add_argument("--ft-holdout", type=int, default=fz["ft_holdout"],
                    help="rolling fine-tunes early-stop on the last N usable dates (held out of the fine-tune)")
    ap.add_argument("--epochs", type=int, default=fz["epochs"], help="max epochs of a from-scratch training")
    ap.add_argument("--patience", type=int, default=fz["patience"])
    ap.add_argument("--stop-on", choices=("loss", "ic"), default=fz.get("stop_on", "loss"),
                    help="early-stopping metric on the validation dates (scratch fits and rolling fine-tunes)")
    ap.add_argument("--min-epochs", type=int, default=fz.get("min_epochs", 0),
                    help="never early-stop before this many epochs")
    ap.add_argument("--restarts", type=int, default=fz.get("restarts", 1),
                    help="independent inits per from-scratch fit; the best validation IC is kept")
    ap.add_argument("--signed-message", action=argparse.BooleanOptionalAction,
                    default=fz.get("signed_message", False),
                    help="signed edges: top-k by |S|, weights S/sum|S| (no relu/softplus)")
    ap.add_argument("--min-history", type=int, default=156)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--rank", type=int, default=fz["rank"])
    ap.add_argument("--topk", type=int, default=fz["topk"])
    ap.add_argument("--score-fn", choices=("relu", "softplus"), default=fz.get("score_fn", "relu"),
                    help="S = score_fn(E_s E_d^T)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--emb-lr", type=float, default=fz["emb_lr"], help="Adam lr of the ticker embeddings")
    ap.add_argument("--emb-l2", type=float, default=fz["emb_l2"],
                    help="L2 penalty on the embedding rows of each date's nodes")
    ap.add_argument("--lambda-ic", type=float, default=1.0)
    ap.add_argument("--l1", type=float, default=fz["l1"], help="penalty on mean(S) of each date's node set")
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


HASHED_ARGS = ("mode", "seed", "rank", "topk", "score_fn", "signed_message", "stop_on", "min_epochs", "restarts", "dropout", "lr", "emb_lr", "emb_l2", "weight_decay",
               "lambda_ic", "l1", "epochs", "patience", "finetune_epochs", "ft_holdout", "window", "retrain_every", "embargo",
               "min_history", "max_nodes", "val_frac", "threads")
PANEL_META_KEYS = ("format", "feature_names", "eligibility", "wf_params", "wf_params_hash", "label_horizons")
_HERE = os.path.dirname(os.path.abspath(__file__))


def source_digest():
    """Hash of the code that defines the model trajectory."""
    h = hashlib.sha256()
    for name in ("model.py", "train_wf.py"):
        with open(os.path.join(_HERE, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def panel_digest(p):
    """Hash of the panel's descriptive meta (not N/T/times: the export grows; times are checked per cutoff)."""
    return hashlib.sha256(json.dumps({k: p.meta.get(k) for k in PANEL_META_KEYS}, sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]


def times_digest(p, upto_bar):
    return hashlib.sha256(np.ascontiguousarray(p.times[: upto_bar + 1], dtype="<i8").tobytes()).hexdigest()[:16]


def params_for(args, variant, p=None):
    """Everything that shapes the model trajectory (threads included: float reductions depend on them)."""
    out = {k: getattr(args, k) for k in HASHED_ARGS}
    out["variant"] = variant
    out["horizon"] = p.horizon if p is not None else HORIZON
    out["source"] = source_digest()
    out["panel_meta"] = panel_digest(p) if p is not None else None
    if variant == "B1":
        out["b1_edges"] = file_digest(args.b1_edges)
    return out


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


def _reg(args):
    return mdl.Reg(lambda_ic=args.lambda_ic, l1=args.l1, emb_l2=args.emb_l2)


def _new_model(args, variant):
    return mdl.GraphNet(variant, rank=args.rank, topk=args.topk, dropout=args.dropout, seed=args.seed,
                        score_fn=getattr(args, "score_fn", "relu"), signed=getattr(args, "signed_message", False))


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
    ns = argparse.Namespace(**{k: p[k] for k in ("rank", "topk", "dropout", "seed", "lr", "weight_decay", "emb_lr")},
                            score_fn=p.get("score_fn", "relu"), signed_message=p.get("signed_message", False))
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
    params = params_for(args, variant, p)
    ph = params_hash(params)
    reg = _reg(args)
    latest = store.latest()
    if latest is not None and latest.get("params_hash") != ph:
        old = latest.get("params") or {}
        diff = {k: (old.get(k), params.get(k)) for k in sorted(set(old) | set(params)) if old.get(k) != params.get(k)}
        raise ResumeRefused(
            f"{store.root}: the stored model was trained with params hash {latest.get('params_hash')}, this run has "
            f"{ph}; differences (stored, now): {json.dumps(diff, sort_keys=True, default=str)}. Refusing to "
            f"resume -- rerun with --fresh to discard the store, or point --store elsewhere.")
    sha = git_sha()
    if latest is not None and latest.get("git_sha") != sha:
        log(f"[{variant}] warning: resuming a store written at git {latest.get('git_sha')} from git {sha}")
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
            if not np.isfinite(s).any():
                continue  # the C++ harness counts any date with rows as scored: never write an all-NaN date
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
        if resume_block >= len(blocks) or int(p.times[reb[blocks[resume_block]["pred"][0]]]) != ck["t"] \
                or times_digest(p, int(reb[blocks[resume_block]["pred"][0]])) != ck.get("times_digest"):
            raise ResumeRefused(f"{store.root}: checkpoint block {resume_block} (cutoff {ck['t']}) does not match "
                                f"this panel's bar times / rebalance calendar up to the cutoff; rerun with --fresh")
        log(f"[{variant}] resuming from {latest['path']} (block {resume_block}, cutoff {ck['t']})")

    for b in blocks:
        j = b["block"]
        if j < resume_block:
            continue
        cutoff_bar = int(reb[b["pred"][0]])
        cutoff = int(p.times[cutoff_bar])
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
        bseed = args.seed * 100_003 + j
        oos = None
        if model is not None:
            # honest out-of-sample score of the previous model on the dates that became usable since it was fit
            prev_last = history[-1]["usable_until_t"] if history else None
            fresh_dates = [x for x in train if prev_last is None or x.t > prev_last]
            model.ensure_tickers(p.tickers, opt)
            oos = {"n_dates": len(fresh_dates), "loss": mdl.evaluate(model, fresh_dates, reg),
                   "ic": mdl.evaluate_ic(model, fresh_dates)}
        warm = args.mode == "rolling" and model is not None
        if warm:
            n_val = min(args.ft_holdout, max(0, len(train) - 1))
            tr, val = train[:len(train) - n_val], train[len(train) - n_val:]
            torch.manual_seed(bseed)
            hist = mdl.fit(model, opt, tr, val, args.finetune_epochs, reg, patience=args.patience, seed=bseed,
                           keep_initial=True, stop_on=args.stop_on)
            restart_ics, chosen = None, None
        else:
            n_val = int(round(args.val_frac * len(train))) if len(train) >= 5 else 0
            tr, val = train[:len(train) - n_val], train[len(train) - n_val:]
            fits = []
            for r in range(max(1, args.restarts)):
                rseed = bseed + 1_000_003 * r
                torch.manual_seed(rseed)
                m_r = _new_model(args, variant)
                m_r.ensure_tickers(p.tickers)
                o_r = mdl.make_optimizer(m_r, args.lr, args.weight_decay, args.emb_lr)
                torch.manual_seed(rseed)
                h_r = mdl.fit(m_r, o_r, tr, val, args.epochs, reg, patience=args.patience, seed=rseed,
                              stop_on=args.stop_on, min_epochs=args.min_epochs)
                fits.append((m_r, o_r, h_r, mdl.evaluate_ic(m_r, val) if val else float("nan")))
            restart_ics = [f[3] for f in fits]
            chosen = int(np.nanargmax(restart_ics)) if np.isfinite(restart_ics).any() else 0
            model, opt, hist = fits[chosen][:3]
            if args.restarts <= 1:
                restart_ics, chosen = [restart_ics[0]], 0
        preds = predict_rows(b["pred"])
        entry = {"block": j, "cutoff_t": cutoff, "kind": "finetune" if warm else "scratch",
                 "n_train": len(tr), "n_val": len(val), "first_train_t": train[0].t, "last_train_t": tr[-1].t
                 if tr else None, "usable_until_t": train[-1].t, "oos": oos,
                 "train_loss": hist["train_loss"], "val_loss": hist["val_loss"], "val_ic": hist["val_ic"],
                 "restart_val_ic": restart_ics, "restart_chosen": chosen,
                 "initial_val_loss": hist["initial_val_loss"], "best_epoch": hist["best_epoch"],
                 "seconds": round(time.time() - t0, 3)}
        history = history + [entry]
        retrained.append(j)
        ck_name = store.save_checkpoint(cutoff, {
            "model_state": model.state_dict(), "opt_state": opt.state_dict(), "tickers": list(model.tickers),
            "params": params, "params_hash": ph, "t": cutoff, "block": j, "history": history,
            "times_digest": times_digest(p, cutoff_bar), "torch": torch.__version__})
        graph_name = None
        if variant == "learned":
            with torch.no_grad():
                graph_name = store.save_graph(cutoff, graph_columns(model, batch(b["pred"][0], False)))
        pred_name = store.save_predictions(cutoff, preds)
        latest = {"t": cutoff, "path": ck_name, "graph": graph_name, "predictions": pred_name, "block": j,
                  "last_pred_t": int(p.times[reb[b["pred"][-1]]]), "git_sha": sha, "params_hash": ph,
                  "params": params, "variant": variant}
        store.write_latest(latest)
        oos_s = f" oos_ic={oos['ic']:+.4f}" if oos else ""
        log(f"[{variant}] block {j} cutoff {cutoff} {entry['kind']} n_train={entry['n_train']} "
            f"epochs={len(hist['train_loss'])} best={hist['best_epoch']} train={hist['train_loss'][-1]:.4f} "
            f"val={hist['val_loss'][-1]:.4f}{oos_s} {entry['seconds']:.1f}s")

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
    for base in ("B0", "B0E"):
        if base in ics:
            for v, ic in ics.items():
                if v != base:
                    common = sorted(set(ic) & set(ics[base]))
                    out[v]["minus_" + base] = mean_t(ic[t] - ics[base][t] for t in common)
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
        extra = "".join(f"  minus {base}: {s['minus_' + base]['mean']:+.4f} (t={s['minus_' + base]['t']:.2f})"
                        for base in ("B0", "B0E") if "minus_" + base in s)
        log(f"OOS IC {v:8s} mean={s['mean']:+.4f} t={s['t']:.2f} dates={s['n']}{extra}")
    run = {"args": vars(args), "variants": {r["variant"]: r for r in results}, "oos_ic": ic,
           "git_sha": git_sha(), "torch": torch.__version__, "seconds": round(time.time() - t0, 3),
           "panel_meta": {k: p.meta.get(k) for k in ("N", "T")}}
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(run, f, indent=1, sort_keys=True, default=str)
    return run


if __name__ == "__main__":
    main()
