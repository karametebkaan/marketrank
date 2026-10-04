#!/usr/bin/env python3
"""M5 walk-forward trainer on the 15-minute panel: learned (prior-anchored) vs Bprior, B0, B0E.

  wf_m5.py --panel DIR --out DIR2 [--variants learned,Bprior,Bshuf,B0E,B0] [--mode rolling|scratch] [--store DIR]
           [--fresh] [--seed 0] [--threads 1]

Calendar (in sessions): the first predicted session is session --min-history (60); every --retrain-every (21,
~monthly) sessions the model is retrained at the cutoff = the first session of the block, on the sessions k with
k + 1 + embargo <= cutoff (a session's labels are realized by its close; --embargo 1 drops one more session),
expanding window (--window 0) or the last --window sessions; it then predicts every bar of the block's sessions.
Batching: one session (all its bars) per optimizer step.
Modes: rolling (deployed): the first block trains from scratch; later blocks warm-start and fine-tune for at most
--finetune-epochs, early-stopping on the last --ft-holdout usable sessions (the pre-fine-tune model is a
candidate). scratch: every block trains a fresh model. From-scratch fits: --restarts inits, each with
validation-IC early stopping (last --val-frac of the sessions, --patience, --min-epochs); the best validation IC
is kept.
Persistence (store.py): <store>/<variant>/checkpoint_<t>.pt, predictions_<t>.parquet, latest.json, written
atomically; a rerun with the same params resumes (predicts new sessions of the last block, retrains later blocks);
a params mismatch refuses unless --fresh.
Evaluation: per-bar Spearman IC of each variant vs label_6; paired differences per bar, averaged per session,
t-stat over sessions (day-clustered SE). Printed and written to run.json.
"""
import argparse
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

import intraday as itd  # noqa: E402
import prior_model as pm  # noqa: E402
from store import Store  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_STORE = os.path.join(REPO, "data", "graphlearn_m5")
DEFAULTS = {"min_history": 60, "retrain_every": 21, "embargo": 1, "window": 0, "epochs": 30, "patience": 5,
            "min_epochs": 6, "restarts": 2, "finetune_epochs": 3, "ft_holdout": 10, "val_frac": 0.2, "rank": 4,
            "g_hidden": 8, "msg_dim": 8, "signed": False, "direction": "out", "lr": 1e-3, "emb_lr": 1e-2, "emb_l2": 1e-3,
            "corr_l2": 1e-3, "weight_decay": 1e-4, "lambda_ic": 1.0, "dropout": 0.1, "max_nodes": 3000}
HASHED = tuple(DEFAULTS) + ("mode", "seed", "threads")
SHUF_SEED = 20261004  # the fixed ticker permutation of the Bshuf control (M6 pre-registration)
BASES = ("Bprior", "Bshuf", "B0E", "B0")


def parse_args(argv):
    d = DEFAULTS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="learned,Bprior,B0E,B0")
    ap.add_argument("--mode", choices=("rolling", "scratch"), default="rolling")
    ap.add_argument("--seed", type=int, default=0)
    for k in ("min_history", "retrain_every", "embargo", "window", "epochs", "patience", "min_epochs", "restarts",
              "finetune_epochs", "ft_holdout", "rank", "g_hidden", "msg_dim", "max_nodes"):
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=d[k])
    for k in ("val_frac", "lr", "emb_lr", "emb_l2", "corr_l2", "weight_decay", "lambda_ic", "dropout"):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=d[k])
    ap.add_argument("--signed", action=argparse.BooleanOptionalAction, default=d["signed"])
    ap.add_argument("--direction", choices=("out", "in"), default=d["direction"])
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--csv", action="store_true", help="also write <variant>.csv (t,ticker,score)")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def setup_torch(threads):
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)


def source_digest():
    h = hashlib.sha256()
    for name in ("intraday.py", "prior_model.py", "wf_m5.py"):
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def panel_digest(p):
    keep = {k: p.meta.get(k) for k in ("format", "timeframe", "label_horizons", "windows", "prior_params")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def times_digest(p, upto_bar):
    return hashlib.sha256(np.ascontiguousarray(p.times[: upto_bar + 1], "<i8").tobytes()).hexdigest()[:16]


def params_for(args, variant, p=None):
    out = {k: getattr(args, k) for k in HASHED}
    out.update(variant=variant, source=source_digest(), panel_meta=panel_digest(p) if p is not None else None)
    return out


def params_hash(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def git_sha():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def plan_blocks(n_sessions, min_history, retrain_every, embargo):
    blocks, start = [], min_history
    while start < n_sessions:
        pred = list(range(start, min(start + retrain_every, n_sessions)))
        train = [k for k in range(start) if k + 1 + embargo <= start]
        blocks.append({"block": len(blocks), "pred": pred, "train": train})
        start += retrain_every
    return blocks


class ResumeRefused(RuntimeError):
    pass


class Data:
    """Session batches of a panel, built lazily and cached (shared by all variants of a run)."""

    def __init__(self, p, args, relabel=None):
        self.p = p
        self.fb = itd.FeatureBuilder(p, args.max_nodes, args.direction, relabel=relabel)
        self.sess = p.sessions()
        self.lmask = itd.label_mask(p.session, p.horizon)
        self.cache = {}

    def batch(self, pos, train=False):
        """All bars of the session (prediction), or only its labeled bars (training: 19 of 26)."""
        key = (pos, train)
        if key not in self.cache:
            sid, a, b = self.sess[pos]
            bars = np.arange(a, b)
            if train:
                bars = bars[self.lmask[bars]]
            self.cache[key] = pm.prepare(self.fb.session_batch(sid, bars))
        return self.cache[key]


def _reg(args):
    return pm.Reg(lambda_ic=args.lambda_ic, emb_l2=args.emb_l2, corr_l2=args.corr_l2)


def new_model(args, variant, tickers, restart=0):
    """Restart r > 0 gets a different init (net weights via torch's seed, embeddings via their ticker seed)."""
    return pm.PriorNet(variant, tickers, rank=args.rank, g_hidden=args.g_hidden, signed=args.signed,
                       dropout=args.dropout, seed=args.seed + 7919 * restart, msg_dim=args.msg_dim)


def predict_cols(model, data, positions):
    cols = {"t": [], "ticker": [], "score": []}
    tk = data.p.tickers
    for pos in positions:
        b = data.batch(pos)
        if b.x.shape[0] == 0:
            continue
        s = pm.predict(model, b)
        nb = b.node_bar.numpy()
        cols["t"].extend(int(b.times[k]) for k in nb)
        cols["ticker"].extend(tk[c] for c in b.node_col)
        cols["score"].extend(s.astype(np.float64).tolist())
    return cols


def run_variant(data, variant, args, log=print):
    p = data.p
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
        raise ResumeRefused(f"{store.root}: stored params hash {latest.get('params_hash')} != {ph}; differences "
                            f"(stored, now): {json.dumps(diff, sort_keys=True, default=str)}; rerun with --fresh")
    blocks = plan_blocks(len(data.sess), args.min_history, args.retrain_every, args.embargo)
    sha = git_sha()
    model = opt = None
    history, resume_block, retrained = [], -1, []
    if latest is not None:
        ck = store.load_checkpoint(latest["path"])
        resume_block = ck["block"]
        if resume_block >= len(blocks):
            raise ResumeRefused(f"{store.root}: checkpoint block {resume_block} beyond this panel's calendar")
        cut_bar = data.sess[blocks[resume_block]["pred"][0]][1]
        if int(p.times[cut_bar]) != ck["t"] or times_digest(p, cut_bar) != ck["times_digest"]:
            raise ResumeRefused(f"{store.root}: checkpoint cutoff {ck['t']} does not match this panel's bar times")
        model = new_model(args, variant, ck["tickers"])
        model.load_state_dict(ck["model_state"])
        opt = pm.make_optimizer(model, args.lr, args.weight_decay, args.emb_lr)
        opt.load_state_dict(ck["opt_state"])
        history = ck["history"]
        log(f"[{variant}] resuming at block {resume_block} (cutoff {ck['t']})")
    for blk in blocks:
        j = blk["block"]
        if j < resume_block:
            continue
        cut_bar = data.sess[blk["pred"][0]][1]
        cutoff = int(p.times[cut_bar])
        if j == resume_block:  # predict the block's sessions that were not exported when it was stored
            prev = store.load_predictions(cutoff) or {"t": [], "ticker": [], "score": []}
            done = set(prev["t"])
            todo = [s for s in blk["pred"] if int(p.times[data.sess[s][1]]) not in done]
            if todo:
                new = predict_cols(model, data, todo)
                for k in prev:
                    prev[k].extend(new[k])
                store.save_predictions(cutoff, prev)
                latest["last_session"] = blk["pred"][-1]
                store.write_latest(latest)
            continue
        t0 = time.time()
        usable = blk["train"][-args.window:] if args.window > 0 else blk["train"]
        train = [x for x in (data.batch(k, True) for k in usable) if bool(x.ymask.any())]
        bseed = args.seed * 100_003 + j
        warm = args.mode == "rolling" and model is not None
        if warm:
            n_val = min(args.ft_holdout, max(0, len(train) - 1))
            tr, val = train[: len(train) - n_val], train[len(train) - n_val:]
            torch.manual_seed(bseed)
            hist = pm.fit(model, opt, tr, val, args.finetune_epochs, reg, patience=args.patience, seed=bseed,
                          keep_initial=True)
            restart_ics = None
        else:
            n_val = int(round(args.val_frac * len(train))) if len(train) >= 5 else 0
            tr, val = train[: len(train) - n_val], train[len(train) - n_val:]
            fits = []
            for r in range(max(1, args.restarts)):
                rseed = bseed + 1_000_003 * r
                torch.manual_seed(rseed)
                m_r = new_model(args, variant, p.tickers, restart=r)
                o_r = pm.make_optimizer(m_r, args.lr, args.weight_decay, args.emb_lr)
                h_r = pm.fit(m_r, o_r, tr, val, args.epochs, reg, patience=args.patience, seed=rseed,
                             min_epochs=args.min_epochs)
                fits.append((m_r, o_r, h_r, pm.evaluate_ic(m_r, val) if val else float("nan")))
            restart_ics = [f[3] for f in fits]
            chosen = int(np.nanargmax(restart_ics)) if np.isfinite(restart_ics).any() else 0
            model, opt, hist = fits[chosen][:3]
        preds = predict_cols(model, data, blk["pred"])
        entry = {"block": j, "cutoff_t": cutoff, "kind": "finetune" if warm else "scratch", "n_train": len(tr),
                 "n_val": len(val), "train_loss": hist["train_loss"], "val_ic": hist["val_ic"],
                 "best_epoch": hist["best_epoch"], "restart_val_ic": restart_ics,
                 "seconds": round(time.time() - t0, 2)}
        history = history + [entry]
        retrained.append(j)
        ck_name = store.save_checkpoint(cutoff, {
            "model_state": model.state_dict(), "opt_state": opt.state_dict(), "tickers": list(model.tickers),
            "params": params, "params_hash": ph, "t": cutoff, "block": j, "history": history,
            "times_digest": times_digest(p, cut_bar), "torch": torch.__version__})
        pred_name = store.save_predictions(cutoff, preds)
        latest = {"t": cutoff, "path": ck_name, "predictions": pred_name, "block": j,
                  "last_session": blk["pred"][-1], "git_sha": sha, "params_hash": ph, "params": params,
                  "variant": variant}
        store.write_latest(latest)
        log(f"[{variant}] block {j} {entry['kind']} n_train={len(tr)} epochs={len(hist['train_loss'])} "
            f"best={hist['best_epoch']} val_ic={hist['val_ic'][-1] if hist['val_ic'] else float('nan'):+.4f} "
            f"{entry['seconds']:.1f}s")
    return {"variant": variant, "params_hash": ph, "store": store.root, "retrained_blocks": retrained,
            "resumed_from_block": resume_block if resume_block >= 0 else None, "history": history}


def load_model(variant_store):
    """The latest checkpointed model of a variant store."""
    st = Store(variant_store)
    ck = st.load_checkpoint(st.latest()["path"])
    pr = ck["params"]
    m = pm.PriorNet(pr["variant"], ck["tickers"], rank=pr["rank"], g_hidden=pr["g_hidden"], signed=pr["signed"],
                    dropout=pr["dropout"], seed=pr["seed"], msg_dim=pr["msg_dim"])
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, ck


# ---------------------------------------------------------------- evaluation

def bar_ic_table(p, preds):
    """{bar index: Spearman(score, label_6)} over the predicted nodes of each labeled bar."""
    bar = {int(t): i for i, t in enumerate(p.times)}
    col = {tk: i for i, tk in enumerate(p.tickers)}
    lm = itd.label_mask(p.session, p.horizon)
    by = {}
    for t, tk, s in zip(preds["t"], preds["ticker"], preds["score"]):
        by.setdefault(bar[int(t)], ([], []))
        by[bar[int(t)]][0].append(col[tk])
        by[bar[int(t)]][1].append(s)
    out = {}
    lab = p.a[itd.LABEL]
    for d, (idx, s) in by.items():
        if not lm[d]:
            continue
        y = np.asarray(lab[d, idx], float)
        s = np.asarray(s, float)
        ok = np.isfinite(y) & np.isfinite(s)
        if ok.sum() >= 3:
            ic = pm.spearman(s[ok], y[ok])
            if np.isfinite(ic):
                out[d] = ic
    return out


def clustered(p, per_bar):
    """Mean of a per-bar series and its t-stat with day-clustered SE: the per-session means are the units."""
    by = {}
    for d, v in per_bar.items():
        by.setdefault(int(p.session[d]), []).append(v)
    days = np.array([np.mean(v) for _, v in sorted(by.items())])
    n = len(days)
    if n < 2:
        return {"mean": float(days.mean()) if n else float("nan"), "t": float("nan"), "n_days": n,
                "n_bars": len(per_bar)}
    sd = days.std(ddof=1)
    months = {}
    for (sid, v) in sorted(by.items()):
        first = int(np.flatnonzero(p.session == sid)[0])
        mkey = time.strftime("%Y-%m", time.gmtime(int(p.times[first])))
        months.setdefault(mkey, []).extend(v)
    pos = [np.mean(v) > 0 for v in months.values()]
    return {"mean": float(np.mean(list(per_bar.values()))), "day_mean": float(days.mean()),
            "t": float(days.mean() / sd * math.sqrt(n)) if sd > 0 else float("nan"), "n_days": n,
            "n_bars": len(per_bar), "months_positive": float(np.mean(pos)) if pos else float("nan")}


def ic_summary(p, all_preds):
    ics = {v: bar_ic_table(p, pr) for v, pr in all_preds.items()}
    out = {v: clustered(p, ic) for v, ic in ics.items()}
    for v in ics:
        for base in BASES:
            if base in ics and base != v:
                common = set(ics[v]) & set(ics[base])
                out[v]["minus_" + base] = clustered(p, {d: ics[v][d] - ics[base][d] for d in common})
    return out, ics


# ---------------------------------------------------------------- main

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    log = (lambda *a: None) if args.quiet else print
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in pm.VARIANTS:
            raise SystemExit(f"unknown variant {v}")
    os.makedirs(args.out, exist_ok=True)
    setup_torch(args.threads)
    t0 = time.time()
    p = itd.load_intraday(args.panel)
    data = Data(p, args)
    shuf = Data(p, args, relabel=itd.shuffle_perm(p.N, SHUF_SEED)) if "Bshuf" in variants else None
    try:
        results = [run_variant(shuf if v == "Bshuf" else data, v, args, log) for v in variants]
    except ResumeRefused as e:
        raise SystemExit(f"refusing to resume: {e}")
    all_preds = {r["variant"]: Store(r["store"]).all_predictions() for r in results}
    if args.csv:
        for v, pr in all_preds.items():
            with open(os.path.join(args.out, f"{v}.csv"), "w") as f:
                f.write("t,ticker,score\n")
                for t, tk, s in zip(pr["t"], pr["ticker"], pr["score"]):
                    f.write(f"{t},{tk},{s:.9g}\n")
    ic, _ = ic_summary(p, all_preds)
    for v in variants:
        s = ic[v]
        extra = "".join(f"  -{b}: {s['minus_' + b]['mean']:+.4f} (t={s['minus_' + b]['t']:.2f})"
                        for b in BASES if "minus_" + b in s)
        log(f"OOS IC {v:7s} mean={s['mean']:+.4f} t={s['t']:.2f} days={s['n_days']}{extra}")
    run = {"args": vars(args), "variants": {r["variant"]: r for r in results}, "oos_ic": ic, "git_sha": git_sha(),
           "torch": torch.__version__, "seconds": round(time.time() - t0, 2)}
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(run, f, indent=1, sort_keys=True, default=str)
    return run


if __name__ == "__main__":
    main()
