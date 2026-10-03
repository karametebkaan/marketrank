"""Synthetic intraday market with a known MarketRank-like prior and a planted connectivity CORRECTION.

Bars: `sessions` sessions of 26 bars. Stocks have sectors and sizes (ldv). Returns:
  ret1[k] = market + sector + idio e[k] (+ a 3x overnight shock on the first bar of a session) + message part
The message part: the idiosyncratic innovation e_j[t'] of bar t' moves the next six bars of every i with
W_t[i, j] != 0 by (beta / 6) * W_t[i, j] * e_j[t'] each (bars t'+1..t'+6, never past the session close).
label_6[t] = sum ret1[t+1..t+6], defined iff bars t+1..t+7 lie in the session of t (intraday.label_mask).
Its predictable part (`signal.f32`, what an oracle knowing W and every past e would predict) is
  sum_{t' <= t, same session} (beta / 6) * (W_t e[t'])_i * max(0, 6 - (t - t')).

Prior (what MarketRank exports): a static base graph P0 (row = src; ~`degree` kept out-edges per row, biased to
the same sector and to larger names, log-normal weights), re-estimated once per session as P_s = rownorm(P0 *
exp(0.25 z)) (a noisy, time-varying estimate on the same support); raw = P_s * the src's dollar volume * noise.
It is written at the first bar of each session (intraday.prior_at: as-of within the session).

Truth W_s by kind:
  "planted"  the corrected chain: rownorm(P_s * exp(delta)) on the prior support with
             delta_ij = 1.0 * (same_sector_ij - mean) + 1.2 * a_i . b_j (a, b rank 2; learnable by g and u^T v),
             the sign flipped on the ~15% of support edges with c_i . d_j above its 85th percentile (learnable
             only by the signed variant), plus one off-prior edge (weight 0.3) on 10% of the rows (not learnable
             by any prior-anchored model).
  "null_a"   W = 0: nothing is predictable.
  "null_b"   W = P_s: the prior alone is predictive (Bprior is the right model; learned must not beat it).
beta is calibrated (bisection) so that the oracle IC -- the mean per-bar Spearman of signal vs label_6 over the
eligible stocks -- equals `target_ic` (null_a: beta = 0). Truth extras go to truth.npz (P0, delta, flip mask,
off-prior edges, beta, the correction IC = IC of the W - P part alone).
"""
import json
import os

import numpy as np

import intraday as itd

BARS = 26


def _rownorm(W):
    s = W.sum(1, keepdims=True)
    return np.where(s > 0, W / np.where(s > 0, s, 1), 0.0)


def _mean_ic(sig, lab, mask_rows, elig):
    """Mean per-bar Spearman(sig, lab) over the rows in mask_rows (eligible, finite columns)."""
    rows = np.flatnonzero(mask_rows)
    ics = []
    for t in rows:
        ok = elig[t] & np.isfinite(lab[t]) & np.isfinite(sig[t])
        if ok.sum() < 3 or np.ptp(sig[t, ok]) == 0:
            continue
        r = itd.avg_ranks_rows(np.vstack([sig[t, ok], lab[t, ok]]))
        a, b = r[0] - r[0].mean(), r[1] - r[1].mean()
        ics.append((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    return float(np.mean(ics)) if ics else 0.0


def make_market(out, n=300, sessions=250, target_ic=0.05, kind="planted", seed=0, degree=20, n_sectors=10,
                sigma=0.003, calib_bars=2000):
    if kind not in ("planted", "null_a", "null_b"):
        raise ValueError(kind)
    rng = np.random.default_rng(seed)
    T = sessions * BARS
    session = np.repeat(np.arange(sessions), BARS)
    sector = rng.integers(0, n_sectors, n)
    size = rng.standard_normal(n)
    same = sector[:, None] == sector[None, :]

    # ---- base prior P0 (row = src)
    P0 = np.zeros((n, n))
    for i in range(n):
        d = int(np.clip(rng.poisson(degree), degree // 2, 2 * degree))
        pref = np.exp(0.5 * size) * np.where(same[i], 4.0, 1.0)
        pref[i] = 0
        js = rng.choice(n, size=d, replace=False, p=pref / pref.sum())
        P0[i, js] = rng.lognormal(0, 1, d) * np.exp(0.3 * size[js])
    P0 = _rownorm(P0)
    support = P0 > 0

    # ---- planted correction
    a, b = rng.standard_normal((n, 2)), rng.standard_normal((n, 2))
    c, dd = rng.standard_normal((n, 2)), rng.standard_normal((n, 2))
    delta = 1.0 * (same - same[support].mean()) + 1.2 * (a @ b.T)
    cd = c @ dd.T
    flip = support & (cd > np.quantile(cd[support], 0.85))
    off = np.zeros((n, n))
    for i in rng.choice(n, size=n // 10, replace=False):
        cand = np.flatnonzero(~support[i] & (np.arange(n) != i))
        off[i, rng.choice(cand)] = 0.3

    # ---- returns: noise, the message part of ret1 at beta = 1, the predictable part of label_6 at beta = 1
    vol_i = sigma * np.exp(0.2 * rng.standard_normal(n))
    e = rng.standard_normal((T, n)) * vol_i
    mkt = rng.standard_normal(T) * sigma * 0.7
    secf = rng.standard_normal((T, n_sectors)) * sigma * 0.5
    noise = e + mkt[:, None] + secf[:, sector]
    first = np.arange(T) % BARS == 0
    noise[first] += 3 * sigma * rng.standard_normal((first.sum(), n))

    # ---- per-session prior estimates and truth (W_s used inline, never stored)
    dv = np.exp(15 + size)
    prior = {}
    src, dst = np.nonzero(support)
    msg = np.zeros((T, n))  # message part of ret1 (beta = 1)
    pred = np.zeros((T, n))  # predictable part of label_6 (beta = 1)
    pp = np.zeros((T, n))  # the part a model using the observed prior P_s as the graph would predict

    def spread(W, b0, into_msg, into_pred):
        we = e[b0: b0 + BARS] @ W.T  # [26, n]: (W e[t'])_i
        for tp in range(BARS):
            if into_msg is not None:
                for k in range(tp + 1, min(tp + 7, BARS)):
                    into_msg[b0 + k] += we[tp] / 6.0
        for t in range(BARS):
            for tp in range(max(0, t - 5), t + 1):
                into_pred[b0 + t] += we[tp] / 6.0 * (6 - (t - tp))

    for s in range(sessions):
        Ps = np.zeros((n, n))
        Ps[src, dst] = P0[src, dst] * np.exp(0.25 * rng.standard_normal(len(src)))
        Ps = _rownorm(Ps)
        raw = Ps[src, dst] * dv[src] * rng.lognormal(0, 0.2, len(src))
        prior[s * BARS] = (src, dst, Ps[src, dst], raw)
        if kind == "planted":
            W = _rownorm(Ps * np.exp(delta) * support)
            W = np.where(flip, -W, W) + off
        elif kind == "null_b":
            W = Ps
        else:
            W = None
        if W is not None:
            spread(W, s * BARS, msg, pred)
        if kind == "planted":
            spread(Ps, s * BARS, None, pp)
    cs_n = np.vstack([np.zeros((1, n)), np.cumsum(noise, 0)])
    cs_m = np.vstack([np.zeros((1, n)), np.cumsum(msg, 0)])
    lm = itd.label_mask(session)
    tt = np.flatnonzero(lm)
    lab_n = np.full((T, n), np.nan)
    lab_m = np.full((T, n), np.nan)
    lab_n[tt] = cs_n[tt + 7] - cs_n[tt + 1]
    lab_m[tt] = cs_m[tt + 7] - cs_m[tt + 1]

    elig = rng.random((T, n)) > 0.01
    elig[:BARS] = False
    calib = np.zeros(T, dtype=bool)
    calib[rng.choice(tt, size=min(calib_bars, len(tt)), replace=False)] = True
    calib &= lm

    def ic_at(beta):
        return _mean_ic(beta * pred, lab_n + beta * lab_m, calib, elig)

    if kind == "null_a" or target_ic <= 0:
        beta = 0.0
    else:
        lo, hi = 0.0, 1.0
        while ic_at(hi) < target_ic and hi < 1e4:
            hi *= 2
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ic_at(mid) < target_ic else (lo, mid)
        beta = 0.5 * (lo + hi)
    ret1 = noise + beta * msg
    label = lab_n + beta * lab_m
    signal = np.where(lm[:, None], beta * pred, np.nan)
    oracle = _mean_ic(signal, label, lm, elig)
    # IC of the correction alone: the predictable part minus what the observed prior would predict
    corr_ic = prior_ic = float("nan")
    if kind == "planted":
        corr_ic = _mean_ic(beta * (pred - pp), label, calib, elig)
        prior_ic = _mean_ic(beta * pp, label, calib, elig)

    cs = np.vstack([np.zeros((1, n)), np.cumsum(ret1, 0)])
    vol = np.full((T, n), np.nan)
    L = 520  # 20 sessions of bars
    sq = np.vstack([np.zeros((1, n)), np.cumsum(ret1 ** 2, 0)])
    idx = np.arange(T)
    lo_ = np.maximum(0, idx - L + 1)
    cntv = (idx - lo_ + 1)[:, None]
    m1 = (cs[idx + 1] - cs[lo_]) / cntv
    m2 = (sq[idx + 1] - sq[lo_]) / cntv
    vol[:] = np.sqrt(np.maximum(m2 - m1 ** 2, 0))
    vol[:BARS] = np.nan
    ldv = 15 + size[None, :] + 0.1 * rng.standard_normal((T, n))
    arrays = {"ret1": ret1, "ldv": ldv, "dvshock": rng.standard_normal((T, n)), "vol": vol,
              "pressure": rng.standard_normal((T, n)) * rng.lognormal(0, 1, (T, n)),
              "elig": elig.astype(np.float32), "active": np.ones((T, n), np.float32), "label_6": label}
    tickers = [f"S{i:04d}" for i in range(n)]
    sectors = [f"SEC{int(x)}" for x in sector]
    day0 = 1_704_200_000  # 2024-01-02 14:13 UTC; times only need to be increasing and monthly-groupable
    times = np.array([day0 + 86400 * (t // BARS) + 900 * (t % BARS) for t in range(T)], dtype=np.int64)
    info = {"kind": kind, "target_ic": target_ic, "beta": beta, "oracle_ic": oracle, "correction_ic": corr_ic,
            "prior_graph_ic": prior_ic,
            "seed": seed, "n": n, "sessions": sessions, "degree": degree}
    itd.write_intraday(out, tickers, sectors, times, session, arrays, prior=prior, extra_meta={"synthetic": info})
    np.ascontiguousarray(signal, dtype="<f4").tofile(os.path.join(out, "signal.f32"))
    np.savez_compressed(os.path.join(out, "truth.npz"), P0=P0.astype(np.float32), delta=delta.astype(np.float32),
                        flip=flip, off=off.astype(np.float32), size=size, sector=sector)
    with open(os.path.join(out, "synth.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def load_signal(d, T, N):
    return np.fromfile(os.path.join(d, "signal.f32"), dtype="<f4").reshape(T, N)
