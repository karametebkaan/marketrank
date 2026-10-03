"""Synthetic markets for the tests: a planted sparse flow graph, a null market, and label shifts.

Weekly returns: R[w+1] = beta * W R[w] + noise. W is nilpotent (targets read only from sources, sources have no
in-edges), so W^2 = 0 and a label shifted by one more week carries no predictable signal. Each target row has
k edges chosen as the top-k of a planted rank-`rank_true` score U V^T (a sparse graph the low-rank embedding
model can represent). Own history carries no signal. Bars split each week's return into 5 bar returns with
zero-sum intra-week noise, so ret1 summed over a week is exactly R[w].
"""
import os

import numpy as np

import panel as pnl


def make_planted_panel(out, n=300, t=1500, beta=0.4, k=3, rank_true=4, null=False, label_shift_weeks=0,
                       seed=0, sigma=0.02, elig_drop=0.02, n_sectors=10):
    rng = np.random.default_rng(seed)
    weeks = t // 5
    t = weeks * 5
    perm = rng.permutation(n)
    sources, targets = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    W = np.zeros((n, n))
    U = rng.standard_normal((n, rank_true))
    V = rng.standard_normal((n, rank_true))
    if not null:
        for i in targets:
            sc = U[i] @ V[sources].T
            W[i, sources[np.argsort(-sc)[:k]]] = 1.0 / np.sqrt(k)
    has_in = W.sum(1) > 0
    R = np.zeros((weeks, n))
    R[0] = rng.normal(0, sigma, n)
    noise_scale = np.where(has_in, np.sqrt(1 - beta ** 2), 1.0)
    for w in range(1, weeks):
        R[w] = beta * (W @ R[w - 1]) + noise_scale * rng.normal(0, sigma, n)
    eta = rng.normal(0, sigma / 5, (weeks, 5, n))
    eta -= eta.mean(axis=1, keepdims=True)
    ret1 = (R[:, None, :] / 5 + eta).reshape(t, n)

    cs = np.vstack([np.zeros((1, n)), np.cumsum(ret1, axis=0)])
    label = np.full((t, n), np.nan)
    d = np.arange(t - 5)
    label[d] = cs[d + 6] - cs[d + 1]  # sum of ret1[d+1 .. d+5]
    if label_shift_weeks:
        s = 5 * label_shift_weeks
        shifted = np.full_like(label, np.nan)
        shifted[: t - s] = label[s:]
        label = shifted

    vol20 = np.full((t, n), np.nan)
    for i in range(19, t):
        vol20[i] = ret1[i - 19: i + 1].std(axis=0)
    ldv = 15 + rng.normal(0, 1, n)[None, :] + 0.1 * rng.standard_normal((t, n))
    elig = (rng.random((t, n)) > elig_drop).astype(np.float32)
    elig[:20] = 0
    arrays = {"ret1": ret1, "ldv": ldv, "dvshock": rng.standard_normal((t, n)), "vol20": vol20,
              "pressure": rng.standard_normal((t, n)) * rng.lognormal(0, 1, (t, n)), "elig": elig,
              "label_w": label}
    tickers = [f"S{i:04d}" for i in range(n)]
    sectors = [f"SEC{int(s)}" for s in rng.integers(0, n_sectors, n)]
    times = 1_600_000_000 + 86400 * np.arange(t)
    pnl.write_panel(out, tickers, sectors, times, np.arange(4, t, 5), arrays)
    return {"W": W, "sources": sources, "targets": targets, "tickers": tickers}


def truncate_panel(src, dst, t_keep, horizon=5):
    """Copy `src` keeping only the first t_keep bars, as an export made at that time would look: labels whose
    window runs past the last bar become NaN."""
    p = pnl.load_panel(src)
    arrays = {name: p.a[name][:t_keep].copy() for name in pnl.FIELDS}
    arrays["label_w"][max(0, t_keep - horizon):] = np.nan
    reb = p.rebalance[p.rebalance < t_keep]
    os.makedirs(dst, exist_ok=True)
    pnl.write_panel(dst, p.tickers, p.sectors, p.times[:t_keep], reb, arrays)
