"""M5 prior-anchored connectivity model and its baselines (one session of 26 bars per optimizer step).

Variants (identical encoder and head; they differ only in the message z_i fed to the head with h_i):
  learned  A_t = row-softmax over the PRIOR's support of (log P_prior + g(pair features) + u_i^T v_j);
           g is a small shared MLP over (log P, log raw, same sector, log size ratio), u, v per-ticker rank-r.
           g's last layer starts at zero and u^T v starts ~0, so at init A_t == P_prior (the Bprior message).
           signed=True: the weights are A_t * tanh(b + g_sign(pair) + u'_i^T v'_j) (b starts at 2: ~+0.96), so
           an edge can carry a negative weight.
  Bprior   A_t = P_prior (row-renormalized over the edges present), fixed: MarketRank's chain as is.
  Bshuf    Bprior on the prior with every edge's dst ticker relabeled by one fixed random permutation (the batch
           builder does the relabeling): the same graph shape and weights on the wrong tickers. Control for "any
           neighbour average helps".
  B0       no message (z = 0).
  B0E      per-ticker embedding (rank r) fed into the head in place of the message; no message.
Message z_i = sum_e A_e m_{col(e)}, m = Linear(h) (msg_dim 8), over the edges e with row(e) = i (intraday.py: row = src by default).
"""
import copy
import zlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import intraday as itd

VARIANTS = ("learned", "Bprior", "Bshuf", "B0", "B0E")


@dataclass
class Reg:
    """loss = mean_bars(MSE) - lambda_ic * mean_bars(Pearson) + emb_l2 * mean(emb rows^2)
              + corr_l2 * mean((logit - log P)^2)   (keeps the learned chain near the prior)"""
    lambda_ic: float = 1.0
    emb_l2: float = 1e-3
    corr_l2: float = 1e-3


def _ticker_init(ticker, seed, shape, scale):
    key = (zlib.crc32(ticker.encode()) * 1_000_003 + 7919 * int(seed)) % (2 ** 62)
    g = torch.Generator().manual_seed(key)
    return scale * torch.randn(*shape, generator=g)


class PriorNet(nn.Module):
    def __init__(self, variant, tickers, rank=4, g_hidden=16, signed=False, dropout=0.1, seed=0, msg_dim=8):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        self.variant, self.rank, self.signed, self.seed = variant, rank, bool(signed), seed
        self.tickers = list(tickers)
        self.enc = nn.Sequential(nn.Linear(itd.N_NODE_FEATURES, 64), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout))
        self.msg_dim = msg_dim
        if variant in ("learned", "Bprior", "Bshuf"):
            self.msg = nn.Linear(32, msg_dim)  # the message carries a small projection of h_j
        head_in = 32 + (rank if variant == "B0E" else msg_dim)
        self.head = nn.Sequential(nn.Linear(head_in, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.emb_names = ()
        n = len(self.tickers)
        if variant == "learned":
            n_out = 2 if self.signed else 1
            self.g = nn.Sequential(nn.Linear(itd.N_PAIR_FEATURES, g_hidden), nn.GELU(), nn.Linear(g_hidden, n_out))
            nn.init.zeros_(self.g[2].weight)
            nn.init.zeros_(self.g[2].bias)
            names = ["U", "V"] + (["Us", "Vs"] if self.signed else [])
            for k, name in enumerate(names):
                init = torch.stack([_ticker_init(t, seed * 31 + k, (rank,), 0.1 / rank ** 0.5) for t in self.tickers]) \
                    if n else torch.zeros(0, rank)
                setattr(self, name, nn.Parameter(init))
            if self.signed:
                self.sign_bias = nn.Parameter(torch.tensor(2.0))
            self.emb_names = tuple(names)
        elif variant == "B0E":
            init = torch.stack([_ticker_init(t, seed * 31 + 7, (rank,), 1.0 / rank ** 0.5) for t in self.tickers])
            self.E = nn.Parameter(init)
            self.emb_names = ("E",)

    def emb_param_names(self):
        return set(self.emb_names)

    # ------------------------------------------------------------ adjacency
    def edge_weights(self, b):
        """Per-edge message weights of the batch's prior support, and the mean squared logit correction."""
        row, col = b.e_row, b.e_col
        zero = b.e_logp.new_zeros(())
        if self.variant in ("Bprior", "Bshuf"):
            return torch.exp(b.e_logp), zero
        if self.variant != "learned":
            raise ValueError("no edges for " + self.variant)
        tr, tc = b.node_col_t[row], b.node_col_t[col]
        gout = self.g(b.e_feat)
        corr = gout[:, 0] + (self.U[tr] * self.V[tc]).sum(1)
        logit = b.e_logp + corr
        n = b.x.shape[0]
        m = torch.full((n,), float("-inf")).scatter_reduce(0, row, logit.detach(), "amax", include_self=True)
        ex = torch.exp(logit - m[row])
        s = torch.zeros(n).index_add(0, row, ex)
        w = ex / s[row]
        if self.signed:
            w = w * torch.tanh(self.sign_bias + gout[:, 1] + (self.Us[tr] * self.Vs[tc]).sum(1))
        return w, (corr ** 2).mean() if len(corr) else zero

    def forward(self, b):
        h = self.enc(b.x)
        zero = h.new_zeros(())
        aux = {"corr_sq": zero, "e_sq": zero}
        if self.emb_names:
            aux["e_sq"] = sum((getattr(self, k)[b.node_col_t] ** 2).mean() for k in self.emb_names)
        if self.variant == "B0":
            z = h.new_zeros(h.shape[0], self.msg_dim)
        elif self.variant == "B0E":
            z = self.E[b.node_col_t]
        else:
            mh = self.msg(h)
            z = torch.zeros_like(mh)
            if len(b.e_row):
                w, aux["corr_sq"] = self.edge_weights(b)
                z = z.index_add(0, b.e_row, w.unsqueeze(1) * mh[b.e_col])
        return self.head(torch.cat([h, z], dim=1)).squeeze(-1), aux


def prepare(b):
    """Attach the torch view of the node columns (embedding rows) once per batch."""
    if not hasattr(b, "node_col_t"):
        b.node_col_t = torch.from_numpy(np.ascontiguousarray(b.node_col, dtype=np.int64))
    return b


def make_optimizer(model, lr=1e-3, weight_decay=1e-4, emb_lr=1e-2):
    emb = model.emb_param_names()
    e = [p for n, p in model.named_parameters() if n in emb]
    net = [p for n, p in model.named_parameters() if n not in emb]
    groups = [{"params": net, "weight_decay": weight_decay}]
    if e:
        groups.append({"params": e, "weight_decay": 0.0, "lr": emb_lr})
    return torch.optim.Adam(groups, lr=lr)


# ---------------------------------------------------------------- loss / metrics

def per_bar_pearson(yhat, y, bar, n_bars):
    """Pearson of yhat vs y within each bar (bars with < 3 nodes get weight 0). Returns (r[n_bars], valid)."""
    cnt = torch.bincount(bar, minlength=n_bars).to(yhat.dtype)
    c = cnt.clamp(min=1)
    ma = torch.zeros(n_bars).index_add(0, bar, yhat) / c
    mb = torch.zeros(n_bars).index_add(0, bar, y) / c
    da, db = yhat - ma[bar], y - mb[bar]
    cov = torch.zeros(n_bars).index_add(0, bar, da * db)
    va = torch.zeros(n_bars).index_add(0, bar, da * da)
    vb = torch.zeros(n_bars).index_add(0, bar, db * db)
    return cov / (torch.sqrt(va * vb) + 1e-8), cnt >= 3


def session_loss(model, b, reg):
    prepare(b)
    yhat, aux = model(b)
    m = b.ymask
    if not bool(m.any()):
        return None
    yh, y, bar = yhat[m], b.y[m], b.node_bar[m]
    r, ok = per_bar_pearson(yh, y, bar, b.n_bars)
    return (F.mse_loss(yh, y) - reg.lambda_ic * r[ok].mean() + reg.emb_l2 * aux["e_sq"]
            + reg.corr_l2 * aux["corr_sq"])


def predict(model, b):
    prepare(b)
    model.eval()
    with torch.no_grad():
        return model(b)[0].numpy()


def spearman(a, b):
    ra, rb = itd.avg_ranks_rows(np.asarray(a, float)[None])[0], itd.avg_ranks_rows(np.asarray(b, float)[None])[0]
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def bar_ics(scores, b):
    """{local bar: Spearman(score, label)} over the labeled nodes of each bar of the batch."""
    m = b.ymask.numpy()
    nb = b.node_bar.numpy()
    y = b.y.numpy()
    out = {}
    for k in np.unique(nb[m]):
        sel = m & (nb == k)
        if sel.sum() >= 3:
            ic = spearman(scores[sel], y[sel])
            if np.isfinite(ic):
                out[int(k)] = ic
    return out


def evaluate_ic(model, batches):
    ics = []
    for b in batches:
        ics.extend(bar_ics(predict(model, b), b).values())
    return float(np.mean(ics)) if ics else float("nan")


def _snapshot(model, opt):
    return copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())


def fit(model, opt, train, val, epochs, reg, patience=None, seed=0, keep_initial=False, min_epochs=0):
    """One session per step, sessions shuffled per epoch (seeded). Early stopping on the validation per-bar
    rank-IC (never before min_epochs); the best epoch's weights AND Adam state are restored. keep_initial: the
    starting model is a candidate (epoch -1), so a fine-tune that does not help is discarded."""
    gen = torch.Generator().manual_seed(seed)
    hist = {"train_loss": [], "val_ic": [], "best_epoch": None, "initial_val_ic": None}
    early = patience is not None and bool(val)
    best, best_state, bad = -float("inf"), None, 0
    if early and keep_initial:
        hist["initial_val_ic"] = evaluate_ic(model, val)
        best = hist["initial_val_ic"] if np.isfinite(hist["initial_val_ic"]) else -float("inf")
        best_state, hist["best_epoch"] = _snapshot(model, opt), -1
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for i in torch.randperm(len(train), generator=gen).tolist():
            opt.zero_grad()
            loss = session_loss(model, train[i], reg)
            if loss is None:
                continue
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        hist["train_loss"].append(tot / max(1, n))
        vi = evaluate_ic(model, val) if val else float("nan")
        hist["val_ic"].append(vi)
        if early:
            key = vi if np.isfinite(vi) else -float("inf")
            if key > best or best_state is None:
                best, bad, hist["best_epoch"] = key, 0, ep
                best_state = _snapshot(model, opt)
            else:
                bad += 1
                if bad >= patience and ep + 1 >= min_epochs:
                    break
    if early and best_state is not None:
        model.load_state_dict(best_state[0])
        opt.load_state_dict(best_state[1])
    model.eval()
    return hist
