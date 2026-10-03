"""Learned-connectivity model: per-date node sets and causal features, the graph net and its variants, training.

Variants (identical except the message-passing matrix A):
  learned  A = row-softmax over the top-k of relu(E_s E_d^T) (per-ticker embeddings, diagonal masked)
  B0       m = 0 (no graph; the decisive baseline)
  B0E      control: B0 plus the same per-ticker embedding fed straight into the head, head([h, E_s[i]]), no
           message -- separates "the graph helps" from "per-ticker parameters help"
  B1       A = a supplied fixed sparse matrix (e.g. MarketRank's P), row-normalized
  B2       A = sector block (uniform over the same-sector nodes, excluding self)
"""
import copy
import zlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURES = ("r1", "r5", "r20", "dvshock", "vol20", "pressure")
N_FEATURES = 2 * len(FEATURES)  # value + missing flag
VARIANTS = ("learned", "B0", "B0E", "B1", "B2")
# Embedding init: entries (1 +- 1)/sqrt(rank), so S = E_s.E_d^T starts at ~1 and is almost always positive: relu is
# not dead for any pair at init and every edge can receive gradient (with mean-0 init half the pairs never learn).


# ---------------------------------------------------------------- features

def avg_ranks(x):
    """1-based average ranks (ties share their mean rank)."""
    x = np.asarray(x)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    first = np.r_[True, xs[1:] != xs[:-1]]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(xs)]
    avg = (starts + ends - 1) / 2.0 + 1.0
    ranks = np.empty(len(x))
    ranks[order] = avg[np.cumsum(first) - 1]
    return ranks


def rank_gauss(v):
    """Cross-sectional rank-gaussianization of the finite entries; NaN stays NaN."""
    v = np.asarray(v, dtype=np.float64)
    out = np.full(v.shape, np.nan)
    m = np.isfinite(v)
    n = int(m.sum())
    if n == 0:
        return out
    u = (avg_ranks(v[m]) - 0.5) / n
    out[m] = torch.special.ndtri(torch.from_numpy(u)).numpy()
    return out


def node_set(p, d, max_nodes, require_label):
    """The walk-forward's decision set at bar d -- elig AND active (active.f32 when exported, else finite
    pressure: inactive nodes are exactly those with NaN pressure) -- with a finite label when training, capped to
    the top max_nodes by ldv[d]. The same mask is used for training and prediction."""
    ok = p.a["elig"][d] == 1
    if "active" in p.a:
        ok &= p.a["active"][d] == 1
    else:
        ok &= np.isfinite(p.a["pressure"][d])
    if require_label:
        ok &= np.isfinite(p.a["label_w"][d])
    idx = np.flatnonzero(ok)
    if len(idx) > max_nodes:
        ldv = p.a["ldv"][d, idx].astype(np.float64)
        ldv = np.where(np.isfinite(ldv), ldv, -np.inf)
        idx = np.sort(idx[np.argsort(-ldv, kind="stable")[:max_nodes]])
    return idx


def _window_sum(r, d, nodes, length):
    w = r[max(0, d - length + 1): d + 1, nodes].astype(np.float64)
    s = np.nansum(w, axis=0)
    s[np.isfinite(w).sum(axis=0) == 0] = np.nan
    return s


def date_features(p, d, nodes):
    """Causal features (bars <= d), each rank-gaussianized over the node set; NaN -> 0 plus a missing flag."""
    r = p.a["ret1"]
    # (pressure is rank-gaussianized like everything else, so no scale normalization is needed)
    press = p.a["pressure"][d, nodes].astype(np.float64)
    raw = [_window_sum(r, d, nodes, 1), _window_sum(r, d, nodes, 5), _window_sum(r, d, nodes, 20),
           p.a["dvshock"][d, nodes].astype(np.float64), p.a["vol20"][d, nodes].astype(np.float64), press]
    x = np.zeros((len(nodes), N_FEATURES), dtype=np.float32)
    for f, v in enumerate(raw):
        g = rank_gauss(v)
        miss = ~np.isfinite(g)
        x[:, f] = np.where(miss, 0.0, g)
        x[:, len(FEATURES) + f] = miss
    return x


@dataclass
class Batch:
    d: int  # bar index
    t: int  # unix time of bar d
    nodes: np.ndarray  # panel ticker indices
    tickers: list
    x: torch.Tensor
    y: torch.Tensor  # rank-gaussianized label (None when predicting)
    sector: torch.Tensor
    b1: tuple = None  # (rows, cols, w) in node-local indices


def make_batch(p, d, max_nodes, train, sector_codes, b1_lookup=None):
    nodes = node_set(p, d, max_nodes, require_label=train)
    y = None
    if train:
        y = torch.from_numpy(rank_gauss(p.a["label_w"][d, nodes]).astype(np.float32))
    b1 = b1_lookup(int(p.times[d]), [p.tickers[i] for i in nodes]) if b1_lookup else None
    return Batch(d=int(d), t=int(p.times[d]), nodes=nodes, tickers=[p.tickers[i] for i in nodes],
                 x=torch.from_numpy(date_features(p, d, nodes)), y=y,
                 sector=torch.from_numpy(sector_codes[nodes].astype(np.int64)), b1=b1)


# ---------------------------------------------------------------- net

@dataclass
class Reg:
    """Loss weights: loss = MSE - lambda_ic * Pearson + l1 * mean(S) + emb_l2 * mean(E[rows]^2)."""
    lambda_ic: float = 1.0
    l1: float = 1e-2
    emb_l2: float = 1e-3


class GraphNet(nn.Module):
    def __init__(self, variant, rank=8, topk=20, dropout=0.1, seed=0, score_fn="relu"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        if score_fn not in ("relu", "softplus"):
            raise ValueError(f"unknown score_fn {score_fn}")
        self.variant, self.rank, self.topk, self.seed, self.score_fn = variant, rank, topk, seed, score_fn
        self._dense = None
        self.enc = nn.Sequential(nn.Linear(N_FEATURES, 64), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout))
        head_in = 32 + (rank if variant == "B0E" else 32)
        self.head = nn.Sequential(nn.Linear(head_in, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.tickers = []  # embedding row -> ticker
        self.row = {}  # ticker -> embedding row
        self.emb_names = {"learned": ("E_s", "E_d"), "B0E": ("E_s",)}.get(variant, ())
        for name in self.emb_names:
            setattr(self, name, nn.Parameter(torch.zeros(0, rank)))

    # --- per-ticker embeddings that persist across dates and universe changes
    def _ticker_init(self, ticker):
        key = (zlib.crc32(ticker.encode()) * 1_000_003 + 7919 * int(self.seed)) % (2 ** 62)
        g = torch.Generator().manual_seed(key)
        return (1.0 + torch.randn(2, self.rank, generator=g)) / self.rank ** 0.5

    def ensure_tickers(self, tickers, optimizer=None):
        """Give every new ticker a fresh (ticker-seeded, order-independent) embedding row; existing and departed
        tickers keep theirs. Adam moments of new rows start at zero."""
        new = []
        for t in tickers:
            if t not in self.row:
                self.row[t] = len(self.tickers)
                self.tickers.append(t)
                new.append(t)
        if not new or not self.emb_names:
            return
        init = torch.stack([self._ticker_init(t) for t in new])  # [n_new, 2, r]
        for j, name in enumerate(self.emb_names):
            part = init[:, j]
            old = getattr(self, name)
            newp = nn.Parameter(torch.cat([old.data, part]))
            if optimizer is not None:
                for group in optimizer.param_groups:
                    group["params"] = [newp if q is old else q for q in group["params"]]
                st = optimizer.state.pop(old, None)
                if st:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in st:
                            st[key] = torch.cat([st[key], torch.zeros_like(part)])
                    optimizer.state[newp] = st
            setattr(self, name, newp)

    def rows_for(self, tickers):
        return torch.tensor([self.row[t] for t in tickers], dtype=torch.int64)

    def scores(self, rows):
        """S = relu(E_s E_d^T) (or softplus: an edge pushed below 0 is not dead and can come back) over the given
        embedding rows (diagonal included)."""
        logits = self.E_s[rows] @ self.E_d[rows].T
        return F.relu(logits) if self.score_fn == "relu" else F.softplus(logits)

    def adjacency(self, rows):
        """Top-k per row of S (diagonal masked) and their row-softmax weights; also the mean off-diagonal S."""
        S = self.scores(rows)
        n = S.shape[0]
        k = min(self.topk, n - 1)
        eye = torch.eye(n, dtype=torch.bool)
        masked = S.masked_fill(eye, float("-inf"))
        vals, idx = torch.topk(masked, k, dim=1)
        pen = S.masked_fill(eye, 0.0).sum() / (n * (n - 1))
        self._dense = torch.softmax(masked, dim=1) if self.training else None
        return idx, torch.softmax(vals, dim=1), pen

    def emb_sq(self, rows):
        """mean(E_s[rows]^2 + E_d[rows]^2): an L2 penalty on the rows of today's nodes only (absent tickers are
        left untouched)."""
        return sum((getattr(self, name)[rows] ** 2).mean() for name in self.emb_names)

    def forward(self, b):
        """Returns (y_hat, aux) with aux = {"s_mean": mean off-diagonal S, "e_sq": embedding L2 term}."""
        h = self.enc(b.x)
        zero = h.new_zeros(())
        aux = {"s_mean": zero, "e_sq": zero}
        n = h.shape[0]
        rows = self.rows_for(b.tickers) if self.emb_names else None
        if rows is not None:
            aux["e_sq"] = self.emb_sq(rows)
        if self.variant == "B0E":
            z = self.E_s[rows]  # the same per-ticker embedding, fed straight into the head; no message
        elif self.variant == "B0" or n < 2:
            z = torch.zeros_like(h)
        elif self.variant == "B2":
            # static sectors (the panel's one sector label per ticker)
            g = int(b.sector.max()) + 1
            sums = torch.zeros(g, h.shape[1]).index_add(0, b.sector, h)
            cnt = torch.bincount(b.sector, minlength=g).to(h.dtype)
            z = (sums[b.sector] - h) / (cnt[b.sector] - 1).clamp(min=1).unsqueeze(1)
        elif self.variant == "B1":
            z = torch.zeros_like(h)
            if b.b1 is not None and len(b.b1[0]):
                br, bc, w = b.b1
                z = z.index_add(0, br, w.unsqueeze(1) * h[bc])
        else:
            idx, w, aux["s_mean"] = self.adjacency(rows)
            z = (w.unsqueeze(-1) * h[idx]).sum(1)
            if self._dense is not None:
                # training only, straight-through: the forward value stays the top-k message, but the dense
                # row-softmax adds a gradient path to every S_ij (only S learns through it). Without it only the
                # k selected edges per row get gradient and true edges outside the top-k are never discovered.
                md = self._dense @ h.detach()
                z = z + md - md.detach()
        return self.head(torch.cat([h, z], dim=1)).squeeze(-1), aux


def make_optimizer(model, lr=1e-3, weight_decay=1e-4, emb_lr=1e-3):
    """Adam; weight decay on the net weights only. The embeddings get their own lr and no weight decay -- they are
    regularized by emb_l2 on the rows of each date's nodes (AdamW-style decay would also shrink the rows of
    tickers absent from the current universe)."""
    emb = [p for n, p in model.named_parameters() if n in ("E_s", "E_d")]
    net = [p for n, p in model.named_parameters() if n not in ("E_s", "E_d")]
    groups = [{"params": net, "weight_decay": weight_decay}]
    if emb:
        groups.append({"params": emb, "weight_decay": 0.0, "lr": emb_lr})
    return torch.optim.Adam(groups, lr=lr)


# ---------------------------------------------------------------- loss / training

def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)


def date_loss(model, b, reg):
    yhat, aux = model(b)
    return (F.mse_loss(yhat, b.y) - reg.lambda_ic * pearson(yhat, b.y) + reg.l1 * aux["s_mean"]
            + reg.emb_l2 * aux["e_sq"])


def evaluate(model, batches, reg):
    if not batches:
        return float("nan")
    model.eval()
    with torch.no_grad():
        return float(np.mean([date_loss(model, b, reg).item() for b in batches]))


def spearman_np(a, b):
    ra, rb = avg_ranks(a), avg_ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def evaluate_ic(model, batches):
    """Mean per-date Spearman of y_hat vs the label (y is a monotone transform of label_w)."""
    ics = [spearman_np(predict(model, b), b.y.numpy()) for b in batches if len(b.nodes) >= 3]
    ics = [x for x in ics if np.isfinite(x)]
    return float(np.mean(ics)) if ics else float("nan")


def _snapshot(model, opt):
    return copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())


def fit(model, opt, train, val, epochs, reg, patience=None, seed=0, keep_initial=False):
    """One date per step, dates shuffled per epoch (seeded). With `val` and `patience`: early stopping, and the
    weights AND the Adam state of the best epoch are restored at the end. keep_initial=True also scores the
    starting model on `val` first (epoch -1), so a fine-tune that never improves the held-out loss keeps the
    pre-fine-tune state."""
    gen = torch.Generator().manual_seed(seed)
    hist = {"train_loss": [], "val_loss": [], "best_epoch": None, "initial_val_loss": None}
    early = patience is not None and bool(val)
    best, best_state, bad = float("inf"), None, 0
    if early and keep_initial:
        best = hist["initial_val_loss"] = evaluate(model, val, reg)
        best_state, hist["best_epoch"] = _snapshot(model, opt), -1
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for i in torch.randperm(len(train), generator=gen).tolist():
            opt.zero_grad()
            loss = date_loss(model, train[i], reg)
            loss.backward()
            opt.step()
            tot += loss.item()
        hist["train_loss"].append(tot / max(1, len(train)))
        vl = evaluate(model, val, reg)
        hist["val_loss"].append(vl)
        if early:
            if vl < best:
                best, bad, hist["best_epoch"] = vl, 0, ep
                best_state = _snapshot(model, opt)
            else:
                bad += 1
                if bad >= patience:
                    break
    if early and best_state is not None:
        model.load_state_dict(best_state[0])
        opt.load_state_dict(best_state[1])
    model.eval()
    return hist


def predict(model, b):
    model.eval()
    with torch.no_grad():
        return model(b)[0].numpy()
