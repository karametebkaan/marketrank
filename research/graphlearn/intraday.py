"""M5 intraday panel: reader/writer, prior-edge lookup, label mask, per-session batches.

The on-disk layout is defined in ONE place (LAYOUT below and `_resolve_layout`). It is the layout this module
assumes for the 15-minute export of M5 Task 1 (`marketrank --export-panel DIR --timeframe 15m`); when Task 1's
meta.json differs, only `_resolve_layout` has to change.

LAYOUT (version "m5-intraday/1", all little-endian):
  meta.json  N, T, tickers[N], sectors[N], times[T] (unix s of the bar start), and the session index of every bar
             as `session` (a list of T ints, non-decreasing; bars of one regular session share it). Optional:
             "files" {name -> file} overriding the array file names, "label_horizons" {"label_6": 6},
             "prior" {"edges": path, "offsets": path}.
  <name>.f32 float32 row-major [T][N] for every name in FIELDS (ret1, ldv, dvshock, vol, pressure, elig, active,
             label_6); NaN = missing. label_6 at bar t = open[t+1+6]/open[t+1]-1, NaN unless bars t+1..t+7 all
             lie in the session of bar t (no overnight returns).
  prior/edges.bin    packed records (t_index u32, src u32, dst u32, P f32, raw f32), 20 bytes, sorted by t_index:
                     MarketRank's kept transition edges at bar t_index (src/dst are ticker columns).
  prior/offsets.bin  uint64 [T+1]: the records of bar t are edges[offsets[t]:offsets[t+1]].
Prior lookup (`prior_at`): the edges of bar t; when bar t has no records, the edges of the latest earlier bar of
the SAME session that has records (as-of within the session, never across the close, never from a later bar).
The real export writes every bar; the synthetic markets write only the first bar of each session (a prior that is
re-estimated once per session) to keep them small.

Message direction: row = src. The node src aggregates from the dst ends of its out-edges, with the row
distribution over its kept out-edges (`direction="in"` swaps the roles: dst aggregates from its in-edges).
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

LAYOUT_VERSION = "m5-intraday/1"
FIELDS = ("ret1", "ldv", "dvshock", "vol", "pressure", "elig", "active", "label_6")
LABEL = "label_6"
HORIZON = 6
EDGE_DTYPE = np.dtype([("t", "<u4"), ("src", "<u4"), ("dst", "<u4"), ("P", "<f4"), ("raw", "<f4")])
NODE_FEATURES = ("r1", "r3", "r6", "r26", "dvshock", "vol", "pressure")
N_NODE_FEATURES = 2 * len(NODE_FEATURES)  # value + missing flag
PAIR_FEATURES = ("logP", "lograw", "same_sector", "size_ratio")
N_PAIR_FEATURES = len(PAIR_FEATURES)


def _resolve_layout(dirpath, meta):
    """The only place that knows file names and meta keys. Returns {"files": {field: path}, "session": int64[T],
    "edges": path or None, "offsets": path or None, "horizon": int}."""
    files = dict(meta.get("files") or {})
    out = {"files": {name: os.path.join(dirpath, files.get(name, name + ".f32")) for name in FIELDS}}
    T = int(meta["T"])
    sess = meta.get("session")
    if sess is None and os.path.exists(os.path.join(dirpath, "session.i32")):
        sess = np.fromfile(os.path.join(dirpath, "session.i32"), dtype="<i4")
    if sess is None:
        raise ValueError(f"{dirpath}/meta.json: no session index per bar ('session')")
    out["session"] = np.asarray(sess, dtype=np.int64)
    if len(out["session"]) != T:
        raise ValueError("meta.json: session must have T entries")
    pr = meta.get("prior") or {}
    e = os.path.join(dirpath, pr.get("edges", os.path.join("prior", "edges.bin")))
    o = os.path.join(dirpath, pr.get("offsets", os.path.join("prior", "offsets.bin")))
    out["edges"] = e if os.path.exists(e) else None
    out["offsets"] = o if os.path.exists(o) else None
    out["horizon"] = int((meta.get("label_horizons") or {}).get(LABEL, HORIZON))
    return out


@dataclass
class IntradayPanel:
    meta: dict
    tickers: list
    sectors: list
    times: np.ndarray  # int64 [T]
    session: np.ndarray  # int64 [T]
    a: dict = field(default_factory=dict)
    edges: np.ndarray = None  # structured EDGE_DTYPE (memmap)
    offsets: np.ndarray = None  # uint64 [T+1]
    horizon: int = HORIZON

    @property
    def T(self):
        return int(self.meta["T"])

    @property
    def N(self):
        return int(self.meta["N"])

    def sessions(self):
        """[(session id, first bar, last bar + 1)] in time order."""
        s = self.session
        starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        ends = np.r_[starts[1:], len(s)]
        return [(int(s[a]), int(a), int(b)) for a, b in zip(starts, ends)]

    def prior_at(self, t):
        """(src, dst, P, raw) of the prior at bar t (see the module doc for the as-of rule); empty arrays if none."""
        if self.offsets is None:
            return _empty_edges()
        s = self.session[t]
        k = t
        while k >= 0 and self.session[k] == s:
            a, b = int(self.offsets[k]), int(self.offsets[k + 1])
            if b > a:
                rec = self.edges[a:b]
                if np.any(rec["t"] != k):
                    raise ValueError(f"prior records of bar {k} carry another t_index")
                return (rec["src"].astype(np.int64), rec["dst"].astype(np.int64), rec["P"].astype(np.float64),
                        rec["raw"].astype(np.float64))
            k -= 1
        return _empty_edges()


def _empty_edges():
    z = np.zeros(0, dtype=np.int64)
    return z, z.copy(), np.zeros(0), np.zeros(0)


def load_intraday(dirpath):
    with open(os.path.join(dirpath, "meta.json")) as f:
        meta = json.load(f)
    T, N = int(meta["T"]), int(meta["N"])
    lay = _resolve_layout(dirpath, meta)
    arrays = {}
    for name, path in lay["files"].items():
        if not os.path.exists(path):
            if name == "active":
                continue
            raise FileNotFoundError(path)
        if os.path.getsize(path) != T * N * 4:
            raise ValueError(f"{path}: {os.path.getsize(path) // 4} floats, expected T*N = {T * N}")
        arrays[name] = np.memmap(path, dtype="<f4", mode="c", shape=(T, N))
    edges = offsets = None
    if lay["edges"] and lay["offsets"]:
        offsets = np.fromfile(lay["offsets"], dtype="<u8")
        if len(offsets) != T + 1:
            raise ValueError(f"{lay['offsets']}: {len(offsets)} offsets, expected T+1 = {T + 1}")
        n_rec = os.path.getsize(lay["edges"]) // EDGE_DTYPE.itemsize
        if n_rec != int(offsets[-1]) or os.path.getsize(lay["edges"]) % EDGE_DTYPE.itemsize:
            raise ValueError(f"{lay['edges']}: {n_rec} records, offsets end at {int(offsets[-1])}")
        edges = np.memmap(lay["edges"], dtype=EDGE_DTYPE, mode="r", shape=(n_rec,)) if n_rec else \
            np.zeros(0, dtype=EDGE_DTYPE)
    tickers, sectors = [str(x) for x in meta["tickers"]], [str(x) for x in meta["sectors"]]
    if len(tickers) != N or len(sectors) != N or len(meta["times"]) != T:
        raise ValueError("meta.json: tickers/sectors need N entries, times T entries")
    return IntradayPanel(meta=meta, tickers=tickers, sectors=sectors, times=np.asarray(meta["times"], np.int64),
                         session=lay["session"], a=arrays, edges=edges, offsets=offsets, horizon=lay["horizon"])


def write_intraday(dirpath, tickers, sectors, times, session, arrays, prior=None, extra_meta=None):
    """Write a panel in LAYOUT. prior: dict bar -> (src, dst, P, raw) (bars absent = no records)."""
    os.makedirs(dirpath, exist_ok=True)
    T, N = len(times), len(tickers)
    meta = {"format": LAYOUT_VERSION, "timeframe": "15m", "N": N, "T": T, "tickers": list(tickers),
            "sectors": list(sectors), "times": [int(t) for t in times], "session": [int(s) for s in session],
            "files": {name: name + ".f32" for name in FIELDS}, "label_horizons": {LABEL: HORIZON},
            "decision_bars": "every bar"}
    if extra_meta:
        meta.update(extra_meta)
    for name in FIELDS:
        arr = np.ascontiguousarray(arrays[name], dtype="<f4")
        if arr.shape != (T, N):
            raise ValueError(f"{name}: shape {arr.shape}, expected {(T, N)}")
        arr.tofile(os.path.join(dirpath, name + ".f32"))
    if prior is not None:
        os.makedirs(os.path.join(dirpath, "prior"), exist_ok=True)
        counts = np.zeros(T, dtype=np.uint64)
        with open(os.path.join(dirpath, "prior", "edges.bin"), "wb") as f:
            for t in sorted(prior):
                src, dst, P, raw = prior[t]
                rec = np.zeros(len(src), dtype=EDGE_DTYPE)
                rec["t"], rec["src"], rec["dst"], rec["P"], rec["raw"] = t, src, dst, P, raw
                order = np.lexsort((rec["dst"], rec["src"]))
                f.write(rec[order].tobytes())
                counts[t] = len(src)
        np.r_[np.zeros(1, np.uint64), np.cumsum(counts)].astype("<u8").tofile(
            os.path.join(dirpath, "prior", "offsets.bin"))
        meta["prior"] = {"edges": "prior/edges.bin", "offsets": "prior/offsets.bin",
                         "record": "t_index u32, src u32, dst u32, P f32, raw f32"}
    with open(os.path.join(dirpath, "meta.json"), "w") as f:
        json.dump(meta, f)


def label_mask(session, horizon=HORIZON):
    """True at bar t iff bars t+1..t+1+horizon all lie in the session of bar t (label_6 may be defined)."""
    session = np.asarray(session)
    T = len(session)
    ok = np.zeros(T, dtype=bool)
    idx = np.arange(T)  # sessions are contiguous runs, so checking the last bar of the window suffices
    j = idx + 1 + horizon
    inb = j < T
    ok[inb] = session[j[inb]] == session[idx[inb]]
    return ok


# ---------------------------------------------------------------- features

def avg_ranks_rows(x):
    """Average 1-based ranks along the last axis of a 2-D float array (finite entries only; ties averaged)."""
    out = np.full(x.shape, np.nan)
    for r in range(x.shape[0]):
        m = np.isfinite(x[r])
        if not m.any():
            continue
        v = x[r, m]
        order = np.argsort(v, kind="stable")
        vs = v[order]
        first = np.r_[True, vs[1:] != vs[:-1]]
        starts = np.flatnonzero(first)
        ends = np.r_[starts[1:], len(vs)]
        avg = (starts + ends - 1) / 2.0 + 1.0
        rk = np.empty(len(v))
        rk[order] = avg[np.cumsum(first) - 1]
        out[r, m] = rk
    return out


def rank_gauss_vec(v):
    """Rank-gaussianize the finite entries of a 1-D array; NaN stays NaN."""
    v = np.asarray(v, dtype=np.float64)
    out = np.full(v.shape, np.nan)
    m = np.isfinite(v)
    n = int(m.sum())
    if n == 0:
        return out
    r = avg_ranks_rows(v[m][None, :])[0]
    out[m] = torch.special.ndtri(torch.from_numpy((r - 0.5) / n)).numpy()
    return out


def _window_sum(cs, cnt, t, length):
    lo = max(0, t - length + 1)
    s = cs[t + 1] - cs[lo]
    n = cnt[t + 1] - cnt[lo]
    s = s.astype(np.float64)
    s[n == 0] = np.nan
    return s


@dataclass
class SessionBatch:
    """All bars of one session, flattened: node k belongs to bar `bar_of[k]` (0-based within the session)."""
    session: int
    bars: np.ndarray  # absolute bar indices of the session, in order
    times: np.ndarray
    node_bar: torch.Tensor  # [n] local bar index
    node_col: np.ndarray  # [n] panel ticker column
    x: torch.Tensor  # [n, N_NODE_FEATURES]
    y: torch.Tensor  # [n] rank-gaussianized label (NaN where no label)
    ymask: torch.Tensor  # [n] bool: in the loss
    e_row: torch.Tensor  # [E] node index (flattened) of the aggregating end
    e_col: torch.Tensor  # [E] node index of the sending end
    e_logp: torch.Tensor  # [E] log prior weight (row-renormalized over the edges present)
    e_feat: torch.Tensor  # [E, N_PAIR_FEATURES]
    n_bars: int


class FeatureBuilder:
    """Builds SessionBatch objects (cached by the caller). Features use bars <= t only."""

    def __init__(self, p, max_nodes=3000, direction="out"):
        if direction not in ("out", "in"):
            raise ValueError(direction)
        self.p, self.max_nodes, self.direction = p, max_nodes, direction
        r = np.asarray(p.a["ret1"], dtype=np.float64)
        fin = np.isfinite(r)
        self.cs = np.vstack([np.zeros((1, p.N)), np.cumsum(np.where(fin, r, 0.0), axis=0)])
        self.cnt = np.vstack([np.zeros((1, p.N), np.int64), np.cumsum(fin, axis=0)])
        sec = sorted(set(p.sectors))
        self.sector_code = np.array([sec.index(s) for s in p.sectors])
        self.lmask = label_mask(p.session, p.horizon)

    def nodes(self, t):
        a = self.p.a
        ok = (a["elig"][t] == 1) & np.isfinite(a["ret1"][t])
        if "active" in a:
            ok &= a["active"][t] == 1
        idx = np.flatnonzero(ok)
        if len(idx) > self.max_nodes:
            ldv = np.where(np.isfinite(a["ldv"][t, idx]), a["ldv"][t, idx], -np.inf)
            idx = np.sort(idx[np.argsort(-ldv, kind="stable")[: self.max_nodes]])
        return idx

    def node_features(self, t, nodes):
        a = self.p.a
        raw = [_window_sum(self.cs, self.cnt, t, L)[nodes] for L in (1, 3, 6, 26)]
        raw += [a[k][t, nodes].astype(np.float64) for k in ("dvshock", "vol", "pressure")]
        x = np.zeros((len(nodes), N_NODE_FEATURES), dtype=np.float32)
        for f, v in enumerate(raw):
            g = rank_gauss_vec(v)
            miss = ~np.isfinite(g)
            x[:, f] = np.where(miss, 0.0, g)
            x[:, len(NODE_FEATURES) + f] = miss
        return x

    def edges(self, t, nodes):
        """Prior edges among `nodes` at bar t in local indices: (row, col, logP_renorm, feat[E,4])."""
        src, dst, P, raw = self.p.prior_at(t)
        if self.direction == "in":
            src, dst = dst, src
        pos = np.full(self.p.N, -1, dtype=np.int64)
        pos[nodes] = np.arange(len(nodes))
        keep = (src < self.p.N) & (dst < self.p.N)
        src, dst, P, raw = src[keep], dst[keep], P[keep], raw[keep]
        r, c = pos[src], pos[dst]
        ok = (r >= 0) & (c >= 0) & (r != c) & np.isfinite(P) & (P > 0)
        r, c, P, raw = r[ok], c[ok], P[ok], raw[ok]
        if len(r) == 0:
            return r, c, np.zeros(0), np.zeros((0, N_PAIR_FEATURES), np.float32)
        rowsum = np.bincount(r, weights=P, minlength=len(nodes))
        logp = np.log(P / rowsum[r])
        feat = np.zeros((len(r), N_PAIR_FEATURES), dtype=np.float32)
        lp = np.log(P)
        sd = lp.std()
        feat[:, 0] = (lp - lp.mean()) / (sd if sd > 0 else 1.0)
        lr = np.where(np.isfinite(raw) & (raw > 0), np.log(np.where(raw > 0, raw, 1.0)), np.nan)
        g = rank_gauss_vec(lr)
        feat[:, 1] = np.where(np.isfinite(g), g, 0.0)
        feat[:, 2] = self.sector_code[nodes[r]] == self.sector_code[nodes[c]]
        ldv = np.asarray(self.p.a["ldv"][t], dtype=np.float64)
        dr = ldv[nodes[r]] - ldv[nodes[c]]
        feat[:, 3] = np.clip(np.where(np.isfinite(dr), dr, 0.0) / 2.0, -3, 3)
        return r, c, logp, feat

    def session_batch(self, sess_id, bars):
        bars = np.asarray(bars)
        xs, ys, ms, nb, ncol, er, ec, el, ef = [], [], [], [], [], [], [], [], []
        off = 0
        lab = self.p.a[LABEL]
        for k, t in enumerate(bars):
            nodes = self.nodes(int(t))
            if len(nodes) == 0:
                continue
            xs.append(self.node_features(int(t), nodes))
            yl = lab[t, nodes].astype(np.float64) if self.lmask[t] else np.full(len(nodes), np.nan)
            yg = rank_gauss_vec(yl)
            m = np.isfinite(yg)
            if m.sum() < 3:
                m[:] = False
            ys.append(np.where(m, yg, np.nan))
            ms.append(m)
            nb.append(np.full(len(nodes), k))
            ncol.append(nodes)
            r, c, lp, f = self.edges(int(t), nodes)
            er.append(r + off)
            ec.append(c + off)
            el.append(lp)
            ef.append(f)
            off += len(nodes)
        cat = (lambda L, dt, w=None: np.concatenate(L).astype(dt) if L else np.zeros((0,) if w is None else (0, w), dt))
        return SessionBatch(
            session=int(sess_id), bars=bars, times=self.p.times[bars], n_bars=len(bars),
            node_bar=torch.from_numpy(cat(nb, np.int64)), node_col=cat(ncol, np.int64),
            x=torch.from_numpy(cat(xs, np.float32, N_NODE_FEATURES)),
            y=torch.from_numpy(cat(ys, np.float32)), ymask=torch.from_numpy(cat(ms, bool)),
            e_row=torch.from_numpy(cat(er, np.int64)), e_col=torch.from_numpy(cat(ec, np.int64)),
            e_logp=torch.from_numpy(cat(el, np.float32)), e_feat=torch.from_numpy(cat(ef, np.float32, N_PAIR_FEATURES)))


def truncate(src, dst, n_sessions):
    """Copy panel `src` keeping its first n_sessions sessions (an export made at that session's close)."""
    p = load_intraday(src)
    sess = p.sessions()
    T = sess[n_sessions - 1][2] if n_sessions <= len(sess) else p.T
    arrays = {name: np.asarray(p.a[name][:T]).copy() for name in FIELDS if name in p.a}
    prior = None
    if p.offsets is not None:
        prior = {}
        for t in range(T):
            a, b = int(p.offsets[t]), int(p.offsets[t + 1])
            if b > a:
                r = p.edges[a:b]
                prior[t] = (r["src"].copy(), r["dst"].copy(), r["P"].copy(), r["raw"].copy())
    extra = {k: v for k, v in p.meta.items() if k not in ("N", "T", "tickers", "sectors", "times", "session", "files",
                                                           "prior", "format", "label_horizons")}
    write_intraday(dst, p.tickers, p.sectors, p.times[:T], p.session[:T], arrays, prior=prior, extra_meta=extra)
