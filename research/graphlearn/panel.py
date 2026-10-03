"""Loader/writer for the panel exported by `marketrank --export-panel DIR`.

DIR/meta.json holds N, T, tickers[N], sectors[N], times[T] (unix s), the rebalance bar indices (either a plain
list or, as the C++ export writes it, {"bars": [...], "mode": ...}), optionally a "files" map name -> file and
"label_horizons" {"label_w": h}. Each field is a float32 little-endian row-major [T][N] file (default `<name>.f32`);
NaN = missing. label_w at bar t = open[t+1+h] / open[t+1] - 1, so it is realized at bar t+1+h.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np

FIELDS = ("ret1", "ldv", "dvshock", "vol20", "pressure", "elig", "label_w")
FILE_NAMES = {name: name + ".f32" for name in FIELDS}


@dataclass
class Panel:
    meta: dict
    tickers: list
    sectors: list
    times: np.ndarray  # int64 [T]
    rebalance: np.ndarray  # int64 bar indices
    a: dict = field(default_factory=dict)  # name -> float32 [T][N]

    @property
    def T(self):
        return int(self.meta["T"])

    @property
    def N(self):
        return int(self.meta["N"])

    @property
    def horizon(self):
        """Label horizon h in bars (label_w at t is known from bar t+1+h on)."""
        return int(self.meta.get("label_horizons", {}).get("label_w", 5))


def load_panel(dirpath):
    with open(os.path.join(dirpath, "meta.json")) as f:
        meta = json.load(f)
    T, N = int(meta["T"]), int(meta["N"])
    files = meta.get("files") or FILE_NAMES
    arrays = {}
    for name in FIELDS:
        path = os.path.join(dirpath, files.get(name, FILE_NAMES[name]))
        size = os.path.getsize(path)
        if size != T * N * 4:
            raise ValueError(f"{path}: {size // 4} floats, expected T*N = {T}*{N} = {T * N}")
        # copy-on-write memmap: pages load lazily (the real export is ~100 MB per array), writes stay in memory
        arrays[name] = np.memmap(path, dtype="<f4", mode="c", shape=(T, N))
    tickers = [str(t) for t in meta["tickers"]]
    sectors = [str(s) for s in meta["sectors"]]
    if len(tickers) != N or len(sectors) != N or len(meta["times"]) != T:
        raise ValueError("meta.json: tickers/sectors must have N entries and times T entries")
    reb = meta["rebalance"]
    if isinstance(reb, dict):
        reb = reb["bars"]
    return Panel(meta=meta, tickers=tickers, sectors=sectors, times=np.asarray(meta["times"], dtype=np.int64),
                 rebalance=np.asarray(reb, dtype=np.int64), a=arrays)


def write_panel(dirpath, tickers, sectors, times, rebalance, arrays, extra_meta=None, horizon=5):
    """Write a panel in the C++ export format (used by tests and the synthetic markets)."""
    os.makedirs(dirpath, exist_ok=True)
    T, N = len(times), len(tickers)
    meta = {"format": "marketrank-panel-export/1", "N": N, "T": T, "tickers": list(tickers),
            "sectors": list(sectors), "times": [int(t) for t in times],
            "rebalance": {"mode": "weekly", "bars": [int(r) for r in rebalance]},
            "files": dict(FILE_NAMES), "feature_names": ["ret1", "ldv", "dvshock", "vol20", "pressure"],
            "label_horizons": {"label_w": int(horizon)}}
    if extra_meta:
        meta.update(extra_meta)
    for name in FIELDS:
        arr = np.ascontiguousarray(arrays[name], dtype="<f4")
        if arr.shape != (T, N):
            raise ValueError(f"{name}: shape {arr.shape}, expected {(T, N)}")
        arr.tofile(os.path.join(dirpath, FILE_NAMES[name]))
    with open(os.path.join(dirpath, "meta.json"), "w") as f:
        json.dump(meta, f)
