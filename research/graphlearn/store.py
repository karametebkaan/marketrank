"""Persistent per-variant model store with atomic writes.

Layout of a variant directory:
  checkpoint_<t>.pt       model + optimizer state + ticker->embedding row index + params + cutoff t + history
  graph_<t>.parquet       learned adjacency at cutoff t: t, src_ticker, dst_ticker, weight, rank_in_row (top-k/row)
  predictions_<t>.parquet the dates predicted by the model of cutoff t: t, ticker, score
  latest.json             {t, path, graph, predictions, block, git_sha, params_hash, ...} -- written last
Every file is written to a temp file in the same directory, fsynced, then renamed over the target, so a crash
leaves either the old file or the new one, never a partial file.
"""
import glob
import json
import os
import re
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import torch

_KNOWN = re.compile(r"^(checkpoint_\d+\.pt|graph_\d+\.parquet|predictions_\d+\.parquet|latest\.json)$")


def atomic_write(path, write_fn):
    """write_fn(tmp_path) writes the content; the result appears at `path` only if it completes."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", suffix=".tmp", dir=d)
    os.close(fd)
    try:
        write_fn(tmp)
        with open(tmp, "rb+") as f:
            os.fsync(f.fileno())
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)  # mkstemp creates 0600; give the usual permissions
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _write_json(obj):
    def w(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
    return w


def _write_parquet(columns):
    def w(tmp):
        pq.write_table(pa.table(columns), tmp)
    return w


class Store:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, name):
        return os.path.join(self.root, name)

    def latest(self):
        p = self.path("latest.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    def clear(self):
        """Remove this store's own files (used by --fresh)."""
        for name in os.listdir(self.root):
            if _KNOWN.match(name) or name.endswith(".tmp"):
                os.remove(self.path(name))

    def save_checkpoint(self, t, obj):
        name = f"checkpoint_{t}.pt"
        atomic_write(self.path(name), lambda tmp: torch.save(obj, tmp))
        return name

    def load_checkpoint(self, name):
        return torch.load(self.path(name), weights_only=False)

    def save_graph(self, t, columns):
        name = f"graph_{t}.parquet"
        atomic_write(self.path(name), _write_parquet(columns))
        return name

    def save_predictions(self, t, columns):
        name = f"predictions_{t}.parquet"
        atomic_write(self.path(name), _write_parquet(columns))
        return name

    def load_predictions(self, t):
        p = self.path(f"predictions_{t}.parquet")
        return pq.read_table(p).to_pydict() if os.path.exists(p) else None

    def all_predictions(self):
        """All stored predictions, ordered by cutoff then as written."""
        files = sorted(glob.glob(self.path("predictions_*.parquet")),
                       key=lambda f: int(re.search(r"_(\d+)\.parquet$", f).group(1)))
        out = {"t": [], "ticker": [], "score": []}
        for f in files:
            tab = pq.read_table(f).to_pydict()
            for k in out:
                out[k].extend(tab[k])
        return out

    def graph_files(self):
        return sorted(glob.glob(self.path("graph_*.parquet")),
                      key=lambda f: int(re.search(r"_(\d+)\.parquet$", f).group(1)))

    def write_latest(self, info):
        atomic_write(self.path("latest.json"), _write_json(info))
