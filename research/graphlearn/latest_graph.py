#!/usr/bin/env python3
"""Print / load the most recent learned graph from a model store.

  latest_graph.py [--store DIR] [--ticker X] [--top 20]

DIR is a variant store (contains latest.json) or a store root (then DIR/learned is used); default
<repo>/data/graphlearn. Edges are (t, src_ticker, dst_ticker, weight, rank_in_row): row src_ticker's prediction
reads dst_ticker's state with that softmax weight (top-k per row).
"""
import argparse
import datetime as dt
import json
import os
import sys

import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_STORE = os.path.join(REPO, "data", "graphlearn")


def resolve_store(store):
    if os.path.exists(os.path.join(store, "latest.json")):
        return store
    sub = os.path.join(store, "learned")
    if os.path.exists(os.path.join(sub, "latest.json")):
        return sub
    raise FileNotFoundError(f"no latest.json in {store} or {sub}")


def read_graph(path):
    """Edge list as a dict of columns."""
    return pq.read_table(path).to_pydict()


def load_latest_graph(store=DEFAULT_STORE, ticker=None):
    """(cutoff t, [edge dicts]) of the latest graph, optionally only edges touching `ticker`."""
    root = resolve_store(store)
    with open(os.path.join(root, "latest.json")) as f:
        info = json.load(f)
    if not info.get("graph"):
        raise ValueError(f"{root}: variant {info.get('variant')} stores no graph")
    g = read_graph(os.path.join(root, info["graph"]))
    keys = list(g)
    rows = [dict(zip(keys, vals)) for vals in zip(*(g[k] for k in keys))]
    if ticker is not None:
        rows = [r for r in rows if ticker in (r["src_ticker"], r["dst_ticker"])]
    return info["t"], rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="print the most recent learned graph")
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--top", type=int, default=20, help="edges to print (by weight); 0 = all")
    a = ap.parse_args(argv)
    t, rows = load_latest_graph(a.store, a.ticker)
    rows.sort(key=lambda r: -r["weight"])
    when = dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d")
    print(f"graph at cutoff {when} (t={t}): {len(rows)} edges" + (f" touching {a.ticker}" if a.ticker else ""))
    for r in rows[: a.top or None]:
        print(f"{r['src_ticker']:>8s} <- {r['dst_ticker']:<8s} w={r['weight']:.4f} rank={r['rank_in_row']}")


if __name__ == "__main__":
    sys.exit(main())
