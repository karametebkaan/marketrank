"""Synthetic-market tests: the learned graph must find a planted graph, must not "find" one in a null market,
and must lose all skill when the labels are moved one more week into the future.

Market (synth.py): N=300, T=1500 bars, weekly rebalances; next-week return of a target stock =
beta * sum_j W_ij * (this week's return of j) + noise, W sparse with k=3 edges per target row, own history carries
no signal. The walk-forwards run exactly as deployed (rolling warm-start mode, default settings).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model as mdl  # noqa: E402
import synth  # noqa: E402
import train_wf  # noqa: E402

BETA = 0.8
MARKETS = {  # name: (variants, market options)
    "planted": ("learned,B0", {}),
    "null": ("learned,B0", {"null": True}),
    "shifted": ("learned", {"label_shift_weeks": 1}),
}
TRAIN_WF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_wf.py")


def run_markets(tmp):
    """Build the three markets, then run their walk-forwards concurrently as CLI processes (each also trains its
    variants in parallel) with the default, deployed settings (rolling mode, retrain every 13)."""
    infos, procs = {}, {}
    for name, (variants, market) in MARKETS.items():
        pdir = os.path.join(tmp, name, "panel")
        infos[name] = synth.make_planted_panel(pdir, n=300, t=1500, beta=BETA, seed=1, **market)
        cmd = [sys.executable, TRAIN_WF, "--panel", pdir, "--out", os.path.join(tmp, name, "out"),
               "--store", os.path.join(tmp, name, "store"), "--variants", variants, "--threads", "3",
               "--jobs", "2", "--quiet"]
        procs[name] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    runs = {}
    for name, pr in procs.items():
        out, _ = pr.communicate()
        if pr.returncode != 0:
            raise RuntimeError(f"{name} walk-forward failed:\n{out}")
        with open(os.path.join(tmp, name, "out", "run.json")) as f:
            runs[name] = json.load(f)
    return runs, infos


def edge_auc(model, info):
    """AUC of the learned scores S_ij for true vs false edges over the rows that have true edges."""
    W, tg = info["W"], info["targets"]
    S = model.scores(model.rows_for(info["tickers"])).detach().numpy()[tg]
    truth = W[tg] > 0
    off_diag = np.ones_like(truth)
    off_diag[np.arange(len(tg)), tg] = False
    pos, neg = S[truth], S[~truth & off_diag]
    r = mdl.avg_ranks(np.r_[pos, neg])
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


class SyntheticMarkets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.runs, cls.infos = run_markets(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_planted_graph_is_found(self):
        ic = self.runs["planted"]["oos_ic"]
        diff = ic["learned"]["minus_B0"]
        print(f"\nplanted: learned IC {ic['learned']['mean']:+.4f}, B0 {ic['B0']['mean']:+.4f}, "
              f"diff {diff['mean']:+.4f} (t={diff['t']:.1f}, n={diff['n']})")
        self.assertGreater(diff["mean"], 0.05)
        self.assertGreater(diff["t"], 3)
        m, _ = train_wf.load_model_from_store(os.path.join(self.tmp.name, "planted", "store", "learned"))
        auc = edge_auc(m, self.infos["planted"])
        print(f"planted: edge AUC on the final retrain {auc:.3f}")
        self.assertGreater(auc, 0.9)

    def test_null_market_gives_no_gain(self):
        diff = self.runs["null"]["oos_ic"]["learned"]["minus_B0"]
        print(f"\nnull: learned - B0 IC {diff['mean']:+.4f} (t={diff['t']:.1f})")
        self.assertLess(diff["t"], 2)

    def test_labels_one_week_later_collapse_ic(self):
        s = self.runs["shifted"]["oos_ic"]["learned"]
        print(f"\nshifted labels: learned IC {s['mean']:+.4f} (t={s['t']:.1f})")
        self.assertLess(abs(s["mean"]), 0.02)


if __name__ == "__main__":
    unittest.main()
