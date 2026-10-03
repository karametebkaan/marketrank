"""power_curve.py: the pre-registered selection rule and the oracle IC."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import panel as pnl  # noqa: E402
import power_curve as pc  # noqa: E402
import synth  # noqa: E402


def res(setting, beta, seed, t_b0, t_b0e, null=False):
    m = {"n": 300, "beta": beta, "seed": seed}
    if null:
        m["null"] = True
    return {"setting": setting, "market": m, "mode": "rolling",
            "minus_B0": {"mean": 0.0, "t": t_b0}, "minus_B0E": {"mean": 0.0, "t": t_b0e}}


def grid(setting, ts, null_ts):
    out = [res(setting, b, s, *ts[i]) for i, (b, s) in
           enumerate((b, s) for s in pc.TUNE_SEEDS for b in pc.TUNE_BETAS)]
    return out + [res(setting, 0.8, s, *null_ts[i], null=True) for i, s in enumerate(pc.TUNE_SEEDS)]


class Selection(unittest.TestCase):
    def test_rule(self):
        strong = [(5, 5)] * 6
        results = (grid("powerful_but_null_fails", strong, [(0.5, 0.5), (2.5, 0.1)])
                   + grid("b0e_explains_it", [(5, 1)] * 6, [(0, 0), (0, 0)])
                   + grid("good", [(3, 3)] * 4 + [(1, 1)] * 2, [(1.9, -1.9), (0, 0)])
                   + grid("good_tie", [(3, 2.5)] * 4 + [(1, 1)] * 2, [(0, 0), (0, 0)]))
        best, table = pc.select_setting(results)
        self.assertEqual(best, "good")
        rows = {r["setting"]: r for r in table}
        self.assertFalse(rows["powerful_but_null_fails"]["admissible"])
        self.assertEqual(rows["b0e_explains_it"]["score"], 0)
        self.assertEqual(rows["good"]["score"], 4)

    def test_nothing_admissible(self):
        self.assertIsNone(pc.select_setting(grid("x", [(5, 5)] * 6, [(3, 3), (0, 0)]))[0])


class Oracle(unittest.TestCase):
    def test_oracle_ic_tracks_beta(self):
        with tempfile.TemporaryDirectory() as d:
            out = []
            for beta in (0.0, 0.8):
                pdir = os.path.join(d, str(beta))
                synth.make_planted_panel(pdir, n=200, t=600, beta=beta, seed=3)
                p = pnl.load_panel(pdir)
                preds = {"t": [], "ticker": []}
                for bar in p.rebalance[10:100]:
                    preds["t"] += [int(p.times[bar])] * p.N
                    preds["ticker"] += p.tickers
                out.append(pc.oracle_ic(p, synth.load_signal(pdir, p.T, p.N), preds))
            self.assertEqual(out[0], 0.0)
            self.assertGreater(out[1], 0.3)


if __name__ == "__main__":
    unittest.main()
