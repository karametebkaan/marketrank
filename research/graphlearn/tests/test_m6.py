"""M6: the pre-registered verdict, the data gate and the decile spread."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import intraday as itd  # noqa: E402
import real_m6  # noqa: E402
import synth_intraday as si  # noqa: E402


def oos(t_e, months, t_s):
    return {"Bprior": {"minus_B0E": {"t": t_e, "months_positive": months}, "minus_Bshuf": {"t": t_s}}}


class Verdict(unittest.TestCase):
    def test_pass_needs_every_condition_on_both_seeds(self):
        good = oos(3.0, 0.8, 2.5)
        self.assertTrue(real_m6.verdict({("rolling", 1): good, ("rolling", 2): good})[0])
        for bad in (oos(1.9, 0.8, 2.5), oos(3.0, 0.6, 2.5), oos(3.0, 0.8, 1.9)):
            self.assertFalse(real_m6.verdict({("rolling", 1): good, ("rolling", 2): bad})[0])
            self.assertFalse(real_m6.verdict({("rolling", 1): bad, ("rolling", 2): good})[0])

    def test_two_thirds_is_inclusive(self):
        r = oos(3.0, 2 / 3, 2.5)
        self.assertTrue(real_m6.verdict({("rolling", 1): r, ("rolling", 2): r})[0])


class GateAndSpread(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.panel = os.path.join(cls.tmp.name, "m")
        si.make_market(cls.panel, n=60, sessions=8, target_ic=0.05, seed=4, degree=5)
        cls.p = itd.load_intraday(cls.panel)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_gate_fails_on_a_small_panel(self):
        g = real_m6.data_gate(self.p)
        self.assertEqual(g["sessions"], 8)
        self.assertFalse(g["passes"])

    def test_spread_of_a_perfect_score_is_positive_and_net_subtracts_cost(self):
        lab = self.p.a[itd.LABEL]
        csv = os.path.join(self.tmp.name, "perfect.csv")
        with open(csv, "w") as f:
            f.write("t,ticker,score\n")
            for d in range(self.p.T):
                for c in range(self.p.N):
                    if np.isfinite(lab[d, c]):
                        f.write(f"{int(self.p.times[d])},{self.p.tickers[c]},{float(lab[d, c])}\n")
        s = real_m6.decile_spread(self.p, csv)
        self.assertGreater(s["gross"], 0)
        self.assertAlmostEqual(s["gross"] - s["net"], real_m6.COST_PER_PERIOD)
        self.assertEqual(s["n_days"], 8)


if __name__ == "__main__":
    unittest.main()
