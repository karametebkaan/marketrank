"""Fast unit tests: panel loader round-trip, causal features, embargo, determinism."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import panel as pnl  # noqa: E402
import model as mdl  # noqa: E402
import synth  # noqa: E402
import train_wf  # noqa: E402


class LoaderRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(1)
        T, N = 7, 4
        arrays = {name: rng.standard_normal((T, N)).astype(np.float32) for name in pnl.FIELDS}
        arrays["ret1"][2, 1] = np.nan
        arrays["elig"] = (rng.random((T, N)) > 0.3).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            pnl.write_panel(d, tickers=["A", "B", "C", "D"], sectors=["X", "Y", "X", "Z"],
                            times=list(range(1000, 1000 + T)), rebalance=[1, 3, 5], arrays=arrays)
            # raw little-endian row-major [T][N]
            raw = np.fromfile(os.path.join(d, "ret1.f32"), dtype="<f4")
            self.assertEqual(raw.size, T * N)
            self.assertEqual(raw[1 * N + 2], arrays["ret1"][1, 2])
            p = pnl.load_panel(d)
            self.assertEqual((p.T, p.N), (T, N))
            self.assertEqual(p.tickers, ["A", "B", "C", "D"])
            self.assertEqual(p.sectors, ["X", "Y", "X", "Z"])
            self.assertEqual(list(p.times), list(range(1000, 1000 + T)))
            self.assertEqual(list(p.rebalance), [1, 3, 5])
            for name in pnl.FIELDS:
                np.testing.assert_array_equal(p.a[name], arrays[name])
            self.assertTrue(np.isnan(p.a["ret1"][2, 1]))

    def test_plain_rebalance_list_and_horizon(self):
        with tempfile.TemporaryDirectory() as d:
            arrays = {name: np.zeros((3, 2), np.float32) for name in pnl.FIELDS}
            pnl.write_panel(d, ["A", "B"], ["X", "X"], [1, 2, 3], [0, 2], arrays, horizon=7)
            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            self.assertEqual(meta["rebalance"]["bars"], [0, 2])  # C++ export layout
            self.assertEqual(pnl.load_panel(d).horizon, 7)
            meta["rebalance"] = [0, 1]
            del meta["label_horizons"], meta["files"]
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(meta, f)
            p = pnl.load_panel(d)
            self.assertEqual(list(p.rebalance), [0, 1])
            self.assertEqual(p.horizon, 5)

    def test_size_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            arrays = {name: np.zeros((3, 2), np.float32) for name in pnl.FIELDS}
            pnl.write_panel(d, ["A", "B"], ["X", "X"], [1, 2, 3], [0], arrays)
            np.zeros(5, np.float32).tofile(os.path.join(d, "vol20.f32"))
            with self.assertRaises(ValueError):
                pnl.load_panel(d)


class Features(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        synth.make_planted_panel(self.tmp.name, n=60, t=200, seed=3)
        self.p = pnl.load_panel(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_features_use_only_past_bars(self):
        d = int(self.p.rebalance[20])
        nodes = mdl.node_set(self.p, d, max_nodes=3000, require_label=True)
        x0 = mdl.date_features(self.p, d, nodes)
        for name in ("ret1", "ldv", "dvshock", "vol20", "pressure", "elig"):
            self.p.a[name][d + 1:] = 123.0  # scramble the future
        x1 = mdl.date_features(self.p, d, nodes)
        np.testing.assert_array_equal(x0, x1)
        self.assertEqual(x0.shape, (len(nodes), 2 * len(mdl.FEATURES)))

    def test_rank_gauss(self):
        v = np.array([3.0, np.nan, 1.0, 2.0, 10.0])
        g = mdl.rank_gauss(v)
        self.assertTrue(np.isnan(g[1]))
        self.assertTrue(g[2] < g[3] < g[0] < g[4])
        self.assertAlmostEqual(float(np.nansum(g)), 0.0, places=5)

    def test_node_set_requires_active(self):
        d = int(self.p.rebalance[20])
        base = mdl.node_set(self.p, d, max_nodes=3000, require_label=True)
        self.p.a["pressure"][d, base[:3]] = np.nan  # inactive at d
        got = mdl.node_set(self.p, d, max_nodes=3000, require_label=False)
        self.assertFalse(set(base[:3]) & set(got))
        self.assertEqual(set(got) & set(base), set(base[3:]))
        self.p.a["active"] = np.ones_like(self.p.a["elig"])  # an exported active mask takes precedence
        self.p.a["active"][d, base[3]] = 0
        got = mdl.node_set(self.p, d, max_nodes=3000, require_label=True)
        self.assertEqual(list(got), list(base[:3]) + list(base[4:]))

    def test_active_array_is_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            arrays = {name: np.ones((3, 2), np.float32) for name in pnl.FIELDS}
            pnl.write_panel(d, ["A", "B"], ["X", "X"], [1, 2, 3], [0], arrays)
            self.assertNotIn("active", pnl.load_panel(d).a)
            np.array([[1, 0]] * 3, dtype="<f4").tofile(os.path.join(d, "active.f32"))
            np.testing.assert_array_equal(pnl.load_panel(d).a["active"][:, 1], 0)

    def test_node_cap_by_ldv(self):
        d = int(self.p.rebalance[20])
        nodes = mdl.node_set(self.p, d, max_nodes=10, require_label=True)
        self.assertEqual(len(nodes), 10)
        elig = np.where((self.p.a["elig"][d] == 1) & np.isfinite(self.p.a["label_w"][d])
                        & np.isfinite(self.p.a["pressure"][d]))[0]
        ldv = self.p.a["ldv"][d]
        self.assertGreaterEqual(ldv[nodes].min(), np.sort(ldv[elig])[-10])


class Embargo(unittest.TestCase):
    def test_no_label_window_overlaps_predictions(self):
        reb = np.arange(4, 1500, 5)
        for every in (1, 4, 13):
            for embargo in (0, 1, 2):
                blocks = train_wf.plan_retrains(reb, min_history=156, retrain_every=every,
                                                embargo=embargo, horizon=5)
                self.assertTrue(blocks)
                self.assertEqual(blocks[0]["pred"][0], 156)
                predicted = [m for b in blocks for m in b["pred"]]
                self.assertEqual(predicted, list(range(156, len(reb))))
                for b in blocks:
                    first_pred_bar = reb[b["pred"][0]]
                    self.assertTrue(b["train"])
                    for k in b["train"]:
                        self.assertNotIn(k, b["pred"])
                        # label_w(d_k) = open[d_k+6]/open[d_k+1]-1 is realized at bar d_k+6, which must be at
                        # or before every predicted date, plus the embargo
                        self.assertLessEqual(reb[k] + 6 + embargo * 5, first_pred_bar)
                        for m in b["pred"]:
                            self.assertLessEqual(reb[k] + 6, reb[m])
                    # nothing usable was left out
                    usable = [k for k in range(b["pred"][0]) if reb[k] + 6 + 5 * embargo <= first_pred_bar]
                    self.assertEqual(b["train"], usable)

    def test_irregular_calendar(self):
        reb = np.array([0, 3, 9, 10, 11, 20, 26, 27, 33, 40])
        blocks = train_wf.plan_retrains(reb, min_history=4, retrain_every=3, embargo=1, horizon=5)
        self.assertEqual(blocks[0]["pred"], [4, 5, 6])
        self.assertEqual(blocks[0]["train"], [0])  # 0+11<=11; 3+11>11
        self.assertEqual(blocks[1]["pred"], [7, 8, 9])
        self.assertEqual(blocks[1]["train"], [0, 1, 2, 3, 4])  # 11+11<=27; 20+11>27


class B1FixedGraph(unittest.TestCase):
    def test_as_of_lookup_and_row_normalization(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.csv")
            with open(path, "w") as f:
                f.write("t,src,dst,w\n100,A,B,1\n100,A,C,3\n100,B,A,2\n100,A,Z,5\n200,C,A,1\n")
            lk = train_wf.B1Lookup(path)
            self.assertIsNone(lk(99, ["A", "B", "C"]))
            rows, cols, w = lk(150, ["A", "B", "C"])  # snapshot 100; Z not in the node set
            got = sorted(zip(rows.tolist(), cols.tolist(), [round(x, 6) for x in w.tolist()]))
            self.assertEqual(got, [(0, 1, 0.25), (0, 2, 0.75), (1, 0, 1.0)])
            rows, cols, w = lk(10_000, ["C", "A"])
            self.assertEqual((rows.tolist(), cols.tolist(), w.tolist()), ([0], [1], [1.0]))

    def test_b1_variant_runs(self):
        with tempfile.TemporaryDirectory() as d:
            synth.make_planted_panel(os.path.join(d, "panel"), n=30, t=300, seed=2)
            with open(os.path.join(d, "p.csv"), "w") as f:
                f.write("t,src,dst,w\n0," + "\n0,".join(f"S{i:04d},S{(i + 1) % 30:04d},1" for i in range(30)) + "\n")
            run = train_wf.main(["--panel", os.path.join(d, "panel"), "--out", os.path.join(d, "out"),
                                 "--store", os.path.join(d, "store"), "--variants", "B1,B0",
                                 "--b1-edges", os.path.join(d, "p.csv"), "--min-history", "30",
                                 "--retrain-every", "20", "--epochs", "2", "--threads", "2", "--quiet"])
            self.assertIn("B1", run["oos_ic"])
            self.assertTrue(os.path.exists(os.path.join(d, "out", "B1.csv")))


class Determinism(unittest.TestCase):
    def test_same_seed_identical_csvs(self):
        with tempfile.TemporaryDirectory() as d:
            synth.make_planted_panel(os.path.join(d, "panel"), n=40, t=400, seed=5)
            outs = []
            for run in ("a", "b"):
                out = os.path.join(d, run)
                train_wf.main(["--panel", os.path.join(d, "panel"), "--out", out,
                               "--variants", "learned,B0,B2", "--retrain-every", "40",
                               "--min-history", "30", "--epochs", "3", "--seed", "7",
                               "--threads", "2", "--quiet", "--store", os.path.join(d, "store_" + run)])
                outs.append(out)
            for v in ("learned", "B0", "B2"):
                with open(os.path.join(outs[0], v + ".csv")) as f0, open(os.path.join(outs[1], v + ".csv")) as f1:
                    a, b = f0.read(), f1.read()
                self.assertEqual(a.splitlines()[0], "t,ticker,score")
                self.assertGreater(len(a.splitlines()), 10)
                self.assertEqual(a, b)
            self.assertTrue(os.path.exists(os.path.join(outs[0], "run.json")))
            self.assertTrue(os.path.exists(os.path.join(outs[0], "learned_edges_0.csv")))


if __name__ == "__main__":
    unittest.main()
