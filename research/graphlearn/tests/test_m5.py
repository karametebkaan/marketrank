"""M5: intraday reader, label mask, prior-support softmax, Bprior, planted correction, nulls, determinism/resume."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import intraday as itd  # noqa: E402
import power_m5  # noqa: E402
import prior_model as pm  # noqa: E402
import synth_intraday as si  # noqa: E402
import wf_m5  # noqa: E402

SMALL = ["--min-history", "20", "--retrain-every", "10", "--epochs", "12", "--patience", "4", "--min-epochs", "3",
         "--restarts", "1", "--finetune-epochs", "2", "--ft-holdout", "4", "--threads", "2", "--quiet"]


def tiny_panel(d, n=5, sessions=3, bars=26, with_prior=True, seed=0):
    rng = np.random.default_rng(seed)
    T = sessions * bars
    arrays = {k: rng.standard_normal((T, n)).astype(np.float32) for k in itd.FIELDS}
    arrays["elig"][:] = 1
    arrays["active"][:] = 1
    session = np.repeat(np.arange(sessions), bars)
    times = 1_700_000_000 + np.arange(T) * 900
    prior = None
    if with_prior:
        prior = {0: (np.array([0, 0, 1]), np.array([1, 2, 0]), np.array([0.25, 0.75, 1.0]), np.array([1., 3., 2.])),
                 30: (np.array([2, 3]), np.array([4, 4]), np.array([1.0, 1.0]), np.array([5., 6.]))}
    itd.write_intraday(d, [f"T{i}" for i in range(n)], [f"S{i % 2}" for i in range(n)], times, session, arrays,
                       prior=prior)
    return arrays, session, prior


class Reader(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            arrays, session, prior = tiny_panel(d)
            p = itd.load_intraday(d)
            self.assertEqual((p.N, p.T), (5, 78))
            for k in itd.FIELDS:
                np.testing.assert_array_equal(np.asarray(p.a[k]), arrays[k])
            np.testing.assert_array_equal(p.session, session)
            self.assertEqual(p.sessions(), [(0, 0, 26), (1, 26, 52), (2, 52, 78)])
            src, dst, P, raw = p.prior_at(0)
            np.testing.assert_array_equal(src, [0, 0, 1])
            np.testing.assert_array_equal(dst, [1, 2, 0])
            np.testing.assert_allclose(P, [0.25, 0.75, 1.0])
            np.testing.assert_allclose(raw, [1, 3, 2])
            # as-of within the session: bar 25 (session 0) sees bar 0's prior; bar 26 (session 1) sees none until 30
            np.testing.assert_array_equal(p.prior_at(25)[0], [0, 0, 1])
            self.assertEqual(len(p.prior_at(26)[0]), 0)
            self.assertEqual(len(p.prior_at(29)[0]), 0)
            np.testing.assert_array_equal(p.prior_at(31)[1], [4, 4])
            self.assertEqual(len(p.prior_at(52)[0]), 0)  # never across the close
            raw_rec = np.fromfile(os.path.join(d, "prior", "edges.bin"), dtype=itd.EDGE_DTYPE)
            self.assertEqual(itd.EDGE_DTYPE.itemsize, 20)
            np.testing.assert_array_equal(raw_rec["t"], [0, 0, 0, 30, 30])
            off = np.fromfile(os.path.join(d, "prior", "offsets.bin"), dtype="<u8")
            self.assertEqual((len(off), int(off[1]), int(off[30]), int(off[31]), int(off[-1])), (79, 3, 3, 5, 5))

    def test_reads_the_cpp_export_layout(self):
        """The C++ --export-panel writes session as an object and the prior files under prior.dir."""
        with tempfile.TemporaryDirectory() as d:
            arrays, _, _ = tiny_panel(d)
            itd.write_intraday(d, [f"T{i}" for i in range(5)], [f"S{i % 2}" for i in range(5)],
                               1_700_000_000 + np.arange(78) * 900, np.repeat(np.arange(3), 26), arrays,
                               prior={0: (np.array([0, 1]), np.array([1, 0]), np.array([1., 1.]), np.array([1., 1.]))})
            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            os.rename(os.path.join(d, "prior", "offsets.bin"), os.path.join(d, "prior", "offsets.u64"))
            meta["session"] = {"count": 3, "dates": ["a", "b", "c"], "index": meta["session"],
                               "slot": [k % 26 for k in range(78)]}
            meta["prior"] = {"dir": "prior", "edges": "edges.bin", "offsets": "offsets.u64", "count": 2}
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(meta, f)
            p = itd.load_intraday(d)
            np.testing.assert_array_equal(p.session, np.repeat(np.arange(3), 26))
            self.assertEqual(len(p.prior_at(5)[0]), 2)

    def test_truncate(self):
        with tempfile.TemporaryDirectory() as d:
            tiny_panel(os.path.join(d, "a"))
            itd.truncate(os.path.join(d, "a"), os.path.join(d, "b"), 2)
            p = itd.load_intraday(os.path.join(d, "b"))
            self.assertEqual(p.T, 52)
            np.testing.assert_array_equal(p.prior_at(31)[1], [4, 4])

    def test_bad_sizes_refused(self):
        with tempfile.TemporaryDirectory() as d:
            tiny_panel(d)
            np.zeros(3, np.float32).tofile(os.path.join(d, "ret1.f32"))
            with self.assertRaises(ValueError):
                itd.load_intraday(d)


class LabelMask(unittest.TestCase):
    def test_mask_never_crosses_the_close(self):
        session = np.repeat(np.arange(3), 26)
        m = itd.label_mask(session, 6)
        # bar t needs bars t+1..t+7 in its session: t = 0..18 of each 26-bar session
        np.testing.assert_array_equal(np.flatnonzero(m[:26]), np.arange(19))
        self.assertEqual(int(m.sum()), 3 * 19)
        self.assertFalse(m[-7:].any())
        self.assertTrue(m[-8])
        # a half day (13 bars) has 6 labeled bars
        m2 = itd.label_mask(np.r_[np.zeros(13, int), np.ones(26, int)], 6)
        self.assertEqual(int(m2[:13].sum()), 6)

    def test_batches_use_masked_labels_only(self):
        with tempfile.TemporaryDirectory() as d:
            tiny_panel(d)  # label_6 is finite at EVERY bar in the file: the mask must still drop bars 19..25
            p = itd.load_intraday(d)
            fb = itd.FeatureBuilder(p)
            b = fb.session_batch(1, np.arange(26, 52))
            labeled_bars = np.unique(b.node_bar.numpy()[b.ymask.numpy()])
            np.testing.assert_array_equal(labeled_bars, np.arange(19))
            self.assertTrue(np.isnan(b.y.numpy()[~b.ymask.numpy()]).all())

    def test_synth_label_is_the_masked_six_bar_sum(self):
        with tempfile.TemporaryDirectory() as d:
            si.make_market(d, n=30, sessions=6, target_ic=0.05, seed=3, degree=5)
            p = itd.load_intraday(d)
            r, lab = np.asarray(p.a["ret1"], float), np.asarray(p.a["label_6"], float)
            m = itd.label_mask(p.session)
            t = np.flatnonzero(m)
            np.testing.assert_allclose(lab[t], np.stack([r[k + 1: k + 7].sum(0) for k in t]), rtol=1e-4, atol=1e-6)
            self.assertTrue(np.isnan(lab[~m]).all())


def edge_batch(n_bars=2, n=6, seed=0):
    rng = np.random.default_rng(seed)
    nb = np.repeat(np.arange(n_bars), n)
    rows, cols, lps = [], [], []
    for k in range(n_bars):
        for i in range(n):
            js = rng.choice([j for j in range(n) if j != i], size=3, replace=False)
            P = rng.random(3) + 0.1
            rows += [k * n + i] * 3
            cols += list(k * n + js)
            lps += list(np.log(P / P.sum()))
    g = torch.Generator().manual_seed(seed)
    b = itd.SessionBatch(session=0, bars=np.arange(n_bars), times=np.arange(n_bars), n_bars=n_bars,
                         node_bar=torch.from_numpy(nb), node_col=np.tile(np.arange(n), n_bars),
                         x=torch.randn(n_bars * n, itd.N_NODE_FEATURES, generator=g),
                         y=torch.randn(n_bars * n, generator=g), ymask=torch.ones(n_bars * n, dtype=torch.bool),
                         e_row=torch.tensor(rows), e_col=torch.tensor(cols),
                         e_logp=torch.tensor(lps, dtype=torch.float32),
                         e_feat=torch.randn(len(rows), itd.N_PAIR_FEATURES, generator=g))
    return pm.prepare(b)


class PriorSoftmax(unittest.TestCase):
    def test_softmax_over_prior_support(self):
        torch.manual_seed(0)
        b = edge_batch()
        m = pm.PriorNet("learned", [f"T{i}" for i in range(6)], rank=4, g_hidden=8)
        with torch.no_grad():  # make the correction non-trivial
            for prm in m.g.parameters():
                prm.copy_(torch.randn_like(prm))
            m.U.copy_(torch.randn_like(m.U))
            m.V.copy_(torch.randn_like(m.V))
        w, _ = m.edge_weights(b)
        n = b.x.shape[0]
        logit = (b.e_logp + m.g(b.e_feat)[:, 0] + (m.U[b.node_col_t[b.e_row]] * m.V[b.node_col_t[b.e_col]]).sum(1))
        dense = torch.full((n, n), float("-inf"))
        dense[b.e_row, b.e_col] = logit.detach()
        ref = torch.softmax(dense, dim=1)
        A = torch.zeros(n, n)
        A[b.e_row, b.e_col] = w.detach()
        torch.testing.assert_close(A, ref)  # zero off the support, row-softmax on it
        torch.testing.assert_close(A.sum(1), torch.ones(n))
        self.assertTrue(torch.all((A > 0) == (dense > float("-inf"))))

    def test_learned_starts_at_the_prior(self):
        b = edge_batch()
        m = pm.PriorNet("learned", [f"T{i}" for i in range(6)], rank=4)
        w, corr = m.edge_weights(b)
        torch.testing.assert_close(w.detach(), torch.exp(b.e_logp), atol=2e-3, rtol=2e-2)

    def test_signed_variant_can_go_negative(self):
        b = edge_batch()
        m = pm.PriorNet("learned", [f"T{i}" for i in range(6)], rank=4, signed=True)
        w, _ = m.edge_weights(b)
        self.assertTrue(torch.all(w > 0))
        with torch.no_grad():
            m.sign_bias.fill_(-3.0)
        w, _ = m.edge_weights(b)
        self.assertTrue(torch.all(w < 0))
        m.train()
        pm.session_loss(m, b, pm.Reg()).backward()
        self.assertTrue(m.Us.grad.abs().sum() > 0 and m.g[2].weight.grad.abs().sum() > 0)


class Bprior(unittest.TestCase):
    def test_bprior_message_is_the_fixed_prior(self):
        b = edge_batch()
        m = pm.PriorNet("Bprior", [f"T{i}" for i in range(6)])
        m.eval()
        captured = {}
        m.head.register_forward_hook(lambda mod, inp, out: captured.setdefault("z", inp[0][:, 32:]))
        m(b)
        n = b.x.shape[0]
        P = torch.zeros(n, n)
        P[b.e_row, b.e_col] = torch.exp(b.e_logp)
        with torch.no_grad():
            ref = P @ m.msg(m.enc(b.x))
        torch.testing.assert_close(captured["z"], ref)
        self.assertEqual(sum(p.numel() for n_, p in m.named_parameters() if n_.startswith("g")), 0)

    def test_renormalized_prior_over_present_nodes(self):
        with tempfile.TemporaryDirectory() as d:
            arrays, _, _ = tiny_panel(d)
            arrays["elig"][5, 2] = 0  # node 2 absent at bar 5: row 0 keeps only 0->1, renormalized to 1
            itd.write_intraday(d, [f"T{i}" for i in range(5)], [f"S{i % 2}" for i in range(5)],
                               1_700_000_000 + np.arange(78) * 900, np.repeat(np.arange(3), 26), arrays,
                               prior={0: (np.array([0, 0, 1]), np.array([1, 2, 0]), np.array([0.25, 0.75, 1.]),
                                          np.array([1., 3., 2.]))})
            fb = itd.FeatureBuilder(itd.load_intraday(d))
            nodes = fb.nodes(5)
            r, c, lp, f = fb.edges(5, nodes)
            self.assertEqual(sorted(zip(nodes[r].tolist(), nodes[c].tolist())), [(0, 1), (1, 0)])
            np.testing.assert_allclose(np.exp(lp), [1.0, 1.0])
            nodes = fb.nodes(6)
            r, c, lp, f = fb.edges(6, nodes)
            np.testing.assert_allclose(sorted(np.exp(lp)), [0.25, 0.75, 1.0], rtol=1e-6)
            self.assertEqual(f.shape[1], itd.N_PAIR_FEATURES)
            # same-sector flag: T0 (S0) -> T2 (S0) is 1, T0 -> T1 is 0
            same = {(int(nodes[a]), int(nodes[b_])): f[k, 2] for k, (a, b_) in enumerate(zip(r, c))}
            self.assertEqual((same[(0, 2)], same[(0, 1)]), (1.0, 0.0))


class Bshuf(unittest.TestCase):
    def test_relabel_moves_dst_and_keeps_weights(self):
        with tempfile.TemporaryDirectory() as d:
            arrays, _, _ = tiny_panel(d)
            itd.write_intraday(d, [f"T{i}" for i in range(5)], [f"S{i % 2}" for i in range(5)],
                               1_700_000_000 + np.arange(78) * 900, np.repeat(np.arange(3), 26), arrays,
                               prior={0: (np.array([0, 0, 1]), np.array([1, 2, 0]), np.array([0.25, 0.75, 1.]),
                                          np.array([1., 3., 2.]))})
            p = itd.load_intraday(d)
            perm = np.array([0, 3, 4, 1, 2])  # 1 -> 3, 2 -> 4, 0 -> 0
            fb, fs = itd.FeatureBuilder(p), itd.FeatureBuilder(p, relabel=perm)
            nodes = fb.nodes(6)
            r, c, lp, _ = fb.edges(6, nodes)
            rs, cs, lps, _ = fs.edges(6, nodes)
            self.assertEqual(sorted(zip(nodes[rs].tolist(), nodes[cs].tolist())), [(0, 3), (0, 4), (1, 0)])
            np.testing.assert_allclose(sorted(np.exp(lps)), sorted(np.exp(lp)), rtol=1e-6)
            with self.assertRaises(ValueError):
                itd.FeatureBuilder(p, relabel=np.array([0, 0, 1, 2, 3]))

    def test_shuffle_perm_is_fixed(self):
        np.testing.assert_array_equal(itd.shuffle_perm(50, 7), itd.shuffle_perm(50, 7))
        self.assertEqual(sorted(itd.shuffle_perm(50, 7).tolist()), list(range(50)))


def run_wf(panel, out, extra=()):
    return wf_m5.main(["--panel", panel, "--out", out, "--store", os.path.join(out, "store"), *SMALL, *extra])


class Synthetic(unittest.TestCase):
    """Small synthetic markets (N=80, 60 sessions, degree 8). The planted market uses a strong signal."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = cls.tmp.name
        cls.info = {}
        for name, kind, ic in (("planted", "planted", 0.30), ("null_a", "null_a", 0.0), ("null_b", "null_b", 0.10)):
            cls.info[name] = si.make_market(os.path.join(cls.d, name), n=80, sessions=60, target_ic=ic, kind=kind,
                                            seed=11, degree=8)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_calibration(self):
        self.assertAlmostEqual(self.info["planted"]["oracle_ic"], 0.30, delta=0.03)
        self.assertAlmostEqual(self.info["null_b"]["oracle_ic"], 0.10, delta=0.02)
        self.assertEqual(self.info["null_a"]["beta"], 0.0)
        self.assertGreater(self.info["planted"]["correction_ic"], 0.05)

    def test_planted_correction_recovered(self):
        out = os.path.join(self.d, "run_planted")
        run = run_wf(os.path.join(self.d, "planted"), out, ["--signed", "--variants", "learned,Bprior,B0E"])
        ic = run["oos_ic"]["learned"]
        print(f"\nplanted: learned {ic['mean']:+.4f}, -Bprior {ic['minus_Bprior']['mean']:+.4f} "
              f"(t={ic['minus_Bprior']['t']:.1f}), -B0E {ic['minus_B0E']['mean']:+.4f} (t={ic['minus_B0E']['t']:.1f})")
        self.assertGreater(ic["minus_Bprior"]["t"], 2)
        self.assertGreater(ic["minus_B0E"]["t"], 2)
        # the learned chain moved toward the planted weights: corr(learned logit correction, planted delta) > 0
        m, _ = wf_m5.load_model(os.path.join(out, "store", "learned"))
        r = power_m5.correction_recovery(m, os.path.join(self.d, "planted"))
        print(f"planted: corr(learned correction, planted delta) = {r:+.3f}")
        self.assertGreater(r, 0.2)

    def test_null_a_nothing_predictive(self):
        run = run_wf(os.path.join(self.d, "null_a"), os.path.join(self.d, "run_a"),
                     ["--variants", "learned,Bprior,B0E"])
        ic = run["oos_ic"]["learned"]
        print(f"\nnull a: -Bprior t={ic['minus_Bprior']['t']:.2f} -B0E t={ic['minus_B0E']['t']:.2f}")
        self.assertLess(abs(ic["minus_Bprior"]["t"]), 2)
        self.assertLess(abs(ic["minus_B0E"]["t"]), 2)

    def test_null_b_prior_only(self):
        run = run_wf(os.path.join(self.d, "null_b"), os.path.join(self.d, "run_b"),
                     ["--variants", "learned,Bprior,B0E"])
        ic = run["oos_ic"]
        print(f"\nnull b: learned-Bprior t={ic['learned']['minus_Bprior']['t']:.2f}, "
              f"Bprior-B0E t={ic['Bprior']['minus_B0E']['t']:.2f}")
        self.assertLess(abs(ic["learned"]["minus_Bprior"]["t"]), 2)  # learned must NOT beat Bprior
        self.assertGreater(ic["Bprior"]["minus_B0E"]["t"], 2)  # while the prior itself is predictive

    def test_null_b_bprior_beats_bshuf(self):
        """Where the prior is the truth, the right tickers matter: Bprior beats the relabeled chain."""
        run = run_wf(os.path.join(self.d, "null_b"), os.path.join(self.d, "run_b_shuf"),
                     ["--variants", "Bprior,Bshuf,B0E"])
        ic = run["oos_ic"]
        print(f"\nnull b: Bprior-Bshuf t={ic['Bprior']['minus_Bshuf']['t']:.2f}, "
              f"Bshuf-B0E t={ic['Bshuf']['minus_B0E']['t']:.2f}")
        self.assertGreater(ic["Bprior"]["minus_Bshuf"]["t"], 2)

    def test_null_a_bprior_does_not_beat_bshuf(self):
        run = run_wf(os.path.join(self.d, "null_a"), os.path.join(self.d, "run_a_shuf"),
                     ["--variants", "Bprior,Bshuf"])
        t = run["oos_ic"]["Bprior"]["minus_Bshuf"]["t"]
        print(f"\nnull a: Bprior-Bshuf t={t:.2f}")
        self.assertLess(abs(t), 2)


class DeterminismResume(unittest.TestCase):
    def test_determinism_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            full = os.path.join(d, "full")
            si.make_market(full, n=40, sessions=45, target_ic=0.2, seed=5, degree=6)
            part = os.path.join(d, "part")
            itd.truncate(full, part, 35)
            ex = ["--variants", "learned,B0E", "--signed"]
            a = run_wf(full, os.path.join(d, "a"), ex)
            b = run_wf(full, os.path.join(d, "b"), ex)
            pa = {v: wf_m5.Store(os.path.join(d, "a", "store", v)).all_predictions() for v in ("learned", "B0E")}
            pb = {v: wf_m5.Store(os.path.join(d, "b", "store", v)).all_predictions() for v in ("learned", "B0E")}
            self.assertEqual(pa, pb)  # determinism
            # resume: run on the first 35 sessions, then on the full export with the same store
            run_wf(part, os.path.join(d, "c"), ex)
            c = run_wf(full, os.path.join(d, "c"), ex)
            self.assertEqual(c["variants"]["learned"]["resumed_from_block"], 1)
            self.assertEqual(c["variants"]["learned"]["retrained_blocks"], [2])
            pc_ = {v: wf_m5.Store(os.path.join(d, "c", "store", v)).all_predictions() for v in ("learned", "B0E")}
            for v in pa:
                np.testing.assert_array_equal(np.array(pc_[v]["t"]), np.array(pa[v]["t"]))
                np.testing.assert_allclose(np.array(pc_[v]["score"]), np.array(pa[v]["score"]), rtol=1e-6,
                                           atol=1e-7)
            self.assertEqual(json.dumps(a["oos_ic"], sort_keys=True), json.dumps(b["oos_ic"], sort_keys=True))
            # a params change refuses to resume
            with self.assertRaises(SystemExit):
                run_wf(full, os.path.join(d, "c"), ["--variants", "learned", "--rank", "3"])


class Selection(unittest.TestCase):
    def pt(self, cfg, kind, ic, ms, mode, seed, t_bp, t_be, params=None):
        return {"config": cfg, "params": params or {}, "market": {"kind": kind, "target_ic": ic, "seed": ms, "n": 300},
                "mode": mode, "seed": seed, "t": {"Bprior": t_bp, "B0E": t_be}}

    def grid(self, cfg, det05, det03=0, null_t=0.5, params=None):
        out, k5, k3 = [], 0, 0
        for ms in power_m5.TUNE_MARKET_SEEDS:
            for mode in ("rolling", "scratch"):
                for seed in power_m5.MODEL_SEEDS:
                    out.append(self.pt(cfg, "planted", 0.05, ms, mode, seed, *((3, 3) if k5 < det05 else (1, 3)),
                                       params))
                    k5 += 1
                    out.append(self.pt(cfg, "planted", 0.03, ms, mode, seed, *((3, 3) if k3 < det03 else (1, 1)),
                                       params))
                    k3 += 1
                    out.append(self.pt(cfg, "planted", 0.02, ms, mode, seed, 0, 0, params))
            for kind in ("null_a", "null_b"):
                for mode in ("rolling", "scratch"):
                    out.append(self.pt(cfg, kind, 0.05 if kind == "null_b" else 0.0, ms, mode,
                                       power_m5.MODEL_SEEDS[0], null_t, null_t if kind == "null_a" else 5, params))
        return out

    def test_rule(self):
        res = (self.grid("A", 10, det03=4, params={"n_params": 100}) + self.grid("B", 12, det03=4,
               params={"n_params": 50}) + self.grid("C", 9) + self.grid("D", 12, null_t=2.5))
        best, table = power_m5.select(res)
        rows = {r["config"]: r for r in table}
        self.assertTrue(rows["A"]["passes"] and rows["B"]["passes"])
        self.assertFalse(rows["C"]["passes"])  # 9 < 10 detections at 0.05
        self.assertFalse(rows["D"]["passes"])  # a null with |t| >= 2
        # A and B tie at 0.02 (0) and 0.03 (4); B has more at 0.05 -> B
        self.assertEqual(best, "B")
        best2, _ = power_m5.select(self.grid("A", 10, det03=4, params={"n_params": 100})
                                   + self.grid("E", 10, det03=4, params={"n_params": 50}))
        self.assertEqual(best2, "E")  # full tie -> fewer params
        self.assertIsNone(power_m5.select(self.grid("C", 9))[0])


if __name__ == "__main__":
    unittest.main()
