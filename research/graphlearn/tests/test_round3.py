"""Round-3 optimization fixes: signed message, IC early stopping with min epochs, restarts."""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model as mdl  # noqa: E402
import synth  # noqa: E402
import train_wf  # noqa: E402
import power_curve as pc  # noqa: E402


def tiny_batch(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return mdl.Batch(d=0, t=0, nodes=np.arange(n), tickers=[f"T{i}" for i in range(n)],
                     x=torch.randn(n, mdl.N_FEATURES, generator=g), y=torch.randn(n, generator=g),
                     sector=torch.arange(n) % 3)


class SignedMessage(unittest.TestCase):
    def model(self):
        torch.manual_seed(0)
        m = mdl.GraphNet("learned", rank=4, topk=3, signed=True)
        m.ensure_tickers([f"T{i}" for i in range(12)])
        return m

    def test_signed_weights_topk_by_magnitude(self):
        m = self.model()
        with torch.no_grad():
            m.E_s.abs_()
            m.E_d.copy_(-m.E_d.abs() - 0.01)  # every logit negative: relu would kill all edges
        rows = m.rows_for(m.tickers)
        S = m.scores(rows)
        self.assertTrue(torch.all(S[~torch.eye(12, dtype=torch.bool)] < 0))
        idx, w, pen = m.adjacency(rows)
        self.assertTrue(torch.all(w < 0))
        torch.testing.assert_close(w.abs().sum(1), torch.ones(12))
        mag = S.abs().masked_fill(torch.eye(12, dtype=torch.bool), -1)
        torch.testing.assert_close(mag.gather(1, idx), torch.topk(mag, 3, dim=1).values)
        self.assertAlmostEqual(float(pen), float(S.abs()[~torch.eye(12, dtype=torch.bool)].mean()), places=5)
        m.train()
        mdl.date_loss(m, tiny_batch(), mdl.Reg()).backward()
        self.assertTrue(torch.all(m.E_s.grad.abs().sum(1) > 0))  # no dead rows

    def test_signed_in_params_hash_and_model(self):
        a = train_wf.parse_args(["--panel", "x", "--out", "y", "--signed-message", "--stop-on", "ic",
                                 "--min-epochs", "7", "--restarts", "3"])
        p = train_wf.params_for(a, "learned")
        self.assertEqual((p["signed_message"], p["stop_on"], p["min_epochs"], p["restarts"]), (True, "ic", 7, 3))
        self.assertTrue(train_wf._new_model(a, "learned").signed)


class ICEarlyStopping(unittest.TestCase):
    def test_min_epochs_and_ic_metric(self):
        torch.manual_seed(0)
        m = mdl.GraphNet("B0", rank=4)
        opt = mdl.make_optimizer(m)
        train, val = [tiny_batch(seed=s) for s in range(3)], [tiny_batch(seed=9)]
        h = mdl.fit(m, opt, train, val, 20, mdl.Reg(), patience=1, seed=0, stop_on="ic", min_epochs=6)
        self.assertGreaterEqual(len(h["train_loss"]), 6)
        self.assertEqual(len(h["val_ic"]), len(h["train_loss"]))
        best = h["best_epoch"]
        self.assertEqual(h["val_ic"][best], max(h["val_ic"]))
        # the restored model reproduces the best epoch's validation IC
        self.assertAlmostEqual(mdl.evaluate_ic(m, val), h["val_ic"][best], places=6)


class Restarts(unittest.TestCase):
    def test_scratch_fit_picks_best_validation_ic(self):
        with tempfile.TemporaryDirectory() as d:
            synth.make_planted_panel(os.path.join(d, "panel"), n=30, t=300, seed=2, beta=0.8)
            run = train_wf.main(["--panel", os.path.join(d, "panel"), "--out", os.path.join(d, "out"),
                                 "--store", os.path.join(d, "store"), "--variants", "learned,B0",
                                 "--min-history", "30", "--retrain-every", "20", "--epochs", "3",
                                 "--min-epochs", "2", "--restarts", "3", "--stop-on", "ic", "--mode", "scratch",
                                 "--threads", "2", "--quiet"])
            for v in ("learned", "B0"):
                for h in run["variants"][v]["history"]:
                    ics = h["restart_val_ic"]
                    self.assertEqual(len(ics), 3)
                    self.assertEqual(h["restart_chosen"], int(np.nanargmax(ics)))
            # rolling: only the first fit restarts
            run = train_wf.main(["--panel", os.path.join(d, "panel"), "--out", os.path.join(d, "out2"),
                                 "--store", os.path.join(d, "store2"), "--variants", "learned",
                                 "--min-history", "30", "--retrain-every", "20", "--epochs", "3",
                                 "--min-epochs", "2", "--restarts", "2", "--stop-on", "ic", "--threads", "2",
                                 "--quiet"])
            hist = run["variants"]["learned"]["history"]
            self.assertEqual(len(hist[0]["restart_val_ic"]), 2)
            self.assertTrue(all(h["restart_val_ic"] is None for h in hist[1:]))
            self.assertTrue(all("val_ic" in h for h in hist))


class Round3Selection(unittest.TestCase):
    def res(self, setting, beta, seed, mode, t0, te, null=False):
        m = {"n": 300, "beta": beta, "seed": seed}
        if null:
            m["null"] = True
        return {"setting": setting, "market": m, "mode": mode, "params": pc.R3_SETTINGS_BY_NAME[setting],
                "minus_B0": {"t": t0}, "minus_B0E": {"t": te}}

    def grid(self, setting, detect, null_t=0.0):
        out = []
        for mode in ("rolling", "scratch"):
            for seed in pc.R3_TUNE_SEEDS:
                for beta in pc.R3_BETAS:
                    t = 5.0 if (beta, seed, mode) in detect else 0.5
                    out.append(self.res(setting, beta, seed, mode, t, t))
                out.append(self.res(setting, 0.0, seed, mode, null_t, null_t, null=True))
        return out

    def test_rule(self):
        names = [n for n, _ in pc.R3_SETTINGS]
        allpts = {(b, s, m) for b in pc.R3_BETAS for s in pc.R3_TUNE_SEEDS for m in ("rolling", "scratch")}
        miss_low = allpts - {(0.3, pc.R3_TUNE_SEEDS[0], "rolling"), (0.3, pc.R3_TUNE_SEEDS[1], "rolling")}
        miss_high = allpts - {(0.6, pc.R3_TUNE_SEEDS[0], "rolling"), (0.6, pc.R3_TUNE_SEEDS[1], "rolling")}
        results = (self.grid(names[0], allpts, null_t=2.5)        # most power but a null fails
                   + self.grid(names[1], miss_low)                # 10/12, misses at beta 0.3
                   + self.grid(names[2], miss_high)               # 10/12, misses at beta 0.6 -> wins
                   + self.grid(names[3], allpts - set(list(allpts)[:3])))  # 9/12 -> fails the target
        best, table = pc.select_round3(results)
        self.assertEqual(best, names[2])
        rows = {r["setting"]: r for r in table}
        self.assertFalse(rows[names[0]]["passes"])
        self.assertFalse(rows[names[3]]["passes"])

    def test_none_passing(self):
        names = [n for n, _ in pc.R3_SETTINGS]
        self.assertIsNone(pc.select_round3(self.grid(names[0], set()))[0])


if __name__ == "__main__":
    unittest.main()
