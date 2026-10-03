"""Persistence: resume equivalence, params-mismatch refusal, atomic writes, new tickers."""
import glob
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import latest_graph  # noqa: E402
import model as mdl  # noqa: E402
import panel as pnl  # noqa: E402
import store as st  # noqa: E402
import synth  # noqa: E402
import train_wf  # noqa: E402

# 120 rebalances; blocks start at positions 30, 50, 70, 90, 110
COMMON = ["--min-history", "30", "--retrain-every", "20", "--epochs", "3", "--finetune-epochs", "2",
          "--threads", "2", "--quiet", "--seed", "3"]


def run(panel_dir, out, store, variants="learned,B0", extra=()):
    return train_wf.main(["--panel", panel_dir, "--out", out, "--store", store, "--variants", variants,
                          *COMMON, *extra])


def read(path):
    with open(path) as f:
        return f.read()


def write_text(path, text):
    with open(path, "w") as f:
        f.write(text)


class Persistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = cls.tmp.name
        cls.full = os.path.join(cls.d, "full")
        synth.make_planted_panel(cls.full, n=40, t=600, seed=11)
        cls.part = os.path.join(cls.d, "part")
        synth.truncate_panel(cls.full, cls.part, 330)  # ends inside block 1 (positions 50..69)
        # reference: everything in one go
        cls.ref = run(cls.full, os.path.join(cls.d, "ref_out"), os.path.join(cls.d, "ref_store"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def sub(self, name):
        return os.path.join(self.d, name)

    def test_resume_equivalence(self):
        store = self.sub("res_store")
        r1 = run(self.part, self.sub("res_out1"), store)
        self.assertEqual(r1["variants"]["learned"]["retrained_blocks"], [0, 1])
        r2 = run(self.full, self.sub("res_out2"), store)
        for v in ("learned", "B0"):
            self.assertEqual(r2["variants"][v]["resumed_from_block"], 1)
            self.assertEqual(r2["variants"][v]["retrained_blocks"], [2, 3, 4])
            self.assertEqual(read(os.path.join(self.sub("res_out2"), v + ".csv")),
                             read(os.path.join(self.sub("ref_out"), v + ".csv")))
        # learned graphs identical too
        for a, b in zip(sorted(glob.glob(os.path.join(store, "learned", "graph_*.parquet"))),
                        sorted(glob.glob(os.path.join(self.sub("ref_store"), "learned", "graph_*.parquet")))):
            self.assertEqual(os.path.basename(a), os.path.basename(b))
            self.assertEqual(latest_graph.read_graph(a), latest_graph.read_graph(b))
        # a rerun with nothing new retrains nothing and changes nothing
        r3 = run(self.full, self.sub("res_out3"), store)
        self.assertEqual(r3["variants"]["learned"]["retrained_blocks"], [])
        self.assertEqual(read(os.path.join(self.sub("res_out3"), "learned.csv")),
                         read(os.path.join(self.sub("ref_out"), "learned.csv")))

    def test_store_layout_and_latest(self):
        root = os.path.join(self.sub("ref_store"), "learned")
        info = json.loads(read(os.path.join(root, "latest.json")))
        for key in ("t", "path", "graph", "predictions", "git_sha", "params_hash"):
            self.assertIn(key, info)
        self.assertEqual(len(glob.glob(os.path.join(root, "checkpoint_*.pt"))), 5)
        self.assertEqual(len(glob.glob(os.path.join(root, "predictions_*.parquet"))), 5)
        g = latest_graph.read_graph(os.path.join(root, info["graph"]))
        self.assertEqual(set(g), {"t", "src_ticker", "dst_ticker", "weight", "rank_in_row"})
        self.assertEqual(set(g["t"]), {info["t"]})
        ck = torch.load(os.path.join(root, info["path"]), weights_only=False)
        for key in ("model_state", "opt_state", "tickers", "params", "t"):
            self.assertIn(key, ck)
        # helper for the server
        t, rows = latest_graph.load_latest_graph(self.sub("ref_store"), ticker=g["src_ticker"][0])
        self.assertEqual(t, info["t"])
        self.assertTrue(rows and all(g["src_ticker"][0] in (r["src_ticker"], r["dst_ticker"]) for r in rows))
        self.assertFalse(glob.glob(os.path.join(root, "*.tmp")))

    def test_params_mismatch_refuses(self):
        store = self.sub("mm_store")
        run(self.part, self.sub("mm_out1"), store, variants="B0")
        with self.assertRaises(SystemExit) as cm:
            run(self.full, self.sub("mm_out2"), store, variants="B0", extra=["--l1", "0.5"])
        self.assertIn("refusing to resume", str(cm.exception))
        self.assertIn("--fresh", str(cm.exception))
        r = run(self.full, self.sub("mm_out3"), store, variants="B0", extra=["--l1", "0.5", "--fresh"])
        self.assertEqual(r["variants"]["B0"]["retrained_blocks"], [0, 1, 2, 3, 4])

    def test_crash_mid_write_then_resume(self):
        store = self.sub("crash_store")
        real = st.Store.save_predictions
        calls = {"n": 0}

        def flaky(self_, t, columns):
            calls["n"] += 1
            if calls["n"] == 3:  # die while writing block 2's predictions
                def partial(tmp):
                    with open(tmp, "wb") as f:
                        f.write(b"PAR1 partial")
                    raise KeyboardInterrupt("simulated crash")
                st.atomic_write(self_.path(f"predictions_{t}.parquet"), partial)
            return real(self_, t, columns)

        with mock.patch.object(st.Store, "save_predictions", flaky):
            with self.assertRaises(KeyboardInterrupt):
                run(self.full, self.sub("crash_out1"), store, variants="learned")
        root = os.path.join(store, "learned")
        self.assertFalse([f for f in os.listdir(root) if f.endswith(".tmp")])
        self.assertEqual(len(glob.glob(os.path.join(root, "predictions_*.parquet"))), 2)
        info = json.loads(read(os.path.join(root, "latest.json")))
        self.assertEqual(info["block"], 1)  # latest still points at the last complete retrain
        r = run(self.full, self.sub("crash_out2"), store, variants="learned")
        self.assertEqual(r["variants"]["learned"]["retrained_blocks"], [2, 3, 4])
        self.assertEqual(read(os.path.join(self.sub("crash_out2"), "learned.csv")),
                         read(os.path.join(self.sub("ref_out"), "learned.csv")))

    def test_new_ticker_on_resume(self):
        # the later export lists one more ticker (first in the list): it gets a fresh row and predictions
        p = pnl.load_panel(self.full)
        rng = np.random.default_rng(0)
        arrays = {}
        for name in pnl.FIELDS:
            col = p.a[name][:, :1].copy()
            if name in ("ret1", "dvshock", "pressure"):
                col = rng.standard_normal(col.shape).astype(np.float32) * 0.01
            arrays[name] = np.hstack([col, p.a[name]])
        grown = self.sub("grown")
        pnl.write_panel(grown, ["NEWCO"] + p.tickers, [p.sectors[0]] + p.sectors, p.times, p.rebalance, arrays)
        store = self.sub("nt_store")
        run(self.part, self.sub("nt_out1"), store, variants="learned")
        before = torch.load(os.path.join(store, "learned", st.Store(os.path.join(store, "learned"))
                                         .latest()["path"]), weights_only=False)
        r = run(grown, self.sub("nt_out2"), store, variants="learned")
        self.assertEqual(r["variants"]["learned"]["retrained_blocks"], [2, 3, 4])
        m, ck = train_wf.load_model_from_store(os.path.join(store, "learned"))
        self.assertEqual(ck["tickers"][:len(before["tickers"])], before["tickers"])  # old rows kept in place
        self.assertEqual(ck["tickers"][-1], "NEWCO")
        self.assertIn("NEWCO", read(os.path.join(self.sub("nt_out2"), "learned.csv")))


class NewTickers(unittest.TestCase):
    def test_embedding_rows_and_adam_state(self):
        torch.manual_seed(0)
        m = mdl.GraphNet("learned", rank=4, topk=2)
        m.ensure_tickers(["A", "B", "C"])
        opt = mdl.make_optimizer(m)
        b = mdl.Batch(d=0, t=0, nodes=np.arange(3), tickers=["A", "B", "C"], x=torch.randn(3, mdl.N_FEATURES),
                      y=torch.randn(3), sector=torch.zeros(3, dtype=torch.int64))
        mdl.date_loss(m, b, 1.0, 1e-4).backward()
        opt.step()
        old_s, old_d = m.E_s.detach().clone(), m.E_d.detach().clone()
        m.ensure_tickers(["D", "A"], opt)  # D new; B and C departed but keep their rows
        self.assertEqual(m.tickers, ["A", "B", "C", "D"])
        torch.testing.assert_close(m.E_s[:3].detach(), old_s)
        torch.testing.assert_close(m.E_d[:3].detach(), old_d)
        fresh = m._ticker_init("D")
        torch.testing.assert_close(m.E_s[3].detach(), fresh[0])
        torch.testing.assert_close(m.E_d[3].detach(), fresh[1])
        state = opt.state[m.E_s]
        self.assertEqual(state["exp_avg"].shape, (4, 4))
        self.assertTrue(torch.all(state["exp_avg"][3] == 0))
        self.assertTrue(any(q is m.E_s for g in opt.param_groups for q in g["params"]))
        # the optimizer keeps working on the grown table
        opt.zero_grad()
        b2 = mdl.Batch(d=0, t=0, nodes=np.arange(2), tickers=["D", "A"], x=torch.randn(2, mdl.N_FEATURES),
                       y=torch.randn(2), sector=torch.zeros(2, dtype=torch.int64))
        mdl.date_loss(m, b2, 1.0, 1e-4).backward()
        opt.step()
        # absent rows are not decayed: only Adam's fading momentum moves them (<= ~lr per step)
        emb_lr = opt.param_groups[1]["lr"]
        self.assertLess(float((m.E_s[1].detach() - old_s[1]).abs().max()), 1.5 * emb_lr)
        # a ticker's fresh row does not depend on when or in what order it is added
        m2 = mdl.GraphNet("learned", rank=4, topk=2)
        m2.ensure_tickers(["D"])
        torch.testing.assert_close(m2.E_s[0].detach(), fresh[0])


class AtomicWrite(unittest.TestCase):
    def test_failed_write_leaves_old_file_and_no_temp(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "latest.json")
            st.atomic_write(path, lambda tmp: write_text(tmp, '{"v": 1}'))

            def boom(tmp):
                with open(tmp, "w") as f:
                    f.write('{"v": 2, "trunc')
                raise RuntimeError("crash")

            with self.assertRaises(RuntimeError):
                st.atomic_write(path, boom)
            self.assertEqual(read(path), '{"v": 1}')
            self.assertEqual(os.listdir(d), ["latest.json"])
            with self.assertRaises(RuntimeError):
                st.atomic_write(os.path.join(d, "new.json"), boom)
            self.assertEqual(os.listdir(d), ["latest.json"])


if __name__ == "__main__":
    unittest.main()
