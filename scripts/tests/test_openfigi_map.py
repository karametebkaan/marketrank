import csv, io, os, sys, tempfile, unittest
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import openfigi_map as om


class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def hit(ticker, st="Common Stock", name="N", figi="F"):
    return {"data": [{"figi": figi, "name": name, "ticker": ticker, "exchCode": "US",
                      "securityType": st}]}


class FakePost:
    def __init__(self, table=None, script=None):
        self.table = table or {}
        self.script = list(script or [])
        self.calls = []

    def __call__(self, url, jobs, headers):
        self.calls.append((url, jobs, dict(headers)))
        if self.script:
            r = self.script.pop(0)
            if r is not None:
                return r
        return 200, {}, [self.table.get(j["idValue"], {"warning": "No identifier found."})
                         for j in jobs]


def write_holdings(d, q, rows):
    os.makedirs(os.path.join(d, "13f"), exist_ok=True)
    with open(os.path.join(d, "13f", "holdings_%s.csv" % q), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cik", "cusip", "issuer", "shares", "value_usd"])
        w.writerows(rows)


def read_map(d):
    with open(os.path.join(d, "13f", "cusip_map.csv")) as f:
        return {r["cusip"]: r for r in csv.DictReader(f)}


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.clk = Clock()

    def tearDown(self):
        self.tmp.cleanup()

    def run_map(self, post, key=None, min_value=1e6, **kw):
        out = io.StringIO()
        with redirect_stdout(out):
            om.run(self.d, min_value, post, api_key=key, now=self.clk.now,
                   sleep=self.clk.sleep, **kw)
        return out.getvalue()

    def many(self, n, v=2e6):
        write_holdings(self.d, "2024Q1", [["1", "C%05d" % i, "I", "10", str(v)] for i in range(n)])

    def test_batching_without_key(self):
        self.many(25)
        p = FakePost()
        self.run_map(p)
        self.assertEqual([len(c[1]) for c in p.calls], [10, 10, 5])
        self.assertEqual(p.calls[0][1][0], {"idType": "ID_CUSIP", "idValue": "C00000", "exchCode": "US"})
        self.assertEqual(p.calls[0][0], "https://api.openfigi.com/v3/mapping")
        self.assertNotIn("X-OPENFIGI-APIKEY", p.calls[0][2])

    def test_batching_with_key(self):
        self.many(250)
        p = FakePost()
        self.run_map(p, key="SECRETKEY")
        self.assertEqual([len(c[1]) for c in p.calls], [100, 100, 50])

    def test_min_value_filter_any_quarter(self):
        write_holdings(self.d, "2024Q1", [["1", "AAA", "I", "1", "5"], ["1", "BBB", "I", "1", "5"]])
        write_holdings(self.d, "2024Q2", [["1", "AAA", "I", "1", "2000000"]])
        p = FakePost()
        self.run_map(p)
        self.assertEqual([j["idValue"] for c in p.calls for j in c[1]], ["AAA"])

    def test_rate_limit_no_key(self):
        self.many(30)
        p = FakePost()
        self.run_map(p)
        # 3 requests at <=25/min: 2.4 s spacing
        self.assertAlmostEqual(self.clk.t, 2 * 2.4, places=6)

    def test_rate_limit_key(self):
        self.many(300)
        self.run_map(FakePost(), key="K")
        self.assertAlmostEqual(self.clk.t, 2 * 0.24, places=6)

    def test_choose_match(self):
        r = {"data": [{"ticker": "X1", "securityType": "Warrant"},
                      {"ticker": "X2", "securityType": "ETP"},
                      {"ticker": "X3", "securityType": "Common Stock"}]}
        self.assertEqual(om.choose(r["data"])["ticker"], "X2")
        r2 = [{"ticker": "W", "securityType": "Warrant"}, {"ticker": "V", "securityType": "Right"}]
        self.assertEqual(om.choose(r2)["ticker"], "W")

    def test_normalise(self):
        self.assertEqual(om.normalise("BRK/B"), "BRK.B")
        self.assertEqual(om.normalise(" aapl "), "AAPL")

    def test_output_and_unmatched_cached(self):
        write_holdings(self.d, "2024Q1", [["1", "AAA", "I", "1", "2e6"], ["1", "ZZZ", "I", "1", "2e6"]])
        p = FakePost({"AAA": hit("BRK/B", name="Berk", figi="G1")})
        self.run_map(p)
        m = read_map(self.d)
        self.assertEqual(m["AAA"]["ticker"], "BRK.B")
        self.assertEqual(m["AAA"]["security_type"], "Common Stock")
        self.assertEqual(m["AAA"]["figi"], "G1")
        self.assertEqual(m["ZZZ"]["ticker"], "")
        # rerun: nothing to fetch
        p2 = FakePost()
        self.run_map(p2)
        self.assertEqual(p2.calls, [])

    def test_unmatched_retried_after_90_days(self):
        write_holdings(self.d, "2024Q1", [["1", "ZZZ", "I", "1", "2e6"]])
        self.run_map(FakePost())
        p = FakePost()
        self.clk_now = None
        old = (datetime.now() - timedelta(days=91)).strftime("%Y-%m-%dT%H:%M:%SZ")
        m = read_map(self.d)
        m["ZZZ"]["fetched_at"] = old
        om.save_cache(os.path.join(self.d, "13f", "cusip_map.csv"), m)
        self.run_map(p)
        self.assertEqual(len(p.calls), 1)

    def test_resume_partial_cache(self):
        self.many(12)
        os.makedirs(os.path.join(self.d, "13f"), exist_ok=True)
        pre = {"C00000": dict(cusip="C00000", ticker="A", name="", security_type="", figi="",
                              fetched_at="2024-01-01T00:00:00Z")}
        om.save_cache(os.path.join(self.d, "13f", "cusip_map.csv"), pre)
        p = FakePost()
        self.run_map(p)
        sent = [j["idValue"] for c in p.calls for j in c[1]]
        self.assertEqual(len(sent), 11)
        self.assertNotIn("C00000", sent)
        self.assertEqual(read_map(self.d)["C00000"]["ticker"], "A")

    def test_error_element_not_cached(self):
        write_holdings(self.d, "2024Q1", [["1", "AAA", "I", "1", "2e6"], ["1", "BBB", "I", "1", "2e6"]])
        script = [(200, {}, [{"error": "boom"}, hit("BBB")])]
        self.run_map(FakePost(script=script))
        m = read_map(self.d)
        self.assertNotIn("AAA", m)
        self.assertEqual(m["BBB"]["ticker"], "BBB")

    def test_429_retry_after(self):
        self.many(3)
        p = FakePost(script=[(429, {"Retry-After": "7"}, None)])
        self.run_map(p)
        self.assertEqual(len(p.calls), 2)
        self.assertIn(7, self.clk.sleeps)
        self.assertEqual(len(read_map(self.d)), 3)

    def test_429_retry_after_capped(self):
        self.many(3)
        p = FakePost(script=[(429, {"Retry-After": "900"}, None)])
        self.run_map(p)
        self.assertIn(60, self.clk.sleeps)

    def test_413_halves_batch(self):
        self.many(10)
        p = FakePost(script=[(413, {}, None)])
        self.run_map(p)
        self.assertEqual([len(c[1]) for c in p.calls], [10, 5, 5])
        self.assertEqual(len(read_map(self.d)), 10)

    def test_commit_every_50_requests(self):
        self.many(600)
        saves = []
        orig = om.save_cache
        om.save_cache = lambda path, m: (saves.append(len(m)), orig(path, m))[1]
        try:
            self.run_map(FakePost(), key="K", min_value=1e6, batch_override=1)
        finally:
            om.save_cache = orig
        self.assertEqual(saves[:2], [50, 100])

    def test_key_header_and_never_printed(self):
        self.many(2)
        p = FakePost()
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            om.run(self.d, 1e6, p, api_key="SECRETKEY123", now=self.clk.now, sleep=self.clk.sleep)
        self.assertEqual(p.calls[0][2]["X-OPENFIGI-APIKEY"], "SECRETKEY123")
        self.assertNotIn("SECRETKEY123", out.getvalue() + err.getvalue())

    def test_coverage_report(self):
        write_holdings(self.d, "2024Q1", [["1", "AAA", "I", "1", "3000000"],
                                          ["1", "BBB", "I", "1", "1000000"]])
        ud = os.path.join(self.d, "universe")
        os.makedirs(ud)
        for name, tick in (("universe_2024-01-01_n1.csv", "OLD"), ("universe_2024-06-01_n1.csv", "AAA")):
            with open(os.path.join(ud, name), "w") as f:
                f.write("ticker,name,sector,exchange,median_dollar_volume\n%s,x,y,z,1\n" % tick)
        out = self.run_map(FakePost({"AAA": hit("AAA"), "BBB": hit("BBB")}))
        self.assertIn("75.0%", out)


if __name__ == "__main__":
    unittest.main()
