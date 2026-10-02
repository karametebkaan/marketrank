import csv, io, os, sys, tempfile, unittest, zipfile
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sec13f

INDEX_HTML = """<html><body>
<a href="/files/structureddata/data/form-13f-data-sets/01jan2024-29feb2024_form13f.zip">a</a>
<a href="/files/structureddata/data/form-13f-data-sets/2023q4_form13f.zip">b</a>
<a href="/files/other.pdf">c</a>
<a href='/files/structureddata/data/form-13f-data-sets/01mar2024-31may2024_form13f.zip'>d</a>
</body></html>"""


def tsv(header, rows):
    out = ["\t".join(header)]
    out += ["\t".join(r) for r in rows]
    return "\n".join(out) + "\n"


def build_zip(path):
    sub_h = ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"]
    sub = [
        ["A1", "10-FEB-2022", "13F-HR", "111", "31-DEC-2021"],      # manager A original (thousands)
        ["A2", "20-MAR-2022", "13F-HR/A", "111", "31-DEC-2021"],    # RESTATEMENT replaces A1
        ["B1", "10-FEB-2024", "13F-HR", "222", "31-DEC-2023"],      # manager B original (dollars)
        ["B2", "20-FEB-2024", "13F-HR/A", "222", "31-DEC-2023"],    # NEW HOLDINGS adds
        ["X1", "10-FEB-2024", "13F-NT", "333", "31-DEC-2023"],      # not 13F-HR
    ]
    cov_h = ["ACCESSION_NUMBER", "REPORTCALENDARORQUARTER", "ISAMENDMENT", "AMENDMENTTYPE"]
    cov = [
        ["A1", "31-DEC-2021", "N", ""],
        ["A2", "31-DEC-2021", "Y", "RESTATEMENT"],
        ["B1", "31-DEC-2023", "N", ""],
        ["B2", "31-DEC-2023", "Y", "NEW HOLDINGS"],
        ["X1", "31-DEC-2023", "N", ""],
    ]
    info_h = ["ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "VALUE",
              "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"]
    info = [
        ["A1", "OLD CO", "COM", "000000001", "999", "999", "SH", ""],   # superseded
        ["A2", "APPLE INC", "COM", "037833100", "100", "10", "SH", ""],  # 100 thousand -> 100000
        ["A2", "APPLE INC", "COM", "037833100", "50", "5", "SH", ""],    # duplicate sums
        ["A2", "PUT CO", "COM", "111111111", "7", "7", "SH", "Put"],     # dropped
        ["A2", "BOND CO", "NOTE", "222222222", "8", "8", "PRN", ""],     # dropped
        ["B1", "MSFT", "COM", "594918104", "2000", "20", "SH", ""],
        ["B2", "NVDA", "COM", "67066G104", "3000", "30", "SH", ""],
        ["X1", "NT CO", "COM", "333333333", "1", "1", "SH", ""],
    ]
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("SUBMISSION.tsv", tsv(sub_h, sub))
        z.writestr("COVERPAGE.tsv", tsv(cov_h, cov))
        z.writestr("INFOTABLE.tsv", tsv(info_h, info))


class Sec13fTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.zip = os.path.join(self.tmp.name, "x_form13f.zip")
        build_zip(self.zip)

    def test_parse_index(self):
        links = sec13f.parse_index(INDEX_HTML)
        self.assertEqual([n for _, n in links],
                         ["01jan2024-29feb2024_form13f.zip", "2023q4_form13f.zip",
                          "01mar2024-31may2024_form13f.zip"])
        self.assertTrue(links[0][0].startswith("/files/") or links[0][0].startswith("http"))

    def test_quarter_of(self):
        self.assertEqual(sec13f.quarter_of("31-DEC-2024"), "2024Q4")
        self.assertEqual(sec13f.quarter_of("30-JUN-2023"), "2023Q2")
        self.assertEqual(sec13f.quarter_of("01-JAN-2020"), "2020Q1")

    def test_read_zip_rows(self):
        rows = sorted(sec13f.aggregate(sec13f.read_zip(self.zip)))
        self.assertEqual(rows, [
            ("2021Q4", "111", "037833100", "APPLE INC", 15, 150000),
            ("2023Q4", "222", "594918104", "MSFT", 20, 2000),
            ("2023Q4", "222", "67066G104", "NVDA", 30, 3000),
        ])

    def test_value_units_cutover(self):
        self.assertEqual(sec13f.value_to_usd(5, "02-JAN-2023"), 5000)
        self.assertEqual(sec13f.value_to_usd(5, "03-JAN-2023"), 5)

    def test_apply_amendments(self):
        subs = {
            "O": {"cik": "1", "period": "31-DEC-2023", "filing_date": "01-FEB-2024", "type": "13F-HR"},
            "R": {"cik": "1", "period": "31-DEC-2023", "filing_date": "05-FEB-2024", "type": "13F-HR/A"},
            "N": {"cik": "1", "period": "31-DEC-2023", "filing_date": "09-FEB-2024", "type": "13F-HR/A"},
        }
        cov = {"O": {"amendment_type": ""}, "R": {"amendment_type": "RESTATEMENT"},
               "N": {"amendment_type": "NEW HOLDINGS"}}
        self.assertEqual(sec13f.apply_amendments(subs, cov), {"R": "replace", "N": "add"})
        del subs["R"], cov["R"]
        self.assertEqual(sec13f.apply_amendments(subs, cov), {"O": "keep", "N": "add"})

    def test_write_quarter(self):
        p = os.path.join(self.tmp.name, "h.csv")
        sec13f.write_quarter([("2023Q4", "2", "B", "b", 1, 2), ("2023Q4", "1", "A", "a", 3, 4)], p)
        with open(p) as f:
            got = list(csv.reader(f))
        self.assertEqual(got, [["cik", "cusip", "issuer", "shares", "value_usd"],
                               ["1", "A", "a", "3", "4"], ["2", "B", "b", "1", "2"]])

    def test_pipeline_outputs(self):
        data = os.path.join(self.tmp.name, "data")
        raw = os.path.join(data, "13f", "raw")
        os.makedirs(raw)
        with open(self.zip, "rb") as f:
            blob = f.read()
        index = '<a href="/d/01jan2024-29feb2024_form13f.zip">z</a>'

        def get(url, ua):
            return (200, index.encode() if url.endswith("index") else blob, 0)
        client = sec13f.SecClient("ua", min_interval=0, get=get, sleep=lambda s: None)
        sec13f.run(data, None, None, "https://x/index", client)
        with open(os.path.join(data, "13f", "holdings_2023Q4.csv")) as f:
            got = list(csv.reader(f))
        self.assertEqual(got[1:], [["222", "594918104", "MSFT", "20", "2000"],
                                   ["222", "67066G104", "NVDA", "30", "3000"]])
        self.assertTrue(os.path.exists(os.path.join(data, "13f", "holdings_2021Q4.csv")))

    def test_abort_on_403(self):
        c = sec13f.SecClient("ua", min_interval=0, get=lambda u, ua: (403, b"", 0), sleep=lambda s: None)
        with self.assertRaises(sec13f.SecAbort) as cm:
            c.get("https://x/p")
        self.assertNotIn("ua", str(cm.exception).replace("User-Agent", "").replace("SEC_USER_AGENT", ""))

    def test_sleep_on_429(self):
        seq = [(429, b"", 7), (200, b"ok", 0)]
        slept = []
        c = sec13f.SecClient("ua", min_interval=0, get=lambda u, ua: seq.pop(0), sleep=slept.append)
        self.assertEqual(c.get("https://x/p"), b"ok")
        self.assertEqual(slept, [7])
        seq[:] = [(429, b"", 500), (200, b"ok", 0)]
        slept.clear()
        c.get("https://x/p")
        self.assertEqual(slept, [60])

    def test_abort_after_20_failures(self):
        c = sec13f.SecClient("ua", min_interval=0, get=lambda u, ua: (500, b"", 0), sleep=lambda s: None)
        with self.assertRaises(sec13f.SecAbort):
            c.get("https://x/p")

    def test_missing_user_agent(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SEC_USER_AGENT", None)
            with self.assertRaises(SystemExit) as cm:
                sec13f.main(["--data", self.tmp.name], env_path=os.path.join(self.tmp.name, "none.env"))
        self.assertEqual(str(cm.exception),
                         'SEC_USER_AGENT is not set: add SEC_USER_AGENT="Your Name your@email" to .env (SEC requires a contact)')

    def test_dotenv_sets_only_missing(self):
        p = os.path.join(self.tmp.name, ".env")
        with open(p, "w") as f:
            f.write('# c\nQQ_A="one two"\nQQ_B=keep\n')
        with mock.patch.dict(os.environ, {"QQ_B": "orig"}):
            os.environ.pop("QQ_A", None)
            sec13f.load_dotenv(p)
            self.assertEqual(os.environ["QQ_A"], "one two")
            self.assertEqual(os.environ["QQ_B"], "orig")
            os.environ.pop("QQ_A", None)

    # ---- fix round 1 ----
    def _zip2(self):
        p = os.path.join(self.tmp.name, "y_form13f.zip")
        sub_h = ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"]
        cov_h = ["ACCESSION_NUMBER", "REPORTCALENDARORQUARTER", "ISAMENDMENT", "AMENDMENTTYPE"]
        info_h = ["ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "VALUE",
                  "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"]
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("SUBMISSION.tsv", tsv(sub_h, [["C1", "10-AUG-2022", "13F-HR", "222", "30-JUN-2022"]]))
            z.writestr("COVERPAGE.tsv", tsv(cov_h, [["C1", "30-JUN-2022", "N", ""]]))
            z.writestr("INFOTABLE.tsv", tsv(info_h, [["C1", "TSLA", "COM", "88160R101", "4", "9", "SH", ""],
                                                      ["C1", "TSLA", "COM", "88160R101", "1", "1", "SH", ""]]))
        return p

    def test_staging_matches_in_memory_three_quarters(self):
        z2 = self._zip2()
        blobs = {}
        for n, p in (("01jan2022-29feb2024_form13f.zip", self.zip), ("01mar2024-31may2024_form13f.zip", z2)):
            with open(p, "rb") as f:
                blobs[n] = f.read()
        index = "".join('<a href="/d/%s">z</a>' % n for n in blobs)

        def get(url, ua):
            if url.endswith("index"):
                return 200, index.encode(), 0
            return 200, blobs[url.rsplit("/", 1)[-1]], 0
        data = os.path.join(self.tmp.name, "data")
        client = sec13f.SecClient("ua", min_interval=0, get=get, sleep=lambda s: None)
        with mock.patch.object(sec13f, "read_meta", wraps=sec13f.read_meta) as rm:
            qs = sec13f.run(data, None, None, "https://x/index", client)
        self.assertEqual(rm.call_count, 2)
        self.assertEqual(qs, ["2021Q4", "2022Q2", "2023Q4"])
        self.assertTrue(os.path.exists(os.path.join(data, "13f", "staging.sqlite")))
        # in-memory reference
        subs, cov = {}, {}
        for p in (self.zip, z2):
            s, c = sec13f.read_meta(p)
            subs.update(s); cov.update(c)
        dec = sec13f.apply_amendments(subs, cov)
        ref = {}
        for p in (self.zip, z2):
            for r in sec13f.aggregate(sec13f.read_zip(p, dec)):
                ref.setdefault(r[0], []).append(r)
        for q, rows in ref.items():
            refp = os.path.join(self.tmp.name, "ref_%s.csv" % q)
            sec13f.write_quarter(rows, refp)
            with open(refp) as a, open(os.path.join(data, "13f", "holdings_%s.csv" % q)) as b:
                self.assertEqual(a.read(), b.read())

    def test_download_streams_in_chunks(self):
        class Body:
            def __init__(self): self.sizes = []; self.n = 3
            def read(self, size=-1):
                self.sizes.append(size)
                self.n -= 1
                return b"x" * 10 if self.n >= 0 else b""
            def close(self): pass
        body = Body()
        c = sec13f.SecClient("ua", min_interval=0, get=lambda u, ua: (200, body, 0), sleep=lambda s: None)
        dest = os.path.join(self.tmp.name, "d.zip")
        c.download("https://x/d.zip", dest)
        self.assertEqual(os.path.getsize(dest), 30)
        self.assertTrue(all(0 < n <= 1 << 20 for n in body.sizes))
        self.assertFalse(os.path.exists(dest + ".part"))

    def test_find_dotenv_order(self):
        cwd = os.path.join(self.tmp.name, "cwd"); par = os.path.join(self.tmp.name, "par")
        root = os.path.join(self.tmp.name, "root")
        for d in (cwd, par, root, os.path.join(par, "data")):
            os.makedirs(d, exist_ok=True)
        data = os.path.join(par, "data")
        self.assertIsNone(sec13f.find_dotenv(data, cwd, root))
        open(os.path.join(root, ".env"), "w").close()
        self.assertEqual(sec13f.find_dotenv(data, cwd, root), os.path.join(root, ".env"))
        open(os.path.join(par, ".env"), "w").close()
        self.assertEqual(sec13f.find_dotenv(data, cwd, root), os.path.join(par, ".env"))
        open(os.path.join(cwd, ".env"), "w").close()
        self.assertEqual(sec13f.find_dotenv(data, cwd, root), os.path.join(cwd, ".env"))

    def test_main_uses_parent_of_data_env(self):
        par = os.path.join(self.tmp.name, "par"); os.makedirs(os.path.join(par, "data"))
        with open(os.path.join(par, ".env"), "w") as f:
            f.write("SEC_USER_AGENT=FAKE-UA-zzz\n")
        seen = {}
        def fake_run(data, qf, qt, url, client, stats=None):
            seen["ua"] = client._ua
            return []
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("SEC_USER_AGENT", None)
            with mock.patch.object(sec13f, "run", fake_run), mock.patch.object(sec13f, "find_dotenv",
                    lambda d, c=None, r=None, _f=sec13f.find_dotenv: _f(d, self.tmp.name, self.tmp.name)):
                sec13f.main(["--data", os.path.join(par, "data")])
        self.assertEqual(seen["ua"], "FAKE-UA-zzz")

    def test_pacing(self):
        t = [0.0]; stamps = []
        def get(u, ua):
            stamps.append(t[0]); return 200, b"", 0
        def sleep(s): t[0] += s
        c = sec13f.SecClient("ua", min_interval=0.15, get=get, sleep=sleep, now=lambda: t[0])
        for _ in range(4):
            c.get("https://x/p")
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        self.assertTrue(all(g >= 0.15 - 1e-9 for g in gaps), gaps)

    def test_exactly_20_calls(self):
        n = [0]
        def get(u, ua):
            n[0] += 1; return 500, b"", 0
        c = sec13f.SecClient("ua", min_interval=0, get=get, sleep=lambda s: None)
        with self.assertRaises(sec13f.SecAbort):
            c.get("https://x/p")
        self.assertEqual(n[0], 20)

    def test_retry_after_on_503(self):
        seq = [(503, b"", 5), (200, b"ok", 0)]
        slept = []
        c = sec13f.SecClient("ua", min_interval=0, get=lambda u, ua: seq.pop(0), sleep=slept.append)
        c.get("https://x/p")
        self.assertEqual(slept, [5])

    def test_ua_never_leaks(self):
        ua = "SECRET-UA-xyz"
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            for status in (403, 500, 404):
                c = sec13f.SecClient(ua, min_interval=0, get=lambda u, a: (status, b"", 0), sleep=lambda s: None)
                with self.assertRaises(Exception) as cm:
                    c.get("https://x/p")
                self.assertNotIn(ua, str(cm.exception))
        self.assertNotIn(ua, buf.getvalue())

    def _custom_zip(self, sub, cov, info, info_bytes=None):
        p = os.path.join(self.tmp.name, "c_form13f.zip")
        sub_h = ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"]
        cov_h = ["ACCESSION_NUMBER", "REPORTCALENDARORQUARTER", "ISAMENDMENT", "AMENDMENTTYPE"]
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("SUBMISSION.tsv", tsv(sub_h, sub))
            z.writestr("COVERPAGE.tsv", tsv(cov_h, cov))
            z.writestr("INFOTABLE.tsv", info_bytes if info_bytes is not None else info)
        return p

    def test_tsv_robustness(self):
        h = "ACCESSION_NUMBER\tNAMEOFISSUER\tTITLEOFCLASS\tCUSIP\tVALUE\tSSHPRNAMT\tSSHPRNAMTTYPE\tPUTCALL\n"
        body = (h +
                "A\tNAME\tCOM\tAAA111111\t5\t12345678901234567\tSH\t\n"   # big int, exact
                "A\tFRAC\tCOM\tBBB222222\t5\t10.5\tSH\t\n"                 # Decimal
                "A\tNOCUSIP\tCOM\t\t5\t1\tSH\t\n"                         # empty cusip
                "A\tSHORT\tCOM\tCCC333333\t5\t2\tSH\n")                    # short row (no PUTCALL)
        raw = b"\xef\xbb\xbf" + body.encode()
        p = self._custom_zip([["A", "10-FEB-2024", "13F-HR", "9", "31-DEC-2023"]],
                             [["A", "31-DEC-2023", "N", ""]], None, raw)
        stats = {}
        rows = list(sec13f.read_zip(p, stats=stats))
        d = {r[2]: r for r in rows}
        self.assertEqual(d["AAA111111"][4], 12345678901234567)
        self.assertEqual(str(d["BBB222222"][4]), "10.5")
        self.assertEqual(d["CCC333333"][3], "SHORT")
        self.assertEqual(stats["empty_cusip"], 1)

    def test_blank_amendment_type_skipped(self):
        subs = {"O": {"cik": "1", "period": "31-DEC-2023", "filing_date": "01-FEB-2024", "type": "13F-HR"},
                "U": {"cik": "1", "period": "31-DEC-2023", "filing_date": "05-FEB-2024", "type": "13F-HR/A"}}
        cov = {"O": {"amendment_type": "", "is_amendment": "N"},
               "U": {"amendment_type": "", "is_amendment": "Y"}}
        stats = {}
        self.assertEqual(sec13f.apply_amendments(subs, cov, stats), {"O": "keep"})
        self.assertEqual(stats["unknown_amendments"], 1)

    def test_to_skips_late_zips(self):
        with open(self.zip, "rb") as fh:
            blob = fh.read()
        fetched = []
        index = ('<a href="/d/01jan2022-31mar2022_form13f.zip">a</a>'
                 '<a href="/d/01jan2030-31mar2030_form13f.zip">b</a>')
        def get(url, ua):
            fetched.append(url)
            return (200, index.encode(), 0) if url.endswith("index") else (200, blob, 0)
        c = sec13f.SecClient("ua", min_interval=0, get=get, sleep=lambda s: None)
        sec13f.run(os.path.join(self.tmp.name, "dd"), None, "2023Q4", "https://x/index", c)
        self.assertEqual(len(fetched), 2)
        self.assertFalse(any("2030" in u for u in fetched))

    @staticmethod
    def _read(path):
        with open(path) as fh:
            return fh.read()

    def test_main_without_dotenv(self):
        ran = {}
        def fake_run(data, qf, qt, url, client, stats=None):
            ran["ok"] = True
            return []
        with mock.patch.object(sec13f, "find_dotenv", lambda *a, **k: None), \
                mock.patch.object(sec13f, "run", fake_run), \
                mock.patch.dict(os.environ, {"SEC_USER_AGENT": "FAKE-UA-set"}):
            sec13f.main(["--data", self.tmp.name])
        self.assertTrue(ran.get("ok"))
        with mock.patch.object(sec13f, "find_dotenv", lambda *a, **k: None), \
                mock.patch.dict(os.environ, {}):
            os.environ.pop("SEC_USER_AGENT", None)
            with self.assertRaises(SystemExit) as cm:
                sec13f.main(["--data", self.tmp.name])
        self.assertEqual(str(cm.exception), sec13f.MISSING_UA)

    def test_read_meta_short_rows(self):
        p = self._custom_zip([["A", "10-FEB-2024", "13F-HR", "9", "31-DEC-2023"]],
                             [["A", "31-DEC-2023", "N", ""]], "")
        with zipfile.ZipFile(p, "a") as z:
            pass
        # rewrite with short rows
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("SUBMISSION.tsv", "ACCESSION_NUMBER\tFILING_DATE\tSUBMISSIONTYPE\tCIK\tPERIODOFREPORT\n"
                       "A\t10-FEB-2024\t13F-HR\t9\t31-DEC-2023\nB\t10-FEB-2024\n")
            z.writestr("COVERPAGE.tsv", "ACCESSION_NUMBER\tREPORTCALENDARORQUARTER\tISAMENDMENT\tAMENDMENTTYPE\n"
                       "A\t31-DEC-2023\nB\n")
        subs, cov = sec13f.read_meta(p)
        self.assertIn("A", subs)
        self.assertEqual(cov["A"]["amendment_type"], "")
        sec13f.apply_amendments(subs, cov)  # must not raise

    # --- VALUE unit repair (normalise_value_units) ---
    PRICES = {"A": 10, "B": 50, "C": 20, "D": 5, "E": 100, "F": 2}

    def _good(self, cik, scale=1, cusips="ABCDEF"):
        return [("q", str(cik), c, c.lower(), 1000, 1000 * self.PRICES[c] * scale) for c in cusips]

    def _norm(self, rows):
        p = os.path.join(self.tmp.name, "h.csv")
        sec13f.write_quarter(rows, p)
        st = sec13f.normalise_value_units(p)
        with open(p) as f:
            got = {(r["cik"], r["cusip"]): r["value_usd"] for r in csv.DictReader(f)}
        return p, st, got

    def test_normalise_value_units(self):
        # Managers 1-6 are correct on A-F. Manager 7 is 1000x too large on all five of its rows (dollars before the
        # cutover), manager 8 1000x too small (thousands after it): both rescaled. Manager 9 holds only an unseen
        # CUSIP (no consensus: untouched). Manager 10 has one row 10^6 too large (value or deflated shares: ambiguous,
        # left for the C++ guard), manager 11 one row whose shares are 10^6 too large: both untouched and counted.
        rows = []
        for m in range(1, 7):
            rows += self._good(m)
        rows += self._good(7, 1000, "ABCDE")
        rows += [r[:5] + (r[5] // 1000,) for r in self._good(8, 1, "BCDEF")]  # reported in thousands
        rows += [("q", "8", "Z", "z", 3, 7)]
        rows += [("q", "9", "Z", "z", 5, 5000000)]
        rows += self._good(10, 1, "BCDEF") + [("q", "10", "A", "a", 100, 1000000000)]
        rows += self._good(11, 1, "BCDEF") + [("q", "11", "A", "a", 100000000, 1000)]
        p, st, got = self._norm(rows)
        self.assertEqual(st, {"managers_down": 1, "managers_up": 1, "rows_inconsistent": 2})
        self.assertEqual(got[("7", "A")], "10000")
        self.assertEqual(got[("7", "E")], "100000")
        self.assertEqual(got[("8", "B")], "50000")
        self.assertEqual(got[("8", "Z")], "7000")   # a manager's factor applies to all its rows
        self.assertEqual(got[("1", "A")], "10000")
        self.assertEqual(got[("9", "Z")], "5000000")
        self.assertEqual(got[("10", "A")], "1000000000")
        self.assertEqual(got[("11", "A")], "1000")
        # Idempotent: a second pass changes nothing and rescales no manager.
        with open(p) as f:
            before = f.read()
        st2 = sec13f.normalise_value_units(p)
        self.assertEqual((st2["managers_down"], st2["managers_up"]), (0, 0))
        with open(p) as f:
            self.assertEqual(f.read(), before)

    def test_normalise_shares_deflated_row_is_not_rewritten(self):
        # 7 holders of A at $10; one row reports shares=1, value=$1e6 (shares deflated, value right). Rewriting the
        # value to $10 would hide the error from the C++ price guard; the row must stay as filed.
        rows = []
        for m in range(1, 8):
            rows += self._good(m)
        rows += self._good(8, 1, "BCDEF") + [("q", "8", "A", "a", 1, 1000000)]
        _, st, got = self._norm(rows)
        self.assertEqual(got[("8", "A")], "1000000")
        self.assertEqual((st["managers_down"], st["managers_up"]), (0, 0))
        self.assertEqual(st["rows_inconsistent"], 1)

    def test_normalise_thin_consensus_from_mis_unit_majority(self):
        # X has 5 holders; 3 of them report thousands after the cutover (on every row, so they are caught on A-F),
        # 2 report dollars. The raw X consensus is 1000x low; it must be recomputed without the mis-unit managers,
        # so the 2 correct X rows are not touched and the 3 others are rescaled up.
        rows = []
        for m in range(1, 7):
            rows += self._good(m)
        for m in (7, 8, 9):
            rows += [r[:5] + (r[5] // 1000,) for r in self._good(m)] + [("q", str(m), "X", "x", 100, 30)]  # X is $300
        for m in (10, 11):
            rows += self._good(m, 1, "ABCDE") + [("q", str(m), "X", "x", 100, 30000)]
        _, st, got = self._norm(rows)
        self.assertEqual(st["managers_up"], 3)
        self.assertEqual(st["managers_down"], 0)
        for m in (7, 8, 9):
            self.assertEqual(got[(str(m), "X")], "30000")
            self.assertEqual(got[(str(m), "A")], "10000")
        for m in (10, 11):
            self.assertEqual(got[(str(m), "X")], "30000")
            self.assertEqual(got[(str(m), "A")], "10000")

    def test_normalise_thin_manager_is_not_rescaled(self):
        # A manager needs >= 5 consensus rows, mostly at the same power of 1000, before it is rescaled.
        # Manager 7: a single row 1000x high. Manager 8: one 40x row and four correct rows. Manager 9: 5 rows of which
        # only 2 are 1000x off (no agreement). Manager 10: 100x on all rows (not a power of 1000). None is rescaled.
        rows = []
        for m in range(1, 7):
            rows += self._good(m)
        rows += [("q", "7", "A", "a", 100, 1000000)]
        rows += self._good(8, 1, "BCDE") + [("q", "8", "A", "a", 100, 40000)]
        rows += self._good(9, 1, "ABC") + self._good(9, 1000, "DE")
        rows += self._good(10, 100)
        _, st, got = self._norm(rows)
        self.assertEqual((st["managers_down"], st["managers_up"]), (0, 0))
        self.assertEqual(got[("7", "A")], "1000000")
        self.assertEqual(got[("8", "A")], "40000")
        self.assertEqual(got[("10", "A")], "1000000")

    def test_run_normalises_units(self):
        with mock.patch.object(sec13f, "normalise_value_units", return_value={}) as nv:
            data = os.path.join(self.tmp.name, "data")
            os.makedirs(os.path.join(data, "13f", "raw"))
            with open(self.zip, "rb") as f:
                blob = f.read()
            index = '<a href="/d/01jan2024-29feb2024_form13f.zip">z</a>'

            def get(url, ua):
                return (200, index.encode() if url.endswith("index") else blob, 0)
            client = sec13f.SecClient("ua", min_interval=0, get=get, sleep=lambda s: None)
            qs = sec13f.run(data, None, None, "https://x/index", client)
            self.assertEqual(nv.call_count, len(qs))

    def test_gitignore_pycache(self):
        root = os.path.join(os.path.dirname(__file__), "..", "..", ".gitignore")
        self.assertIn("__pycache__/", self._read(root).split())


if __name__ == "__main__":
    unittest.main()
