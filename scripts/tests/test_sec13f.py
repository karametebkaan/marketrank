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
        rows = sorted(sec13f.read_zip(self.zip))
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


if __name__ == "__main__":
    unittest.main()
