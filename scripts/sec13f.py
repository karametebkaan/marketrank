#!/usr/bin/env python3
"""Ingest SEC Form 13F data sets into per-quarter holdings CSVs (stdlib only).

Usage: python3 scripts/sec13f.py --data DIR [--from 2016Q1] [--to 2026Q3] [--index-url URL]

Output: DIR/13f/holdings_<YYYYQn>.csv with columns cik,cusip,issuer,shares,value_usd.

VALUE units: the SEC reported VALUE in thousands of dollars for filings made before
2023-01-03 and in whole dollars from 2023-01-03 on (by FILING_DATE). Pre-cutover
values are multiplied by 1000 so value_usd is always dollars.

Filters: 13F-HR / 13F-HR/A only; per (CIK, period) the latest RESTATEMENT replaces
earlier filings and later NEW HOLDINGS amendments add rows; info-table rows need
SSHPRNAMTTYPE == SH and empty PUTCALL. Duplicate (cik, cusip) rows are summed.
Reads SEC_USER_AGENT from the environment (or .env, loaded without printing).
"""
import argparse
import csv
import io
import os
import re
import sys
import time
import urllib.request
import urllib.error
import zipfile
from datetime import datetime
from urllib.parse import urljoin

DEFAULT_INDEX = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
CUTOVER = datetime(2023, 1, 3)
MAX_RETRY_AFTER_S = 60
MAX_CONSECUTIVE_FAILURES = 20
MISSING_UA = ('SEC_USER_AGENT is not set: add SEC_USER_AGENT="Your Name your@email" '
              'to .env (SEC requires a contact)')
csv.field_size_limit(1 << 28)


class SecAbort(Exception):
    pass


def _urllib_get(url, ua):
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read(), 0
    except urllib.error.HTTPError as e:
        try:
            ra = int(e.headers.get("Retry-After", "0") or 0)
        except ValueError:
            ra = 0
        return e.code, b"", ra
    except (urllib.error.URLError, OSError):
        return 0, b"", 0


class SecClient:
    def __init__(self, ua, min_interval=0.15, get=_urllib_get, sleep=time.sleep, now=time.monotonic):
        self._ua, self._min, self._get, self._sleep, self._now = ua, min_interval, get, sleep, now
        self._last = None
        self._fails = 0

    def get(self, url):
        delay = 1.0
        while True:
            if self._last is not None and self._min > 0:
                gap = self._min - (self._now() - self._last)
                if gap > 0:
                    self._sleep(gap)
            self._last = self._now()
            status, body, retry_after = self._get(url, self._ua)
            if status == 200:
                self._fails = 0
                return body
            if status == 403:
                raise SecAbort("SEC returned HTTP 403 for %s: blocked or User-Agent rejected; "
                               "check SEC_USER_AGENT and wait before retrying" % url)
            if not (status == 0 or status == 429 or status >= 500):
                raise RuntimeError("SEC GET %s failed (HTTP %d)" % (url, status))
            self._fails += 1
            if self._fails >= MAX_CONSECUTIVE_FAILURES:
                raise SecAbort("SEC GET %s failed %d consecutive times (last HTTP %d)"
                               % (url, self._fails, status))
            if status == 429 and retry_after > 0:
                self._sleep(min(retry_after, MAX_RETRY_AFTER_S))
            else:
                self._sleep(delay)
            delay = min(delay * 2, MAX_RETRY_AFTER_S)


def load_dotenv(path):
    """Minimal .env parser: sets only variables that are missing. Prints nothing."""
    try:
        f = open(path, encoding="utf-8")
    except OSError:
        return
    with f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:]
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            if k and k not in os.environ:
                os.environ[k] = v


def parse_index(html):
    out, seen = [], set()
    for m in re.finditer(r'href\s*=\s*["\']([^"\']+_form13f\.zip)["\']', html, re.I):
        url = m.group(1)
        if url not in seen:
            seen.add(url)
            out.append((url, url.rsplit("/", 1)[-1]))
    return out


def _parse_date(s):
    s = s.strip()
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.title() if fmt == "%d-%b-%Y" else s, fmt)
        except ValueError:
            pass
    raise ValueError("unrecognised date: %r" % s)


def quarter_of(period_str):
    d = _parse_date(period_str)
    return "%dQ%d" % (d.year, (d.month - 1) // 3 + 1)


def value_to_usd(value, filing_date):
    return value if _parse_date(filing_date) >= CUTOVER else value * 1000


def _num(s):
    s = (s or "").strip().replace(",", "")
    return int(float(s)) if s else 0


def _table(z, name):
    for n in z.namelist():
        if n.rsplit("/", 1)[-1].upper() == name:
            return csv.DictReader(io.TextIOWrapper(z.open(n), encoding="utf-8", errors="replace", newline=""),
                                  delimiter="\t", quoting=csv.QUOTE_NONE)
    return iter(())


def read_meta(path):
    """(submissions, coverpages) dicts keyed by accession, parsed by header name."""
    subs, cov = {}, {}
    with zipfile.ZipFile(path) as z:
        for r in _table(z, "SUBMISSION.TSV"):
            subs[r["ACCESSION_NUMBER"]] = {"cik": r["CIK"].strip(), "period": r["PERIODOFREPORT"],
                                           "filing_date": r["FILING_DATE"], "type": r["SUBMISSIONTYPE"].strip()}
        for r in _table(z, "COVERPAGE.TSV"):
            cov[r["ACCESSION_NUMBER"]] = {"amendment_type": (r.get("AMENDMENTTYPE") or "").strip().upper(),
                                          "is_amendment": (r.get("ISAMENDMENT") or "").strip().upper()}
    return subs, cov


def apply_amendments(submissions, coverpages):
    """{accession: 'keep'|'replace'|'add'} for the filings whose rows to use.

    Per (cik, period), only 13F-HR / 13F-HR/A. Filings are ordered by filing date;
    the latest RESTATEMENT (or the original if none) is the base and supersedes
    everything before it; NEW HOLDINGS amendments filed after the base add rows.
    Superseded filings are absent from the result.
    """
    groups = {}
    for acc, s in submissions.items():
        if s["type"] in ("13F-HR", "13F-HR/A"):
            groups.setdefault((s["cik"], quarter_of(s["period"])), []).append(acc)
    out = {}
    for accs in groups.values():
        accs.sort(key=lambda a: (_parse_date(submissions[a]["filing_date"]), a))

        def kind(a):
            c = coverpages.get(a, {})
            if submissions[a]["type"] == "13F-HR/A" or c.get("is_amendment") == "Y":
                return "replace" if c.get("amendment_type") == "RESTATEMENT" else "add"
            return "keep"
        kinds = {a: kind(a) for a in accs}
        base = max([i for i, a in enumerate(accs) if kinds[a] in ("replace", "keep")], default=None)
        if base is None:
            # only NEW HOLDINGS amendments seen (original in another data set): keep them all
            for a in accs:
                out[a] = "add"
            continue
        for a in accs[base:]:
            out[a] = kinds[a]
    return out


def read_zip(path, decisions=None):
    """Yield (quarter, cik, cusip, issuer, shares, value_usd) rows, summed per (quarter,cik,cusip)."""
    if decisions is None:
        decisions = apply_amendments(*read_meta(path))
    subs, _ = read_meta(path)
    agg = {}
    with zipfile.ZipFile(path) as z:
        for r in _table(z, "INFOTABLE.TSV"):
            acc = r["ACCESSION_NUMBER"]
            if acc not in decisions or acc not in subs:
                continue
            if (r.get("SSHPRNAMTTYPE") or "").strip().upper() != "SH" or (r.get("PUTCALL") or "").strip():
                continue
            s = subs[acc]
            key = (quarter_of(s["period"]), s["cik"], (r["CUSIP"] or "").strip().upper())
            cur = agg.setdefault(key, [r["NAMEOFISSUER"].strip(), 0, 0])
            cur[1] += _num(r["SSHPRNAMT"])
            cur[2] += value_to_usd(_num(r["VALUE"]), s["filing_date"])
    for (q, cik, cusip), (issuer, sh, val) in sorted(agg.items()):
        yield (q, cik, cusip, issuer, sh, val)


def write_quarter(rows, path):
    rows = sorted(rows, key=lambda r: (int(r[1]) if str(r[1]).isdigit() else 0, r[1], r[2]))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["cik", "cusip", "issuer", "shares", "value_usd"])
        for r in rows:
            w.writerow([r[1], r[2], r[3], r[4], r[5]])


def _zip_end_quarter(name):
    m = re.match(r"(\d{4})q([1-4])_", name, re.I)
    if m:
        return "%sQ%s" % (m.group(1), m.group(2))
    m = re.match(r"\d{2}[a-z]{3}\d{4}-(\d{2}[a-z]{3}\d{4})_", name, re.I)
    if m:
        d = datetime.strptime(m.group(1).title(), "%d%b%Y")
        return "%dQ%d" % (d.year, (d.month - 1) // 3 + 1)
    return None


def run(data_dir, q_from, q_to, index_url, client):
    raw = os.path.join(data_dir, "13f", "raw")
    os.makedirs(raw, exist_ok=True)
    links = parse_index(client.get(index_url).decode("utf-8", "replace"))
    paths = []
    for url, name in links:
        eq = _zip_end_quarter(name)
        if q_from and eq and eq < q_from:
            continue  # data set ends before the requested range
        dest = os.path.join(raw, name)
        if not (os.path.exists(dest) and os.path.getsize(dest) > 0 and zipfile.is_zipfile(dest)):
            blob = client.get(urljoin(index_url, url))
            with open(dest + ".part", "wb") as f:
                f.write(blob)
            os.replace(dest + ".part", dest)
        paths.append(dest)
    subs, cov = {}, {}
    for p in sorted(paths):
        s, c = read_meta(p)
        subs.update(s)
        cov.update(c)
    decisions = apply_amendments(subs, cov)
    by_q = {}
    for p in sorted(paths):
        for row in read_zip(p, decisions):
            q = row[0]
            if (q_from and q < q_from) or (q_to and q > q_to):
                continue
            slot = by_q.setdefault(q, {}).setdefault((row[1], row[2]), [row[3], 0, 0])
            slot[1] += row[4]
            slot[2] += row[5]
    for q, d in by_q.items():
        write_quarter([(q, k[0], k[1], v[0], v[1], v[2]) for k, v in d.items()],
                      os.path.join(data_dir, "13f", "holdings_%s.csv" % q))
    return sorted(by_q)


def main(argv=None, env_path=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--from", dest="q_from")
    ap.add_argument("--to", dest="q_to")
    ap.add_argument("--index-url", default=DEFAULT_INDEX)
    args = ap.parse_args(argv)
    load_dotenv(env_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise SystemExit(MISSING_UA)
    try:
        qs = run(args.data, args.q_from, args.q_to, args.index_url, SecClient(ua))
    except SecAbort as e:
        raise SystemExit(str(e))
    print("wrote %d quarters" % len(qs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
