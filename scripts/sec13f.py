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
import itertools
import re
import sqlite3
import sys
import time
import urllib.request
import urllib.error
import zipfile
from datetime import datetime
from decimal import Decimal
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
    """Return (status, body, retry_after). On 200 the body is the open response (file-like)."""
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
    try:
        return 200, urllib.request.urlopen(req, timeout=120), 0
    except urllib.error.HTTPError as e:
        try:
            ra = int(e.headers.get("Retry-After", "0") or 0)
        except ValueError:
            ra = 0
        return e.code, b"", ra
    except (urllib.error.URLError, OSError):
        return 0, b"", 0


CHUNK = 1 << 20


class SecClient:
    def __init__(self, ua, min_interval=0.15, get=_urllib_get, sleep=time.sleep, now=time.monotonic):
        self._ua, self._min, self._get, self._sleep, self._now = ua, min_interval, get, sleep, now
        self._last = None
        self._fails = 0

    def _fetch(self, url):
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
            if status in (429, 503) and retry_after > 0:
                self._sleep(min(retry_after, MAX_RETRY_AFTER_S))
            else:
                self._sleep(delay)
            delay = min(delay * 2, MAX_RETRY_AFTER_S)

    def get(self, url):
        body = self._fetch(url)
        if isinstance(body, (bytes, bytearray)):
            return bytes(body)
        try:
            return body.read()
        finally:
            getattr(body, "close", lambda: None)()

    def download(self, url, dest):
        """Stream to dest.part in ~1 MB chunks, then rename to dest."""
        body = self._fetch(url)
        part = dest + ".part"
        try:
            with open(part, "wb") as f:
                if isinstance(body, (bytes, bytearray)):
                    f.write(body)
                else:
                    while True:
                        chunk = body.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
        finally:
            getattr(body, "close", lambda: None)()
        os.replace(part, dest)


def load_dotenv(path):
    """Minimal .env parser: sets only variables that are missing. Prints nothing."""
    if not path:
        return
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


def find_dotenv(data_dir, cwd=None, root=None):
    """First existing .env among: cwd, parent of data_dir, repo root."""
    root = root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    for d in (cwd or os.getcwd(), os.path.dirname(os.path.abspath(data_dir)), root):
        p = os.path.join(d, ".env")
        if os.path.isfile(p):
            return p
    return None


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
    if not s:
        return 0
    try:
        return int(s)
    except ValueError:
        d = Decimal(s)
        return int(d) if d == d.to_integral_value() else d


def _table(z, name):
    for n in z.namelist():
        if n.rsplit("/", 1)[-1].upper() == name:
            return csv.DictReader(io.TextIOWrapper(z.open(n), encoding="utf-8-sig", errors="replace", newline=""),
                                  delimiter="\t", quoting=csv.QUOTE_NONE)
    return iter(())


def read_meta(path):
    """(submissions, coverpages) dicts keyed by accession, parsed by header name."""
    subs, cov = {}, {}
    with zipfile.ZipFile(path) as z:
        for r in _table(z, "SUBMISSION.TSV"):
            g = lambda k: (r.get(k) or "").strip()
            if not g("ACCESSION_NUMBER") or not g("PERIODOFREPORT") or not g("FILING_DATE"):
                continue  # unusable row (short or blank)
            subs[g("ACCESSION_NUMBER")] = {"cik": g("CIK"), "period": g("PERIODOFREPORT"),
                                           "filing_date": g("FILING_DATE"), "type": g("SUBMISSIONTYPE")}
        for r in _table(z, "COVERPAGE.TSV"):
            acc = (r.get("ACCESSION_NUMBER") or "").strip()
            if acc:
                cov[acc] = {"amendment_type": (r.get("AMENDMENTTYPE") or "").strip().upper(),
                            "is_amendment": (r.get("ISAMENDMENT") or "").strip().upper()}
    return subs, cov


def apply_amendments(submissions, coverpages, stats=None):
    """{accession: 'keep'|'replace'|'add'} for the filings whose rows to use.

    Per (cik, period), only 13F-HR / 13F-HR/A. Filings are ordered by filing date;
    the latest RESTATEMENT (or the original if none) is the base and supersedes
    everything before it; NEW HOLDINGS amendments filed after the base add rows.
    Amendments with a blank/unrecognised AMENDMENTTYPE are unknown: skipped and
    counted in stats["unknown_amendments"] (never double-counted).
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
                t = c.get("amendment_type", "")
                return {"RESTATEMENT": "replace", "NEW HOLDINGS": "add"}.get(t, "unknown")
            return "keep"
        kinds = {}
        for a in accs:
            k = kind(a)
            if k == "unknown":
                if stats is not None:
                    stats["unknown_amendments"] = stats.get("unknown_amendments", 0) + 1
            else:
                kinds[a] = k
        accs = [a for a in accs if a in kinds]
        base = max([i for i, a in enumerate(accs) if kinds[a] in ("replace", "keep")], default=None)
        if base is None:
            # only NEW HOLDINGS amendments seen (original in another data set): keep them all
            for a in accs:
                out[a] = "add"
            continue
        for a in accs[base:]:
            out[a] = kinds[a]
    return out


def read_zip(path, decisions=None, subs=None, stats=None):
    """Stream filtered rows (quarter, cik, cusip, issuer, shares, value_usd), one per info-table row.

    Not aggregated (bounded memory); see aggregate(). Pass subs/decisions to avoid re-reading metadata.
    """
    if subs is None or decisions is None:
        subs, cov = read_meta(path)
        decisions = apply_amendments(subs, cov, stats) if decisions is None else decisions
    with zipfile.ZipFile(path) as z:
        for r in _table(z, "INFOTABLE.TSV"):
            acc = r.get("ACCESSION_NUMBER") or ""
            if acc not in decisions or acc not in subs:
                continue
            if (r.get("SSHPRNAMTTYPE") or "").strip().upper() != "SH" or (r.get("PUTCALL") or "").strip():
                continue
            cusip = (r.get("CUSIP") or "").strip().upper()
            if not cusip:
                if stats is not None:
                    stats["empty_cusip"] = stats.get("empty_cusip", 0) + 1
                continue
            s = subs[acc]
            yield (quarter_of(s["period"]), s["cik"], cusip, (r.get("NAMEOFISSUER") or "").strip(),
                   _num(r.get("SSHPRNAMT")), value_to_usd(_num(r.get("VALUE")), s["filing_date"]))


def aggregate(rows):
    """In-memory sum of rows per (quarter, cik, cusip); sorted. For tests and small inputs."""
    agg = {}
    for q, cik, cusip, issuer, sh, val in rows:
        cur = agg.setdefault((q, cik, cusip), [issuer, 0, 0])
        cur[1] += sh
        cur[2] += val
    return [(q, cik, cusip, v[0], v[1], v[2]) for (q, cik, cusip), v in sorted(agg.items())]


def _write_rows(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["cik", "cusip", "issuer", "shares", "value_usd"])
        for r in rows:
            w.writerow([r[1], r[2], r[3], r[4], r[5]])


def _cik_key(r):
    return (int(r[1]) if str(r[1]).isdigit() else 0, r[1], r[2])


def write_quarter(rows, path):
    _write_rows(sorted(rows, key=_cik_key), path)


def _zip_range_quarters(name):
    """(start_quarter, end_quarter) from a data-set filename, or (None, None)."""
    def q(d):
        return "%dQ%d" % (d.year, (d.month - 1) // 3 + 1)
    m = re.match(r"(\d{4})q([1-4])_", name, re.I)
    if m:
        qq = "%sQ%s" % (m.group(1), m.group(2))
        return qq, qq
    m = re.match(r"(\d{2}[a-z]{3}\d{4})-(\d{2}[a-z]{3}\d{4})_", name, re.I)
    if m:
        a, b = (datetime.strptime(x.title(), "%d%b%Y") for x in m.groups())
        return q(a), q(b)
    return None, None


def _next_quarter(qs):
    y, n = int(qs[:4]), int(qs[5])
    return "%dQ%d" % (y + (n == 4), n % 4 + 1)


def run(data_dir, q_from, q_to, index_url, client, stats=None):
    stats = stats if stats is not None else {}
    base = os.path.join(data_dir, "13f")
    raw = os.path.join(base, "raw")
    os.makedirs(raw, exist_ok=True)
    links = parse_index(client.get(index_url).decode("utf-8", "replace"))
    paths = []
    for url, name in links:
        start, end = _zip_range_quarters(name)
        if q_from and end and end < q_from:
            continue  # data set ends before the requested range
        # Filings come in after the period ends, so a data set starting one quarter past --to
        # can still hold late period<=to filings; skip only sets starting later than that.
        if q_to and start and start > _next_quarter(q_to):
            continue
        dest = os.path.join(raw, name)
        if not (os.path.exists(dest) and os.path.getsize(dest) > 0 and zipfile.is_zipfile(dest)):
            client.download(urljoin(index_url, url), dest)
        paths.append(dest)
    paths.sort()
    metas = {}
    all_subs, all_cov = {}, {}
    for p in paths:
        metas[p] = read_meta(p)  # once per ZIP
        all_subs.update(metas[p][0])
        all_cov.update(metas[p][1])
    decisions = apply_amendments(all_subs, all_cov, stats)
    del all_subs, all_cov

    stage = os.path.join(base, "staging.sqlite")
    for ext in ("", "-journal", "-wal", "-shm"):
        if os.path.exists(stage + ext):
            os.remove(stage + ext)
    db = sqlite3.connect(stage)
    try:
        db.execute("CREATE TABLE h (q TEXT, cik TEXT, cusip TEXT, issuer TEXT, shares TEXT, value TEXT)")
        for p in paths:
            subs = metas[p][0]
            rows = (r for r in read_zip(p, decisions, subs, stats)
                    if not ((q_from and r[0] < q_from) or (q_to and r[0] > q_to)))
            while True:
                batch = list(itertools.islice(rows, 50000))
                if not batch:
                    break
                db.executemany("INSERT INTO h VALUES (?,?,?,?,?,?)",
                               [(a, b, c, d, str(e), str(f)) for a, b, c, d, e, f in batch])
            db.commit()
            metas[p] = None
        db.execute("CREATE INDEX hq ON h(q)")
        quarters = [r[0] for r in db.execute("SELECT DISTINCT q FROM h ORDER BY q")]
        for q in quarters:
            cur = db.execute("SELECT cik, cusip, issuer, shares, value FROM h WHERE q=? "
                             "ORDER BY CAST(cik AS INTEGER), cik, cusip, rowid", (q,))

            def grouped():
                for (cik, cusip), grp in itertools.groupby(cur, key=lambda r: (r[0], r[1])):
                    issuer, sh, val = None, 0, 0
                    for _, _, iss, a, b in grp:
                        if issuer is None:
                            issuer = iss
                        sh += _num(a)
                        val += _num(b)
                    yield (q, cik, cusip, issuer, sh, val)
            _write_rows(grouped(), os.path.join(base, "holdings_%s.csv" % q))
    finally:
        db.close()
    return quarters


def main(argv=None, env_path=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--from", dest="q_from")
    ap.add_argument("--to", dest="q_to")
    ap.add_argument("--index-url", default=DEFAULT_INDEX)
    args = ap.parse_args(argv)
    load_dotenv(env_path or find_dotenv(args.data))
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise SystemExit(MISSING_UA)
    stats = {}
    try:
        qs = run(args.data, args.q_from, args.q_to, args.index_url, SecClient(ua), stats)
    except SecAbort as e:
        raise SystemExit(str(e))
    if stats.get("unknown_amendments"):
        print("warning: skipped %d amendments with unknown AMENDMENTTYPE" % stats["unknown_amendments"],
              file=sys.stderr)
    if stats.get("empty_cusip"):
        print("warning: skipped %d rows with empty CUSIP" % stats["empty_cusip"], file=sys.stderr)
    print("wrote %d quarters" % len(qs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
