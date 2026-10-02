#!/usr/bin/env python3
"""Map 13F CUSIPs to tickers via the OpenFIGI v3 mapping API (stdlib only).

Usage: python3 scripts/openfigi_map.py --data DIR [--min-value 1e6]

Reads DIR/13f/holdings_*.csv, maps every CUSIP held at >= min-value in any quarter that is not
already in DIR/13f/cusip_map.csv, and appends results to that cache (cusip,ticker,name,
security_type,figi,fetched_at). Unmatched CUSIPs are cached with an empty ticker and retried
after 90 days; per-job errors are not cached (retried next run). OPENFIGI_API_KEY (optional) is
loaded from .env or the environment and is never printed.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec13f  # noqa: E402

URL = "https://api.openfigi.com/v3/mapping"
COLS = ["cusip", "ticker", "name", "security_type", "figi", "fetched_at"]
PREFERRED = {"Common Stock", "ETP", "ADR", "REIT", "Closed-End Fund", "Open-End Fund", "MLP"}
RETRY_NO_MATCH_DAYS = 90
COMMIT_EVERY = 50
MAX_RETRY_AFTER_S = 60
MAX_429_RETRIES = 8
MAX_FAILURES = 5
LIMITS = {False: (25, 10), True: (250, 100)}  # has_key -> (requests/min, jobs/request)


def _urllib_post(url, jobs, headers):
    req = urllib.request.Request(url, data=json.dumps(jobs).encode(), method="POST",
                                 headers=dict(headers, **{"Content-Type": "application/json"}))
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, dict(r.headers), json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), None
    except (urllib.error.URLError, OSError, ValueError):
        return 0, {}, None


def normalise(t):
    return (t or "").strip().upper().replace("/", ".")


def choose(results):
    for r in results:
        if r.get("securityType") in PREFERRED:
            return r
    return results[0]


def load_cache(path):
    if not os.path.isfile(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["cusip"]: r for r in csv.DictReader(f)}


def save_cache(path, m):
    tmp = path + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLS, extrasaction="ignore")
        w.writeheader()
        for k in sorted(m):
            w.writerow(m[k])
    os.replace(tmp, path)


def clean_cusip(c):
    return (c or "").strip().upper()


def aggregate_rows(rows, min_value, wanted, acc):
    """Stream rows into wanted (cusips >= min_value) and acc[cusip] += value. O(unique cusips)."""
    for r in rows:
        try:
            c, v = clean_cusip(r["cusip"]), float(r["value_usd"])
        except (KeyError, ValueError):
            continue
        if not c:
            continue
        acc[c] = acc.get(c, 0.0) + v
        if v >= min_value:
            wanted.add(c)


def read_holdings(data_dir, min_value):
    """Single streaming pass per file. Returns (wanted, {quarter: {cusip: value}})."""
    wanted, by_q = set(), {}
    for p in sorted(glob.glob(os.path.join(data_dir, "13f", "holdings_*.csv"))):
        q = os.path.basename(p)[len("holdings_"):-len(".csv")]
        with open(p, newline="", encoding="utf-8") as f:
            aggregate_rows(csv.DictReader(f), min_value, wanted, by_q.setdefault(q, {}))
    return wanted, by_q


def _fresh_nomatch(row, now_dt):
    try:
        t = datetime.strptime(row["fetched_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (KeyError, ValueError):
        return False
    return now_dt - t < timedelta(days=RETRY_NO_MATCH_DAYS)


def coverage(data_dir, by_q, cache):
    """Returns (snapshot name, latest (q, share), [(q, share)], all-quarters share) or None."""
    ups = sorted(glob.glob(os.path.join(data_dir, "universe", "universe_*.csv")))
    if not ups or not by_q:
        return None
    with open(ups[-1], newline="", encoding="utf-8") as f:
        uni = {normalise(r["ticker"]) for r in csv.DictReader(f)}
    mapped = {c for c, r in cache.items() if r.get("ticker") and normalise(r["ticker"]) in uni}
    per, tot_all, got_all = [], 0.0, 0.0
    for q in sorted(by_q):
        tot = sum(by_q[q].values())
        got = sum(v for c, v in by_q[q].items() if c in mapped)
        per.append((q, got / tot if tot else 0.0))
        tot_all += tot
        got_all += got
    return os.path.basename(ups[-1]), per[-1], per, (got_all / tot_all if tot_all else 0.0)


def run(data_dir, min_value, post, api_key=None, now=time.monotonic, sleep=time.sleep,
        batch_override=None):
    path = os.path.join(data_dir, "13f", "cusip_map.csv")
    wanted, by_q = read_holdings(data_dir, min_value)
    cache = load_cache(path)
    now_dt = datetime.now(timezone.utc)
    want = sorted(wanted)
    todo = [c for c in want if c not in cache or
            (not cache[c].get("ticker") and not _fresh_nomatch(cache[c], now_dt))]
    per_min, batch = LIMITS[bool(api_key)]
    if batch_override:
        batch = batch_override
    interval = 60.0 / per_min
    headers = {"X-OPENFIGI-APIKEY": api_key} if api_key else {}
    print("%d cusips wanted, %d cached, %d to map" % (len(want), len(want) - len(todo), len(todo)))

    state = {"last": None, "requests": 0, "saved": 0}
    dirty = False

    def request(jobs):
        if state["last"] is not None:
            wait = state["last"] + interval - now()
            if wait > 0:
                sleep(wait)
        state["last"] = now()
        state["requests"] += 1
        return post(URL, jobs, headers)

    i, failures, retries429 = 0, 0, 0
    size = batch
    try:
        while i < len(todo):
            chunk = todo[i:i + size]
            jobs = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in chunk]
            status, hdrs, body = request(jobs)
            if status == 429:
                retries429 += 1
                if retries429 > MAX_429_RETRIES:
                    raise SystemExit("OpenFIGI: persistent 429, progress saved")
                try:
                    ra = int(float({k.lower(): v for k, v in hdrs.items()}.get("retry-after", "")))
                except (ValueError, OverflowError):
                    ra = 2 ** retries429
                sleep(min(max(ra, 1), MAX_RETRY_AFTER_S))
                continue
            retries429 = 0
            if status == 413:
                if size == 1:
                    raise SystemExit("OpenFIGI: 413 on a single job")
                size = max(1, size // 2)
                continue
            if status != 200 or not isinstance(body, list) or len(body) != len(chunk):
                failures += 1
                if failures >= MAX_FAILURES:
                    raise SystemExit("OpenFIGI: %d consecutive failures (last status %s), progress saved"
                                     % (failures, status))
                sleep(min(2 ** failures, MAX_RETRY_AFTER_S))
                continue
            failures = 0
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            for c, el in zip(chunk, body):
                if not isinstance(el, dict) or "error" in el:
                    continue  # retry on a later run
                data = el.get("data") or []
                if data:
                    m = choose(data)
                    cache[c] = dict(cusip=c, ticker=normalise(m.get("ticker")), name=m.get("name") or "",
                                    security_type=m.get("securityType") or "", figi=m.get("figi") or "",
                                    fetched_at=stamp)
                else:  # warning: definitive no-match
                    cache[c] = dict(cusip=c, ticker="", name="", security_type="", figi="",
                                    fetched_at=stamp)
                dirty = True
            i += len(chunk)
            if dirty and state["requests"] - state["saved"] >= COMMIT_EVERY:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                save_cache(path, cache)
                state["saved"], dirty = state["requests"], False
    finally:
        if dirty:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_cache(path, cache)
    cov = coverage(data_dir, by_q, cache)
    if cov:
        snap, (lq, lshare), per, allshare = cov
        print("coverage: %.1f%% of %s 13F dollar value maps to a ticker in %s"
              % (lshare * 100, lq, snap))
        for q, sh in per:
            print("  %s: %.1f%%" % (q, sh * 100))
        print("  all quarters: %.1f%%" % (allshare * 100))
    else:
        print("coverage: no universe snapshot found")
    return cache


def main(argv=None, env_path=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--min-value", type=float, default=1e6)
    args = ap.parse_args(argv)
    sec13f.load_dotenv(env_path or sec13f.find_dotenv(args.data))
    run(args.data, args.min_value, _urllib_post, api_key=os.environ.get("OPENFIGI_API_KEY") or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
