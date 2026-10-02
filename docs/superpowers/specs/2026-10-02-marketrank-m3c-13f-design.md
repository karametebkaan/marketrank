# MarketRank M3c: observed paired flows from SEC Form 13F

**Status:** user-approved direction, 2026-10-02. The user chose to start now, in parallel with M3a, and to map CUSIPs with OpenFIGI.

**Why.** MarketRank needs paired flows: who sold what to buy what. The flows used so far are *estimated* from public bars. Form 13F gives *observed* account-level pairing. Every institutional manager with more than $100M in US equities files its full long holdings each quarter. Comparing a manager's holdings quarter over quarter shows which positions it cut (money out) and which it added (money in), inside the same account. Summed over about 5,000+ managers, that is an observed flow matrix T_{i→j}, from public EDGAR data.

## 1. Data
- **Source:** SEC "Form 13F Data Sets", quarterly ZIP files linked from `https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets`.
  - Discover the file list from that page; file naming changed over time.
  - Each ZIP holds `SUBMISSION.tsv`, `COVERPAGE.tsv` and `INFOTABLE.tsv`.
  - Fetch with the same SEC fair-access rules as the sector sync: `SEC_USER_AGENT` from `.env` (never printed), ≥ 150 ms between requests, abort on 403, honour Retry-After.
- **Filters:**
  - Report type 13F-HR only. Drop notice filings (13F-NT).
  - Per (CIK, period) keep the latest filing. A restatement amendment replaces the holdings; a "new holdings" amendment adds to them.
  - Info-table rows: keep `SSHPRNAMTTYPE = SH` with `PUTCALL` empty. That means shares only, no options and no principal amounts.
- **Quarters:** all quarters that overlap the lake. After the M3a backfill, that is 2016-Q1 onward.

## 2. CUSIP → ticker
- Use the OpenFIGI v3 `/v3/mapping` endpoint with `idType = ID_CUSIP`, preferring `exchCode = US` and common or ETP security types.
  - `OPENFIGI_API_KEY` in `.env` is optional and raises rate limits; never print it.
  - Without a key, respect the free limits (25 requests/min, 10 jobs per request).
  - Cache the results in `data/13f/cusip_map.csv` and resume across runs.
- **Map only CUSIPs that matter:** those held with a position ≥ $1M in any quarter. Report the share of total 13F dollar value that maps to our universe.

## 3. Observed flow matrix per quarter
- For each manager m and each pair of consecutive quarters (q−1, q), and each ticker:
  - Δshares = shares_q − shares_{q−1}. A ticker missing in a quarter counts as 0 shares.
  - Dollar change d = Δshares × P, where P is the ticker's mean close over quarter q's bars from our lake. Use the 13F reported value only as a fallback. The lake price is used for consistency, because the trade dates are unknown.
- Each manager's **sources** are the positions with d < 0 (out = −d). Its **sinks** are those with d > 0 (in = d).
  - **Pairing:** F^m_{i→j} = out_i · in_j / Σ_k in_k, scaled by min(1, Σin/Σout) so the paired dollars never exceed the smaller side.
  - The remainder is the manager's net cash in or out. It is not paired and goes into an unpaired tally.
- **T_q = Σ_m F^m.** Each manager's contribution is computed sparsely, then merged.
- **Excluded:** corporate actions, i.e. splits handled through split-adjusted shares and cash-out mergers. Do this by adjusting shares with the lake's split ratio at quarter end. A position that vanishes because its ticker was delisted counts as a source.

## 4. Comparison with the estimated flows
For each quarter q overlapping the lake:
- **Estimated T̂_q:** the slow-flux contributions summed over q's bars. Use a pipeline run under the marketrank preset with accumulation restricted to the quarter, i.e. a fresh FluxAccumulator per quarter with a very long half-life.
- **Metrics:**
  - edge agreement: Spearman correlation of weights over the union of the top-5,000 edges of each matrix, and the overlap of the top-k (k = 500, 2,000) edge sets;
  - node agreement: Spearman correlation of in-dollars and of out-dollars;
  - **MarketRank agreement:** solve π on T_q (observed) and on T̂_q with the same solver (p = 0.15). Report the Spearman correlation of π and the overlap of the top 50.
  - Shuffled baselines for each metric: permute node labels and recompute.
- **Calibration:** grid over the affinity λ ∈ {0, 0.5, 1} and the pressure modes (dollar, sqrt). Report which setting agrees best with observed flows, and by how much.
- **Honest limits:**
  - 13F is quarterly, long-only and institutional, with a 45-day lag. It omits shorts, retail and intra-quarter round trips.
  - Fund inflows and outflows appear as unpaired cash, not pairs.

## 5. Outputs
- **CLI:**
  - `--sync-13f [--13f-from YYYYQn]` downloads, parses and maps;
  - `--compare-13f`, in replay mode, builds T_q, compares and calibrates.
- **Data:** `data/13f/` holds the raw ZIP cache, per-quarter holdings CSVs, `cusip_map.csv`, and per-quarter `t13f_<q>.csv` edge lists.
- **Report:** `data/13f/report.md` plus a JSON file, then a paper section and a README paragraph.
- **Engineering rules** are those of M1–M3a: deterministic, no O(N²), no `.env` reads by agents, tests with synthetic holdings and fake HTTP.
