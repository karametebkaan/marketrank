# MarketRank M3c: Observed Paired Flows from SEC Form 13F — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an *observed* stock-to-stock flow matrix per quarter from SEC Form 13F filings: each manager's quarter-over-quarter cuts are paired against its adds. Then measure how well MarketRank's *estimated* flows and π agree with it, and calibrate the estimator.

**Architecture:**
- **Ingestion** (download, parse, filter) and **CUSIP→ticker mapping** are stdlib-only Python scripts, which is simpler for ZIP, TSV and HTTP work. They write CSVs under `data/13f/`.
- **The C++ module `src/flows13f/`**:
  - loads those CSVs and lake prices;
  - builds the paired matrix T_q;
  - computes the estimated quarter matrix T̂_q from the pipeline;
  - solves π on both and reports agreement and calibration.

  It is reached through `marketrank --compare-13f`.

**Tech Stack:** Python 3 stdlib (`urllib`, `zipfile`, `csv`, `json`, `unittest`), C++20, doctest, and the existing lake, CSV and solver code.

**Spec:** `docs/superpowers/specs/2026-10-02-marketrank-m3c-13f-design.md`

## Global Constraints

**Credentials.** SEC requests use `SEC_USER_AGENT`, loaded from `.env` by the script itself. Optional OpenFIGI requests use `OPENFIGI_API_KEY`. **Never print either value.** Agents never read, print or grep `.env`; the scripts load it.

**SEC fair access.** At least 150 ms between requests. Abort on 403. Honour Retry-After, capped at 60 s. Abort after 20 consecutive failures.

**OpenFIGI.**
- Without a key: ≤ 25 requests/min, ≤ 10 jobs per request.
- With a key: ≤ 250 requests/min, ≤ 100 jobs per request.
- Results are cached and resumable in `data/13f/cusip_map.csv`.

**13F filters:**
- 13F-HR only.
- Per (CIK, period) keep the latest filing. A RESTATEMENT replaces the holdings; NEW HOLDINGS amendments add rows.
- Info-table rows must have `SSHPRNAMTTYPE == SH` and an empty `PUTCALL`.

**Pairing (per manager, per quarter):**
- d = Δshares · P̄_q, where P̄_q is the mean close over quarter q's lake bars.
- Sources have d < 0 and sinks d > 0.
- F_{i→j} = out_i · in_j / Σin · min(1, Σin/Σout).
- The remainder is unpaired.
- T_q = Σ_managers F.

**Engineering rules:**
- Deterministic; no O(N²).
- C++ tests use synthetic holdings. Python tests use a fake HTTP layer and fixture ZIPs; no network in tests.
- Never run `--mode alpaca`, `--refetch-full`, `--maintain` or `--migrate-cache`. Only Task 5 makes real SEC and OpenFIGI requests.

**Repo:** work in the worktree `/home/kkaramete/stocks-13f`, branch `m3c-13f`. It is merged to master at the end; M3a runs in parallel on master. Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Data lives in `/home/kkaramete/stocks/data` (pass `--data /home/kkaramete/stocks/data` or use it as cwd) and is never committed.

---

### Task 1: 13F dataset ingestion (`scripts/sec13f.py`)

**Files:**
- Create `scripts/sec13f.py`, `scripts/tests/test_sec13f.py` and `scripts/tests/fixtures/` (a tiny synthetic ZIP built in setUp).

**Interfaces:**
- **Usage:** `python3 scripts/sec13f.py --data DIR [--from 2016Q1] [--to 2026Q3] [--index-url URL]`.
- **Discovery:** fetch the index page and parse every link ending in `_form13f.zip`. Map each one to the report quarter(s) it covers, using the filename and the PERIODOFREPORT values inside.
- **Download:** fetch each ZIP into `DIR/13f/raw/`, skipping files already present with matching size.
- **Parse:** stream `SUBMISSION.tsv`, `COVERPAGE.tsv` and `INFOTABLE.tsv` straight from the ZIP.
- **Filter:** apply the Global Constraints filters, including the amendment rules.
- **Output:** write `DIR/13f/holdings_<YYYYQn>.csv` with columns `cik,cusip,issuer,shares,value_usd`.
  - VALUE units changed from thousands to dollars for filings from 2023-01-03 on. Normalise to dollars using the filing date and document the rule.
  - Aggregate duplicate (cik, cusip) rows by summing.
- **Functions** (testable): `parse_index(html) -> list[(url, name)]`, `quarter_of(period_str) -> "YYYYQn"`, `read_zip(path) -> iterator of holdings rows`, `apply_amendments(submissions, coverpages) -> {accession: keep|replace|add}`, `write_quarter(rows, path)`.
- **HTTP:** a `SecClient(ua, min_interval=0.15, get=urllib-based)`. Tests inject `get`.
  - On 403, raise `SecAbort`. On 429, sleep for Retry-After.
  - Error messages contain only paths, never the UA.

- [ ] **Step 1: Write the tests (unittest):**
  - `parse_index` on a fixture HTML page with three links.
  - `quarter_of("31-DEC-2024") == "2024Q4"`.
  - A fixture ZIP with:
    - two managers;
    - an amendment RESTATEMENT replacing manager A;
    - a NEW HOLDINGS amendment adding a row for manager B;
    - one put row, which must be dropped;
    - one PRN row, which must be dropped;
    - a pre-2023 value given in thousands.

    The output CSV must equal an expected list.
  - The fake client aborts on 403 and sleeps on 429.
  - With `SEC_USER_AGENT` missing, the script exits with the same message wording as the sector sync.
- [ ] **Step 2: Run the tests and see them fail.** `python3 -m unittest discover -s scripts/tests -v`
- [ ] **Step 3: Implement.** Load `.env` with a minimal parser that sets only missing environment variables and never prints anything.
- [ ] **Step 4: Run the tests; they pass.**
- [ ] **Step 5: Commit** with message `feat(13f): SEC Form 13F dataset ingestion with amendment handling`.

---

### Task 2: CUSIP → ticker mapping (`scripts/openfigi_map.py`)

**Files:**
- Create `scripts/openfigi_map.py` and `scripts/tests/test_openfigi_map.py`.

**Interfaces:**
- **Usage:** `python3 scripts/openfigi_map.py --data DIR [--min-value 1e6]`.
- **Input:**
  - collect the CUSIPs from every `holdings_*.csv` held with a position ≥ min-value in any quarter;
  - drop those already in `DIR/13f/cusip_map.csv`;
  - map the rest in batches.
- **Request:** POST to `https://api.openfigi.com/v3/mapping` with jobs `{"idType":"ID_CUSIP","idValue":cusip,"exchCode":"US"}`.
- **Choosing a match:** pick the first result whose `securityType` is in {Common Stock, ETP, ADR, REIT, Closed-End Fund, Open-End Fund, MLP}. Otherwise take the first result.
- **Ticker form:** normalise to the universe's form, converting the class-share separator `/` to `.`.
- **Output:** `cusip_map.csv` with columns `cusip,ticker,name,security_type,figi,fetched_at`. A CUSIP with no match is recorded with an empty ticker and not retried for 90 days.
- **Rate limits:** apply the limits from the Global Constraints, keyed on whether `OPENFIGI_API_KEY` is set. Commit to disk every 50 requests.
- **Coverage report:** print the share of total 13F dollar value whose CUSIP maps to a ticker in the latest universe snapshot.

- [ ] **Step 1: Write the tests:**
  - batching at 10 and 100 jobs;
  - rate limiting via an injected clock;
  - choosing a match from the security types;
  - ticker normalisation, e.g. `BRK/B` becomes `BRK.B`;
  - resume from a partial cache;
  - an unmatched CUSIP is cached;
  - the API key is sent as the `X-OPENFIGI-APIKEY` header and never printed.
- [ ] **Step 2: Run the tests; they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests; they pass.**
- [ ] **Step 5: Commit** with message `feat(13f): OpenFIGI CUSIP to ticker mapping with cache and rate limits`.

---

### Task 3: Observed flow matrix (`src/flows13f/observed.*`)

**Files:**
- Create `src/flows13f/observed.hpp`, `src/flows13f/observed.cpp` and `tests/test_13f_observed.cpp`.

**Interfaces:**
```cpp
namespace mr {
struct Holding { std::uint64_t cik; std::string ticker; double shares; double value_usd; };
// Reads holdings_<q>.csv joined with cusip_map.csv; rows whose CUSIP has no ticker are dropped (counted).
struct QuarterHoldings { std::string quarter; std::vector<Holding> rows; double dropped_value = 0, total_value = 0; };
QuarterHoldings load_quarter(const std::filesystem::path& dir, const std::string& quarter);
struct FlowEdge { std::uint32_t from, to; double dollars; };
struct ObservedFlows { std::string quarter; std::vector<FlowEdge> edges; double paired = 0, unpaired_in = 0, unpaired_out = 0; std::size_t managers = 0; };
// Node indices are positions in `tickers`; prices = mean close of quarter q per ticker (NaN = unknown -> value_usd/shares
// fallback when finite, else the position is skipped). split_ratio(ticker) adjusts q-1 shares to q's basis (1.0 = none).
ObservedFlows observed_flows(const QuarterHoldings& prev, const QuarterHoldings& cur, const std::vector<std::string>& tickers,
                             const std::vector<double>& prices, const std::function<double(const std::string&)>& split_ratio);
}  // namespace mr
```

**Rules:**
- **Grouping:** group by cik, so each manager is processed independently. The output is deterministic: sort by (from, to) and sum duplicates.
- **Per-manager pairing:** for each manager, d_i = (shares_cur − ratio·shares_prev)·P_i. Pair sources with sinks exactly as the Global Constraints formula states.
- **Bounding the cost:** the pairing is O(sources·sinks) per manager. If sources·sinks > 1e6, keep the top 1,000 sinks by in-dollars and spread the remainder proportionally over the kept sinks. Document that threshold.
- **Totals:** accumulate paired, unpaired_in and unpaired_out.

- [ ] **Step 1: Write the tests:**
  - **(a)** A single manager cuts A by $100 and adds $60 to B and $40 to C. The edges are A→B 60 and A→C 40, with unpaired 0.
  - **(b)** The adds total more than the cuts. The edges are scaled by Σout/Σin, and the excess counts as unpaired_in.
  - **(c)** Two managers are summed.
  - **(d)** Split adjustment: a 2:1 split with unchanged real holdings produces no flow.
  - **(e)** A position missing in prev is a new buy; a position missing in cur is a full sell.
  - **(f)** Edge order and totals are deterministic.
  - **(g)** The fallback price is used when a lake price is NaN.
- [ ] **Step 2: Run the tests; they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests; they pass.**
- [ ] **Step 5: Commit** with message `feat(13f): observed per-manager paired flow matrix per quarter`.

---

### Task 4: Comparison, calibration and CLI (`src/flows13f/compare.*`)

**Files:**
- Create `src/flows13f/compare.hpp`, `src/flows13f/compare.cpp` and `tests/test_13f_compare.cpp`.
- Modify `src/cli/args.*` and `src/main.cpp` to add `--compare-13f` and `--13f-quarters Q1,Q2,…`.

**Interfaces:**
```cpp
namespace mr {
// Sum of the pipeline's slow-flux bar contributions over the bars of one quarter (fresh accumulator, half-life 1e9).
std::vector<FlowEdge> estimated_quarter_flows(const Panel&, const CoreParams&, std::size_t first_bar, std::size_t last_bar);
struct Agreement {
  double edge_spearman, top500_overlap, top2000_overlap, in_spearman, out_spearman, pi_spearman, top50_pi_overlap;
  double shuf_edge_spearman, shuf_pi_spearman;  // label-permutation baselines (seed 13)
};
Agreement compare_flows(const std::vector<FlowEdge>& observed, const std::vector<FlowEdge>& estimated, std::size_t n);
// pi on an edge list: row-normalised out-shares, p = 0.15, dangling -> teleport (same solver as the engine).
std::vector<double> pi_of(const std::vector<FlowEdge>&, std::size_t n);
}  // namespace mr
```

**Rules:**
- **Edge Spearman:** computed over the union of each list's top-5,000 edges. A pair missing from one list counts as 0 in that list.
- **Top-k overlaps:** |A ∩ B| / k.
- **Node in/out Spearman:** computed over nodes with any flow.
- **π:** computed with the engine's `stationary` solver on a Csr built from the edges. There is no pruning beyond what is necessary.
- **Shuffled baseline:** permute the estimated matrix's node labels with a fixed seed and recompute.

**CLI behaviour.** `--compare-13f` runs in replay mode and loads the panel. For each requested quarter present in `data/13f`:
1. build the observed matrix with `observed_flows` (from the previous and current quarter's holdings, lake prices and split ratios);
2. build the estimated matrix for the quarter's bars;
3. run `compare_flows`;
4. repeat for the calibration grid λ ∈ {0, 0.5, 1} × pressure ∈ {dollar, sqrt}.

It writes `data/13f/report.md` and `report.json`.

- [ ] **Step 1: Write the tests:**
  - **(a)** `compare_flows` on identical inputs gives every Spearman = 1 and every overlap = 1.
  - **(b)** On independent random inputs every Spearman is within ±0.1 of 0 (fixed seed).
  - **(c)** `pi_of` reproduces the A/B/C example: B 0.36016, C 0.34665, A 0.29319.
  - **(d)** `estimated_quarter_flows` on a synthetic panel equals the summed `bar_flux_sparse` rows over those bars.
  - **(e)** CLI parsing of the new flags.
- [ ] **Step 2: Run the tests; they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests; they pass.** Then run the full suite.
- [ ] **Step 5: Commit** with message `feat(13f): estimated-vs-observed flow comparison, MarketRank agreement, calibration grid`.

---

### Task 5: Real run and write-up (designated live task)

- [ ] **Step 1: Ingest.** Run `python3 scripts/sec13f.py --data /home/kkaramete/stocks/data --from <first quarter overlapping the lake>`. Record the files fetched, the bytes, the managers per quarter and the rows kept.
- [ ] **Step 2: Map CUSIPs.** Run `python3 scripts/openfigi_map.py --data /home/kkaramete/stocks/data`. Record the requests made and the share of 13F dollar value covered.
- [ ] **Step 3: Compare.** Run `./build/marketrank --mode replay --timeframe 1d --compare-13f` with cwd `/home/kkaramete/stocks` (read-only lake). Record the agreement tables and the calibration winner.
- [ ] **Step 4: Write up.** Add a "Observed flows from 13F" section to the README and a paper section ("Observed paired flows from SEC Form 13F"). Each gives the method, coverage, agreement against shuffled baselines, the calibration result and the honest limits. Hand the paper text to the controller; don't revise the conclusion. The controller revises the conclusion after both M3a and M3c are in.
- [ ] **Step 5: Commit** with message `results(13f): observed vs estimated flows`.

## Self-review
- **Spec coverage:**
  - §1 data → Task 1
  - §2 mapping → Task 2
  - §3 pairing → Task 3
  - §4 comparison and calibration → Task 4
  - §5 outputs → Tasks 4–5

  The spec's `--sync-13f` CLI is replaced by the two Python scripts. That is simpler for ZIP and HTTP work, and is recorded here as a plan deviation.
- **Types:** `FlowEdge` is shared by Tasks 3 and 4, and `QuarterHoldings` is produced by Task 3's loader.
