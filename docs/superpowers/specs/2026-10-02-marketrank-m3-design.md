# MarketRank M3: backfill, walk-forward test, then optimizer or better flows

**Status:** approved by the user on 2026-10-02, with the §7 defaults accepted as proposed, except that **rebalancing is weekly** (user decision, same day: any edge is expected within about a week; monthly is a secondary comparison).
**Builds on:** `2026-10-01-marketrank-design.md` (model, engine, Fluxscape).

## 0. The one question M3 answers

> If we had followed MarketRank's signals month by month, would the portfolio have beaten simply holding the current mix, after trading costs?

M3 answers it in two steps:
- **Backfill:** download many more years of daily prices, so that luck can be told apart from a pattern.
- **Walk-forward:** step through those years one month at a time. Each month the rule is decided using only data up to that month, and it is scored on the month that follows. The model never sees the future.

The outcome decides what comes next:
- **If yes, consistently:** build the optimizer that moves the portfolio, and paper-track it in a shadow ledger before anyone trusts it (§4).
- **If no:** that was learned cheaply and without risking money, and the work turns to better flow data: ETF flows, 13F, signed order flow (§5).

**Non-goals:** placing orders, and live trading. MarketRank stays advisory.

## 1. Backfill

- **Source:** Alpaca `/v2/stocks/bars`, SIP feed, `adjustment=all`, timeframe 1d. Bars go back to **2016-01-01**, which is as far as Alpaca's SIP daily history reliably extends.
  - It uses the existing sync. `--lookback-days` or a new `--backfill-from YYYY-MM-DD` sets the window start, and `covered_from` back-fill fetches only what is missing.
  - The split/adjustment check (spec §4.3) applies unchanged.
- **Size estimate:** 10K tickers × about 2,450 bars ≈ 24M bars, roughly 250–350 MB of Parquet. The previous run gave 302 requests for 2.38M bars, which scales to about 3,000 requests: about 17 min at 180 per minute, plus commit time.
- **Universe over time.** Each month uses a point-in-time liquidity ranking: the top N by trailing median dollar volume, computed from bars available up to that month. The current snapshot's universe is never used for the past.
  - **Survivorship bias remains.** Alpaca's asset list contains only today's tickers, so stocks delisted before today are missing. The bias favours strategies that held survivors.
  - Mitigation:
    - report the bias as a limitation;
    - re-run the test on a large-cap sub-universe (top 500 by point-in-time liquidity), where delistings are rare;
    - treat any edge that exists only in small names as suspect.
- **Runtime check:** a full causal replay over about 2,450 bars at about 146 ms per step takes about 6 minutes per configuration at 10K nodes. Peak memory must stay under 4 GB; measure it.

## 2. Walk-forward harness: `marketrank --walkforward`

### 2.1 Timeline and causality
- The pipeline already steps bar by bar and is causal. The harness records, at each **rebalance date** (default monthly: the last trading day of each month), the signal state computed from bars up to that date.
- Trades execute at the **next day's open**, the tradeable open-to-open convention already used in the eval. Holdings are marked at each close.
- The first 12 months are warm-up only: the accumulators fill and the trailing statistics form.

### 2.2 Candidate signals
Each signal is a cross-sectional z-score over active stocks:
- the MarketRank score π·N;
- the heartbeat Δlog(π·N) over 1, 5 and 20 bars;
- π relative to size, log(π/s);
- hotness h, including its *sign-flipped* version. The eval found a significant reversal: hot names underperform.
- inflow momentum, from the fast accumulator against the slow one;
- the existing forecast score.

**Research table.** For every signal and each horizon (1, 5, 20 trading days), report the walk-forward mean IC, its t-statistic, and its stability by year. Every configuration that is run goes into a registry, so nothing is cherry-picked.

### 2.3 Strategy: a constrained tilt of the base portfolio
- **Base:** the current mix (AAPL 60%, VOO 15%, NVDA 7%, LLY 6%, NVO 5%, NKE 3.5%, F 3.5%). Over history it is held fixed. ETFs are allowed, per the user's decision.
- **Tilt:** a fixed **tilt budget**, by default 20% of portfolio value, is spread over the top-K names by the chosen signal (K = 10 by default). The tilt is funded pro-rata from the base holdings.
- **Constraints:**
  - long-only;
  - a cap of 10% of portfolio value on any single tilt name;
  - eligibility: a trailing median dollar volume of at least $50M;
  - a cap on turnover per rebalance.
- **Costs:** 10 bps per side by default, with a sensitivity check at 0, 10 and 25 bps. At $1M of capital, market impact is negligible for eligible names. The cost model is still explicit and testable.
- **Benchmarks:**
  - (a) buy-and-hold of the base mix;
  - (b) the base mix rebalanced monthly;
  - (c) VOO.

### 2.4 Gated blend (spec §7.2b)
- **Weights:** simplex weights over the candidate signals are learned on a trailing window, by default 36 months. Learning uses purging and an embargo, so that no training label overlaps the test month.
- **Gate:** the blend is used only when its trailing out-of-sample IC has t > 2. Otherwise the strategy holds the base portfolio with no tilt. A dead signal therefore costs only the trading costs that the gate avoids.

### 2.5 Metrics and statistics
- Cumulative return, plus annualized return and volatility.
- Excess return over each benchmark, the information ratio, and the Sharpe ratio.
- Maximum drawdown, turnover, cost drag, and the hit rate by year.
- **Uncertainty:** a block-bootstrap 95% interval on annualized excess return, with 1-month blocks. Because many configurations are tried, report the **deflated Sharpe ratio**.
- **Outputs:**
  - `walkforward/<run-id>/` holding results.json, trades.csv and equity.csv;
  - `--walkforward-report` producing a markdown summary;
  - an equity-curve SVG (strategy against the benchmarks) for the README and the paper.

## 3. Decision gate (proposed thresholds; the user confirms them)

**"Yes, consistently"** requires all of these:
1. annualized excess return over buy-and-hold of the base mix > 0 at 10 bps, with a bootstrap 95% interval above 0;
2. excess return positive in at least 60% of calendar years;
3. a deflated Sharpe probability above 0.95;
4. a maximum drawdown no worse than the base mix's plus 5 percentage points;
5. the result holds on the large-cap sub-universe (survivorship check).

Anything less is **"no"**, and M3 continues with §5.

**Amendment (2026-10-02, after the M3a final review).**
- **c3 is applied to the excess series.** From 2026-10-02, criterion 3 reads: the deflated probability of the
  daily information ratio of the excess return over buy-and-hold of the base mix is above 0.95. In formula terms it
  is `deflated_sharpe(ir_daily, T, skew_e, kurt_e, trial_ir_var, n_ir_trials)`, where e is the daily excess. The
  trials are the registry rows that carry an `ir_daily`.
  - An undefined IR fails c3. This happens when the excess has zero variance or there are no days.
  - The deflated total Sharpe is still reported, as "DSR(total), informational".
- **Why it changed.** The final review found that the original wording deflated the strategy's total Sharpe ratio.
  A portfolio that is 80% the base mix has a high total Sharpe whatever the tilt does, so that criterion did not
  test the excess. It "passed" (0.989) while the excess was −4.09% a year.
- **The M3a verdict is unchanged under either definition.** It is "no". Under the old c3 the run fails on c1, c2
  and c5; under the new one it fails c3 as well (DSR of the excess ≈ 3×10⁻⁵ on the main run).
- **Pre-registered secondaries for the next experiment (M3c).** These are reported, not gated.
  1. Every strategy against the base rebalanced on the same calendar (`bench:rebalanced`), with the same metrics:
     excess, CI95, IR and DSR of the excess.
  2. Base-free sleeves, one per signal and one for the blend. Each sleeve has tilt 1 and holds the top k names at
     1/k each, with the same costs and turnover cap; a flat (gate-closed) blend holds the benchmark. They are
     measured against an equal-weight portfolio of each rebalance's eligible names, rebalanced on the same calendar
     (`bench:ew_eligible`).
  - Both secondaries remove the hindsight-selected base, which biases c1 against tilting away from it.
  - Sleeves are not registry trials.

## 4. If yes: optimizer and shadow ledger
- **Optimizer:** the same tilt, generalized as a turnover-penalized mean-variance or fractional-Kelly tilt of the base, with a covariance estimated from trailing returns.
- **Schedule:** it produces proposals daily or weekly. It stays advisory and never places orders.
- **Shadow ledger:** a paper portfolio of $1M, starting from the base mix, that follows the proposals.
  - Every change to a strategy parameter creates a **new strategy version**. Old versions keep running, and history is never reset (user requirement).
  - It is stored in the lake/catalog.
- **UI:** a ledger panel showing the equity curve against buy-and-hold, plus the current proposals. It follows the fixed-UI rule: no knobs.

## 5. If no (or in parallel later): better flow data
- **ETF creation/redemption flows:** the daily change in shares outstanding times NAV gives observed dollars. These are pushed down to constituents through ETF holdings.
- **13F paired flows** (approved for M3): each manager's quarterly position cuts paired against that manager's adds, aggregated across all managers. This is an observed account-level T, used to calibrate the inferred pairing (λ, the lag) and to validate it.
- **Signed order flow:** Lee–Ready classification on trades, for the S&P 500 at minute resolution.
- **Lagged pairing:** an outflow at bar t can be absorbed by inflows over t…t+L.
- **`--flows` CSV ingest** (`t,from,to,dollars`): plugs in observed pairs directly.

Each of these is re-tested with the same walk-forward harness (§2).

## 6. Engineering rules (unchanged from M1–M2)
- Bit-identical results across thread counts: index-owned writes and serial floating-point reductions.
- No O(N²) work.
- Causal by construction. A test asserts that changing any future bar leaves every past rebalance decision bit-identical.
- Synthetic fixtures for every metric. A planted-signal test must show that the harness detects a known edge and finds no edge in noise.

## 7. Defaults (accepted 2026-10-02; changes come later as new strategy versions)
1. Rebalance frequency: monthly by default, with weekly as an option.
2. Tilt budget (20%) and K (10).
3. Cost assumption (10 bps per side).
4. Whether the base mix stays fixed as the core holding, or the optimizer may also resize AAPL's 60% position.
5. The thresholds in §3 (c3 amended 2026-10-02 to the IR of the excess, see §3; the secondaries listed there are
   pre-registered for M3c).
