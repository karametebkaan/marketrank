# MarketRank M3a — Backfill, Walk-Forward Test, Decision Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer the M3 question: would following MarketRank's signals month by month have beaten buy-and-hold of the current mix, after trading costs? Answer it with an honest, causal walk-forward over a backfilled history, ending in a go/no-go decision gate.

**Architecture:**
- A new `src/walkforward/` module makes **one causal pass** of `CorePipeline` over the whole panel.
  - On every bar it updates the signal trackers and accumulates information coefficients (ICs) for the research table, sampled without overlap.
  - On each rebalance date it stores cross-sectional z-scores, eligibility and the forward label.
- After the pass:
  - the backtests (single-signal strategies, the gated blend and the three benchmarks) run cheaply from the stored records;
  - metrics, a block bootstrap and the deflated Sharpe feed the decision gate;
  - results, a run registry, a markdown report and an equity SVG are written.
- The backfill reuses the existing sync with a long `--lookback-days`.

**Tech Stack:** C++20, OpenMP, doctest, nlohmann/json, the existing DuckDB/Parquet lake, and Python stdlib for the SVG.

**Spec:** `docs/superpowers/specs/2026-10-02-marketrank-m3-design.md` (§0–§3, §6–§7). The §4 optimizer/ledger and §5 observed flows are later, separate plans, gated by this plan's decision.

## Global Constraints

**Determinism and causality**
- Results must be bit-identical across thread counts. That means index-owned writes and serial floating-point reductions; do not use OpenMP reductions.
- No O(N²) work anywhere.
- Every decision is causal. Changing any bar after `t+1` leaves every rebalance decision made at a date ≤ t bit-identical.

**Trading conventions**
- Execute trades at the next day's open (open-to-open), and mark holdings at each close.
- Advisory only: no orders, no live trading.

**Defaults from spec §7 (accepted 2026-10-02)**
- Monthly rebalancing on the last trading day of the month.
- Tilt budget 20% over the top K = 10 names.
- Per-name tilt cap 10%; eligible names have trailing median dollar volume ≥ $50M; turnover cap 0.5 per rebalance.
- Costs 10 bps per side, with sensitivity runs at 0 / 10 / 25.
- Blend: trailing window 36 months, embargo 1 month, gate on t > 2 over the last 24 out-of-sample months, with at least 12 months required.
- 12-month warm-up.

**Base portfolio** (held fixed as the core): AAPL 60%, VOO 15%, NVDA 7%, LLY 6%, NVO 5%, NKE 3.5%, F 3.5%. ETFs are allowed.

**Benchmarks:** (a) buy-and-hold of the base mix; (b) the base mix rebalanced monthly; (c) VOO.

**Decision gate (spec §3).** All of these must hold:
1. Annualized excess return over (a) > 0 at 10 bps, and the lower bound of the bootstrap 95% CI > 0.
2. Excess return positive in ≥ 60% of calendar years.
3. Deflated Sharpe probability > 0.95.
4. Max drawdown ≤ base max drawdown + 0.05.
5. Holds on the large-cap sub-universe (point-in-time top 500).

**Project rules**
- Namespace `mr`. Binary `build/marketrank`, tests `build/marketrank_tests`. Tests are doctest files in `tests/`; CMake globs `src/*.cpp` and `tests/*.cpp`.
- No new web-UI controls (the UI is fixed). Developer knobs are CLI flags only.
- Never read `.env`. Only Task 8 runs `--mode alpaca`.
- Commit directly on master, staging explicit paths. Trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/walkforward/stats.hpp/.cpp` | Spearman rank correlation, mean/t, normal CDF and inverse CDF, block bootstrap |
| `src/walkforward/signals.hpp/.cpp` | `Signal` enum, `SignalTracker` (raw signals from frames, using its own history), cross-sectional z-score |
| `src/walkforward/universe.hpp/.cpp` | point-in-time eligibility, forward open-to-open returns, rebalance-date calendar |
| `src/walkforward/backtest.hpp/.cpp` | portfolio simulator: base plus tilt, costs, turnover cap, next-open execution, daily marking, benchmarks |
| `src/walkforward/blend.hpp/.cpp` | gated simplex blend with trailing window, embargo and out-of-sample gate |
| `src/walkforward/metrics.hpp/.cpp` | performance metrics, deflated Sharpe, `DecisionGate` |
| `src/walkforward/walkforward.hpp/.cpp` | the causal pass driver, `WalkForwardResult`, research IC table |
| `src/walkforward/report.hpp/.cpp` | results.json, equity.csv, trades.csv, registry.csv, report.md |
| `scripts/render_equity.py` | equity-curve SVG, stdlib only |
| `src/cli/args.*`, `src/main.cpp` | `--walkforward` and `--wf-*` flags |
| `tests/test_wf_*.cpp` | one test file per module |

---

### Task 1: Statistics primitives

**Files:**
- Create: `src/walkforward/stats.hpp`, `src/walkforward/stats.cpp`
- Test: `tests/test_wf_stats.cpp`

**Interfaces:**
- Produces:
  - `double mr::spearman(std::span<const double> a, std::span<const double> b)`: average ranks for ties. Pairs where either value is non-finite are ignored. Returns NaN if fewer than 3 pairs remain or a variance is zero.
  - `struct MeanT { double mean, t; std::size_t n; }` and `MeanT mr::mean_t(std::span<const double> x)`: NaNs are skipped, t = mean/(sd/√n), and t is NaN if n < 2 or sd = 0.
  - `double mr::norm_cdf(double)` and `double mr::norm_inv(double p)`, using the Acklam rational approximation with |error| < 1.2e-9.
  - `struct CI { double lo, hi; }` and `CI mr::block_bootstrap_mean_ci(std::span<const double> x, std::size_t block, std::size_t reps, std::uint64_t seed, double level)`: a circular block bootstrap of the mean. It uses `std::mt19937_64` with `rng() % n` for block starts, so it is deterministic.

- [ ] **Step 1: Write the failing tests**
```cpp
#include <doctest/doctest.h>
#include <cmath>
#include <vector>
#include "walkforward/stats.hpp"
using namespace mr;

TEST_CASE("spearman handles ties, NaN and degenerate input") {
  std::vector<double> a{1, 2, 3, 4, 5}, b{10, 20, 30, 40, 50};
  CHECK(spearman(a, b) == doctest::Approx(1.0));
  std::vector<double> c{5, 4, 3, 2, 1};
  CHECK(spearman(a, c) == doctest::Approx(-1.0));
  std::vector<double> t{1, 1, 2, 2, 3}, u{1, 2, 3, 4, 5};
  CHECK(spearman(t, u) == doctest::Approx(0.9486833).epsilon(1e-6));  // average ranks
  std::vector<double> n{1, NAN, 3, 4, 5};
  CHECK(spearman(n, b) == doctest::Approx(1.0));
  std::vector<double> k{2, 2, 2, 2, 2};
  CHECK(std::isnan(spearman(k, b)));
}

TEST_CASE("mean_t and normal functions") {
  std::vector<double> x{1, 2, 3, NAN};
  const MeanT m = mean_t(x);
  CHECK(m.n == 3);
  CHECK(m.mean == doctest::Approx(2.0));
  CHECK(m.t == doctest::Approx(2.0 / (1.0 / std::sqrt(3.0))));
  CHECK(norm_cdf(0) == doctest::Approx(0.5));
  CHECK(norm_cdf(1.96) == doctest::Approx(0.9750021).epsilon(1e-6));
  CHECK(norm_inv(0.975) == doctest::Approx(1.959964).epsilon(1e-6));
  CHECK(norm_inv(norm_cdf(-2.3)) == doctest::Approx(-2.3).epsilon(1e-8));
}

TEST_CASE("block bootstrap CI covers the mean and is deterministic") {
  std::vector<double> x(500);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = 0.001 + 0.01 * std::sin(0.7 * static_cast<double>(i));
  const CI a = block_bootstrap_mean_ci(x, 21, 2000, 7, 0.95), b = block_bootstrap_mean_ci(x, 21, 2000, 7, 0.95);
  CHECK(a.lo == b.lo);
  CHECK(a.hi == b.hi);
  double mean = 0;
  for (double v : x) mean += v;
  mean /= static_cast<double>(x.size());
  CHECK(a.lo < mean);
  CHECK(a.hi > mean);
}
```
- [ ] **Step 2: Run the tests and confirm they fail.** `cmake --build build -j && ./build/marketrank_tests -tc="spearman*,mean_t*,block bootstrap*"`. Expected: the build fails because `walkforward/stats.hpp` is missing.
- [ ] **Step 3: Implement.**
```cpp
// stats.hpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
namespace mr {
double spearman(std::span<const double> a, std::span<const double> b);
struct MeanT { double mean = 0, t = 0; std::size_t n = 0; };
MeanT mean_t(std::span<const double> x);
double norm_cdf(double x);
double norm_inv(double p);
struct CI { double lo = 0, hi = 0; };
CI block_bootstrap_mean_ci(std::span<const double> x, std::size_t block, std::size_t reps, std::uint64_t seed,
                           double level);
}  // namespace mr
```
```cpp
// stats.cpp (core parts)
#include "walkforward/stats.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
namespace mr {
namespace {
std::vector<double> avg_ranks(const std::vector<double>& v) {
  std::vector<std::size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return v[a] < v[b] || (v[a] == v[b] && a < b); });
  std::vector<double> r(v.size());
  for (std::size_t i = 0; i < idx.size();) {
    std::size_t j = i;
    while (j + 1 < idx.size() && v[idx[j + 1]] == v[idx[i]]) ++j;
    const double rank = 0.5 * static_cast<double>(i + j) + 1.0;
    for (std::size_t k = i; k <= j; ++k) r[idx[k]] = rank;
    i = j + 1;
  }
  return r;
}
}  // namespace
double spearman(std::span<const double> a, std::span<const double> b) {
  std::vector<double> x, y;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    if (std::isfinite(a[i]) && std::isfinite(b[i])) { x.push_back(a[i]); y.push_back(b[i]); }
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (x.size() < 3) return nan;
  const auto rx = avg_ranks(x), ry = avg_ranks(y);
  const double n = static_cast<double>(rx.size()), m = (n + 1) / 2;
  double sxy = 0, sxx = 0, syy = 0;
  for (std::size_t i = 0; i < rx.size(); ++i) {
    sxy += (rx[i] - m) * (ry[i] - m); sxx += (rx[i] - m) * (rx[i] - m); syy += (ry[i] - m) * (ry[i] - m);
  }
  return (sxx > 0 && syy > 0) ? sxy / std::sqrt(sxx * syy) : nan;
}
// mean_t: serial sums; norm_cdf = 0.5*erfc(-x/sqrt2); norm_inv: Acklam (low/central/high regions);
// block_bootstrap_mean_ci: for r in reps: draw ceil(n/block) starts s = rng()%n, take x[(s+k)%n] for k<block
// until n values; record the mean; sort the means; return quantiles at (1-level)/2 and 1-(1-level)/2
// (index = floor(q*(reps-1))).
}  // namespace mr
```
Write out `mean_t`, `norm_cdf`, `norm_inv` (Acklam coefficients a1..a6, b1..b5, c1..c6, d1..d4, with plow = 0.02425) and `block_bootstrap_mean_ci` in full, following the comment above.
- [ ] **Step 4: Run the tests; they pass.** Then run the full `./build/marketrank_tests`; the build has no warnings.
- [ ] **Step 5: Commit** with `git add src/walkforward/stats.* tests/test_wf_stats.cpp` and the message `feat(wf): statistics primitives (spearman, mean/t, normal, block bootstrap)`.

---

### Task 2: Signals and cross-sectional z-scores

**Files:**
- Create: `src/walkforward/signals.hpp`, `src/walkforward/signals.cpp`
- Test: `tests/test_wf_signals.cpp`

**Interfaces:**
- Consumes: `mr::Frame` (pipeline/core_pipeline.hpp: `active`, `pi`, `h`, `inflow`, `size_ref`, `forecasts[0].score`), `mr::market_rank_score`, and `mr::size_shares` (graph/hotness.hpp).
- Produces:
  - `enum class mr::Signal { Score, Pulse1, Pulse5, Pulse20, PiRelSize, NegHotness, InflowMom5, Forecast }`, plus `constexpr std::size_t kSignals = 8`, `std::string_view to_string(Signal)` and `Signal parse_signal(std::string_view)`. The names are "score", "pulse1", "pulse5", "pulse20", "pi_rel_size", "neg_hotness", "inflow_mom5" and "forecast".
  - `class mr::SignalTracker { SignalTracker(std::size_t n); std::array<std::vector<double>, kSignals> update(const Frame& f); }`. Its raw signals:
    - `Score` = log(π·N), and `PulseK` = log(πN)_t − log(πN)_{t−K}. NaN if the stock is inactive now or K steps ago. A ring buffer holds the last 20 log(πN) vectors.
    - `PiRelSize` = log(π/size_share).
    - `NegHotness` = −h.
    - `InflowMom5` = log(inflow_t/inflow_{t−5}), NaN unless both values are > 0.
    - `Forecast` = `forecasts[0].score` when it exists, else NaN.
  - `std::vector<double> mr::zscore(std::span<const double> x, const std::vector<bool>& mask)`: over finite masked entries, (x − mean)/sd, winsorized at ±3. Entries outside the mask, or not finite, are NaN. If sd = 0 all values are 0.

- [ ] **Step 1: Write the failing tests.**
  1. Hand-build three `Frame`s with n = 4 and `pi = {0.4, 0.3, 0.2, 0.1}`, with node 2's π rising each frame (renormalised so the total stays 1), and node 3 inactive in frame 2.
  2. Check that `Pulse1` for node 2 equals `log(pi2(t)*4) - log(pi2(t-1)*4)` exactly.
  3. Check that node 3's `Pulse1` is NaN in frame 2 and in frame 3, because node 3 was inactive at t−1.
  4. Check that `Pulse5` stays NaN until five steps have passed.
  5. Check that `NegHotness == -h`.
  6. `zscore`: input `{1,2,3,NaN}` with mask `{1,1,1,1}` gives mean 0 and sd 1 over the finite values, and NaN at index 3. A constant input gives all 0. A 10σ outlier is clipped to 3.
- [ ] **Step 2: Run the tests; they fail** because the header is missing.
- [ ] **Step 3: Implement**, following the interface rules exactly. Keep the history as `std::array<std::vector<double>, 21>`, each holding log(πN) with NaN where the stock is inactive, plus a separate ring for inflow. Index with `(head - k + 21) % 21`.
- [ ] **Step 4: The tests pass**, with no warnings.
- [ ] **Step 5: Commit** with message `feat(wf): signal tracker and cross-sectional z-scores`.

---

### Task 3: Universe, forward returns, rebalance calendar

**Files:**
- Create: `src/walkforward/universe.hpp`, `src/walkforward/universe.cpp`
- Test: `tests/test_wf_universe.cpp`

**Interfaces:**
- Consumes: `mr::Panel` (`times`, `tickers`, `open`, `close`, `volume`, `vwap`, `idx(t,i)`).
- Produces:
  - `enum class mr::Rebalance { Monthly, Weekly }` and `parse_rebalance`.
  - `std::vector<std::size_t> mr::rebalance_dates(const Panel&, Rebalance, std::size_t warmup_bars)`.
    - Monthly: the last bar index of each calendar month (UTC date of `times[t]`).
    - Weekly: the last bar of each ISO week.
    - Only indices ≥ `warmup_bars` and ≤ T−2 count, so that a next open exists.
  - `std::vector<bool> mr::eligible_at(const Panel&, std::size_t t, std::size_t window, double min_dollar_volume, std::size_t top_n)`.
    - Uses the trailing median of `vwap*volume` over bars [t−window+1, t], with non-finite values skipped and at least window/2 values required.
    - A stock is eligible if its median ≥ the minimum and it ranks in the top_n by that median (top_n = 0 means no rank limit).
    - Ties are broken by lower index. It is causal: only bars ≤ t are read.
  - `std::vector<double> mr::forward_oo_return(const Panel&, std::size_t t, std::size_t h)`: `open[t+1+h]/open[t+1] - 1` per stock. It is NaN if either open is non-finite, or if t+1+h ≥ T. This is the label: the signal is known at the close of t, entry is at the open of t+1, exit at the open of t+1+h.

- [ ] **Step 1: Write the failing tests.**
  - Build a 3-stock `Panel` with daily times from 2024-01-29 through 2024-03-05, skipping weekends.
  - `rebalance_dates(Monthly, 0)` returns the indices of 2024-01-31 and 2024-02-29. March has no next open yet at its end, so it is excluded.
  - Weekly returns the Fridays.
  - `eligible_at` with window 3: a stock with dollar volume 100M passes `min=50e6`; one with 10M fails; `top_n=1` keeps only the largest.
  - Change a bar after t; the result is unchanged (causal).
  - `forward_oo_return(t, 2)` equals `open[t+3]/open[t+1]-1` exactly. A NaN open gives NaN.
- [ ] **Step 2: Run; the tests fail.**
- [ ] **Step 3: Implement.** Convert dates using the existing `core/time.hpp` helpers: read them, and use the UTC civil date of each `TimePoint`. Medians come from `std::nth_element` on a copy, run serially.
- [ ] **Step 4: The tests pass.**
- [ ] **Step 5: Commit** with message `feat(wf): point-in-time eligibility, forward returns, rebalance calendar`.

---

### Task 4: Backtest simulator with costs and benchmarks

**Files:**
- Create: `src/walkforward/backtest.hpp`, `src/walkforward/backtest.cpp`
- Test: `tests/test_wf_backtest.cpp`

**Interfaces:**
- Consumes: `Panel`, `rebalance_dates`, `eligible_at`.
- Produces:
```cpp
namespace mr {
struct BaseWeight { std::string ticker; double weight; };
struct BacktestParams {
  std::vector<BaseWeight> base;   // the core mix (sums to <= 1; remainder is cash)
  double tilt = 0.20;             // share of value moved toward the signal
  std::size_t k = 10;             // names in the tilt
  double max_name_tilt = 0.10;    // cap on one name's tilt weight
  double cost_bps = 10;           // per side, on traded notional
  double max_turnover = 0.5;      // per rebalance, sum |delta w| / 2; larger moves are scaled down
};
// One decision per rebalance date: the score vector (NaN = not a candidate) and the eligibility mask;
// an empty score vector means "hold the base, no tilt" (gate closed / no signal).
struct Decision { std::size_t date; std::vector<double> score; std::vector<bool> eligible; };
enum class BenchKind { None, BuyHoldBase, RebalancedBase, Single };  // Single = 100% in one ticker
struct EquityCurve {
  std::vector<TimePoint> t;      // every bar from the first execution day on
  std::vector<double> value;     // marked at the close; starts at 1.0
  double costs = 0;              // total cost paid (as fraction of starting value)
  double turnover = 0;           // sum over rebalances
  std::vector<std::string> trades_csv;  // "t,ticker,delta_weight,price"
};
std::vector<double> target_weights(const Panel&, const BacktestParams&, const Decision&);  // size N
EquityCurve simulate(const Panel&, const BacktestParams&, const std::vector<Decision>&, BenchKind bench = BenchKind::None,
                     const std::string& single_ticker = "");
}  // namespace mr
```
**Rules:**
- **Target weights.** The target is (1−tilt)·base + tilt·tilt-part. The tilt part gives equal weight min(1/k, max_name_tilt/tilt) to each of the top-k eligible names by score; ties go to the lower index. Any unallocated tilt goes to cash. If the decision's score is empty, the target is the base alone.
- **Execution.** Each decision is executed at the **open of date+1**:
  1. Value the holdings at that open, using the last finite price as a forward fill.
  2. Compute the target notional and the deltas.
  3. Apply the turnover cap by scaling the deltas.
  4. Pay the cost: cost_bps·1e−4 × Σ|Δnotional|, taken from cash.
  5. Trade at the open price. A stock with no finite open that day keeps its current shares.
- **Marking.** Every close, value = Σ shares × close, forward-filled, plus cash.
- **Benchmarks.**
  - `BuyHoldBase`: buy the base mix at the first execution and never trade again.
  - `RebalancedBase`: trade to the base weights at every decision date, with costs.
  - `Single`: 100% in `single_ticker`.

- [ ] **Step 1: Write the failing tests.**
  - On a deterministic 4-stock panel with 60 bars and known open and close prices:
    - **(a)** Buy-and-hold of {A:0.5, B:0.5} equals the analytic value at every close.
    - **(b)** Costs: one rebalance moving 100% from A to B at 10 bps costs exactly 2×(0.001·value_at_open)×0.5. That is turnover 1.0 under the convention Σ|Δw|/2 = 1, with notional 2×value traded. Assert the exact cost and the post-trade value.
    - **(c)** The turnover cap of 0.5 halves the move.
    - **(d)** A stock with no open on its execution day keeps its shares.
  - **(e) Planted edge.** Build a panel where stock X always gains +1% between consecutive opens and every other stock is flat. A decision scoring X highest beats `BuyHoldBase` of {Y:1.0} after costs. A decision with random scores (fixed seed) does not beat it by more than costs plus 1e−12.
  - **(f)** Running twice gives identical curves.
- [ ] **Step 2: Run; the tests fail.**
- [ ] **Step 3: Implement** `target_weights` and `simulate` exactly as specified. Keep holdings as a vector of shares plus a cash scalar, and process bars in order.
- [ ] **Step 4: The tests pass.**
- [ ] **Step 5: Commit** with message `feat(wf): portfolio simulator with next-open execution, costs, turnover cap, benchmarks`.

---

### Task 5: Gated simplex blend

**Files:**
- Create: `src/walkforward/blend.hpp`, `src/walkforward/blend.cpp`
- Test: `tests/test_wf_blend.cpp`

**Interfaces:**
- Consumes: `mr::spearman` and `mr::mean_t` (Task 1).
- Produces:
```cpp
namespace mr {
struct BlendParams { std::size_t train_months = 36, embargo = 1, gate_months = 24, gate_min = 12; double gate_t = 2.0; };
// Per rebalance month m: z[s] = cross-sectional z-scores (size N, NaN = not eligible), label = forward return to the
// next rebalance execution (size N), both from the walk-forward pass.
struct MonthRecord { std::array<std::vector<double>, kSignals> z; std::vector<double> label; };
struct BlendStep { std::array<double, kSignals> w{}; bool gate_open = false; double oos_ic = NAN; std::vector<double> score; };
std::vector<BlendStep> run_blend(const std::vector<MonthRecord>& months, const BlendParams& p);
}  // namespace mr
```
**Rules for month m:**
- **Training set:** months [m − embargo − train_months, m − embargo − 1], clipped at 0. The months at or after m − embargo are excluded, because their labels overlap month m.
- **Weights:**
  - IC_s(j) = spearman(z_s(j), label(j)) for each training month j.
  - w_s = max(0, mean_j IC_s(j)), normalised so the weights sum to 1.
  - If the sum is 0, or there are fewer than 6 training months, then w = 0 and no score.
- **Score:** score(m) = Σ_s w_s · z_s(m). It is NaN where every z is NaN.
- **Out-of-sample IC:** oos_ic(m) = spearman(score(m), label(m)). It is used *only* by later months.
- **Gate:** the gate for m is open if the mean_t of oos_ic over months [m − embargo − gate_months, m − embargo − 1] has n ≥ gate_min and t > gate_t.
- **Effect:** a closed gate means an empty score, i.e. hold the base.

- [ ] **Step 1: Write the failing tests.**
  - **(a) Planted signal.** 80 months, N = 200. Signal 0's z equals the label plus noise (σ = 0.5); the other signals are pure noise. After enough history, w[0] > 0.8, and the gate opens by month 36 + 1 + 12.
  - **(b) Pure noise.** All signals are noise. The gate stays closed in at least 95% of the months after warm-up, using a fixed seed.
  - **(c) Causality.** Changing `label` of months ≥ m − 1 leaves `run_blend(...)[m]` weights and gate bit-identical.
  - **(d)** Weights are always ≥ 0 and sum to 1 or 0.
- [ ] **Step 2: Run; the tests fail.**
- [ ] **Step 3: Implement** serially, in month order.
- [ ] **Step 4: The tests pass.**
- [ ] **Step 5: Commit** with message `feat(wf): gated simplex-weighted walk-forward blend with embargo`.

---

### Task 6: Metrics, deflated Sharpe, decision gate

**Files:**
- Create: `src/walkforward/metrics.hpp`, `src/walkforward/metrics.cpp`
- Test: `tests/test_wf_metrics.cpp`

**Interfaces:**
- Consumes: `EquityCurve` (Task 4), and `block_bootstrap_mean_ci`, `norm_cdf` and `norm_inv` (Task 1).
- Produces:
```cpp
namespace mr {
struct Perf {
  double cum_return, ann_return, ann_vol, sharpe, max_drawdown;  // 252 trading days; rf = 0
  double ann_excess, ir; CI excess_ci95; double year_hit_rate; std::size_t years;
  double skew, kurt;  // of daily strategy returns (for the deflated Sharpe)
};
Perf performance(const EquityCurve& s, const EquityCurve& bench);  // aligned on common dates
// Bailey & Lopez de Prado deflated Sharpe probability. sr = per-day Sharpe, T = days, trial_sr_var = variance of the
// per-day Sharpe across all registered trials, n_trials >= 1 (n_trials == 1 -> SR* = 0).
double deflated_sharpe(double sr, std::size_t T, double skew, double kurt, double trial_sr_var, std::size_t n_trials);
struct GateResult { bool c1, c2, c3, c4, c5, pass; std::string reason; };
GateResult decision_gate(const Perf& strat_vs_buyhold, double base_max_dd, double dsr, bool largecap_ok);
}  // namespace mr
```
**Formulas:**
- Daily returns r_d = V_d/V_{d−1} − 1, and excess e_d = r_d − rb_d.
- ann_return = (V_end/V_0)^(252/T) − 1, ann_vol = sd(r)·√252, sharpe = mean(r)/sd(r)·√252.
- ann_excess = mean(e)·252, IR = mean(e)/sd(e)·√252.
- excess_ci95 = the block bootstrap of e (block 21, 2000 reps, seed 11), times 252.
- year_hit_rate = the share of calendar years with Σe > 0. A year counts only if it has ≥ 120 days.
- max_drawdown = max over d of 1 − V_d/max_{u≤d} V_u.
- DSR:
  - SR* = √var · ((1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e))), with γ = 0.5772156649.
  - DSR = Φ((sr − SR*)·√(T−1) / √(1 − skew·sr + (kurt−1)/4·sr²)), where kurt is the raw kurtosis (3 for a normal distribution).
- Gate:
  - c1 is ann_excess > 0 && excess_ci95.lo > 0.
  - c2 is year_hit_rate ≥ 0.6.
  - c3 is dsr > 0.95.
  - c4 is strat max_drawdown ≤ base_max_dd + 0.05.
  - c5 is largecap_ok.
  - pass is all five. `reason` lists every failed condition.

- [ ] **Step 1: Write the failing tests.**
  - Two analytic curves:
    - the strategy grows at a constant 0.1% per day;
    - the benchmark is flat.
  - With those curves, check:
    - ann_excess ≈ 0.001·252;
    - max_drawdown = 0;
    - a curve that drops 20% then recovers has max_drawdown = 0.2;
    - year_hit_rate on 2 synthetic years (one positive, one negative) is 0.5.
  - DSR:
    - with n_trials = 1, sr = 0.1, T = 1000, skew 0, kurt 3, it equals Φ(0.1·√999/√(1 + 0.5·0.01));
    - more trials lower the DSR.
  - The gate's pass/fail and reason strings, for each condition separately.
- [ ] **Step 2: Run; the tests fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: The tests pass.**
- [ ] **Step 5: Commit** with message `feat(wf): performance metrics, deflated Sharpe, decision gate`.

---

### Task 7: Walk-forward driver, CLI, outputs and report

**Files:**
- Create: `src/walkforward/walkforward.hpp/.cpp`, `src/walkforward/report.hpp/.cpp`, `scripts/render_equity.py`
- Modify: `src/cli/args.hpp/.cpp` (flags), `src/main.cpp` (dispatch), `README.md` (a short "Walk-forward" usage section)
- Test: `tests/test_wf_driver.cpp`

**Interfaces:**
- Consumes: everything from Tasks 1–6, plus `CorePipeline` and `CoreParams::market_rank()`.
- Produces:
```cpp
namespace mr {
struct WalkForwardParams {
  CoreParams core = CoreParams::market_rank();
  Rebalance rebalance = Rebalance::Monthly;
  std::size_t warmup_bars = 252, elig_window = 20, top_n = 0; double min_dollar_volume = 50e6;
  std::vector<int> ic_horizons{1, 5, 20};
  BacktestParams bt; BlendParams blend;
};
struct IcRow { Signal s; int h; MeanT all; std::vector<std::pair<int, double>> by_year; };  // year -> mean IC
struct WalkForwardResult {
  std::vector<IcRow> ic_table;
  std::vector<std::size_t> dates;              // rebalance bar indices
  std::vector<MonthRecord> months;             // stored z-scores and labels per rebalance
  std::vector<std::vector<bool>> eligible;     // per rebalance
  std::vector<BlendStep> blend;
  // Equity curves: "blend", one per signal ("sig:<name>"), "bench:buyhold", "bench:rebalanced", "bench:VOO".
  std::vector<std::pair<std::string, EquityCurve>> curves;
};
WalkForwardResult run_walkforward(const Panel& panel, const WalkForwardParams& p);
// Writes results.json, equity.csv, trades.csv, report.md under out_dir/<run_id>/ and appends registry.csv rows
// (one per strategy curve) under out_dir; returns the run directory.
std::filesystem::path write_report(const WalkForwardResult&, const WalkForwardParams&, const Panel&,
                                   const std::filesystem::path& out_dir, const std::string& run_id);
}  // namespace mr
```
**Driver rules.**
- One pass for t = 0..T−1: `frame = pipe.step(panel, t)`, then `raw = tracker.update(frame)`.
- **IC research table.** For each horizon h, and each t ≥ warmup with (t − warmup) % h == 0 (non-overlapping samples) and t+1+h < T:
  - ic = spearman(zscore(raw_s, eligible_at(t)), forward_oo_return(t, h)) for every signal;
  - accumulate it into the overall mean_t and the per-year means.
- **At each rebalance date** d_m, store:
  - the z-scores of every signal (masked by `eligible_at(d_m)` ∧ `frame.active`);
  - the label = forward_oo_return(d_m, d_{m+1} − d_m), or NaN for the last month;
  - the eligibility mask.
- **After the pass:**
  - `run_blend(months)` gives the blend decisions.
  - Each signal's strategy uses its own z as the score.
  - Simulate every strategy and the benchmarks.
- **Causality** holds because every stored quantity at d_m depends only on bars ≤ d_m, except the label, which the blend uses only after the embargo.

**CLI.**
- `--walkforward` requires `--mode replay` and runs `run_walkforward` plus `write_report`, then exits.
- Flags:
  - `--wf-rebalance monthly|weekly`, `--wf-top-n N`;
  - `--wf-cost-bps X`, `--wf-tilt X`, `--wf-k N`;
  - `--wf-out DIR` (default `<data>/walkforward`);
  - `--wf-run-id ID` (default `<UTC timestamp>-<8-hex hash of params>`).
- Add the flags to the usage text and `describe`.

**Report.**
- `report.md` contains:
  - the IC research table (signal × horizon: mean IC, t, n, positive-year share);
  - per-strategy Perf against buy-and-hold, with costs;
  - the blend weight history (mean w) and the gate-open share;
  - the decision gate, using the large-cap criterion from a sibling run if one is passed with `--wf-largecap-run ID`, otherwise marked "pending".
- `registry.csv` columns are `run_id,strategy,params_hash,cost_bps,top_n,sharpe_daily,T,ann_excess`. `deflated_sharpe` uses all registry rows for its trial count and variance.
- `scripts/render_equity.py <run_dir>` writes `equity.svg` (strategy curves against the benchmarks, log scale, legend), using only the standard library.

- [ ] **Step 1: Write the failing tests** in `tests/test_wf_driver.cpp`. Use the synthetic market (`generate_synthetic` into a temp BarStore, then `build_panel`) with about 400 daily bars and warmup 100.
  - **(a)** `run_walkforward` produces months whose dates are the last bars of each month, and the curves include every expected key.
  - **(b) Causality.** Copy the panel and replace every open/close/volume after bar d_k + 1 with different finite values. For every month ≤ k, the stored z-scores, eligibility, blend weights and gate are bit-identical, and the blend decision score is too.
  - **(c) Determinism.** Results are identical with `omp_set_num_threads(1)` and with 8 threads. Save and restore the thread count.
  - **(d)** `write_report` creates all the files, and registry.csv grows by one row per strategy on each run.
  - **(e)** `parse_cli` accepts `--walkforward` and the `--wf-*` flags, and rejects `--walkforward` without replay mode.
- [ ] **Step 2: Run; the tests fail.**
- [ ] **Step 3: Implement** the driver, the report, the CLI and the SVG script.
- [ ] **Step 4: The tests pass.** Run the full suite and the synthetic smoke check: `./build/marketrank --mode synthetic` must still work. Then run `./build/marketrank --mode replay --timeframe 1d --walkforward --wf-run-id smoke` on the existing one-year lake. With 12 months of warm-up it reports few months, and that's expected; it just has to exit 0 and write a report.
- [ ] **Step 5: Commit** with message `feat(wf): causal walk-forward driver, CLI, report, registry and equity SVG`.

---

### Task 8: Real backfill and the decision run (designated live task)

**Files:**
- Modify: `README.md` (results summary in "What we are trying to find out"), `paper/marketrank.tex` (results paragraph and figure), `docs/img/walkforward-equity.svg` (new)

**Steps:**
- [ ] **Step 1: Backfill.** This is the only `--mode alpaca` step in the plan. First back up `data/lake` to the scratchpad. Then run:
  `./build/marketrank --mode alpaca --timeframe 1d --universe snapshot --lookback-days 3750 --top 5`.
  - That covers 2016-01 onward.
  - Record the wall time, the request count from the log, the lake size, and the stale-ticker count.
  - Re-run once if any tickers were stale.
- [ ] **Step 2: Replay sanity check.** `./build/marketrank --mode replay --timeframe 1d --lookback-days 3750 --top 10` must exit 0. Record the peak memory with `/usr/bin/time -v`; it must be under 4 GB.
- [ ] **Step 3: Main run.** `./build/marketrank --mode replay --timeframe 1d --lookback-days 3750 --walkforward --wf-run-id main-10bps`. Then repeat with `--wf-cost-bps 0` and `25` as the cost-sensitivity runs.
- [ ] **Step 4: Large-cap run.** Run with `--wf-top-n 500 --wf-run-id largecap-10bps`. Then run the gate report on `main-10bps` with `--wf-largecap-run largecap-10bps`.
- [ ] **Step 5: Render** the equity SVG for main-10bps, copy it to `docs/img/walkforward-equity.svg`, and look at it.
- [ ] **Step 6: Write up honestly** in the README and the paper:
  - the IC table highlights;
  - the blend's gate-open share;
  - the strategy against buy-and-hold, with ann_excess, CI, IR, max drawdown, year hit rate and DSR;
  - **the decision**: "yes, consistently" leads to the M3b optimizer plan, "no" leads to the M3c observed-flows plan.
  - Report the survivorship-bias caveat.
- [ ] **Step 7: Commit** with message `results(wf): backfill to 2016 and the M3 decision run`.

---

## Self-review

- **Spec coverage:**
  - §1 backfill → Task 8 Step 1; point-in-time universe → Task 3; survivorship caveat and large-cap re-test → Task 8 Steps 4 and 6; runtime and memory check → Task 8 Step 2.
  - §2.1 timeline and causality → Tasks 3 and 7.
  - §2.2 signals and the research table → Tasks 2 and 7.
  - §2.3 tilt, constraints, costs and benchmarks → Task 4.
  - §2.4 gated blend → Task 5.
  - §2.5 metrics, bootstrap, DSR and outputs → Tasks 6 and 7.
  - §3 gate → Task 6 and Task 8 Step 4.
  - §6 engineering rules → the causality and determinism tests in Tasks 5 and 7.
  - §4 and §5 are deferred to the M3b and M3c plans by design.
- **Placeholders:** Task 1's long-form numerics (Acklam coefficients, bootstrap loop) are described exactly, with formulas, rather than pasted, which is acceptable for well-known constants. The implementer copies the standard Acklam coefficients.
- **Type consistency:** `Signal`/`kSignals` (Task 2) are used by `MonthRecord`/`BlendStep` (Task 5) and `IcRow` (Task 7). `Decision`/`EquityCurve` (Task 4) are used by `performance` (Task 6) and the driver (Task 7). `Rebalance` comes from Task 3.
