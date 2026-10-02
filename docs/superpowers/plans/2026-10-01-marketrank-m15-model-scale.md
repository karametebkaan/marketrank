# MarketRank Milestone 1.5 — Model A–E and 10K Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace milestone 1's dense flux model with a sparse, OpenMP-parallel model that implements switches A–E (size-neutral pressure, gravity lift, two-sided pruning, hotness reference, retention). Scale it to a liquidity-ranked universe of about 10,000 Alpaca stocks, and add an evaluation harness so the defaults are chosen from evidence.

**Architecture:**
- **New units, all O(N·k) or O(N·W):**
  - `ReturnWindow`: correlation as dot products of unit vectors.
  - `PressureModel` (A).
  - `bar_flux_sparse` and `FluxAccumulator`: sparse edges plus exact row and column totals.
  - `build_transition(FluxAccumulator, TransitionParams)` (B, C, E).
  - `relative_hotness` (D).
- **Rewritten:** `CorePipeline` is rewritten around these units, and the dense `FluxBuilder` and dense `build_transition` are deleted.
- **Universe:** the universe layer gains Alpaca `/v2/assets` parsing, liquidity ranking, CSV snapshots and history back-fill.
- **CLI:** a new `src/cli/args` module, plus `--eval`.
- **OpenMP:** parallel loops write disjoint slots, and floating-point sums are reduced serially in index order, so results are bit-identical at any thread count.

**Tech Stack:** C++20, GCC 13, CMake 3.28, OpenMP (libgomp), nlohmann/json, cpp-httplib (OpenSSL), doctest.

**Spec:** `docs/superpowers/specs/2026-10-01-marketrank-design.md`. This plan implements §3, §4.1 (rate limit, assets), §5, §5.1, §5.2 and the "Milestone 1.5 additions" in §11.

## Global Constraints

- Work and commit directly on `master` (no feature branches). Every commit message ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- C++20, GCC 13.3, CMake ≥ 3.24. Namespace `fx`, headers `#pragma once`. Build warning-free under `-Wall -Wextra -Wpedantic`.
- Dependencies: FetchContent nlohmann/json, cpp-httplib, doctest; system OpenMP via `find_package(OpenMP REQUIRED)`. No others.
- **No O(N²) memory or per-bar O(N²) work anywhere in the pipeline.** N reaches 10,000.
- **OpenMP determinism rule:** inside a parallel loop, write only to slots owned by the loop index. Floating-point accumulation across indices happens serially, in index order, after the loop. No `reduction(+:...)` on doubles.
- Defaults (spec §5):
  - Pressure and correlation: `pressure=relative`, `adv_window=20`, `corr_window=60`, `lambda=1.0` (λ ∈ [0,1]).
  - Flux and accumulators: `sink_candidates=256`, `sinks_per_source=64`, half-lives slow 20 / fast 3 / long 120, `row_cap=256`.
  - Transition and hotness: `lift=excess`, `k_out=20`, `k_in=10`, `retention=1.0`, `h_ref=uniform`.
  - Solver and forecast: `alpha=0.85`, `beta=0.5`, horizons {1,4,8}.
- Legacy (`CoreParams::legacy()`): `pressure=dollar`, `lift=off`, `k_in=0`, `retention=0`, `h_ref=uniform`. Everything else stays at the defaults.
- Power iteration: tolerance ‖Δπ‖₁ < 1e-10, max 1000 iterations. Teleport is uniform over active nodes.
- Alpaca:
  - Data host `data.alpaca.markets`; trading host `paper-api.alpaca.markets` (env `APCA_TRADING_HOST`).
  - Client rate limit: `min_request_interval_ms = 334`.
  - Backoff on 429/5xx: 0.5 s doubling to 30 s, max 6 retries.
  - `end ≤ now − 16 min`.
- Never read, print or log `/home/kkaramete/stocks/.env`.
- Build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`. Tests: `./build/fluxtests` (one case: `-tc="<name>"`).

## File Structure

```
CMakeLists.txt                         + OpenMP
src/graph/return_window.hpp/.cpp       NEW  rolling returns -> unit vectors (corr = dot)
src/graph/pressure.hpp/.cpp            NEW  (A) PressureMode, PressureModel (ADV medians)
src/graph/sparse_flux.hpp/.cpp         NEW  bar_flux_sparse, FluxAccumulator
src/graph/transition.hpp/.cpp          NEW  (B,C,E) LiftMode, TransitionParams, build_transition(acc)
src/graph/hotness.hpp/.cpp             NEW  (D) HotRef, relative_hotness
src/graph/csr.hpp/.cpp                 MOD  drop dense build_transition; add transpose, left_multiply_transposed
src/graph/markov_solver.cpp            MOD  transpose once per solve, parallel pull multiply
src/graph/flux_builder.hpp/.cpp        DELETE
src/pipeline/core_pipeline.hpp/.cpp    REWRITE  CoreParams (A-E, legacy, validate), new step
src/pipeline/evaluation.hpp/.cpp       NEW  gini, spearman, floor_share, sector_coherence, evaluate, grid
src/market/synthetic_market.*          MOD  size_sigma (heavy-tailed sizes)
src/market/bar_store.*                 MOD  first_time
src/market/market_sync.cpp             REWRITE  history back-fill + incremental tail
src/market/alpaca_client.*             MOD  rate limit, public get(), trading_host
src/market/universe.*                  MOD  load_funds, from_securities(secs, funds)
src/market/asset_universe.hpp/.cpp     NEW  assets parse, rules, ranking, snapshots
src/cli/args.hpp/.cpp                  NEW  CliArgs, parse_cli, describe
src/main.cpp                           REWRITE  universe flow, rank output, --eval
tests/flux_test_util.hpp               NEW  acc_from_dense, legacy_tp
tests/test_*.cpp                       per unit (listed per task)
```

---

### Task 1: OpenMP build and ReturnWindow

**Files:**
- Modify: `CMakeLists.txt`
- Create: `src/graph/return_window.hpp`, `src/graph/return_window.cpp`
- Test: `tests/test_return_window.cpp`

**Interfaces:**
- Consumes: nothing new.
- Produces: `class fx::ReturnWindow { ReturnWindow(std::size_t n, std::size_t window); void push(std::span<const double> returns); const std::vector<double>& unit_vectors(); double correlation(std::size_t i, std::size_t j); std::size_t size() const; std::size_t window() const; std::size_t count() const; }`.
  - `unit_vectors()` is n×window row-major (`[i*window + s]`). Each row is the centered, unit-length return vector over the current samples. A row is all zeros if fewer than 2 samples are available or the variance is below 1e-24. Non-finite returns count as 0.
  - The constructor throws `std::invalid_argument` if window < 2.

- [ ] **Step 1: Add OpenMP to the build.** In `CMakeLists.txt`, after the line `find_package(OpenSSL REQUIRED)`, add:
```cmake
find_package(OpenMP REQUIRED)
```
and change the fluxcore link line to:
```cmake
target_link_libraries(fluxcore PUBLIC nlohmann_json::nlohmann_json httplib::httplib
                      OpenSSL::SSL OpenSSL::Crypto OpenMP::OpenMP_CXX)
```
Run `cmake -S . -B build && cmake --build build -j && ./build/fluxtests`. Expected: `Status: SUCCESS!`, no warnings.

- [ ] **Step 2: Write the failing tests** in `tests/test_return_window.cpp`:
```cpp
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "graph/return_window.hpp"

using namespace fx;

namespace {
double pearson(const std::vector<std::vector<double>>& hist, std::size_t i, std::size_t j,
               std::size_t w) {
  const std::size_t m = std::min(hist.size(), w), start = hist.size() - m;
  double mi = 0, mj = 0;
  for (std::size_t t = start; t < hist.size(); ++t) {
    mi += hist[t][i];
    mj += hist[t][j];
  }
  mi /= static_cast<double>(m);
  mj /= static_cast<double>(m);
  double sij = 0, sii = 0, sjj = 0;
  for (std::size_t t = start; t < hist.size(); ++t) {
    const double a = hist[t][i] - mi, b = hist[t][j] - mj;
    sij += a * b;
    sii += a * a;
    sjj += b * b;
  }
  return sij / std::sqrt(sii * sjj);
}
}  // namespace

TEST_CASE("ReturnWindow correlation matches brute-force Pearson across wraps") {
  const std::size_t n = 4, W = 10;
  ReturnWindow rw(n, W);
  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0.0, 0.01);
  std::vector<std::vector<double>> hist;
  for (int step = 1; step <= 35; ++step) {
    std::vector<double> r(n);
    for (auto& x : r) x = noise(rng);
    r[3] = 0.7 * r[0] + noise(rng);
    rw.push(r);
    hist.push_back(r);
    if (step == 7 || step == 10 || step == 11 || step == 23 || step == 35) {
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          INFO("step " << step << " i " << i << " j " << j);
          const double expected = i == j ? 1.0 : pearson(hist, i, j, W);
          CHECK(rw.correlation(i, j) == doctest::Approx(expected).epsilon(1e-9));
        }
      }
    }
  }
  CHECK(rw.count() == W);
}

TEST_CASE("ReturnWindow treats NaN as 0 and constant series as uncorrelated") {
  ReturnWindow rw(3, 5);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  rw.push(std::vector<double>{0.01, 0.02, nan});
  rw.push(std::vector<double>{-0.01, 0.02, nan});
  rw.push(std::vector<double>{0.02, 0.02, nan});
  CHECK(rw.count() == 3);
  CHECK(rw.correlation(0, 1) == 0.0);
  CHECK(rw.correlation(0, 2) == 0.0);
  CHECK(rw.correlation(1, 1) == 0.0);
  CHECK(rw.correlation(0, 0) == doctest::Approx(1.0));
}

TEST_CASE("ReturnWindow needs window >= 2 and two samples") {
  CHECK_THROWS_AS(ReturnWindow(2, 1), std::invalid_argument);
  ReturnWindow rw(2, 4);
  rw.push(std::vector<double>{0.01, -0.01});
  CHECK(rw.correlation(0, 1) == 0.0);
}
```

- [ ] **Step 3: Run the build to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL with `graph/return_window.hpp: No such file or directory`.

- [ ] **Step 4: Implement**

`src/graph/return_window.hpp`:
```cpp
#pragma once
#include <cstddef>
#include <span>
#include <vector>

namespace fx {

// Rolling window of the last `window` returns per node. Pearson correlation is a dot product of
// the centered, unit-length vectors: corr(i, j) = u_i . u_j. Memory O(n * window).
class ReturnWindow {
 public:
  ReturnWindow(std::size_t n, std::size_t window);

  void push(std::span<const double> returns);  // non-finite -> 0
  const std::vector<double>& unit_vectors();   // n x window, row-major
  double correlation(std::size_t i, std::size_t j);
  std::size_t size() const { return n_; }
  std::size_t window() const { return w_; }
  std::size_t count() const { return count_; }

 private:
  std::size_t n_, w_;
  std::vector<double> ring_;  // [slot * n + i]
  std::size_t head_ = 0, count_ = 0;
  std::vector<double> unit_;  // [i * w + s]
  bool dirty_ = true;
};

}  // namespace fx
```

`src/graph/return_window.cpp`:
```cpp
#include "graph/return_window.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fx {

ReturnWindow::ReturnWindow(std::size_t n, std::size_t window)
    : n_(n), w_(window), ring_(n * window, 0.0), unit_(n * window, 0.0) {
  if (window < 2) throw std::invalid_argument("ReturnWindow: window must be >= 2");
}

void ReturnWindow::push(std::span<const double> returns) {
  double* slot = &ring_[head_ * n_];
  for (std::size_t i = 0; i < n_; ++i) slot[i] = std::isfinite(returns[i]) ? returns[i] : 0.0;
  head_ = (head_ + 1) % w_;
  count_ = std::min(count_ + 1, w_);
  dirty_ = true;
}

const std::vector<double>& ReturnWindow::unit_vectors() {
  if (!dirty_) return unit_;
  const std::size_t c = count_;
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < n_; ++i) {
    double* u = &unit_[i * w_];
    std::fill(u, u + w_, 0.0);
    if (c < 2) continue;
    double mean = 0;
    for (std::size_t s = 0; s < c; ++s) mean += ring_[s * n_ + i];
    mean /= static_cast<double>(c);
    double norm2 = 0;
    for (std::size_t s = 0; s < c; ++s) {
      u[s] = ring_[s * n_ + i] - mean;
      norm2 += u[s] * u[s];
    }
    if (norm2 < 1e-24) {
      std::fill(u, u + w_, 0.0);
      continue;
    }
    const double inv = 1.0 / std::sqrt(norm2);
    for (std::size_t s = 0; s < c; ++s) u[s] *= inv;
  }
  dirty_ = false;
  return unit_;
}

double ReturnWindow::correlation(std::size_t i, std::size_t j) {
  const auto& u = unit_vectors();
  double d = 0;
  for (std::size_t s = 0; s < w_; ++s) d += u[i * w_ + s] * u[j * w_ + s];
  return d;
}

}  // namespace fx
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!`. Check warnings with `touch src/graph/return_window.cpp && cmake --build build 2>&1 | grep -i warning` (expect nothing).

- [ ] **Step 6: Commit**
```bash
git add CMakeLists.txt src/graph/return_window.* tests/test_return_window.cpp
git commit -m "feat: OpenMP build and O(N*W) rolling return window

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Pressure model (A) and heavy-tailed synthetic sizes

**Files:**
- Create: `src/graph/pressure.hpp`, `src/graph/pressure.cpp`
- Modify: `src/market/synthetic_market.hpp`, `src/market/synthetic_market.cpp`
- Test: `tests/test_pressure.cpp`, `tests/test_synthetic_market.cpp` (append one case)

**Interfaces:**
- Produces:
  - `enum class fx::PressureMode { Dollar, Sqrt, Relative }`, `PressureMode fx::parse_pressure_mode(std::string_view)` (`"dollar"|"sqrt"|"relative"`; throws `std::invalid_argument`), `std::string_view fx::to_string(PressureMode)`
  - `class fx::PressureModel { PressureModel(std::size_t n, PressureMode mode, std::size_t adv_window = 20); std::vector<double> step(std::span<const double> returns, std::span<const double> volume, std::span<const double> vwap); std::vector<double> median_dollar_volume() const; }`
    - Dollar: `r·v·vw`. Sqrt: `r·√(v·vw)`. Relative: `r·v / median(previous ≤ adv_window recorded volumes)`, and 0 if there is no history.
    - A bar is recorded (volume and v·vw) when v and vw are finite with v ≥ 0 and vw > 0, even if the return is NaN.
    - A non-finite return or a missing bar gives pressure 0.
  - `SyntheticConfig::size_sigma` (double, default 0). When > 0, each stock's base volume is `1e6·exp(N(0, size_sigma))`, drawn from a separate RNG seeded `seed ^ 0x5eed`. When 0, behavior is unchanged.

- [ ] **Step 1: Write the failing tests**

`tests/test_pressure.cpp`:
```cpp
#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "graph/pressure.hpp"

using namespace fx;

TEST_CASE("dollar and sqrt pressure") {
  PressureModel d(2, PressureMode::Dollar), s(2, PressureMode::Sqrt);
  std::vector<double> r = {0.02, -0.01}, v = {100, 400}, vw = {10, 25};
  auto pd = d.step(r, v, vw);
  CHECK(pd[0] == doctest::Approx(0.02 * 1000));
  CHECK(pd[1] == doctest::Approx(-0.01 * 10000));
  auto ps = s.step(r, v, vw);
  CHECK(ps[0] == doctest::Approx(0.02 * std::sqrt(1000.0)));
  CHECK(ps[1] == doctest::Approx(-0.01 * 100.0));
}

TEST_CASE("relative pressure uses the median volume of previous bars only") {
  PressureModel m(1, PressureMode::Relative, 3);
  auto bar = [&](double r, double v) {
    return m.step(std::vector<double>{r}, std::vector<double>{v}, std::vector<double>{10.0})[0];
  };
  CHECK(bar(0.01, 100) == 0.0);
  CHECK(bar(0.01, 300) == doctest::Approx(0.01 * 3.0));
  CHECK(bar(-0.02, 50) == doctest::Approx(-0.02 * 50 / 200.0));
  CHECK(bar(0.01, 1000) == doctest::Approx(0.01 * 1000 / 100.0));
  CHECK(bar(0.01, 10) == doctest::Approx(0.01 * 10 / 300.0));
}

TEST_CASE("missing data gives zero pressure and missing bars are not recorded") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  PressureModel m(2, PressureMode::Relative, 5);
  m.step(std::vector<double>{nan, 0.01}, std::vector<double>{100, nan},
         std::vector<double>{10, 10});
  auto p = m.step(std::vector<double>{0.01, 0.01}, std::vector<double>{200, 100},
                  std::vector<double>{10, 10});
  CHECK(p[0] == doctest::Approx(0.01 * 2.0));
  CHECK(p[1] == 0.0);
  auto mdv = m.median_dollar_volume();
  CHECK(mdv[0] == doctest::Approx(1500.0));
  CHECK(mdv[1] == doctest::Approx(1000.0));
}

TEST_CASE("pressure mode strings") {
  for (auto m : {PressureMode::Dollar, PressureMode::Sqrt, PressureMode::Relative})
    CHECK(parse_pressure_mode(to_string(m)) == m);
  CHECK_THROWS_AS(parse_pressure_mode("cap"), std::invalid_argument);
  CHECK_THROWS_AS(PressureModel(1, PressureMode::Relative, 0), std::invalid_argument);
}
```

Append to `tests/test_synthetic_market.cpp`:
```cpp
TEST_CASE("size_sigma makes stock sizes heavy-tailed") {
  SyntheticConfig cfg;
  cfg.sectors = 10;
  cfg.per_sector = 20;
  cfg.bars = 30;
  cfg.size_sigma = 1.5;
  BarStore s(test::temp_dir("syn_tail"));
  auto secs = generate_synthetic(cfg, s);
  double lo = 1e300, hi = 0;
  for (const auto& sec : secs) {
    double mean = 0;
    for (const Bar& b : s.bars(sec.ticker, cfg.tf)) mean += b.v / 30.0;
    lo = std::min(lo, mean);
    hi = std::max(hi, mean);
  }
  CHECK(hi / lo > 100.0);
}
```

- [ ] **Step 2: Run the build to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL. `graph/pressure.hpp: No such file or directory`, and `size_sigma` is not a member.

- [ ] **Step 3: Implement the pressure model**

`src/graph/pressure.hpp`:
```cpp
#pragma once
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace fx {

enum class PressureMode { Dollar, Sqrt, Relative };

PressureMode parse_pressure_mode(std::string_view s);
std::string_view to_string(PressureMode m);

// Spec 5 (A). Tracks per-node trailing volumes so relative pressure uses ADV from previous bars.
class PressureModel {
 public:
  PressureModel(std::size_t n, PressureMode mode, std::size_t adv_window = 20);

  std::vector<double> step(std::span<const double> returns, std::span<const double> volume,
                           std::span<const double> vwap);
  std::vector<double> median_dollar_volume() const;  // 0 where no history

 private:
  static double median_of(const double* first, std::size_t count);
  std::size_t n_, w_;
  PressureMode mode_;
  std::vector<double> vol_, dollar_;  // [i * w + slot]
  std::vector<std::size_t> count_, head_;
};

}  // namespace fx
```

`src/graph/pressure.cpp`:
```cpp
#include "graph/pressure.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace fx {

PressureMode parse_pressure_mode(std::string_view s) {
  if (s == "dollar") return PressureMode::Dollar;
  if (s == "sqrt") return PressureMode::Sqrt;
  if (s == "relative") return PressureMode::Relative;
  throw std::invalid_argument("unknown pressure mode: " + std::string(s));
}

std::string_view to_string(PressureMode m) {
  switch (m) {
    case PressureMode::Dollar: return "dollar";
    case PressureMode::Sqrt: return "sqrt";
    case PressureMode::Relative: return "relative";
  }
  return "?";
}

PressureModel::PressureModel(std::size_t n, PressureMode mode, std::size_t adv_window)
    : n_(n),
      w_(adv_window),
      mode_(mode),
      vol_(n * adv_window, 0.0),
      dollar_(n * adv_window, 0.0),
      count_(n, 0),
      head_(n, 0) {
  if (adv_window == 0) throw std::invalid_argument("PressureModel: adv_window must be >= 1");
}

double PressureModel::median_of(const double* first, std::size_t count) {
  std::vector<double> v(first, first + count);
  const std::size_t mid = count / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
  if (count % 2 == 1) return v[mid];
  const double upper = v[mid];
  const double lower = *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid));
  return 0.5 * (lower + upper);
}

std::vector<double> PressureModel::step(std::span<const double> returns,
                                        std::span<const double> volume,
                                        std::span<const double> vwap) {
  std::vector<double> p(n_, 0.0);
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < n_; ++i) {
    const double r = returns[i], v = volume[i], vw = vwap[i];
    const bool have_bar = std::isfinite(v) && std::isfinite(vw) && v >= 0 && vw > 0;
    if (std::isfinite(r) && have_bar) {
      switch (mode_) {
        case PressureMode::Dollar: p[i] = r * v * vw; break;
        case PressureMode::Sqrt: p[i] = r * std::sqrt(v * vw); break;
        case PressureMode::Relative:
          if (count_[i] > 0) {
            const double adv = median_of(&vol_[i * w_], count_[i]);
            if (adv > 0) p[i] = r * v / adv;
          }
          break;
      }
    }
    if (have_bar) {
      vol_[i * w_ + head_[i]] = v;
      dollar_[i * w_ + head_[i]] = v * vw;
      head_[i] = (head_[i] + 1) % w_;
      count_[i] = std::min(count_[i] + 1, w_);
    }
  }
  return p;
}

std::vector<double> PressureModel::median_dollar_volume() const {
  std::vector<double> m(n_, 0.0);
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < n_; ++i)
    if (count_[i] > 0) m[i] = median_of(&dollar_[i * w_], count_[i]);
  return m;
}

}  // namespace fx
```

- [ ] **Step 4: Add size_sigma to the synthetic market**

In `src/market/synthetic_market.hpp`, add this field after `TimePoint start = ...;` in `SyntheticConfig`:
```cpp
  double size_sigma = 0.0;  // > 0: base volume 1e6 * exp(N(0, size_sigma)) per stock
```
In `src/market/synthetic_market.cpp`:
- Add `#include <cmath>` if it is missing.
- Right after `std::vector<double> price(static_cast<std::size_t>(n), 100.0);`, insert:
```cpp
  std::vector<double> base(static_cast<std::size_t>(n));
  std::mt19937_64 size_rng(cfg.seed ^ 0x5eedULL);
  std::normal_distribution<double> size_dist(0.0, cfg.size_sigma > 0 ? cfg.size_sigma : 1.0);
  for (int i = 0; i < n; ++i)
    base[static_cast<std::size_t>(i)] =
        cfg.size_sigma > 0 ? 1e6 * std::exp(size_dist(size_rng)) : 1e6 * (1 + i % 5);
```
- Replace the line `const double base = 1e6 * (1 + i % 5);` and its use in `const double v = base * ...` with:
```cpp
      const double v = base[ui] * (1.0 + 30.0 * std::abs(r)) * mult;
```
(`ui` is already defined as `static_cast<std::size_t>(i)` in that loop. If the existing variable name differs, use it.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!`. Every pre-existing test still passes, including the planted-rotation test, which proves `size_sigma = 0` is unchanged. No warnings.

- [ ] **Step 6: Commit**
```bash
git add src/graph/pressure.* src/market/synthetic_market.* tests/test_pressure.cpp tests/test_synthetic_market.cpp
git commit -m "feat: size-neutral pressure modes and heavy-tailed synthetic sizes

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Sparse per-bar flux and flux accumulator

**Files:**
- Create: `src/graph/sparse_flux.hpp`, `src/graph/sparse_flux.cpp`
- Test: `tests/test_sparse_flux.cpp`

**Interfaces:**
- Consumes: `ReturnWindow::unit_vectors()` layout from Task 1 (tests only).
- Produces:
  - `struct fx::WEdge { std::uint32_t j; double w; }`
  - `struct fx::BarFlux { std::vector<std::vector<WEdge>> rows; std::vector<double> out, in; }`. Rows are ascending in j and hold exact shares. `out` is the exact per-node outflow and `in` the exact per-node inflow over all pairs.
  - `struct fx::SparseFluxParams { double lambda = 1.0; std::size_t sink_candidates = 256; std::size_t sinks_per_source = 64; }`
  - `BarFlux fx::bar_flux_sparse(std::span<const double> pressure, std::span<const double> unit, std::size_t w, const SparseFluxParams&)`. Affinity is used only if `lambda > 0 && w > 0 && unit.size() == n*w`.
  - `class fx::FluxAccumulator { FluxAccumulator(std::size_t n, double halflife, std::size_t row_cap = 256); void add(const BarFlux&); const std::vector<std::vector<WEdge>>& rows() const; const std::vector<double>& out() const; const std::vector<double>& in() const; double total() const; std::size_t edge_count() const; std::size_t size() const; }`
    - `add` decays everything by `2^(−1/halflife)`, then merges by column and caps each row at `row_cap` by weight. Ties go to the lower j, and rows stay ascending in j.
    - It throws `std::invalid_argument` if halflife ≤ 0 or row_cap = 0. An infinite half-life never decays.

- [ ] **Step 1: Write the failing tests** in `tests/test_sparse_flux.cpp`:
```cpp
#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "graph/return_window.hpp"
#include "graph/sparse_flux.hpp"

using namespace fx;

namespace {
std::vector<double> dense_flux(const std::vector<double>& p, const std::vector<double>& u,
                               std::size_t w, double lambda) {
  const std::size_t n = p.size();
  auto a = [&](std::size_t i, std::size_t j) {
    if (u.empty()) return 1.0;
    double d = 0;
    for (std::size_t s = 0; s < w; ++s) d += u[i * w + s] * u[j * w + s];
    return 1.0 + lambda * d;
  };
  std::vector<double> F(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (!(p[i] < 0)) continue;
    double denom = 0;
    for (std::size_t k = 0; k < n; ++k)
      if (p[k] > 0) denom += p[k] * a(i, k);
    for (std::size_t j = 0; j < n; ++j)
      if (p[j] > 0) F[i * n + j] = -p[i] * p[j] * a(i, j) / denom;
  }
  return F;
}

struct Market {
  std::vector<double> p, u;
  std::size_t w = 8;
  explicit Market(std::size_t n) {
    std::mt19937 rng(11);
    std::normal_distribution<double> d(0.0, 1.0);
    ReturnWindow rw(n, w);
    for (std::size_t t = 0; t < w; ++t) {
      std::vector<double> r(n);
      for (auto& x : r) x = d(rng) * 0.01;
      rw.push(r);
    }
    u = rw.unit_vectors();
    p.resize(n);
    for (auto& x : p) x = d(rng);
  }
};
}  // namespace

TEST_CASE("sparse flux equals the dense formula when every sink is kept") {
  const std::size_t n = 12;
  Market m(n);
  SparseFluxParams sp;
  sp.lambda = 0.8;
  sp.sink_candidates = n;
  sp.sinks_per_source = n;
  BarFlux bf = bar_flux_sparse(m.p, m.u, m.w, sp);
  auto F = dense_flux(m.p, m.u, m.w, 0.8);
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<double> row(n, 0.0);
    for (const auto& e : bf.rows[i]) row[e.j] = e.w;
    const double out_expected = m.p[i] < 0 ? -m.p[i] : 0.0;
    double rs = 0, col = 0;
    for (std::size_t j = 0; j < n; ++j) {
      CHECK(row[j] == doctest::Approx(F[i * n + j]).epsilon(1e-12));
      rs += row[j];
      col += F[j * n + i];
    }
    CHECK(rs == doctest::Approx(out_expected));
    CHECK(bf.out[i] == doctest::Approx(out_expected));
    CHECK(bf.in[i] == doctest::Approx(col).epsilon(1e-12));
  }
}

TEST_CASE("truncated edges keep exact shares and exact column totals") {
  const std::size_t n = 30;
  Market m(n);
  SparseFluxParams sp;
  sp.lambda = 1.0;
  sp.sink_candidates = 8;
  sp.sinks_per_source = 3;
  BarFlux bf = bar_flux_sparse(m.p, m.u, m.w, sp);
  auto F = dense_flux(m.p, m.u, m.w, 1.0);
  double sum_in = 0, sum_out = 0;
  for (std::size_t i = 0; i < n; ++i) {
    CHECK(bf.rows[i].size() <= 3);
    for (std::size_t e = 1; e < bf.rows[i].size(); ++e) CHECK(bf.rows[i][e - 1].j < bf.rows[i][e].j);
    for (const auto& e : bf.rows[i]) CHECK(e.w == doctest::Approx(F[i * n + e.j]).epsilon(1e-12));
    double col = 0;
    for (std::size_t k = 0; k < n; ++k) col += F[k * n + i];
    CHECK(bf.in[i] == doctest::Approx(col).epsilon(1e-12));
    sum_in += bf.in[i];
    sum_out += bf.out[i];
  }
  CHECK(sum_in == doctest::Approx(sum_out));
}

TEST_CASE("edges go to the top candidate sinks by pressure (no affinity)") {
  std::vector<double> p = {-2.0, 5.0, 1.0, 3.0, 4.0};
  SparseFluxParams sp;
  sp.lambda = 0.0;
  sp.sink_candidates = 3;
  sp.sinks_per_source = 2;
  BarFlux bf = bar_flux_sparse(p, {}, 0, sp);
  REQUIRE(bf.rows[0].size() == 2);
  CHECK(bf.rows[0][0].j == 1);
  CHECK(bf.rows[0][1].j == 4);
  CHECK(bf.rows[0][0].w == doctest::Approx(2.0 * 5.0 / 13.0));
  CHECK(bf.in[2] == doctest::Approx(2.0 * 1.0 / 13.0));
  CHECK(bf.out[0] == 2.0);
}

TEST_CASE("no sinks or no sources gives no flux") {
  BarFlux a = bar_flux_sparse(std::vector<double>{-1, -2, 0}, {}, 0, SparseFluxParams{});
  for (double x : a.out) CHECK(x == 0.0);
  for (const auto& r : a.rows) CHECK(r.empty());
  BarFlux b = bar_flux_sparse(std::vector<double>{1, 2, 0}, {}, 0, SparseFluxParams{});
  for (double x : b.in) CHECK(x == 0.0);
}

TEST_CASE("accumulator decays by half-life, merges by column and caps rows") {
  BarFlux bar;
  bar.rows = {{{1, 3.0}, {2, 1.0}}, {}, {}};
  bar.out = {4.0, 0, 0};
  bar.in = {0, 3.0, 1.0};
  FluxAccumulator acc(3, 1.0, 256);
  acc.add(bar);
  acc.add(bar);
  REQUIRE(acc.rows()[0].size() == 2);
  CHECK(acc.rows()[0][0].j == 1);
  CHECK(acc.rows()[0][0].w == doctest::Approx(4.5));
  CHECK(acc.out()[0] == doctest::Approx(6.0));
  CHECK(acc.in()[2] == doctest::Approx(1.5));
  CHECK(acc.total() == doctest::Approx(6.0));
  CHECK(acc.edge_count() == 2);
  FluxAccumulator capped(3, 1.0, 1);
  capped.add(bar);
  REQUIRE(capped.rows()[0].size() == 1);
  CHECK(capped.rows()[0][0].j == 1);
}

TEST_CASE("infinite half-life never decays; invalid settings throw") {
  BarFlux bar;
  bar.rows = {{{1, 2.0}}, {}};
  bar.out = {2.0, 0};
  bar.in = {0, 2.0};
  FluxAccumulator acc(2, std::numeric_limits<double>::infinity());
  acc.add(bar);
  acc.add(bar);
  CHECK(acc.rows()[0][0].w == 4.0);
  CHECK_THROWS_AS(FluxAccumulator(2, 0.0), std::invalid_argument);
  CHECK_THROWS_AS(FluxAccumulator(2, 1.0, 0), std::invalid_argument);
}
```

- [ ] **Step 2: Run the build to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL with `graph/sparse_flux.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/graph/sparse_flux.hpp`:
```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fx {

struct WEdge {
  std::uint32_t j;
  double w;
};

struct BarFlux {
  std::vector<std::vector<WEdge>> rows;  // per source: kept edges, exact shares, ascending j
  std::vector<double> out;               // exact outflow per node
  std::vector<double> in;                // exact inflow per node (all source-sink pairs)
};

struct SparseFluxParams {
  double lambda = 1.0;                // affinity a_ij = 1 + lambda * rho_ij, lambda in [0, 1]
  std::size_t sink_candidates = 256;  // C: candidate sinks with the largest pressure
  std::size_t sinks_per_source = 64;  // M: edges kept per source
};

// Spec 5. unit: n x w unit vectors (ReturnWindow::unit_vectors) or empty for no affinity.
BarFlux bar_flux_sparse(std::span<const double> pressure, std::span<const double> unit,
                        std::size_t w, const SparseFluxParams& params);

class FluxAccumulator {
 public:
  FluxAccumulator(std::size_t n, double halflife, std::size_t row_cap = 256);

  void add(const BarFlux& bar);
  const std::vector<std::vector<WEdge>>& rows() const { return rows_; }
  const std::vector<double>& out() const { return out_; }
  const std::vector<double>& in() const { return in_; }
  double total() const;
  std::size_t edge_count() const;
  std::size_t size() const { return n_; }

 private:
  std::size_t n_;
  double decay_;
  std::size_t cap_;
  std::vector<std::vector<WEdge>> rows_;
  std::vector<double> out_, in_;
};

}  // namespace fx
```

`src/graph/sparse_flux.cpp`:
```cpp
#include "graph/sparse_flux.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fx {
namespace {

double dot(const double* a, const double* b, std::size_t w) {
  double s = 0;
  for (std::size_t k = 0; k < w; ++k) s += a[k] * b[k];
  return s;
}

bool by_weight_desc(const WEdge& x, const WEdge& y) {
  return x.w > y.w || (x.w == y.w && x.j < y.j);
}

bool by_column(const WEdge& x, const WEdge& y) { return x.j < y.j; }

}  // namespace

BarFlux bar_flux_sparse(std::span<const double> pressure, std::span<const double> unit,
                        std::size_t w, const SparseFluxParams& params) {
  const std::size_t n = pressure.size();
  BarFlux bf;
  bf.rows.resize(n);
  bf.out.assign(n, 0.0);
  bf.in.assign(n, 0.0);
  const bool affinity = params.lambda > 0 && w > 0 && unit.size() == n * w;
  const double lambda = affinity ? params.lambda : 0.0;

  std::vector<std::uint32_t> sinks, sources;
  double P = 0;
  for (std::size_t k = 0; k < n; ++k) {
    if (pressure[k] > 0) {
      sinks.push_back(static_cast<std::uint32_t>(k));
      P += pressure[k];
    } else if (pressure[k] < 0) {
      sources.push_back(static_cast<std::uint32_t>(k));
    }
  }
  if (sinks.empty() || sources.empty()) return bf;

  std::vector<double> g(affinity ? w : 0, 0.0);  // g = sum_k p_k u_k
  for (std::size_t s = 0; s < g.size(); ++s)
    for (auto k : sinks) g[s] += pressure[k] * unit[k * w + s];

  std::vector<std::uint32_t> cand(sinks);
  std::sort(cand.begin(), cand.end(), [&](auto a, auto b) {
    return pressure[a] > pressure[b] || (pressure[a] == pressure[b] && a < b);
  });
  if (cand.size() > params.sink_candidates) cand.resize(params.sink_candidates);

  // Per source: a_i = |p_i| / D_i and its kept edges (parallel, disjoint writes).
  std::vector<double> a_of(n, 0.0);
#pragma omp parallel
  {
    std::vector<WEdge> scored;
#pragma omp for schedule(dynamic, 64)
    for (std::size_t si = 0; si < sources.size(); ++si) {
      const std::size_t i = sources[si];
      const double* ui = affinity ? &unit[i * w] : nullptr;
      const double denom = P + (affinity ? lambda * dot(ui, g.data(), w) : 0.0);
      if (!(denom > 0)) continue;
      const double outflow = -pressure[i];
      const double a = outflow / denom;
      a_of[i] = a;
      bf.out[i] = outflow;
      scored.clear();
      for (auto j : cand) {
        const double score =
            pressure[j] * (1.0 + (affinity ? lambda * dot(ui, &unit[j * w], w) : 0.0));
        if (score > 0) scored.push_back({j, score});
      }
      const std::size_t m = std::min(params.sinks_per_source, scored.size());
      std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(m),
                        scored.end(), by_weight_desc);
      auto& row = bf.rows[i];
      row.reserve(m);
      for (std::size_t e = 0; e < m; ++e) row.push_back({scored[e].j, a * scored[e].w});
      std::sort(row.begin(), row.end(), by_column);
    }
  }

  // Serial, index-ordered reductions (determinism): A = sum a_i, Au = sum a_i u_i.
  double A = 0;
  std::vector<double> Au(affinity ? w : 0, 0.0);
  for (auto i : sources) {
    if (a_of[i] == 0) continue;
    A += a_of[i];
    for (std::size_t s = 0; s < Au.size(); ++s) Au[s] += a_of[i] * unit[i * w + s];
  }

#pragma omp parallel for schedule(static)
  for (std::size_t sk = 0; sk < sinks.size(); ++sk) {
    const std::size_t j = sinks[sk];
    bf.in[j] = pressure[j] * (A + (affinity ? lambda * dot(&unit[j * w], Au.data(), w) : 0.0));
  }
  return bf;
}

FluxAccumulator::FluxAccumulator(std::size_t n, double halflife, std::size_t row_cap)
    : n_(n), decay_(std::exp2(-1.0 / halflife)), cap_(row_cap), rows_(n), out_(n, 0.0), in_(n, 0.0) {
  if (!(halflife > 0)) throw std::invalid_argument("FluxAccumulator: halflife must be > 0");
  if (row_cap == 0) throw std::invalid_argument("FluxAccumulator: row_cap must be >= 1");
}

void FluxAccumulator::add(const BarFlux& bar) {
#pragma omp parallel
  {
    std::vector<WEdge> merged;
#pragma omp for schedule(dynamic, 64)
    for (std::size_t i = 0; i < n_; ++i) {
      out_[i] = decay_ * out_[i] + bar.out[i];
      in_[i] = decay_ * in_[i] + bar.in[i];
      auto& row = rows_[i];
      for (auto& e : row) e.w *= decay_;
      const auto& add = bar.rows[i];
      if (add.empty()) continue;
      merged.clear();
      merged.reserve(row.size() + add.size());
      std::size_t a = 0, b = 0;
      while (a < row.size() || b < add.size()) {
        if (b == add.size() || (a < row.size() && row[a].j < add[b].j)) {
          merged.push_back(row[a++]);
        } else if (a == row.size() || add[b].j < row[a].j) {
          merged.push_back(add[b++]);
        } else {
          merged.push_back({row[a].j, row[a].w + add[b].w});
          ++a;
          ++b;
        }
      }
      if (merged.size() > cap_) {
        std::nth_element(merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(cap_),
                         merged.end(), by_weight_desc);
        merged.resize(cap_);
        std::sort(merged.begin(), merged.end(), by_column);
      }
      row.swap(merged);
    }
  }
}

double FluxAccumulator::total() const {
  double s = 0;
  for (double x : out_) s += x;
  return s;
}

std::size_t FluxAccumulator::edge_count() const {
  std::size_t c = 0;
  for (const auto& r : rows_) c += r.size();
  return c;
}

}  // namespace fx
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests -tc="*flux*,*accumulator*,*sinks*"` and then `./build/fluxtests`.
Expected: `Status: SUCCESS!`, no warnings (touch-rebuild check).

- [ ] **Step 5: Commit**
```bash
git add src/graph/sparse_flux.* tests/test_sparse_flux.cpp
git commit -m "feat: sparse per-bar flux with exact totals and decayed edge accumulator

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Transition builder with lift (B), two-sided pruning (C) and retention (E)

**Files:**
- Create: `src/graph/transition.hpp`, `src/graph/transition.cpp`, `tests/flux_test_util.hpp`
- Test: `tests/test_transition.cpp`

**Interfaces:**
- Consumes: `FluxAccumulator`, `WEdge`, `BarFlux` (Task 3); `Csr` (existing `src/graph/csr.hpp`, fields `n, row_ptr, col, val, raw`).
- Produces:
  - `enum class fx::LiftMode { Off, Excess, Ratio }`, `parse_lift_mode(std::string_view)` (`"off"|"excess"|"ratio"`; throws `std::invalid_argument`), `to_string(LiftMode)`
  - `struct fx::TransitionParams { LiftMode lift = LiftMode::Excess; std::size_t k_out = 20; std::size_t k_in = 10; double retention = 1.0; }`
  - `Csr fx::build_transition(const FluxAccumulator& acc, const TransitionParams& params, const std::vector<bool>& active = {})`
    - Rows are ascending in column, including the self-loop. `val` is row-stochastic.
    - `raw` is the accumulated un-lifted F_ij for kept edges, and `retention·in_i` for a self-loop (0 for inactive rows).
    - Inactive rows are `{i: 1.0}` with raw 0, and they receive no edges.
    - This is a new overload. The old dense `build_transition(std::span<const double>, n, k, active)` stays until Task 5 deletes it.
  - `tests/flux_test_util.hpp`: `FluxAccumulator fx::test::acc_from_dense(const std::vector<double>& F, std::size_t n)` (out = row sums, in = column sums, never decays) and `TransitionParams fx::test::legacy_tp(std::size_t k)` (Off, k_out = k, k_in = 0, retention = 0).

- [ ] **Step 1: Write the test helper and the failing tests**

`tests/flux_test_util.hpp`:
```cpp
#pragma once
#include <cstdint>
#include <limits>
#include <vector>

#include "graph/sparse_flux.hpp"
#include "graph/transition.hpp"

namespace fx::test {

// Accumulator holding exactly the dense n x n matrix F: out = row sums, in = column sums.
inline FluxAccumulator acc_from_dense(const std::vector<double>& F, std::size_t n) {
  BarFlux bar;
  bar.rows.resize(n);
  bar.out.assign(n, 0.0);
  bar.in.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double w = F[i * n + j];
      if (w == 0) continue;
      bar.rows[i].push_back({static_cast<std::uint32_t>(j), w});
      bar.out[i] += w;
      bar.in[j] += w;
    }
  }
  FluxAccumulator acc(n, std::numeric_limits<double>::infinity(), n == 0 ? 1 : n);
  acc.add(bar);
  return acc;
}

inline TransitionParams legacy_tp(std::size_t k) {
  TransitionParams tp;
  tp.lift = LiftMode::Off;
  tp.k_out = k;
  tp.k_in = 0;
  tp.retention = 0.0;
  return tp;
}

}  // namespace fx::test
```

`tests/test_transition.cpp`:
```cpp
#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "flux_test_util.hpp"
#include "graph/transition.hpp"

using namespace fx;

namespace {
void check_stochastic(const Csr& P) {
  for (std::size_t i = 0; i < P.n; ++i) {
    double s = 0;
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) {
      s += P.val[e];
      if (e > P.row_ptr[i]) CHECK(P.col[e - 1] < P.col[e]);
    }
    CHECK(s == doctest::Approx(1.0));
  }
}
}  // namespace

TEST_CASE("legacy settings: top-k rows, normalized, self-loops for empty rows") {
  std::vector<double> F = {0, 5, 1, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0};
  Csr P = build_transition(test::acc_from_dense(F, 4), test::legacy_tp(2));
  REQUIRE(P.row_ptr.size() == 5);
  REQUIRE(P.row_ptr[1] == 2);
  CHECK(P.col[0] == 1);
  CHECK(P.val[0] == doctest::Approx(5.0 / 8.0));
  CHECK(P.raw[0] == 5.0);
  CHECK(P.col[1] == 3);
  CHECK(P.val[1] == doctest::Approx(3.0 / 8.0));
  REQUIRE(P.row_ptr[2] - P.row_ptr[1] == 1);
  CHECK(P.col[P.row_ptr[1]] == 1);
  CHECK(P.val[P.row_ptr[1]] == 1.0);
  CHECK(P.raw[P.row_ptr[1]] == 0.0);
  check_stochastic(P);
}

TEST_CASE("ties at the k-th weight go to the lower column") {
  std::vector<double> F = {0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Csr P = build_transition(test::acc_from_dense(F, 4), test::legacy_tp(2));
  REQUIRE(P.row_ptr[1] == 2);
  CHECK(P.col[0] == 1);
  CHECK(P.col[1] == 2);
}

TEST_CASE("inactive nodes get only a self-loop and receive no edges") {
  std::vector<double> F = {0, 1, 1, 1, 0, 1, 1, 1, 0};
  TransitionParams tp;  // defaults, including retention
  Csr P = build_transition(test::acc_from_dense(F, 3), tp, {true, true, false});
  REQUIRE(P.row_ptr[3] - P.row_ptr[2] == 1);
  CHECK(P.col[P.row_ptr[2]] == 2);
  CHECK(P.val[P.row_ptr[2]] == 1.0);
  CHECK(P.raw[P.row_ptr[2]] == 0.0);
  for (std::size_t e = 0; e < P.row_ptr[2]; ++e) CHECK(P.col[e] != 2);
  check_stochastic(P);
}

TEST_CASE("excess and ratio lift prefer above-gravity edges over big expected ones") {
  // out = [12, 2, 31], in = [2, 40, 3], total = 45.
  // F01 = 10 vs E01 = 10.667 (below gravity); F02 = 2 vs E02 = 0.8 (above gravity).
  std::vector<double> F = {0, 10, 2, 1, 0, 1, 1, 30, 0};
  TransitionParams off = test::legacy_tp(1);
  Csr Poff = build_transition(test::acc_from_dense(F, 3), off);
  CHECK(Poff.col[Poff.row_ptr[0]] == 1);
  for (LiftMode mode : {LiftMode::Excess, LiftMode::Ratio}) {
    TransitionParams tp = off;
    tp.lift = mode;
    Csr P = build_transition(test::acc_from_dense(F, 3), tp);
    REQUIRE(P.row_ptr[1] - P.row_ptr[0] == 1);
    CHECK(P.col[P.row_ptr[0]] == 2);
    CHECK(P.val[P.row_ptr[0]] == 1.0);
    CHECK(P.raw[P.row_ptr[0]] == 2.0);
    check_stochastic(P);
  }
}

TEST_CASE("k_in keeps a column's strongest inbound edge even when no row selects it") {
  std::vector<double> F = {0, 0, 0, 1, 5, 0, 0, 2, 5, 0, 0, 1, 5, 0, 0, 0};
  TransitionParams tp = test::legacy_tp(1);
  Csr without = build_transition(test::acc_from_dense(F, 4), tp);
  CHECK(without.row_ptr[2] - without.row_ptr[1] == 1);
  tp.k_in = 1;
  Csr with = build_transition(test::acc_from_dense(F, 4), tp);
  REQUIRE(with.row_ptr[2] - with.row_ptr[1] == 2);
  CHECK(with.col[with.row_ptr[1]] == 0);
  CHECK(with.col[with.row_ptr[1] + 1] == 3);
  check_stochastic(with);
}

TEST_CASE("retention gives a self-loop of in / (in + out)") {
  std::vector<double> F = {0, 1, 3, 0};  // out = [1, 3], in = [3, 1]
  TransitionParams tp = test::legacy_tp(5);
  tp.retention = 1.0;
  Csr P = build_transition(test::acc_from_dense(F, 2), tp);
  REQUIRE(P.row_ptr[1] == 2);
  CHECK(P.col[0] == 0);
  CHECK(P.val[0] == doctest::Approx(0.75));
  CHECK(P.raw[0] == doctest::Approx(3.0));
  CHECK(P.col[1] == 1);
  CHECK(P.val[1] == doctest::Approx(0.25));
  CHECK(P.val[P.row_ptr[1]] == doctest::Approx(0.75));      // row 1 -> node 0
  CHECK(P.val[P.row_ptr[1] + 1] == doctest::Approx(0.25));  // row 1 self
  check_stochastic(P);
}

TEST_CASE("lift mode strings") {
  for (auto m : {LiftMode::Off, LiftMode::Excess, LiftMode::Ratio})
    CHECK(parse_lift_mode(to_string(m)) == m);
  CHECK_THROWS_AS(parse_lift_mode("max"), std::invalid_argument);
}
```

- [ ] **Step 2: Run the build to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL with `graph/transition.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/graph/transition.hpp`:
```cpp
#pragma once
#include <cstddef>
#include <string_view>
#include <vector>

#include "graph/csr.hpp"
#include "graph/sparse_flux.hpp"

namespace fx {

enum class LiftMode { Off, Excess, Ratio };

LiftMode parse_lift_mode(std::string_view s);
std::string_view to_string(LiftMode m);

struct TransitionParams {
  LiftMode lift = LiftMode::Excess;  // (B)
  std::size_t k_out = 20;            // (C) per-row top edges
  std::size_t k_in = 10;             // (C) per-column top edges
  double retention = 1.0;            // (E)
};

// Spec 5 (B, C, E). Row-stochastic; rows ascending by column; raw = un-lifted F (self: retention*in).
Csr build_transition(const FluxAccumulator& acc, const TransitionParams& params,
                     const std::vector<bool>& active = {});

}  // namespace fx
```

`src/graph/transition.cpp`:
```cpp
#include "graph/transition.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace fx {

LiftMode parse_lift_mode(std::string_view s) {
  if (s == "off") return LiftMode::Off;
  if (s == "excess") return LiftMode::Excess;
  if (s == "ratio") return LiftMode::Ratio;
  throw std::invalid_argument("unknown lift mode: " + std::string(s));
}

std::string_view to_string(LiftMode m) {
  switch (m) {
    case LiftMode::Off: return "off";
    case LiftMode::Excess: return "excess";
    case LiftMode::Ratio: return "ratio";
  }
  return "?";
}

namespace {
struct Cand {
  std::uint32_t i, j;
  double w, raw;
};
}  // namespace

Csr build_transition(const FluxAccumulator& acc, const TransitionParams& params,
                     const std::vector<bool>& active) {
  const std::size_t n = acc.size();
  auto is_active = [&](std::size_t i) { return active.empty() || active[i]; };
  const auto& out = acc.out();
  const auto& in = acc.in();
  const double total = acc.total();

  // 1. Lifted candidates per row (parallel; rows are disjoint).
  std::vector<std::vector<Cand>> per_row(n);
#pragma omp parallel for schedule(dynamic, 64)
  for (std::size_t i = 0; i < n; ++i) {
    if (!is_active(i)) continue;
    for (const auto& e : acc.rows()[i]) {
      if (e.j == i || !is_active(e.j) || !(e.w > 0)) continue;
      double w = e.w;
      if (params.lift != LiftMode::Off) {
        const double expected = total > 0 ? out[i] * in[e.j] / total : 0.0;
        if (params.lift == LiftMode::Excess) {
          w = e.w - expected;
        } else {
          w = expected > 0 ? e.w / expected - 1.0 : 0.0;
        }
      }
      if (w > 0) per_row[i].push_back({static_cast<std::uint32_t>(i), e.j, w, e.w});
    }
  }
  std::vector<std::size_t> row_start(n + 1, 0);
  for (std::size_t i = 0; i < n; ++i) row_start[i + 1] = row_start[i] + per_row[i].size();
  std::vector<Cand> cands;
  cands.reserve(row_start[n]);
  for (auto& r : per_row) cands.insert(cands.end(), r.begin(), r.end());
  per_row.clear();

  auto by_weight = [&](std::size_t a, std::size_t b) {
    const Cand &x = cands[a], &y = cands[b];
    if (x.w != y.w) return x.w > y.w;
    if (x.i != y.i) return x.i < y.i;
    return x.j < y.j;
  };
  std::vector<char> keep(cands.size(), 0);

  // 2a. Top k_out per row.
#pragma omp parallel
  {
    std::vector<std::size_t> idx;
#pragma omp for schedule(dynamic, 64)
    for (std::size_t i = 0; i < n; ++i) {
      idx.clear();
      for (std::size_t c = row_start[i]; c < row_start[i + 1]; ++c) idx.push_back(c);
      const std::size_t m = std::min(params.k_out, idx.size());
      std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(m), idx.end(),
                        by_weight);
      for (std::size_t e = 0; e < m; ++e) keep[idx[e]] = 1;
    }
  }
  // 2b. Top k_in per column (each candidate belongs to exactly one column).
  if (params.k_in > 0) {
    std::vector<std::vector<std::size_t>> by_col(n);
    for (std::size_t c = 0; c < cands.size(); ++c) by_col[cands[c].j].push_back(c);
#pragma omp parallel for schedule(dynamic, 64)
    for (std::size_t j = 0; j < n; ++j) {
      auto& col = by_col[j];
      const std::size_t m = std::min(params.k_in, col.size());
      std::partial_sort(col.begin(), col.begin() + static_cast<std::ptrdiff_t>(m), col.end(),
                        by_weight);
      for (std::size_t e = 0; e < m; ++e) keep[col[e]] = 1;
    }
  }

  // 3. Assemble rows (serial).
  Csr P;
  P.n = n;
  P.row_ptr.reserve(n + 1);
  P.row_ptr.push_back(0);
  for (std::size_t i = 0; i < n; ++i) {
    double offsum = 0;
    for (std::size_t c = row_start[i]; c < row_start[i + 1]; ++c)
      if (keep[c]) offsum += cands[c].w;
    const double retained = is_active(i) ? params.retention * in[i] : 0.0;
    auto push_self = [&](double val) {
      P.col.push_back(static_cast<std::uint32_t>(i));
      P.val.push_back(val);
      P.raw.push_back(retained);
    };
    if (!(offsum > 0)) {
      push_self(1.0);
    } else {
      const double denom = out[i] + retained;
      const double self_mass = denom > 0 ? retained / denom : 0.0;
      const double off_mass = 1.0 - self_mass;
      bool self_done = !(self_mass > 0);
      for (std::size_t c = row_start[i]; c < row_start[i + 1]; ++c) {
        if (!keep[c]) continue;
        if (!self_done && cands[c].j > i) {
          push_self(self_mass);
          self_done = true;
        }
        P.col.push_back(cands[c].j);
        P.val.push_back(off_mass * cands[c].w / offsum);
        P.raw.push_back(cands[c].raw);
      }
      if (!self_done) push_self(self_mass);
    }
    P.row_ptr.push_back(P.col.size());
  }
  return P;
}

}  // namespace fx
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!`, no warnings.

- [ ] **Step 5: Commit**
```bash
git add src/graph/transition.* tests/flux_test_util.hpp tests/test_transition.cpp
git commit -m "feat: transition builder with gravity lift, two-sided pruning and retention

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Hotness reference (D), parallel solver, pipeline rewrite, dense code removal

**Files:**
- Create: `src/graph/hotness.hpp`, `src/graph/hotness.cpp`
- Modify: `src/graph/csr.hpp`, `src/graph/csr.cpp`, `src/graph/markov_solver.cpp`, `src/pipeline/core_pipeline.hpp`, `src/pipeline/core_pipeline.cpp`
- Delete: `src/graph/flux_builder.hpp`, `src/graph/flux_builder.cpp`, `tests/test_flux_builder.cpp`
- Test: `tests/test_hotness.cpp` (new), `tests/test_csr.cpp` (rewrite), `tests/test_markov_solver.cpp` (one edit), `tests/test_core_pipeline.cpp` (edits plus new cases)

**Interfaces:**
- Consumes: `ReturnWindow` (T1), `PressureModel` (T2), `bar_flux_sparse`/`FluxAccumulator`/`SparseFluxParams` (T3), `build_transition(acc, ...)`/`TransitionParams` (T4), the existing `stationary`, `hotness`, `forecast`, `Panel`.
- Produces (milestones 2 and 3 build on these):
  - `enum class fx::HotRef { Uniform, Size, LongRun }`, `parse_hot_ref(std::string_view)` (`"uniform"|"size"|"longrun"`), `to_string(HotRef)`
  - `std::vector<double> fx::relative_hotness(std::span<const double> pi, std::span<const double> ref)`. It returns `pi_i/ref_i − 1` with ref normalized to sum 1. Non-positive or non-finite ref entries are floored at 1e-12·max, and an all-zero ref falls back to uniform.
  - `Csr fx::transpose(const Csr&)` and `std::vector<double> fx::left_multiply_transposed(const Csr& PT, std::span<const double> x)`, a parallel pull product equal to `left_multiply(P, x)`.
  - `CoreParams` (fields exactly as below), `CoreParams::legacy()`, `CoreParams::validate()` (throws `std::invalid_argument`). `Frame` is unchanged. `CorePipeline(std::size_t n, CoreParams)` validates its params. `step` and `run_panel_last` keep their signatures.

- [ ] **Step 1: Write the failing tests**

`tests/test_hotness.cpp`:
```cpp
#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "graph/hotness.hpp"

using namespace fx;

TEST_CASE("relative hotness against uniform, size and degenerate references") {
  std::vector<double> pi = {0.5, 0.25, 0.25};
  auto u = relative_hotness(pi, std::vector<double>{1, 1, 1});
  CHECK(u[0] == doctest::Approx(0.5));
  CHECK(u[1] == doctest::Approx(-0.25));
  auto s = relative_hotness(pi, std::vector<double>{2, 1, 1});  // ref = [0.5, 0.25, 0.25]
  for (double x : s) CHECK(x == doctest::Approx(0.0));
  auto z = relative_hotness(pi, std::vector<double>{0, 0, 0});  // uniform fallback
  CHECK(z[0] == doctest::Approx(0.5));
  auto f = relative_hotness(pi, std::vector<double>{1, 0, 1});  // floored, finite
  CHECK(f[1] > 1e6);
}

TEST_CASE("hot reference strings") {
  for (auto r : {HotRef::Uniform, HotRef::Size, HotRef::LongRun}) CHECK(parse_hot_ref(to_string(r)) == r);
  CHECK_THROWS_AS(parse_hot_ref("cap"), std::invalid_argument);
}
```

Replace the whole of `tests/test_csr.cpp` with:
```cpp
#include <doctest/doctest.h>

#include <vector>

#include "graph/csr.hpp"

using namespace fx;

TEST_CASE("left_multiply computes x * P") {
  Csr P;
  P.n = 2;
  P.row_ptr = {0, 1, 2};
  P.col = {1, 1};
  P.val = {1.0, 1.0};
  P.raw = {1.0, 0.0};
  auto y = left_multiply(P, std::vector<double>{0.3, 0.7});
  CHECK(y[0] == doctest::Approx(0.0));
  CHECK(y[1] == doctest::Approx(1.0));
}

TEST_CASE("transposed pull product equals left_multiply") {
  Csr P;
  P.n = 3;
  P.row_ptr = {0, 2, 3, 5};
  P.col = {0, 2, 1, 0, 1};
  P.val = {0.25, 0.75, 1.0, 0.5, 0.5};
  P.raw = {1, 3, 1, 1, 1};
  std::vector<double> x = {0.2, 0.3, 0.5};
  const Csr PT = transpose(P);
  CHECK(PT.n == 3);
  auto a = left_multiply(P, x);
  auto b = left_multiply_transposed(PT, x);
  for (std::size_t i = 0; i < 3; ++i) CHECK(b[i] == doctest::Approx(a[i]));
}
```

In `tests/test_markov_solver.cpp`:
- Add `#include "flux_test_util.hpp"`.
- Replace `Csr P = build_transition(F, 4, 4);` with:
```cpp
  Csr P = build_transition(test::acc_from_dense(F, 4), test::legacy_tp(4));
```

In `tests/test_core_pipeline.cpp`:
- **(a)** Replace the existing `TEST_CASE("planted rotation makes the receiving sector the top hill") { ... }` with this helper (inside an anonymous namespace placed above it) and two cases:
```cpp
namespace {
void check_rotation(const CoreParams& params) {
  SyntheticConfig cfg;
  BarStore store(test::temp_dir("pipeline"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  Panel panel = build_panel(store, tickers, cfg.tf);
  Frame f = run_panel_last(panel, params);
  CHECK(f.solve.converged);
  REQUIRE(f.h.size() == 50);
  REQUIRE(f.forecasts.size() == 3);
  CHECK(f.forecasts[2].k == 8);
  CHECK(f.t == panel.times.back());
  std::map<std::string, double> sector_mean;
  for (std::size_t i = 0; i < secs.size(); ++i) sector_mean[secs[i].sector] += f.h[i] / 10.0;
  for (const auto& [sector, mean] : sector_mean) {
    INFO(sector << " mean h " << mean);
    if (sector != "Sector1") CHECK(sector_mean["Sector1"] > mean);
  }
  CHECK(sector_mean["Sector1"] > 0);
  CHECK(sector_mean["Sector0"] < sector_mean["Sector1"]);
}
}  // namespace

TEST_CASE("planted rotation makes the receiving sector the top hill (defaults)") {
  check_rotation(CoreParams{});
}

TEST_CASE("planted rotation makes the receiving sector the top hill (legacy)") {
  check_rotation(CoreParams::legacy());
}
```
- **(b)** In `TEST_CASE("frame carries raw pruned weights and the fast matrix")`, change both `run_panel_last(p, CoreParams{})` calls to `run_panel_last(p, CoreParams::legacy())`. Under legacy settings, val = raw / row-sum still holds.
- **(c)** Append at the end of the file:
```cpp
TEST_CASE("hotness references size and longrun give finite relative hotness") {
  for (HotRef ref : {HotRef::Size, HotRef::LongRun}) {
    CoreParams p;
    p.h_ref = ref;
    Frame f = run_panel_last(small_panel(false), p);
    CHECK(f.solve.converged);
    for (double h : f.h) CHECK(std::isfinite(h));
  }
}

TEST_CASE("CoreParams::validate rejects bad settings") {
  CoreParams a;
  a.alpha = 0;
  CHECK_THROWS_AS(a.validate(), std::invalid_argument);
  CoreParams b;
  b.flux.lambda = 1.5;
  CHECK_THROWS_AS(b.validate(), std::invalid_argument);
  CoreParams c;
  c.horizons.clear();
  CHECK_THROWS_AS(c.validate(), std::invalid_argument);
  CoreParams d;
  d.transition.k_out = 0;
  CHECK_THROWS_AS(d.validate(), std::invalid_argument);
  CoreParams e;
  e.flux.sink_candidates = 10;
  e.flux.sinks_per_source = 20;
  CHECK_THROWS_AS(e.validate(), std::invalid_argument);
  CHECK_NOTHROW(CoreParams{}.validate());
  CHECK_NOTHROW(CoreParams::legacy().validate());
  CHECK_THROWS_AS(CorePipeline(4, a), std::invalid_argument);
}
```
Make sure the file includes `<cmath>`, `<map>` and `<stdexcept>`.

- [ ] **Step 2: Run the build to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL. `graph/hotness.hpp` is missing, and `CoreParams::legacy`, `transpose` and `left_multiply_transposed` are not declared.

- [ ] **Step 3: Implement hotness and the CSR transpose**

`src/graph/hotness.hpp`:
```cpp
#pragma once
#include <span>
#include <string_view>
#include <vector>

namespace fx {

enum class HotRef { Uniform, Size, LongRun };  // (D)

HotRef parse_hot_ref(std::string_view s);
std::string_view to_string(HotRef r);

// h_i = pi_i / ref_i - 1 with ref normalized to sum 1 (spec 5 D).
std::vector<double> relative_hotness(std::span<const double> pi, std::span<const double> ref);

}  // namespace fx
```

`src/graph/hotness.cpp`:
```cpp
#include "graph/hotness.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace fx {

HotRef parse_hot_ref(std::string_view s) {
  if (s == "uniform") return HotRef::Uniform;
  if (s == "size") return HotRef::Size;
  if (s == "longrun") return HotRef::LongRun;
  throw std::invalid_argument("unknown hotness reference: " + std::string(s));
}

std::string_view to_string(HotRef r) {
  switch (r) {
    case HotRef::Uniform: return "uniform";
    case HotRef::Size: return "size";
    case HotRef::LongRun: return "longrun";
  }
  return "?";
}

std::vector<double> relative_hotness(std::span<const double> pi, std::span<const double> ref) {
  const std::size_t n = pi.size();
  std::vector<double> r(n, 0.0);
  double mx = 0;
  for (std::size_t i = 0; i < n; ++i) {
    r[i] = std::isfinite(ref[i]) && ref[i] > 0 ? ref[i] : 0.0;
    mx = std::max(mx, r[i]);
  }
  if (!(mx > 0)) std::fill(r.begin(), r.end(), 1.0), mx = 1.0;
  double sum = 0;
  for (auto& x : r) {
    x = std::max(x, 1e-12 * mx);
    sum += x;
  }
  std::vector<double> h(n);
  for (std::size_t i = 0; i < n; ++i) h[i] = pi[i] / (r[i] / sum) - 1.0;
  return h;
}

}  // namespace fx
```

Replace `src/graph/csr.hpp` with:
```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fx {

struct Csr {
  std::size_t n = 0;
  std::vector<std::size_t> row_ptr;
  std::vector<std::uint32_t> col;
  std::vector<double> val;
  std::vector<double> raw;  // accumulated un-lifted flux for kept edges; retention*in for self-loops
};

std::vector<double> left_multiply(const Csr& P, std::span<const double> x);  // y = x P
Csr transpose(const Csr& P);
// y = x P computed from PT = transpose(P) as a parallel pull (deterministic).
std::vector<double> left_multiply_transposed(const Csr& PT, std::span<const double> x);

}  // namespace fx
```

Replace `src/graph/csr.cpp` with:
```cpp
#include "graph/csr.hpp"

namespace fx {

std::vector<double> left_multiply(const Csr& P, std::span<const double> x) {
  std::vector<double> y(P.n, 0.0);
  for (std::size_t i = 0; i < P.n; ++i) {
    const double xi = x[i];
    if (xi == 0) continue;
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) y[P.col[e]] += xi * P.val[e];
  }
  return y;
}

Csr transpose(const Csr& P) {
  Csr T;
  T.n = P.n;
  T.row_ptr.assign(P.n + 1, 0);
  for (auto c : P.col) ++T.row_ptr[c + 1];
  for (std::size_t i = 0; i < P.n; ++i) T.row_ptr[i + 1] += T.row_ptr[i];
  T.col.resize(P.col.size());
  T.val.resize(P.val.size());
  T.raw.resize(P.raw.size());
  std::vector<std::size_t> next(T.row_ptr.begin(), T.row_ptr.end() - 1);
  for (std::size_t i = 0; i < P.n; ++i) {
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) {
      const std::size_t slot = next[P.col[e]]++;
      T.col[slot] = static_cast<std::uint32_t>(i);
      T.val[slot] = P.val[e];
      if (!P.raw.empty()) T.raw[slot] = P.raw[e];
    }
  }
  return T;
}

std::vector<double> left_multiply_transposed(const Csr& PT, std::span<const double> x) {
  std::vector<double> y(PT.n, 0.0);
#pragma omp parallel for schedule(dynamic, 256)
  for (std::size_t j = 0; j < PT.n; ++j) {
    double s = 0;
    for (auto e = PT.row_ptr[j]; e < PT.row_ptr[j + 1]; ++e) s += x[PT.col[e]] * PT.val[e];
    y[j] = s;
  }
  return y;
}

}  // namespace fx
```

- [ ] **Step 4: Make the solver transpose once per call**

In `src/graph/markov_solver.cpp`, keep the public signatures but route the iterations through a transposed matrix. Replace the bodies of `damped_step`, `stationary` and `propagate` with the code below. It adds an anonymous-namespace helper `step_t`; `hotness` is unchanged.
```cpp
namespace {
std::vector<double> step_t(const Csr& PT, double alpha, std::span<const double> pi) {
  std::vector<double> next = left_multiply_transposed(PT, pi);
  const double teleport = (1.0 - alpha) / static_cast<double>(PT.n);
  double sum = 0;
  for (double& x : next) {
    x = alpha * x + teleport;
    sum += x;
  }
  for (double& x : next) x /= sum;
  return next;
}
}  // namespace

std::vector<double> damped_step(const Csr& P, double alpha, std::span<const double> pi) {
  return step_t(transpose(P), alpha, pi);
}

SolveResult stationary(const Csr& P, double alpha, std::span<const double> warm_start, double tol,
                       int max_iter) {
  const Csr PT = transpose(P);
  SolveResult r;
  if (warm_start.size() == P.n) {
    r.pi.assign(warm_start.begin(), warm_start.end());
  } else {
    r.pi.assign(P.n, 1.0 / static_cast<double>(P.n));
  }
  for (r.iterations = 1; r.iterations <= max_iter; ++r.iterations) {
    std::vector<double> next = step_t(PT, alpha, r.pi);
    r.residual = 0;
    for (std::size_t i = 0; i < P.n; ++i) r.residual += std::abs(next[i] - r.pi[i]);
    r.pi = std::move(next);
    if (r.residual < tol) {
      r.converged = true;
      return r;
    }
  }
  r.iterations = max_iter;
  return r;
}

std::vector<double> propagate(const Csr& P, double alpha, std::span<const double> pi, int k) {
  const Csr PT = transpose(P);
  std::vector<double> x(pi.begin(), pi.end());
  for (int s = 0; s < k; ++s) x = step_t(PT, alpha, x);
  return x;
}
```
Keep the file's existing includes (`<cmath>`), and add `<utility>` if `std::move` needs it.

- [ ] **Step 5: Rewrite the pipeline and delete the dense code**

Run: `git rm src/graph/flux_builder.hpp src/graph/flux_builder.cpp tests/test_flux_builder.cpp`

Replace `src/pipeline/core_pipeline.hpp` with:
```cpp
#pragma once
#include <optional>
#include <vector>

#include "core/types.hpp"
#include "graph/csr.hpp"
#include "graph/forecaster.hpp"
#include "graph/hotness.hpp"
#include "graph/markov_solver.hpp"
#include "graph/pressure.hpp"
#include "graph/return_window.hpp"
#include "graph/sparse_flux.hpp"
#include "graph/transition.hpp"
#include "market/panel.hpp"

namespace fx {

struct CoreParams {
  PressureMode pressure = PressureMode::Relative;  // (A)
  std::size_t adv_window = 20;
  std::size_t corr_window = 60;
  SparseFluxParams flux;  // lambda, sink_candidates, sinks_per_source
  double halflife_slow = 20;
  double halflife_fast = 3;
  double halflife_long = 120;  // used only when h_ref == LongRun
  std::size_t row_cap = 256;
  TransitionParams transition;     // (B) lift, (C) k_out / k_in, (E) retention
  HotRef h_ref = HotRef::Uniform;  // (D)
  double alpha = 0.85;
  double beta = 0.5;
  std::vector<int> horizons{1, 4, 8};

  static CoreParams legacy();  // milestone-1 behaviour (spec 5)
  void validate() const;       // throws std::invalid_argument
};

struct Frame {
  TimePoint t = 0;
  std::vector<bool> active;         // size n; false = no data in the whole panel
  std::vector<double> pi, h;        // inactive: pi = 0, h = NaN
  SolveResult solve;
  std::vector<Forecast> forecasts;  // parallel to CoreParams::horizons
  Csr P;                            // slow (equilibrium) transition matrix
  Csr P_fast;                       // fast transition matrix
  double compute_ms = 0;
};

class CorePipeline {
 public:
  CorePipeline(std::size_t n, CoreParams params);
  Frame step(const Panel& panel, std::size_t t);

 private:
  std::size_t n_;
  CoreParams params_;
  PressureModel pressure_;
  ReturnWindow window_;
  FluxAccumulator slow_, fast_;
  std::optional<FluxAccumulator> long_;
  std::vector<bool> active_;          // computed from the panel on the first step
  std::vector<double> prev_pi_;       // full size n, 0 for inactive
  std::vector<double> prev_long_pi_;  // active sub-index, warm start for the long-run solve
};

Frame run_panel_last(const Panel& panel, const CoreParams& params);

}  // namespace fx
```

Replace `src/pipeline/core_pipeline.cpp` with:
```cpp
#include "pipeline/core_pipeline.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace fx {

namespace {

// Restrict a full-size transition matrix to the active sub-index (columns remapped).
Csr compact(const Csr& P, const std::vector<std::size_t>& map, std::size_t n_active) {
  Csr C;
  C.n = n_active;
  C.row_ptr.push_back(0);
  for (std::size_t i = 0; i < P.n; ++i) {
    if (map[i] == static_cast<std::size_t>(-1)) continue;
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) {
      C.col.push_back(static_cast<std::uint32_t>(map[P.col[e]]));
      C.val.push_back(P.val[e]);
      C.raw.push_back(P.raw[e]);
    }
    C.row_ptr.push_back(C.col.size());
  }
  return C;
}

const CoreParams& validated(const CoreParams& p) {
  p.validate();
  return p;
}

}  // namespace

CoreParams CoreParams::legacy() {
  CoreParams p;
  p.pressure = PressureMode::Dollar;
  p.transition.lift = LiftMode::Off;
  p.transition.k_in = 0;
  p.transition.retention = 0.0;
  p.h_ref = HotRef::Uniform;
  return p;
}

void CoreParams::validate() const {
  auto fail = [](const char* what) {
    throw std::invalid_argument(std::string("CoreParams: ") + what);
  };
  if (!(alpha > 0 && alpha <= 1)) fail("alpha must be in (0, 1]");
  if (!(flux.lambda >= 0 && flux.lambda <= 1)) fail("lambda must be in [0, 1]");
  if (!(halflife_slow > 0 && halflife_fast > 0 && halflife_long > 0)) fail("half-lives must be > 0");
  if (transition.k_out == 0) fail("k_out must be >= 1");
  if (!(transition.retention >= 0)) fail("retention must be >= 0");
  if (horizons.empty()) fail("horizons must not be empty");
  for (int k : horizons)
    if (k < 1) fail("horizons must be >= 1");
  if (corr_window < 2) fail("corr_window must be >= 2");
  if (adv_window < 1) fail("adv_window must be >= 1");
  if (flux.sinks_per_source < 1 || flux.sink_candidates < flux.sinks_per_source)
    fail("need 1 <= sinks_per_source <= sink_candidates");
  if (row_cap < transition.k_out) fail("row_cap must be >= k_out");
}

CorePipeline::CorePipeline(std::size_t n, CoreParams params)
    : n_(n),
      params_(validated(params)),
      pressure_(n, params_.pressure, params_.adv_window),
      window_(n, params_.corr_window),
      slow_(n, params_.halflife_slow, params_.row_cap),
      fast_(n, params_.halflife_fast, params_.row_cap) {
  if (params_.h_ref == HotRef::LongRun) long_.emplace(n, params_.halflife_long, params_.row_cap);
}

Frame CorePipeline::step(const Panel& panel, std::size_t t) {
  if (t == 0 || t >= panel.T()) throw std::invalid_argument("CorePipeline::step: t out of range");
  if (panel.N() != n_) throw std::invalid_argument("CorePipeline::step: panel size mismatch");
  const auto t0 = std::chrono::steady_clock::now();
  const double nan = std::numeric_limits<double>::quiet_NaN();

  if (active_.empty()) {
    active_.assign(n_, false);
    for (std::size_t s = 0; s < panel.T(); ++s)
      for (std::size_t i = 0; i < n_; ++i)
        if (std::isfinite(panel.close[panel.idx(s, i)])) active_[i] = true;
  }
  std::vector<std::size_t> map(n_, static_cast<std::size_t>(-1));
  std::size_t n_active = 0;
  for (std::size_t i = 0; i < n_; ++i)
    if (active_[i]) map[i] = n_active++;
  if (n_active == 0) throw std::runtime_error("no nodes with data");

  std::vector<double> returns(n_, nan), volume(n_, nan), vwap(n_, nan);
  for (std::size_t i = 0; i < n_; ++i) {
    const double c = panel.close[panel.idx(t, i)];
    const double c_prev = panel.close[panel.idx(t - 1, i)];
    if (std::isfinite(c) && std::isfinite(c_prev) && c_prev > 0) returns[i] = c / c_prev - 1.0;
    volume[i] = panel.volume[panel.idx(t, i)];
    vwap[i] = panel.vwap[panel.idx(t, i)];
  }
  const std::vector<double> pressure = pressure_.step(returns, volume, vwap);
  window_.push(returns);
  std::span<const double> unit;
  if (params_.flux.lambda > 0) unit = window_.unit_vectors();
  const BarFlux bar = bar_flux_sparse(pressure, unit, window_.window(), params_.flux);
  slow_.add(bar);
  fast_.add(bar);
  if (long_) long_->add(bar);

  Frame f;
  f.t = panel.times[t];
  f.active = active_;
  f.P = build_transition(slow_, params_.transition, active_);
  f.P_fast = build_transition(fast_, params_.transition, active_);

  const Csr Pa = compact(f.P, map, n_active);
  const Csr Pa_fast = compact(f.P_fast, map, n_active);
  std::vector<double> prev_a;
  if (prev_pi_.size() == n_) {
    for (std::size_t i = 0; i < n_; ++i)
      if (active_[i]) prev_a.push_back(prev_pi_[i]);
  }
  f.solve = stationary(Pa, params_.alpha, prev_a);
  const std::vector<double> pi_a = f.solve.pi;

  std::vector<double> h_a;
  switch (params_.h_ref) {
    case HotRef::Uniform:
      h_a = hotness(pi_a);
      break;
    case HotRef::Size: {
      const std::vector<double> mdv = pressure_.median_dollar_volume();
      std::vector<double> ref;
      ref.reserve(n_active);
      for (std::size_t i = 0; i < n_; ++i)
        if (active_[i]) ref.push_back(mdv[i]);
      h_a = relative_hotness(pi_a, ref);
      break;
    }
    case HotRef::LongRun: {
      const Csr Pl = compact(build_transition(*long_, params_.transition, active_), map, n_active);
      const SolveResult lr = stationary(Pl, params_.alpha, prev_long_pi_);
      prev_long_pi_ = lr.pi;
      h_a = relative_hotness(pi_a, lr.pi);
      break;
    }
  }

  f.pi.assign(n_, 0.0);
  f.h.assign(n_, nan);
  for (std::size_t i = 0; i < n_; ++i) {
    if (!active_[i]) continue;
    f.pi[i] = pi_a[map[i]];
    f.h[i] = h_a[map[i]];
  }
  f.solve.pi = f.pi;
  for (int k : params_.horizons) {
    Forecast fa = forecast(Pa_fast, params_.alpha, pi_a, prev_a, k, params_.beta);
    Forecast full;
    full.k = fa.k;
    full.pi_k.assign(n_, 0.0);
    full.score.assign(n_, nan);
    for (std::size_t i = 0; i < n_; ++i) {
      if (!active_[i]) continue;
      full.pi_k[i] = fa.pi_k[map[i]];
      full.score[i] = fa.score[map[i]];
    }
    f.forecasts.push_back(std::move(full));
  }
  prev_pi_ = f.pi;
  f.compute_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return f;
}

Frame run_panel_last(const Panel& panel, const CoreParams& params) {
  if (panel.T() < 2) throw std::runtime_error("need at least two bars to build flux");
  CorePipeline pipeline(panel.N(), params);
  Frame last;
  for (std::size_t t = 1; t < panel.T(); ++t) last = pipeline.step(panel, t);
  return last;
}

}  // namespace fx
```

`src/main.cpp` still compiles unchanged (it only uses `CoreParams params;`). Search for any leftover use of `top_k` or `FluxBuilder` with `grep -rn "top_k\|FluxBuilder\|flux_builder" src tests` and fix any hits.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cmake -S . -B build && cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!` and no warnings.
**If the defaults rotation test fails, do not loosen it and do not change the defaults.** Report BLOCKED with the per-sector means that the INFO lines print. That result is a modeling finding the controller must take to the user.

- [ ] **Step 7: Commit**
```bash
git add -A src/graph src/pipeline tests
git commit -m "feat: A-E pipeline on sparse flux; hotness reference; parallel solver; drop dense flux

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Evaluation harness

**Files:**
- Create: `src/pipeline/evaluation.hpp`, `src/pipeline/evaluation.cpp`
- Test: `tests/test_evaluation.cpp`

**Interfaces:**
- Consumes: `CorePipeline`, `CoreParams`, `Frame` (T5); `Panel`; `Security`; `generate_synthetic` with `size_sigma` (T2).
- Produces:
  - `double fx::gini(std::span<const double>)`
  - `double fx::spearman(std::span<const double>, std::span<const double>)` (average ranks for ties; NaN if the sizes differ, there are fewer than 3 points, or the variance is zero)
  - `double fx::floor_share(const Frame&, double alpha)` (relative tolerance 1e-6)
  - `double fx::sector_coherence(const Frame&, const std::vector<Security>&)` (known sector = not `""`, `"Unclassified"` or `"Extra"`; off-diagonal raw weight in `f.P`)
  - `struct fx::EvalMetrics { double floor_share, gini, sector_coherence, ic_mean, ic_t, ic_h_mean, ic_h_t; std::size_t ic_samples; double mean_frame_ms; }`
  - `EvalMetrics fx::evaluate(const Panel&, const std::vector<Security>& nodes, const CoreParams&, std::size_t eval_bars)`
    - Evaluated frames are `t ∈ [max(1, T − eval_bars), T−1]`.
    - The pipeline starts `warmup = max(corr_window, ceil(3·halflife_slow))` bars earlier, but not before t = 1.
    - Frame metrics are averaged over the evaluated frames. IC uses the +1 score (or h) at t against the return from t to t+1.
    - It throws `std::runtime_error` if T < 3 and `std::invalid_argument` if nodes.size() ≠ N.
  - `struct fx::EvalConfig { std::string name; CoreParams params; }`, `std::vector<EvalConfig> fx::evaluation_grid()`. The grid has 9 entries in this order: `legacy`, `+A relative`, `+B excess`, `+C k_in=10`, `+D size`, `+D longrun`, `+E retention`, `defaults`, `defaults+longrun`.

- [ ] **Step 1: Write the failing tests** in `tests/test_evaluation.cpp`:
```cpp
#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/evaluation.hpp"
#include "test_util.hpp"

using namespace fx;

TEST_CASE("gini of equal and concentrated distributions") {
  CHECK(gini(std::vector<double>{1, 1, 1, 1}) == doctest::Approx(0.0));
  CHECK(gini(std::vector<double>{0, 0, 0, 1}) == doctest::Approx(0.75));
  CHECK(gini(std::vector<double>{}) == 0.0);
}

TEST_CASE("spearman with ties and degenerate inputs") {
  CHECK(spearman(std::vector<double>{1, 2, 3, 4}, std::vector<double>{10, 20, 30, 40}) ==
        doctest::Approx(1.0));
  CHECK(spearman(std::vector<double>{1, 2, 3, 4}, std::vector<double>{4, 3, 2, 1}) ==
        doctest::Approx(-1.0));
  CHECK(spearman(std::vector<double>{1, 1, 2, 3}, std::vector<double>{1, 2, 3, 4}) ==
        doctest::Approx(4.5 / std::sqrt(22.5)));
  CHECK(std::isnan(spearman(std::vector<double>{1, 2}, std::vector<double>{1, 2})));
  CHECK(std::isnan(spearman(std::vector<double>{1, 1, 1}, std::vector<double>{1, 2, 3})));
}

TEST_CASE("floor share and sector coherence on a hand-built frame") {
  Frame f;
  f.active = {true, true, true, false};
  const double floor = (1 - 0.85) / 3.0;
  f.pi = {floor, floor * (1 + 1e-8), 1 - 2 * floor, 0};
  CHECK(floor_share(f, 0.85) == doctest::Approx(2.0 / 3.0));

  std::vector<Security> nodes = {{"A", "A", "Tech"}, {"B", "B", "Tech"}, {"C", "C", "Energy"},
                                 {"D", "D", "Unclassified"}};
  f.active = {true, true, true, true};
  f.P.n = 4;
  f.P.row_ptr = {0, 2, 3, 4, 5};
  f.P.col = {1, 2, 1, 3, 3};  // 0->1 same, 0->2 diff, 1->1 self, 2->3 unknown, 3->3 self
  f.P.val = {0.5, 0.5, 1, 1, 1};
  f.P.raw = {2, 1, 3, 5, 0};
  CHECK(sector_coherence(f, nodes) == doctest::Approx(2.0 / 3.0));
}

TEST_CASE("evaluation grid covers legacy, each switch and the defaults") {
  auto g = evaluation_grid();
  REQUIRE(g.size() == 9);
  CHECK(g.front().name == "legacy");
  CHECK(g[7].name == "defaults");
  for (const auto& c : g) CHECK_NOTHROW(c.params.validate());
}

TEST_CASE("defaults leave fewer nodes at the teleport floor than legacy on a heavy-tailed market") {
  SyntheticConfig cfg;
  cfg.sectors = 10;
  cfg.per_sector = 20;
  cfg.bars = 160;
  cfg.size_sigma = 1.5;
  BarStore store(test::temp_dir("eval"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  Panel panel = build_panel(store, tickers, cfg.tf);
  const EvalMetrics legacy = evaluate(panel, secs, CoreParams::legacy(), 30);
  const EvalMetrics defaults = evaluate(panel, secs, CoreParams{}, 30);
  INFO("legacy floor " << legacy.floor_share << ", defaults floor " << defaults.floor_share);
  CHECK(defaults.floor_share < legacy.floor_share);
  CHECK(defaults.floor_share < 0.2);
  CHECK(legacy.ic_samples > 0);
  CHECK(std::isfinite(defaults.ic_mean));
  CHECK(defaults.sector_coherence >= 0.0);
  CHECK(defaults.sector_coherence <= 1.0);
  CHECK_THROWS_AS(evaluate(panel, std::vector<Security>{}, CoreParams{}, 30), std::invalid_argument);
}
```

- [ ] **Step 2: Run the build to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL with `pipeline/evaluation.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/pipeline/evaluation.hpp`:
```cpp
#pragma once
#include <span>
#include <string>
#include <vector>

#include "market/panel.hpp"
#include "market/universe.hpp"
#include "pipeline/core_pipeline.hpp"

namespace fx {

double gini(std::span<const double> x);
double spearman(std::span<const double> a, std::span<const double> b);
double floor_share(const Frame& f, double alpha);
double sector_coherence(const Frame& f, const std::vector<Security>& nodes);

struct EvalMetrics {
  double floor_share = 0, gini = 0, sector_coherence = 0;
  double ic_mean = 0, ic_t = 0;      // +1 forecast score vs next-bar return
  double ic_h_mean = 0, ic_h_t = 0;  // hotness h vs next-bar return
  std::size_t ic_samples = 0;
  double mean_frame_ms = 0;
};

// Spec 5.2.
EvalMetrics evaluate(const Panel& panel, const std::vector<Security>& nodes,
                     const CoreParams& params, std::size_t eval_bars);

struct EvalConfig {
  std::string name;
  CoreParams params;
};

std::vector<EvalConfig> evaluation_grid();

}  // namespace fx
```

`src/pipeline/evaluation.cpp`:
```cpp
#include "pipeline/evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace fx {

double gini(std::span<const double> x) {
  const std::size_t n = x.size();
  if (n == 0) return 0.0;
  std::vector<double> v(x.begin(), x.end());
  std::sort(v.begin(), v.end());
  double sum = 0, weighted = 0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += v[i];
    weighted += static_cast<double>(i + 1) * v[i];
  }
  if (!(sum > 0)) return 0.0;
  const double nd = static_cast<double>(n);
  return 2.0 * weighted / (nd * sum) - (nd + 1.0) / nd;
}

namespace {

std::vector<double> ranks(std::span<const double> x) {
  const std::size_t n = x.size();
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](auto a, auto b) { return x[a] < x[b]; });
  std::vector<double> r(n);
  for (std::size_t i = 0; i < n;) {
    std::size_t j = i;
    while (j + 1 < n && x[order[j + 1]] == x[order[i]]) ++j;
    const double avg = 0.5 * static_cast<double>(i + j) + 1.0;
    for (std::size_t k = i; k <= j; ++k) r[order[k]] = avg;
    i = j + 1;
  }
  return r;
}

void summarize(const std::vector<double>& v, double& mean, double& tstat) {
  mean = 0;
  tstat = 0;
  if (v.empty()) return;
  for (double x : v) mean += x;
  mean /= static_cast<double>(v.size());
  if (v.size() < 2) return;
  double ss = 0;
  for (double x : v) ss += (x - mean) * (x - mean);
  const double sd = std::sqrt(ss / static_cast<double>(v.size() - 1));
  if (sd > 0) tstat = mean / (sd / std::sqrt(static_cast<double>(v.size())));
}

bool known_sector(const std::string& s) { return !s.empty() && s != "Unclassified" && s != "Extra"; }

}  // namespace

double spearman(std::span<const double> a, std::span<const double> b) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (a.size() != b.size() || a.size() < 3) return nan;
  const auto ra = ranks(a), rb = ranks(b);
  const double n = static_cast<double>(a.size());
  double ma = 0, mb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    ma += ra[i];
    mb += rb[i];
  }
  ma /= n;
  mb /= n;
  double sab = 0, saa = 0, sbb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double da = ra[i] - ma, db = rb[i] - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  if (!(saa > 0 && sbb > 0)) return nan;
  return sab / std::sqrt(saa * sbb);
}

double floor_share(const Frame& f, double alpha) {
  std::size_t n_active = 0;
  for (bool a : f.active) n_active += a ? 1 : 0;
  if (n_active == 0) return 0.0;
  const double floor = (1.0 - alpha) / static_cast<double>(n_active);
  std::size_t at_floor = 0;
  for (std::size_t i = 0; i < f.active.size(); ++i)
    if (f.active[i] && f.pi[i] <= floor * (1.0 + 1e-6)) ++at_floor;
  return static_cast<double>(at_floor) / static_cast<double>(n_active);
}

double sector_coherence(const Frame& f, const std::vector<Security>& nodes) {
  double same = 0, total = 0;
  for (std::size_t i = 0; i < f.P.n; ++i) {
    if (!f.active[i] || !known_sector(nodes[i].sector)) continue;
    for (auto e = f.P.row_ptr[i]; e < f.P.row_ptr[i + 1]; ++e) {
      const std::size_t j = f.P.col[e];
      if (j == i || !f.active[j] || !known_sector(nodes[j].sector)) continue;
      total += f.P.raw[e];
      if (nodes[j].sector == nodes[i].sector) same += f.P.raw[e];
    }
  }
  return total > 0 ? same / total : 0.0;
}

EvalMetrics evaluate(const Panel& panel, const std::vector<Security>& nodes,
                     const CoreParams& params, std::size_t eval_bars) {
  const std::size_t T = panel.T(), N = panel.N();
  if (T < 3) throw std::runtime_error("evaluate: need at least three bars");
  if (nodes.size() != N) throw std::invalid_argument("evaluate: nodes/panel size mismatch");
  const std::size_t e0 = T > eval_bars + 1 ? T - eval_bars : 1;
  const std::size_t warmup = std::max<std::size_t>(
      params.corr_window, static_cast<std::size_t>(std::ceil(3.0 * params.halflife_slow)));
  const std::size_t s0 = e0 > warmup + 1 ? e0 - warmup : 1;

  CorePipeline pipe(N, params);
  EvalMetrics m;
  std::vector<double> ics, ics_h;
  std::size_t frames = 0;
  double ms = 0;
  for (std::size_t t = s0; t < T; ++t) {
    const Frame f = pipe.step(panel, t);
    if (t < e0) continue;
    ++frames;
    ms += f.compute_ms;
    m.floor_share += floor_share(f, params.alpha);
    std::vector<double> pis;
    for (std::size_t i = 0; i < N; ++i)
      if (f.active[i]) pis.push_back(f.pi[i]);
    m.gini += gini(pis);
    m.sector_coherence += sector_coherence(f, nodes);
    if (t + 1 < T) {
      std::vector<double> s, h, r;
      for (std::size_t i = 0; i < N; ++i) {
        const double c0 = panel.close[panel.idx(t, i)], c1 = panel.close[panel.idx(t + 1, i)];
        const double sc = f.forecasts.front().score[i];
        if (!f.active[i] || !std::isfinite(c0) || !std::isfinite(c1) || !(c0 > 0) ||
            !std::isfinite(sc) || !std::isfinite(f.h[i]))
          continue;
        s.push_back(sc);
        h.push_back(f.h[i]);
        r.push_back(c1 / c0 - 1.0);
      }
      const double ic = spearman(s, r), ich = spearman(h, r);
      if (std::isfinite(ic)) ics.push_back(ic);
      if (std::isfinite(ich)) ics_h.push_back(ich);
    }
  }
  if (frames > 0) {
    const double fr = static_cast<double>(frames);
    m.floor_share /= fr;
    m.gini /= fr;
    m.sector_coherence /= fr;
    m.mean_frame_ms = ms / fr;
  }
  summarize(ics, m.ic_mean, m.ic_t);
  summarize(ics_h, m.ic_h_mean, m.ic_h_t);
  m.ic_samples = ics.size();
  return m;
}

std::vector<EvalConfig> evaluation_grid() {
  const CoreParams L = CoreParams::legacy();
  std::vector<EvalConfig> g;
  g.push_back({"legacy", L});
  {
    CoreParams p = L;
    p.pressure = PressureMode::Relative;
    g.push_back({"+A relative", p});
  }
  {
    CoreParams p = L;
    p.transition.lift = LiftMode::Excess;
    g.push_back({"+B excess", p});
  }
  {
    CoreParams p = L;
    p.transition.k_in = 10;
    g.push_back({"+C k_in=10", p});
  }
  {
    CoreParams p = L;
    p.h_ref = HotRef::Size;
    g.push_back({"+D size", p});
  }
  {
    CoreParams p = L;
    p.h_ref = HotRef::LongRun;
    g.push_back({"+D longrun", p});
  }
  {
    CoreParams p = L;
    p.transition.retention = 1.0;
    g.push_back({"+E retention", p});
  }
  g.push_back({"defaults", CoreParams{}});
  {
    CoreParams p;
    p.h_ref = HotRef::LongRun;
    g.push_back({"defaults+longrun", p});
  }
  return g;
}

}  // namespace fx
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!`, no warnings. If the heavy-tail floor-share test fails, report the INFO values (legacy and defaults floor share) as BLOCKED rather than loosening it.

- [ ] **Step 5: Commit**
```bash
git add src/pipeline/evaluation.* tests/test_evaluation.cpp
git commit -m "feat: evaluation harness (floor share, Gini, sector coherence, IC) and A-E grid

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Alpaca rate limit, asset universe, snapshots and history back-fill

**Files:**
- Modify: `src/market/alpaca_client.hpp`, `src/market/alpaca_client.cpp`, `src/market/bar_store.hpp`, `src/market/bar_store.cpp`, `src/market/market_sync.hpp` (comment only), `src/market/market_sync.cpp` (rewrite), `src/market/universe.hpp`, `src/market/universe.cpp`, `.env.example`, `.gitignore`
- Create: `src/market/asset_universe.hpp`, `src/market/asset_universe.cpp`
- Test: `tests/test_alpaca.cpp` (edits plus new cases), `tests/test_asset_universe.cpp` (new)

**Interfaces:**
- Consumes: `AlpacaClient`/`FetchResult`/`HttpGet` (existing), `BarStore`, `Universe`, `read_csv_file`, `aggregate_session_hours`.
- Produces:
  - `AlpacaConfig` gains `int min_request_interval_ms = 334;` and `std::string trading_host = "paper-api.alpaca.markets";`. `alpaca_config_from_env` reads the optional `APCA_TRADING_HOST`.
  - `std::string AlpacaClient::get(const std::string& path)` returns the body of a successful GET, with retry and rate limit; it throws `std::runtime_error` like `fetch_bars` batches do. Every request, including those from `fetch_bars`, is spaced by at least `min_request_interval_ms`.
  - `std::optional<TimePoint> BarStore::first_time(const std::string&, Timeframe) const`
  - `sync_bars`:
    1. It first back-fills `[start, max first_time]` in one request group for tickers whose cached first bar is later than `start` plus a tolerance (Hour 4 d, Day 5 d, Week 14 d).
    2. It then fetches the incremental tail from each ticker's `last_time` (or from `start`).
    3. It saves every ticker that received bars and returns the stale tickers.
  - `std::vector<Fund> fx::load_funds(const std::filesystem::path&)`; `Universe::from_securities(std::vector<Security>, std::vector<Fund> funds = {})`
  - The `asset_universe.hpp` API is exactly as in Step 3.

- [ ] **Step 1: Write the failing tests**

In `tests/test_alpaca.cpp`:
- **(a)** In the `test_config()` helper, add `c.min_request_interval_ms = 0;`.
- **(b)** In `TEST_CASE("sync_bars fetches incrementally and saves")`, the second `sync_bars` call now first back-fills, because the cached first bar (2026-09-29) is more than 5 days after start (2026-09-01). Replace its final assertions `REQUIRE_FALSE(paths.empty()); CHECK(paths[0].find("start=2026-09-30T04:00:00Z") != std::string::npos);` with:
```cpp
  REQUIRE_FALSE(paths.empty());
  bool tail = false;
  for (const auto& p : paths) tail = tail || p.find("start=2026-09-30T04:00:00Z") != std::string::npos;
  CHECK(tail);
```
- **(c)** Append:
```cpp
TEST_CASE("sync_bars back-fills history before the first cached bar") {
  BarStore store(test::temp_dir("backfill"));
  store.merge("AAPL", Timeframe::Day, {{utc_seconds(2026, 9, 29, 4), 1, 1, 1, 1, 1, 1}});
  CHECK(store.first_time("AAPL", Timeframe::Day).value() == utc_seconds(2026, 9, 29, 4));
  std::vector<std::string> paths;
  AlpacaClient client(test_config(), [&](const std::string& path) {
    paths.push_back(path);
    return HttpResponse{200, R"({"bars":{}})"};
  });
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 9, 1), utc_seconds(2026, 10, 1));
  REQUIRE(paths.size() == 2);
  CHECK(paths[0].find("start=2026-09-01T00:00:00Z") != std::string::npos);
  CHECK(paths[0].find("end=2026-09-29T04:00:00Z") != std::string::npos);
  CHECK(paths[1].find("start=2026-09-29T04:00:00Z") != std::string::npos);

  paths.clear();
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 9, 27), utc_seconds(2026, 10, 1));
  CHECK(paths.size() == 1);  // first bar within tolerance of start: no back-fill
}

TEST_CASE("client spaces requests by min_request_interval_ms and get returns the body") {
  AlpacaConfig c = test_config();
  c.min_request_interval_ms = 40;
  int calls = 0;
  AlpacaClient client(c, [&](const std::string&) {
    ++calls;
    return HttpResponse{200, "[]"};
  });
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 3; ++i) CHECK(client.get("/v2/assets") == "[]");
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  CHECK(calls == 3);
  CHECK(ms >= 80);
  AlpacaClient bad(test_config(), [](const std::string&) { return HttpResponse{403, "no"}; });
  CHECK_THROWS_AS(bad.get("/v2/assets"), std::runtime_error);
}
```
Add `#include <chrono>` to the file if it is missing.

`tests/test_asset_universe.cpp`:
```cpp
#include <doctest/doctest.h>

#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "market/asset_universe.hpp"
#include "test_util.hpp"

using namespace fx;

TEST_CASE("parse_assets reads symbols, names, exchanges and tradable flags") {
  auto a = parse_assets(R"([
    {"symbol":"AAPL","name":"Apple Inc. Common Stock","exchange":"NASDAQ","tradable":true,"status":"active"},
    {"symbol":"XYZW","name":null,"exchange":"NYSE","tradable":false},
    {"symbol":"BRK.B","name":"Berkshire Hathaway Inc.","exchange":"NYSE","tradable":true},
    {"name":"no symbol"}])");
  REQUIRE(a.size() == 3);
  CHECK(a[0].symbol == "AAPL");
  CHECK(a[0].exchange == "NASDAQ");
  CHECK(a[0].tradable);
  CHECK(a[1].name.empty());
  CHECK_FALSE(a[1].tradable);
  CHECK_THROWS_AS(parse_assets("{}"), std::runtime_error);
  CHECK_THROWS_AS(parse_assets("not json"), std::runtime_error);
}

TEST_CASE("universe rules filter exchanges, warrants, units, rights and ETF-like names") {
  UniverseRules rules;
  auto ok = [&](std::string sym, std::string name, std::string exch, bool tradable = true) {
    return passes_universe_rules({sym, name, exch, tradable}, rules);
  };
  CHECK(ok("AAPL", "Apple Inc. Common Stock", "NASDAQ"));
  CHECK(ok("FRT", "Federal Realty Investment Trust", "NYSE"));
  CHECK(ok("UNIT", "Uniti Group Inc.", "NASDAQ"));
  CHECK_FALSE(ok("ACMW", "Acme Corp Warrant", "NASDAQ"));
  CHECK_FALSE(ok("ACMU", "Acme Acquisition Corp Units", "NASDAQ"));
  CHECK_FALSE(ok("ACMR", "Acme Acquisition Corp Rights", "NASDAQ"));
  CHECK_FALSE(ok("SPY", "SPDR S&P 500 ETF Trust", "ARCA"));
  CHECK_FALSE(ok("TQQQ", "ProShares UltraPro QQQ", "NASDAQ"));
  CHECK_FALSE(ok("ABCD", "Abcd Holdings", "OTC"));
  CHECK_FALSE(ok("AAPL", "Apple Inc.", "NASDAQ", false));
  rules.always_include = {"SPY"};
  CHECK(ok("SPY", "SPDR S&P 500 ETF Trust", "ARCA"));
  rules.exclude = {"AAPL"};
  CHECK_FALSE(ok("AAPL", "Apple Inc. Common Stock", "NASDAQ"));
}

TEST_CASE("rank_by_liquidity uses the median dollar volume of the last bars") {
  BarStore store(test::temp_dir("rank"));
  auto bar = [](TimePoint t, double v, double vw) { return Bar{t, 1, 1, 1, 1, v, vw}; };
  store.merge("BIG", Timeframe::Day, {bar(1, 1000, 10), bar(2, 10, 10), bar(3, 900, 10)});
  store.merge("MID", Timeframe::Day, {bar(1, 1, 10), bar(2, 300, 10), bar(3, 300, 10)});
  store.merge("SMALL", Timeframe::Day, {bar(3, 10, 10)});
  std::vector<AssetInfo> assets = {{"SMALL", "", "NYSE", true},
                                   {"MID", "", "NYSE", true},
                                   {"BIG", "", "NYSE", true},
                                   {"NONE", "", "NYSE", true}};
  auto ranked = rank_by_liquidity(assets, store, 2, 2);
  REQUIRE(ranked.size() == 2);
  CHECK(ranked[0].asset.symbol == "BIG");
  CHECK(ranked[0].median_dollar_volume == doctest::Approx(4550.0));
  CHECK(ranked[1].asset.symbol == "MID");
  CHECK(rank_by_liquidity(assets, store, 2, 10).size() == 3);
}

TEST_CASE("snapshots round-trip and the latest one is found") {
  auto dir = test::temp_dir("snap");
  Universe sp = Universe::from_securities({{"AAPL", "Apple Inc.", "Information Technology"}});
  std::vector<RankedAsset> ranked = {{{"AAPL", "Apple Inc.", "NASDAQ", true}, 5e9},
                                     {{"ZZZ", "Zeta, Inc.", "NYSE", true}, 1e6}};
  write_universe_snapshot(dir / "universe_2026-09-01.csv", ranked, sp);
  write_universe_snapshot(dir / "universe_2026-10-01.csv", ranked, sp);
  test::write_file(dir / "funds.csv", "ticker,tracks\nVOO,sp500\n");
  auto latest = latest_snapshot(dir);
  REQUIRE(latest.has_value());
  CHECK(latest->filename() == "universe_2026-10-01.csv");
  CHECK(snapshot_date(*latest).value() == "2026-10-01");
  CHECK_FALSE(snapshot_date(dir / "funds.csv").has_value());
  Universe u = load_snapshot(*latest, dir / "funds.csv");
  REQUIRE(u.nodes().size() == 2);
  CHECK(u.nodes()[0].sector == "Information Technology");
  CHECK(u.nodes()[1].name == "Zeta, Inc.");
  CHECK(u.nodes()[1].sector == "Unclassified");
  CHECK(u.is_fund("VOO"));
  CHECK_FALSE(latest_snapshot(dir / "missing").has_value());
  CHECK_FALSE(std::filesystem::exists(dir / "universe_2026-10-01.csv.tmp"));
}

TEST_CASE("read_ticker_list skips comments and blanks; a missing file is empty") {
  auto dir = test::temp_dir("tickers");
  auto p = test::write_file(dir / "x.csv", "# header\nAAPL\n\n  MSFT \r\n");
  CHECK(read_ticker_list(p) == std::set<std::string>{"AAPL", "MSFT"});
  CHECK(read_ticker_list(dir / "none.csv").empty());
}
```

- [ ] **Step 2: Run the build to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL. `market/asset_universe.hpp` is missing, and `first_time`, `get` and `min_request_interval_ms` are unknown.

- [ ] **Step 3: Implement**

**Alpaca client**, in `src/market/alpaca_client.hpp`:
- Add `#include <chrono>`.
- In `AlpacaConfig`, add:
```cpp
  int min_request_interval_ms = 334;  // <= 180 requests/minute
  std::string trading_host = "paper-api.alpaca.markets";
```
- In `class AlpacaClient` public, add `std::string get(const std::string& path);`.
- In private, add `void throttle();` and `std::optional<std::chrono::steady_clock::time_point> last_request_;`.

In `src/market/alpaca_client.cpp`:
- In `alpaca_config_from_env()`, before `return c;`, add:
```cpp
  if (const char* th = std::getenv("APCA_TRADING_HOST"); th && *th) c.trading_host = th;
```
- Add:
```cpp
void AlpacaClient::throttle() {
  if (config_.min_request_interval_ms <= 0) return;
  const auto gap = std::chrono::milliseconds(config_.min_request_interval_ms);
  const auto now = std::chrono::steady_clock::now();
  if (last_request_ && now - *last_request_ < gap) std::this_thread::sleep_for(gap - (now - *last_request_));
  last_request_ = std::chrono::steady_clock::now();
}

std::string AlpacaClient::get(const std::string& path) { return get_with_retry(path).body; }
```
- In `get_with_retry`, call `throttle();` as the first statement inside the `for` loop, before `HttpResponse res = get_(path);`.

**Bar store:** declare `std::optional<TimePoint> first_time(const std::string& ticker, Timeframe tf) const;` in `bar_store.hpp`, and add to `bar_store.cpp`:
```cpp
std::optional<TimePoint> BarStore::first_time(const std::string& ticker, Timeframe tf) const {
  const auto& b = bars(ticker, tf);
  if (b.empty()) return std::nullopt;
  return b.front().t;
}
```

**Sync:** replace `src/market/market_sync.cpp` with:
```cpp
#include "market/market_sync.hpp"

#include <algorithm>
#include <map>
#include <set>

#include "market/session_calendar.hpp"

namespace fx {
namespace {

TimePoint backfill_tolerance(Timeframe tf) {
  switch (tf) {
    case Timeframe::Hour: return 4 * 86400;
    case Timeframe::Day: return 5 * 86400;
    case Timeframe::Week: return 14 * 86400;
  }
  return 5 * 86400;
}

void fetch_into(AlpacaClient& client, BarStore& store, const std::vector<std::string>& group,
                Timeframe tf, TimePoint from, TimePoint to, std::set<std::string>& touched,
                std::set<std::string>& stale) {
  if (group.empty() || from >= to) return;
  FetchResult r = client.fetch_bars(group, alpaca_timeframe(tf), from, to);
  for (auto& [ticker, bars] : r.bars) {
    store.merge(ticker, tf, tf == Timeframe::Hour ? aggregate_session_hours(bars) : bars);
    touched.insert(ticker);
  }
  stale.insert(r.stale.begin(), r.stale.end());
}

}  // namespace

std::vector<std::string> sync_bars(AlpacaClient& client, BarStore& store,
                                   const std::vector<std::string>& tickers, Timeframe tf,
                                   TimePoint start, TimePoint end) {
  std::set<std::string> touched, stale;
  // 1. Back-fill history missing before the first cached bar (one group, up to the latest first bar).
  std::vector<std::string> backfill;
  TimePoint backfill_to = start;
  for (const auto& t : tickers) {
    const auto first = store.first_time(t, tf);
    if (first && *first > start + backfill_tolerance(tf)) {
      backfill.push_back(t);
      backfill_to = std::max(backfill_to, *first);
    }
  }
  fetch_into(client, store, backfill, tf, start, std::min(backfill_to, end), touched, stale);
  // 2. Incremental tail from each ticker's last cached bar (inclusive: refreshes a partial bar).
  std::map<TimePoint, std::vector<std::string>> by_start;
  for (const auto& t : tickers) by_start[store.last_time(t, tf).value_or(start)].push_back(t);
  for (const auto& [from, group] : by_start) fetch_into(client, store, group, tf, from, end, touched, stale);
  for (const auto& t : touched) store.save(t, tf);
  return {stale.begin(), stale.end()};
}

}  // namespace fx
```
In `market_sync.hpp`, update the comment above `sync_bars` to: `// Back-fills missing history, then fetches each ticker's tail; saves what succeeded; returns stale tickers.`

**Universe:**
- In `universe.hpp`, change the declaration to `static Universe from_securities(std::vector<Security> securities, std::vector<Fund> funds = {});`, and declare `std::vector<Fund> load_funds(const std::filesystem::path& funds_csv);` next to `load_portfolio`.
- In `universe.cpp`, add:
```cpp
std::vector<Fund> load_funds(const std::filesystem::path& funds_csv) {
  std::vector<Fund> funds;
  const CsvRows fu = read_csv_file(funds_csv);
  for (std::size_t r = 1; r < fu.size(); ++r)
    if (fu[r].size() >= 2) funds.push_back({fu[r][0], fu[r][1]});
  return funds;
}
```
- Replace `Universe::from_securities` with:
```cpp
Universe Universe::from_securities(std::vector<Security> securities, std::vector<Fund> funds) {
  Universe u;
  u.funds_ = std::move(funds);
  for (auto& s : securities) u.add_node(std::move(s));
  return u;
}
```
- In `Universe::load`, replace the funds-reading loop with `u.funds_ = load_funds(funds_csv);`.

**Asset universe**, `src/market/asset_universe.hpp`:
```cpp
#pragma once
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "market/bar_store.hpp"
#include "market/universe.hpp"

namespace fx {

struct AssetInfo {
  std::string symbol, name, exchange;
  bool tradable = false;
};

std::vector<AssetInfo> parse_assets(const std::string& json);  // throws std::runtime_error

struct UniverseRules {
  std::set<std::string> always_include;  // S&P 500, portfolio holdings, include.csv
  std::set<std::string> exclude;         // exclude.csv
};

// Spec 3: tradable; exclude list; always_include; exchange; warrant/unit/right/ETF-like names.
bool passes_universe_rules(const AssetInfo& a, const UniverseRules& rules);

struct RankedAsset {
  AssetInfo asset;
  double median_dollar_volume = 0;
};

// Median v*vwap over the last `window` Day bars; no bars = dropped; descending (ties by symbol).
std::vector<RankedAsset> rank_by_liquidity(const std::vector<AssetInfo>& assets,
                                           const BarStore& store, std::size_t window,
                                           std::size_t top_n);

// ticker,name,sector,exchange,median_dollar_volume (sector from sp500, else Unclassified).
void write_universe_snapshot(const std::filesystem::path& path,
                             const std::vector<RankedAsset>& ranked, const Universe& sp500);
std::optional<std::string> snapshot_date(const std::filesystem::path& path);  // "YYYY-MM-DD"
std::optional<std::filesystem::path> latest_snapshot(const std::filesystem::path& dir);
Universe load_snapshot(const std::filesystem::path& path, const std::filesystem::path& funds_csv);
std::set<std::string> read_ticker_list(const std::filesystem::path& path);

}  // namespace fx
```

`src/market/asset_universe.cpp`:
```cpp
#include "market/asset_universe.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <regex>
#include <stdexcept>

#include "core/csv.hpp"

namespace fx {

std::vector<AssetInfo> parse_assets(const std::string& json) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(json);
  } catch (const nlohmann::json::exception&) {
    throw std::runtime_error("assets response is not valid JSON");
  }
  if (!j.is_array()) throw std::runtime_error("assets response is not a JSON array");
  std::vector<AssetInfo> out;
  out.reserve(j.size());
  for (const auto& a : j) {
    if (!a.is_object() || !a.contains("symbol") || !a["symbol"].is_string()) continue;
    AssetInfo info;
    info.symbol = a["symbol"].get<std::string>();
    if (a.contains("name") && a["name"].is_string()) info.name = a["name"].get<std::string>();
    if (a.contains("exchange") && a["exchange"].is_string())
      info.exchange = a["exchange"].get<std::string>();
    info.tradable = a.contains("tradable") && a["tradable"].is_boolean() && a["tradable"].get<bool>();
    out.push_back(std::move(info));
  }
  return out;
}

bool passes_universe_rules(const AssetInfo& a, const UniverseRules& rules) {
  if (!a.tradable) return false;
  if (rules.exclude.count(a.symbol)) return false;
  if (rules.always_include.count(a.symbol)) return true;
  static const std::set<std::string> kExchanges = {"NYSE", "NASDAQ", "ARCA",
                                                   "NYSEARCA", "AMEX", "BATS"};
  if (!kExchanges.count(a.exchange)) return false;
  static const std::regex kExcluded(
      R"((warrant|\bunits?\b|\brights?\b|\betf\b|\betn\b|ishares|spdr|proshares|direxion|\bfund\b|\bindex\b))",
      std::regex::icase);
  return !std::regex_search(a.name, kExcluded);
}

std::vector<RankedAsset> rank_by_liquidity(const std::vector<AssetInfo>& assets,
                                           const BarStore& store, std::size_t window,
                                           std::size_t top_n) {
  std::vector<RankedAsset> ranked;
  std::vector<double> dv;
  for (const auto& a : assets) {
    const auto& bars = store.bars(a.symbol, Timeframe::Day);
    if (bars.empty()) continue;
    dv.clear();
    const std::size_t from = bars.size() > window ? bars.size() - window : 0;
    for (std::size_t k = from; k < bars.size(); ++k)
      dv.push_back(bars[k].v * (bars[k].vw > 0 ? bars[k].vw : bars[k].c));
    std::sort(dv.begin(), dv.end());
    const std::size_t m = dv.size();
    const double median = m % 2 ? dv[m / 2] : 0.5 * (dv[m / 2 - 1] + dv[m / 2]);
    ranked.push_back({a, median});
  }
  std::sort(ranked.begin(), ranked.end(), [](const RankedAsset& x, const RankedAsset& y) {
    return x.median_dollar_volume > y.median_dollar_volume ||
           (x.median_dollar_volume == y.median_dollar_volume && x.asset.symbol < y.asset.symbol);
  });
  if (ranked.size() > top_n) ranked.resize(top_n);
  return ranked;
}

namespace {

std::string csv_field(const std::string& s) {
  if (s.find_first_of(",\"\n") == std::string::npos) return s;
  std::string q = "\"";
  for (char ch : s) {
    if (ch == '"') q += '"';
    q += ch;
  }
  return q + "\"";
}

const std::regex& snapshot_name() {
  static const std::regex re(R"(universe_(\d{4}-\d{2}-\d{2})\.csv)");
  return re;
}

}  // namespace

void write_universe_snapshot(const std::filesystem::path& path,
                             const std::vector<RankedAsset>& ranked, const Universe& sp500) {
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  const std::string tmp = path.string() + ".tmp";
  {
    std::ofstream out(tmp);
    if (!out) throw std::runtime_error("cannot write " + tmp);
    out << "ticker,name,sector,exchange,median_dollar_volume\n" << std::setprecision(15);
    for (const auto& r : ranked) {
      const auto idx = sp500.index_of(r.asset.symbol);
      const std::string sector = idx ? sp500.nodes()[*idx].sector : "Unclassified";
      out << csv_field(r.asset.symbol) << ',' << csv_field(r.asset.name) << ','
          << csv_field(sector) << ',' << csv_field(r.asset.exchange) << ','
          << r.median_dollar_volume << '\n';
    }
    if (!out) throw std::runtime_error("write failed for " + tmp);
  }
  std::filesystem::rename(tmp, path);
}

std::optional<std::string> snapshot_date(const std::filesystem::path& path) {
  std::smatch m;
  const std::string name = path.filename().string();
  if (std::regex_match(name, m, snapshot_name())) return m[1].str();
  return std::nullopt;
}

std::optional<std::filesystem::path> latest_snapshot(const std::filesystem::path& dir) {
  if (!std::filesystem::is_directory(dir)) return std::nullopt;
  std::optional<std::filesystem::path> best;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file() || !snapshot_date(entry.path())) continue;
    if (!best || entry.path().filename() > best->filename()) best = entry.path();
  }
  return best;
}

Universe load_snapshot(const std::filesystem::path& path, const std::filesystem::path& funds_csv) {
  const CsvRows rows = read_csv_file(path);
  std::vector<Security> secs;
  for (std::size_t r = 1; r < rows.size(); ++r)
    if (rows[r].size() >= 3) secs.push_back({rows[r][0], rows[r][1], rows[r][2]});
  return Universe::from_securities(std::move(secs), load_funds(funds_csv));
}

std::set<std::string> read_ticker_list(const std::filesystem::path& path) {
  std::set<std::string> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    const auto last = line.find_last_not_of(" \t\r");
    out.insert(line.substr(first, last - first + 1));
  }
  return out;
}

}  // namespace fx
```

**Config files:**
- Append to `.env.example`:
```
# Trading API host used for the asset list (paper keys work with paper-api)
APCA_TRADING_HOST=paper-api.alpaca.markets
```
- Append to `.gitignore`: `data/universe/universe_*.csv`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!`, no warnings.

- [ ] **Step 5: Commit**
```bash
git add src/market tests/test_alpaca.cpp tests/test_asset_universe.cpp .env.example .gitignore
git commit -m "feat: Alpaca asset universe with liquidity ranking, snapshots, rate limit and back-fill

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: CLI arguments, universe flow and --eval

**Files:**
- Create: `src/cli/args.hpp`, `src/cli/args.cpp`
- Modify: `src/main.cpp` (rewrite)
- Test: `tests/test_cli_args.cpp`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `enum class fx::UniverseSource { Auto, Sp500, Snapshot }`
  - `struct fx::CliArgs` (fields as below), `CliArgs fx::parse_cli(const std::vector<std::string>&)` (throws `std::invalid_argument`; `--legacy` applies first regardless of position; calls `params.validate()`), `std::string fx::cli_usage()`, `std::string fx::describe(const CoreParams&)`
  - CLI flags:
    - Data and universe: `--mode synthetic|replay|alpaca`, `--timeframe 1h|1d|1w`, `--lookback-days N`, `--top N`, `--data DIR`, `--universe auto|sp500|snapshot`, `--universe-size N`, `--refresh-universe`.
    - Evaluation and threads: `--eval`, `--eval-bars N`, `--threads N`.
    - Model: `--legacy`, `--pressure dollar|sqrt|relative`, `--lift off|excess|ratio`, `--k-out N`, `--k-in N`, `--retention X`, `--h-ref uniform|size|longrun`, `--lambda X`.
    - Help: `--help`.

- [ ] **Step 1: Write the failing tests** in `tests/test_cli_args.cpp`:
```cpp
#include <doctest/doctest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "cli/args.hpp"

using namespace fx;

TEST_CASE("cli defaults") {
  CliArgs a = parse_cli({});
  CHECK(a.mode == "synthetic");
  CHECK(a.tf == Timeframe::Day);
  CHECK(a.lookback_days == 365);
  CHECK(a.universe == UniverseSource::Auto);
  CHECK(a.universe_size == 10000);
  CHECK_FALSE(a.eval);
  CHECK(a.threads == 0);
  CHECK(a.params.transition.lift == LiftMode::Excess);
  CHECK(describe(a.params).find("lift=excess") != std::string::npos);
}

TEST_CASE("--legacy applies first regardless of position") {
  CliArgs a = parse_cli({"--pressure", "sqrt", "--legacy", "--k-in", "5"});
  CHECK(a.params.pressure == PressureMode::Sqrt);
  CHECK(a.params.transition.lift == LiftMode::Off);
  CHECK(a.params.transition.k_in == 5);
  CHECK(a.params.transition.retention == 0.0);
}

TEST_CASE("model, universe and eval flags") {
  CliArgs a = parse_cli({"--mode", "replay", "--timeframe", "1h", "--lift", "ratio", "--h-ref",
                         "longrun", "--retention", "0.5", "--lambda", "0.3", "--k-out", "12",
                         "--universe", "snapshot", "--universe-size", "2000", "--refresh-universe",
                         "--eval", "--eval-bars", "60", "--threads", "4", "--top", "7"});
  CHECK(a.mode == "replay");
  CHECK(a.lookback_days == 60);
  CHECK(a.params.transition.lift == LiftMode::Ratio);
  CHECK(a.params.h_ref == HotRef::LongRun);
  CHECK(a.params.transition.retention == 0.5);
  CHECK(a.params.flux.lambda == 0.3);
  CHECK(a.params.transition.k_out == 12);
  CHECK(a.universe == UniverseSource::Snapshot);
  CHECK(a.universe_size == 2000);
  CHECK(a.refresh_universe);
  CHECK(a.eval);
  CHECK(a.eval_bars == 60);
  CHECK(a.threads == 4);
  CHECK(a.top == 7);
}

TEST_CASE("bad cli values throw") {
  CHECK_THROWS_AS(parse_cli({"--mode", "live"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--lift", "max"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--lambda", "2"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--k-out"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--top", "abc"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--universe", "world"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--bogus"}), std::invalid_argument);
  CHECK(parse_cli({"--help"}).help);
}
```

- [ ] **Step 2: Run the build to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL with `cli/args.hpp: No such file or directory`.

- [ ] **Step 3: Implement args**

`src/cli/args.hpp`:
```cpp
#pragma once
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "pipeline/core_pipeline.hpp"

namespace fx {

enum class UniverseSource { Auto, Sp500, Snapshot };

struct CliArgs {
  std::string mode = "synthetic";
  Timeframe tf = Timeframe::Day;
  int lookback_days = -1;  // resolved per timeframe: 1h 60, 1d 365, 1w 1825
  std::size_t top = 15;
  std::filesystem::path data = "data";
  UniverseSource universe = UniverseSource::Auto;
  std::size_t universe_size = 10000;
  bool refresh_universe = false;
  bool eval = false;
  std::size_t eval_bars = 120;
  int threads = 0;  // 0 = OpenMP default
  bool help = false;
  CoreParams params;
};

CliArgs parse_cli(const std::vector<std::string>& args);
std::string cli_usage();
std::string describe(const CoreParams& p);

}  // namespace fx
```

`src/cli/args.cpp`:
```cpp
#include "cli/args.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace fx {
namespace {

std::size_t to_size(const std::string& flag, const std::string& v) {
  std::size_t pos = 0;
  long long x = 0;
  try {
    x = std::stoll(v, &pos);
  } catch (const std::exception&) {
    throw std::invalid_argument(flag + " expects a non-negative integer, got '" + v + "'");
  }
  if (pos != v.size() || x < 0)
    throw std::invalid_argument(flag + " expects a non-negative integer, got '" + v + "'");
  return static_cast<std::size_t>(x);
}

double to_double(const std::string& flag, const std::string& v) {
  std::size_t pos = 0;
  double x = 0;
  try {
    x = std::stod(v, &pos);
  } catch (const std::exception&) {
    throw std::invalid_argument(flag + " expects a number, got '" + v + "'");
  }
  if (pos != v.size()) throw std::invalid_argument(flag + " expects a number, got '" + v + "'");
  return x;
}

}  // namespace

CliArgs parse_cli(const std::vector<std::string>& args) {
  CliArgs a;
  if (std::find(args.begin(), args.end(), "--legacy") != args.end()) a.params = CoreParams::legacy();
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& flag = args[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= args.size()) throw std::invalid_argument("missing value for " + flag);
      return args[++i];
    };
    if (flag == "--mode") a.mode = value();
    else if (flag == "--timeframe") a.tf = parse_timeframe(value());
    else if (flag == "--lookback-days") a.lookback_days = static_cast<int>(to_size(flag, value()));
    else if (flag == "--top") a.top = to_size(flag, value());
    else if (flag == "--data") a.data = value();
    else if (flag == "--universe") {
      const std::string v = value();
      if (v == "auto") a.universe = UniverseSource::Auto;
      else if (v == "sp500") a.universe = UniverseSource::Sp500;
      else if (v == "snapshot") a.universe = UniverseSource::Snapshot;
      else throw std::invalid_argument("--universe must be auto, sp500 or snapshot");
    } else if (flag == "--universe-size") a.universe_size = to_size(flag, value());
    else if (flag == "--refresh-universe") a.refresh_universe = true;
    else if (flag == "--eval") a.eval = true;
    else if (flag == "--eval-bars") a.eval_bars = to_size(flag, value());
    else if (flag == "--threads") a.threads = static_cast<int>(to_size(flag, value()));
    else if (flag == "--legacy") continue;
    else if (flag == "--pressure") a.params.pressure = parse_pressure_mode(value());
    else if (flag == "--lift") a.params.transition.lift = parse_lift_mode(value());
    else if (flag == "--k-out") a.params.transition.k_out = to_size(flag, value());
    else if (flag == "--k-in") a.params.transition.k_in = to_size(flag, value());
    else if (flag == "--retention") a.params.transition.retention = to_double(flag, value());
    else if (flag == "--h-ref") a.params.h_ref = parse_hot_ref(value());
    else if (flag == "--lambda") a.params.flux.lambda = to_double(flag, value());
    else if (flag == "--help" || flag == "-h") a.help = true;
    else throw std::invalid_argument("unknown flag " + flag);
  }
  if (a.mode != "synthetic" && a.mode != "replay" && a.mode != "alpaca")
    throw std::invalid_argument("--mode must be synthetic, replay or alpaca");
  if (a.lookback_days < 0)
    a.lookback_days = a.tf == Timeframe::Hour ? 60 : a.tf == Timeframe::Day ? 365 : 5 * 365;
  a.params.validate();
  return a;
}

std::string cli_usage() {
  return "usage: fluxscape [--mode synthetic|replay|alpaca] [--timeframe 1h|1d|1w]\n"
         "                 [--lookback-days N] [--top N] [--data DIR] [--threads N]\n"
         "                 [--universe auto|sp500|snapshot] [--universe-size N] [--refresh-universe]\n"
         "                 [--eval] [--eval-bars N]\n"
         "                 [--legacy] [--pressure dollar|sqrt|relative] [--lift off|excess|ratio]\n"
         "                 [--k-out N] [--k-in N] [--retention X] [--h-ref uniform|size|longrun]\n"
         "                 [--lambda X]\n";
}

std::string describe(const CoreParams& p) {
  std::ostringstream s;
  s << "pressure=" << to_string(p.pressure) << " lift=" << to_string(p.transition.lift)
    << " k_out=" << p.transition.k_out << " k_in=" << p.transition.k_in
    << " retention=" << p.transition.retention << " h_ref=" << to_string(p.h_ref)
    << " lambda=" << p.flux.lambda << " alpha=" << p.alpha;
  return s.str();
}

}  // namespace fx
```

- [ ] **Step 4: Rewrite main**

Replace `src/main.cpp` with:
```cpp
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/args.hpp"
#include "core/time.hpp"
#include "market/alpaca_client.hpp"
#include "market/asset_universe.hpp"
#include "market/market_sync.hpp"
#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "market/universe.hpp"
#include "pipeline/core_pipeline.hpp"
#include "pipeline/evaluation.hpp"

namespace {
namespace fs = std::filesystem;

fx::TimePoint now_utc() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string today_string() { return fx::format_rfc3339(now_utc()).substr(0, 10); }

std::int64_t days_since(const std::string& ymd) {
  const int y = std::stoi(ymd.substr(0, 4));
  const auto m = static_cast<unsigned>(std::stoi(ymd.substr(5, 2)));
  const auto d = static_cast<unsigned>(std::stoi(ymd.substr(8, 2)));
  return fx::floor_div(now_utc(), 86400) - fx::days_from_civil(y, m, d);
}

fx::Universe sp500_universe(const fs::path& data) {
  return fx::Universe::load(data / "universe" / "sp500.csv", data / "universe" / "funds.csv");
}

fs::path ensure_snapshot(const fx::CliArgs& args, const fx::AlpacaConfig& cfg,
                         fx::AlpacaClient& data_client, fx::BarStore& store,
                         const fx::PortfolioSpec& portfolio) {
  const fs::path dir = args.data / "universe";
  if (auto latest = fx::latest_snapshot(dir); latest && !args.refresh_universe) {
    if (auto date = fx::snapshot_date(*latest); date && days_since(*date) < 7) return *latest;
  }
  fx::AlpacaConfig trading_cfg = cfg;
  trading_cfg.host = cfg.trading_host;
  fx::AlpacaClient trading(trading_cfg);
  std::cerr << "fetching asset list from " << cfg.trading_host << "...\n";
  const auto assets =
      fx::parse_assets(trading.get("/v2/assets?status=active&asset_class=us_equity"));
  const fx::Universe sp = sp500_universe(args.data);
  fx::UniverseRules rules;
  for (const auto& t : sp.node_tickers()) rules.always_include.insert(t);
  for (const auto& h : portfolio.holdings)
    if (!sp.is_fund(h.ticker)) rules.always_include.insert(h.ticker);
  for (const auto& t : fx::read_ticker_list(dir / "include.csv")) rules.always_include.insert(t);
  rules.exclude = fx::read_ticker_list(dir / "exclude.csv");
  std::vector<fx::AssetInfo> candidates;
  for (const auto& a : assets)
    if (fx::passes_universe_rules(a, rules)) candidates.push_back(a);
  std::vector<std::string> symbols;
  for (const auto& a : candidates) symbols.push_back(a.symbol);
  store.load_all(symbols, fx::Timeframe::Day);
  const fx::TimePoint end = now_utc() - 16 * 60;
  std::cerr << assets.size() << " assets, " << candidates.size()
            << " candidates; fetching 40 days of daily bars to rank liquidity...\n";
  const auto stale = fx::sync_bars(data_client, store, symbols, fx::Timeframe::Day,
                                   end - 40 * 86400, end);
  if (!stale.empty()) std::cerr << stale.size() << " stale tickers during ranking\n";
  const auto ranked = fx::rank_by_liquidity(candidates, store, 20, args.universe_size);
  const fs::path path = dir / ("universe_" + today_string() + ".csv");
  fx::write_universe_snapshot(path, ranked, sp);
  std::cerr << "wrote " << ranked.size() << "-ticker universe snapshot " << path.string() << "\n";
  return path;
}

void print_row(std::size_t rank, const fx::Security& s, double h, double pi, double score) {
  std::printf("%5zu  %-7s %-24.24s %+10.4f  %.7f  %+9.4f\n", rank, s.ticker.c_str(),
              s.sector.c_str(), h, pi, score);
}

int run_eval(const fx::CliArgs& args, const fx::Panel& panel, const fx::Universe& universe) {
  std::printf("evaluation: last %zu of %zu bars, nodes=%zu, timeframe=%s, threads=%d\n\n",
              args.eval_bars, panel.T(), panel.N(), std::string(fx::to_string(args.tf)).c_str(),
              omp_get_max_threads());
  std::printf("%-18s %6s %6s %6s %9s %6s %9s %6s %9s\n", "config", "floor", "gini", "coher",
              "IC(score)", "t", "IC(h)", "t", "ms/frame");
  for (const auto& c : fx::evaluation_grid()) {
    const fx::EvalMetrics m = fx::evaluate(panel, universe.nodes(), c.params, args.eval_bars);
    std::printf("%-18s %6.3f %6.3f %6.3f %+9.4f %+6.2f %+9.4f %+6.2f %9.1f\n", c.name.c_str(),
                m.floor_share, m.gini, m.sector_coherence, m.ic_mean, m.ic_t, m.ic_h_mean,
                m.ic_h_t, m.mean_frame_ms);
    std::fflush(stdout);
  }
  return 0;
}

int run_rank(const fx::CliArgs& args, const fx::Panel& panel, const fx::Universe& universe,
             const std::optional<fx::PortfolioSpec>& portfolio) {
  const auto t0 = std::chrono::steady_clock::now();
  const fx::Frame f = fx::run_panel_last(panel, args.params);
  const double total_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("mode=%s timeframe=%s nodes=%zu bars=%zu last=%s threads=%d\n", args.mode.c_str(),
              std::string(fx::to_string(args.tf)).c_str(), panel.N(), panel.T(),
              fx::format_rfc3339(f.t).c_str(), omp_get_max_threads());
  std::printf("params: %s\n", fx::describe(args.params).c_str());
  std::printf("solver: %s in %d iterations (residual %.2e); last frame %.1f ms, run %.0f ms\n",
              f.solve.converged ? "converged" : "NOT converged", f.solve.iterations,
              f.solve.residual, f.compute_ms, total_ms);
  std::printf("edges: %zu slow, %zu fast; %.1f%% of active nodes at the teleport floor\n\n",
              f.P.col.size(), f.P_fast.col.size(), 100.0 * fx::floor_share(f, args.params.alpha));

  const auto& nodes = universe.nodes();
  std::vector<std::size_t> hills;
  for (std::size_t i = 0; i < panel.N(); ++i)
    if (f.active[i]) hills.push_back(i);
  const std::size_t inactive = panel.N() - hills.size();
  auto by_ticker = [&](std::size_t a, std::size_t b) { return nodes[a].ticker < nodes[b].ticker; };
  std::sort(hills.begin(), hills.end(), [&](auto a, auto b) {
    return f.h[a] > f.h[b] || (f.h[a] == f.h[b] && by_ticker(a, b));
  });
  std::vector<std::size_t> valleys(hills.rbegin(), hills.rend());
  std::stable_sort(valleys.begin(), valleys.end(), [&](auto a, auto b) { return f.h[a] < f.h[b]; });
  const auto& score = f.forecasts.front().score;
  const std::size_t top = std::min(args.top, hills.size());
  if (inactive > 0) std::printf("%zu inactive (no data)\n\n", inactive);
  std::printf("HILLS (money accumulating)            hotness         pi     score+%d\n",
              f.forecasts.front().k);
  for (std::size_t r = 0; r < top; ++r)
    print_row(r + 1, nodes[hills[r]], f.h[hills[r]], f.pi[hills[r]], score[hills[r]]);
  std::printf("\nVALLEYS (money draining)\n");
  for (std::size_t r = 0; r < top; ++r) {
    const std::size_t i = valleys[r];
    print_row(valleys.size() - r, nodes[i], f.h[i], f.pi[i], score[i]);
  }
  if (portfolio) {
    std::printf("\nPORTFOLIO HOLDINGS\n");
    for (const auto& hld : portfolio->holdings) {
      if (auto i = universe.index_of(hld.ticker); i && !f.active[*i]) {
        std::printf("  %-6s %5.1f%%  (no data)\n", hld.ticker.c_str(), hld.weight * 100);
      } else if (i) {
        std::printf("  %-6s %5.1f%%  hotness %+9.4f  score+%d %+9.4f\n", hld.ticker.c_str(),
                    hld.weight * 100, f.h[*i], f.forecasts.front().k, score[*i]);
      } else {
        std::printf("  %-6s %5.1f%%  (fund: look-through hotness arrives in milestone 3)\n",
                    hld.ticker.c_str(), hld.weight * 100);
      }
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const fx::CliArgs args = fx::parse_cli(std::vector<std::string>(argv + 1, argv + argc));
    if (args.help) {
      std::cout << fx::cli_usage();
      return 0;
    }
    if (args.threads > 0) omp_set_num_threads(args.threads);
    fx::BarStore store(args.data / "cache");
    fx::Universe universe;
    std::optional<fx::PortfolioSpec> portfolio;

    if (args.mode == "synthetic") {
      fx::SyntheticConfig cfg;
      cfg.tf = args.tf;
      universe = fx::Universe::from_securities(fx::generate_synthetic(cfg, store));
    } else {
      portfolio = fx::load_portfolio(args.data / "portfolio.json");
      const fs::path dir = args.data / "universe";
      std::optional<fx::AlpacaConfig> cfg;
      std::optional<fx::AlpacaClient> client;
      if (args.mode == "alpaca") {
        fx::load_dotenv(".env");
        cfg = fx::alpaca_config_from_env();
        if (!cfg) throw std::runtime_error("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set (.env)");
        client.emplace(*cfg);
      }
      std::optional<fs::path> snapshot;
      if (args.universe != fx::UniverseSource::Sp500) {
        snapshot = args.mode == "alpaca" ? ensure_snapshot(args, *cfg, *client, store, *portfolio)
                                         : fx::latest_snapshot(dir);
        if (!snapshot && args.universe == fx::UniverseSource::Snapshot)
          throw std::runtime_error("no universe snapshot; run --mode alpaca first");
      }
      universe = snapshot ? fx::load_snapshot(*snapshot, dir / "funds.csv")
                          : sp500_universe(args.data);
      universe.add_extras(*portfolio);
      store.load_all(universe.price_tickers(), args.tf);
      if (client) {
        const fx::TimePoint end = now_utc() - 16 * 60;
        const fx::TimePoint start = end - static_cast<fx::TimePoint>(args.lookback_days) * 86400;
        std::cerr << "syncing " << universe.price_tickers().size() << " tickers ("
                  << fx::to_string(args.tf) << ") from " << fx::format_rfc3339(start) << "...\n";
        const auto stale =
            fx::sync_bars(*client, store, universe.price_tickers(), args.tf, start, end);
        if (!stale.empty()) std::cerr << stale.size() << " stale tickers\n";
      }
    }

    const fx::Panel panel = fx::build_panel(store, universe.node_tickers(), args.tf);
    if (panel.T() < 2) throw std::runtime_error("not enough cached bars; run with --mode alpaca first");
    return args.eval ? run_eval(args, panel, universe) : run_rank(args, panel, universe, portfolio);
  } catch (const std::exception& e) {
    std::cerr << "fluxscape: " << e.what() << "\n";
    return 1;
  }
}
```

- [ ] **Step 5: Run the tests and the offline CLI checks**

Run: `cmake --build build -j && ./build/fluxtests`
Expected: `Status: SUCCESS!`, no warnings.

Run: `./build/fluxscape --mode synthetic --top 5`
Expected: exit 0, a `params:` line, `solver: converged`, and top hills mostly in `Sector1`.

Run: `./build/fluxscape --mode replay --universe sp500 --timeframe 1d --top 10`
Expected: exit 0 on the existing S&P daily cache. The "teleport floor" percentage should be far below milestone 1's 80% (406/504 nodes). Paste the full output into the report.

Run: `./build/fluxscape --mode replay --universe sp500 --timeframe 1d --eval --eval-bars 120`
Expected: a 9-row table. Paste it into the report verbatim.

- [ ] **Step 6: Commit**
```bash
git add src/cli src/main.cpp tests/test_cli_args.cpp
git commit -m "feat: CLI for A-E switches, Alpaca universe flow, threads and --eval

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Determinism, 10K benchmark, live 10K run and README

**Files:**
- Create: `tests/test_scale.cpp`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above. Produces nothing new for later tasks.

- [ ] **Step 1: Write the tests** in `tests/test_scale.cpp`:
```cpp
#include <doctest/doctest.h>
#include <omp.h>

#include <algorithm>
#include <string>
#include <vector>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/core_pipeline.hpp"
#include "test_util.hpp"

using namespace fx;

namespace {
Panel synthetic_panel(int sectors, int per_sector, int bars, const std::string& tag) {
  SyntheticConfig cfg;
  cfg.sectors = sectors;
  cfg.per_sector = per_sector;
  cfg.bars = bars;
  cfg.size_sigma = 1.5;
  cfg.rotation_start = bars / 2;
  BarStore store(test::temp_dir(tag));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  return build_panel(store, tickers, cfg.tf);
}
}  // namespace

TEST_CASE("results are bit-identical with 1 thread and with many threads") {
  const Panel panel = synthetic_panel(10, 30, 90, "determinism");
  for (const CoreParams& params : {CoreParams{}, CoreParams::legacy()}) {
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
    const Frame one = run_panel_last(panel, params);
    omp_set_num_threads(std::max(saved, 4));
    const Frame many = run_panel_last(panel, params);
    omp_set_num_threads(saved);
    CHECK(one.pi == many.pi);
    CHECK(one.P.col == many.P.col);
    CHECK(one.P.val == many.P.val);
    CHECK(one.forecasts.front().score == many.forecasts.front().score);
  }
}

TEST_CASE("bench: 10,000-node synthetic daily frame under 1 s" * doctest::skip()) {
  const Panel panel = synthetic_panel(50, 200, 40, "bench");
  CorePipeline pipe(panel.N(), CoreParams{});
  double worst = 0, sum = 0;
  int counted = 0;
  for (std::size_t t = 1; t < panel.T(); ++t) {
    const Frame f = pipe.step(panel, t);
    if (t >= 20) {
      worst = std::max(worst, f.compute_ms);
      sum += f.compute_ms;
      ++counted;
    }
  }
  MESSAGE("threads " << omp_get_max_threads() << ", mean frame ms " << sum / counted
                     << ", worst " << worst);
  CHECK(worst < 1000.0);
}
```
(`CHECK(one.pi == many.pi)` compares vectors exactly, which is intended: bit-identical.)

- [ ] **Step 2: Run them**

Run: `cmake --build build -j && ./build/fluxtests -tc="results are bit-identical*"`
Expected: PASS. If it fails, a parallel loop violates the determinism rule. Find it by bisecting with `omp_set_num_threads(1)` around each stage and fix the loop. Never weaken the test.

Run: `./build/fluxtests -tc="bench*" --no-skip`
Expected: PASS, with a MESSAGE showing the mean and worst frame time. If the worst frame takes ≥ 1000 ms, profile with `perf record -g ./build/fluxtests -tc="bench*" --no-skip` and `perf report --stdio | head -60`, then optimize the hottest stage without changing results (the determinism test must still pass). Report the before and after timings.

- [ ] **Step 3: Live 10K universe run**

These commands read market data from Alpaca using the existing `.env`. Never print the `.env` contents.

Run, with a timeout of 30 minutes (the first run builds the snapshot and syncs about 10,000 tickers at ≤ 180 requests per minute):
`./build/fluxscape --mode alpaca --universe-size 10000 --timeframe 1d --top 15 2>&1 | tail -70`
Expected: on stderr, the asset and candidate counts, "wrote N-ticker universe snapshot" with N close to 10000, then the sync. On stdout, `solver: converged`, the floor percentage, hills, valleys and portfolio holdings. Paste it into the report.

Run: `./build/fluxscape --mode replay --timeframe 1d --eval --eval-bars 60 2>&1 | tail -15`
This uses the 10K snapshot automatically. Expected: the 9-row table, with ms/frame for N ≈ 10,000. Paste it into the report.

If the live run fails (network, auth or rate limit), paste the error. That counts as DONE_WITH_CONCERNS, not BLOCKED.

- [ ] **Step 4: Update the README** by replacing the `## Run` section of `README.md` with:
````markdown
## Run

```bash
./build/fluxscape --mode synthetic                          # no keys needed
cp .env.example .env                                        # add Alpaca keys
./build/fluxscape --mode alpaca --universe-size 10000       # build/reuse 10K liquidity snapshot, sync, solve
./build/fluxscape --mode replay                             # cached data, latest snapshot (or --universe sp500)
./build/fluxscape --mode replay --eval --eval-bars 120      # compare A-E settings (floor, Gini, coherence, IC)
./build/fluxscape --mode replay --legacy                    # milestone-1 behaviour
./build/fluxscape --help                                    # all switches (--pressure, --lift, --k-in, ...)
python3 scripts/fetch_sp500.py                              # refresh the S&P 500 list
```

`--threads N` sets OpenMP threads; results are bit-identical for any thread count.
`data/universe/include.csv` / `exclude.csv` (one ticker per line) override the universe filters.

Advisory and experimental. The flux is inferred from price and volume co-movement,
not observed order flow.

## How it works

1. **Universe.** Start from every tradable US stock on Alpaca. Drop warrants, units, rights and
   ETF-like products. Rank what's left by median daily dollar volume and keep the top N (default
   10,000). Your portfolio holdings are always included.
2. **Buying and selling pressure.** For each bar, each stock's pressure is its return times how
   unusual its volume is compared with its own normal (`r · V / ADV`). Positive pressure means
   net buying (a sink); negative means net selling (a source). Comparing a stock's volume with its
   own normal stops mega-caps from dominating just because they're big.
3. **Money flux, with no fitted model.** Each source's outflow is split across the sinks in
   proportion to their pressure, and tilted toward stocks it moves with (rolling return
   correlation). The result is a directed, weighted graph: edge i→j is the estimated money moving
   from selling i into buying j. Only the strongest edges are stored, so memory grows with N, not N².
4. **Memory of the flow.** Edges and totals build up with exponential decay: a slow memory
   (half-life 20 bars) for the equilibrium and a fast memory (3 bars) for the newest flow.
5. **Keep the real structure.** Subtract the flow you'd expect from size alone ("big buyers meet
   big sellers"). Keep each stock's strongest outgoing and incoming edges. Let net buyers retain
   part of what flows in, so money collects where it's being bought.
6. **Markov chain → steady state.** Normalize each stock's edges into transition probabilities
   and solve for the stationary distribution π, the PageRank-style steady state. π is where the
   market's money settles if the current flow pattern continues.
7. **Hotness.** Hotness `h = π / reference − 1`: hills (h > 0) are where money accumulates and
   valleys (h < 0) are where it drains. The reference can be uniform, size, or each stock's own
   long-run normal.
8. **Forecast.** Push the steady state one, four and eight bars forward through the fast flow,
   and add its recent drift. The result is a score for where the money is heading next.
9. **Evaluate before trusting.** `--eval` compares every setting on historical data. It reports
   how much of the graph is dead, how concentrated π is, whether flows cluster by sector, and
   whether the scores rank next-bar returns (information coefficient).
10. **Next milestones.** The steady state becomes a 3D landscape: stocks are laid out by flux and
    snapped to a grid, with inverse-distance-weighted terrain drawn in deck.gl. Then the portfolio
    optimizer moves your holdings "uphill", and the hourly, daily and weekly suggestions are
    paper-traded so their real profit and loss is tracked.
````

- [ ] **Step 5: Run the full suite and commit**

Run: `./build/fluxtests`
Expected: `Status: SUCCESS!` (the benchmark is skipped by default).
```bash
git add tests/test_scale.cpp README.md
git commit -m "test: thread-count determinism and 10K benchmark; README for A-E and 10K universe

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
