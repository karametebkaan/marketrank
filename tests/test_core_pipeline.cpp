#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/core_pipeline.hpp"
#include "test_util.hpp"

using namespace fx;

TEST_CASE("planted rotation makes the receiving sector the top hill") {
  SyntheticConfig cfg;  // 5 sectors x 10, rotation Sector0 -> Sector1 from bar 150
  BarStore store(test::temp_dir("pipeline"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  Panel panel = build_panel(store, tickers, cfg.tf);

  Frame f = run_panel_last(panel, CoreParams{});
  CHECK(f.solve.converged);
  REQUIRE(f.h.size() == 50);
  REQUIRE(f.forecasts.size() == 3);
  CHECK(f.forecasts[2].k == 8);
  CHECK(f.t == panel.times.back());

  std::map<std::string, double> sector_mean;
  for (std::size_t i = 0; i < secs.size(); ++i) sector_mean[secs[i].sector] += f.h[i] / 10.0;
  for (const auto& [sector, mean] : sector_mean) {
    if (sector != "Sector1") CHECK(sector_mean["Sector1"] > mean);
  }
  CHECK(sector_mean["Sector1"] > 0);
  CHECK(sector_mean["Sector0"] < sector_mean["Sector1"]);
}

TEST_CASE("pipeline requires at least two bars") {
  Panel p;
  p.times = {100};
  p.tickers = {"A"};
  p.close = {1};
  p.volume = {1};
  p.vwap = {1};
  CHECK_THROWS_AS(run_panel_last(p, CoreParams{}), std::runtime_error);
}

namespace {
Panel small_panel(bool dead_node) {
  Panel p;
  const std::size_t T = 6, N = 4;
  p.tickers = {"A", "B", "DEAD", "C"};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t t = 0; t < T; ++t) {
    p.times.push_back(static_cast<TimePoint>(100 + t));
    for (std::size_t i = 0; i < N; ++i) {
      const bool dead = dead_node && i == 2;
      const double c = 10.0 + static_cast<double>(i) + 0.5 * static_cast<double>(t) *
                                                           (i % 2 == 0 ? 1.0 : -0.3);
      p.close.push_back(dead ? nan : c);
      p.volume.push_back(dead ? nan : 1000.0 * (1.0 + static_cast<double>(i)));
      p.vwap.push_back(dead ? nan : c);
    }
  }
  return p;
}
}  // namespace

TEST_CASE("a ticker with no bars is inactive and takes no part in the solve") {
  Panel p = small_panel(true);
  Frame f = run_panel_last(p, CoreParams{});
  REQUIRE(f.active.size() == 4);
  CHECK(f.active == std::vector<bool>{true, true, false, true});
  CHECK(f.pi[2] == 0.0);
  CHECK(std::isnan(f.h[2]));
  for (const auto& fc : f.forecasts) CHECK(std::isnan(fc.score[2]));
  double sum = 0;
  for (std::size_t i = 0; i < 4; ++i)
    if (i != 2) sum += f.pi[i];
  CHECK(sum == doctest::Approx(1.0));
  double hsum = 0;
  for (std::size_t i = 0; i < 4; ++i)
    if (i != 2) hsum += f.h[i];
  CHECK(hsum == doctest::Approx(0.0).epsilon(1e-9));  // h = N_active * pi - 1
  // dead node: self-loop only, nothing flows into it
  CHECK(f.P.row_ptr[3] - f.P.row_ptr[2] == 1);
  CHECK(f.P.raw[f.P.row_ptr[2]] == 0.0);
  for (std::size_t e = 0; e < f.P.col.size(); ++e)
    if (f.P.col[e] == 2) CHECK(e == f.P.row_ptr[2]);
}

TEST_CASE("frame carries raw pruned weights and the fast matrix") {
  Panel p = small_panel(false);
  Frame f = run_panel_last(p, CoreParams{});
  REQUIRE(f.P.raw.size() == f.P.val.size());
  CHECK(f.P_fast.n == 4);
  CHECK(f.P_fast.raw.size() == f.P_fast.val.size());
  CHECK(f.active == std::vector<bool>(4, true));
  for (std::size_t i = 0; i < 4; ++i) {
    double s = 0;
    for (auto e = f.P.row_ptr[i]; e < f.P.row_ptr[i + 1]; ++e) s += f.P.raw[e];
    if (s > 0)
      for (auto e = f.P.row_ptr[i]; e < f.P.row_ptr[i + 1]; ++e)
        CHECK(f.P.val[e] == doctest::Approx(f.P.raw[e] / s));
  }
}

TEST_CASE("step rejects out-of-range t and mismatched panels") {
  Panel p = small_panel(false);
  CorePipeline pipe(4, CoreParams{});
  CHECK_THROWS_AS(pipe.step(p, 0), std::invalid_argument);
  CHECK_THROWS_AS(pipe.step(p, p.T()), std::invalid_argument);
  CorePipeline wrong(5, CoreParams{});
  CHECK_THROWS_AS(wrong.step(p, 1), std::invalid_argument);
  CHECK_NOTHROW(pipe.step(p, 1));
}
