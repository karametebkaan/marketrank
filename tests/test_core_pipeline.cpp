#include <doctest/doctest.h>

#include <map>

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
