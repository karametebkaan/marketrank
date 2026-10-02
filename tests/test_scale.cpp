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
