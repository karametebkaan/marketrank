#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <string>
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

TEST_CASE("spearman drops non-finite pairs before ranking") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  CHECK(spearman(std::vector<double>{1, nan, 2, 3, 4, 5},
                 std::vector<double>{10, 15, 20, inf, 40, 50}) == doctest::Approx(1.0));
  CHECK(spearman(std::vector<double>{4, 3, nan, 2, 1}, std::vector<double>{1, 2, 3, 3, 4}) ==
        doctest::Approx(-1.0));
  CHECK(std::isnan(spearman(std::vector<double>{1, 2, nan, 3}, std::vector<double>{1, 2, 3, nan})));
}

namespace {
Panel eval_panel(std::vector<Security>& secs) {
  SyntheticConfig cfg;
  cfg.bars = 70;
  cfg.rotation_start = 30;
  BarStore store(test::temp_dir("evalrobust"));
  secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  return build_panel(store, tickers, cfg.tf);
}
}  // namespace

TEST_CASE("eval_bars beyond the panel evaluates every bar") {
  std::vector<Security> secs;
  const Panel panel = eval_panel(secs);
  const EvalMetrics all = evaluate(panel, secs, CoreParams{}, panel.T());
  const EvalMetrics huge =
      evaluate(panel, secs, CoreParams{}, std::numeric_limits<std::size_t>::max());
  CHECK(all.ic_samples > 0);
  CHECK(huge.ic_samples == all.ic_samples);
  CHECK(huge.ic_mean == all.ic_mean);
  CHECK(huge.floor_share == all.floor_share);
}

TEST_CASE("the IC uses the k = 1 forecast wherever it sits in the horizons") {
  std::vector<Security> secs;
  const Panel panel = eval_panel(secs);
  CoreParams one, four_one, four;
  one.horizons = {1};
  four_one.horizons = {4, 1};
  four.horizons = {4};
  const EvalMetrics a = evaluate(panel, secs, one, 30);
  const EvalMetrics b = evaluate(panel, secs, four_one, 30);
  const EvalMetrics c = evaluate(panel, secs, four, 30);
  REQUIRE(a.ic_samples > 0);
  CHECK(b.ic_mean == a.ic_mean);
  CHECK(b.ic_t == a.ic_t);
  CHECK(c.ic_mean != a.ic_mean);  // no k = 1: falls back to the first horizon
}

TEST_CASE("an infinite slow half-life does not break the warm-up") {
  std::vector<Security> secs;
  const Panel panel = eval_panel(secs);
  CoreParams p;
  p.halflife_slow = std::numeric_limits<double>::infinity();
  const EvalMetrics m = evaluate(panel, secs, p, 30);
  CHECK(m.mean_frame_ms >= 0);
}
