#include <doctest/doctest.h>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "test_util.hpp"

using namespace mr;

TEST_CASE("synthetic market is deterministic and well formed") {
  SyntheticConfig cfg;
  BarStore a(test::temp_dir("syn_a")), b(test::temp_dir("syn_b"));
  auto sa = generate_synthetic(cfg, a);
  auto sb = generate_synthetic(cfg, b);
  REQUIRE(sa.size() == 50);
  CHECK(sa[13].ticker == "S1_03");
  CHECK(sa[13].sector == "Sector1");
  const auto& ba = a.bars("S1_03", cfg.tf);
  REQUIRE(ba.size() == 300);
  CHECK(ba[299].c == b.bars("S1_03", cfg.tf)[299].c);
  CHECK(ba[1].t - ba[0].t == 86400);
  for (const Bar& bar : ba) {
    CHECK(bar.h >= std::max(bar.o, bar.c));
    CHECK(bar.l <= std::min(bar.o, bar.c));
    CHECK(bar.v > 0);
  }
}

TEST_CASE("planted rotation moves sector returns apart") {
  SyntheticConfig cfg;
  BarStore s(test::temp_dir("syn_rot"));
  auto secs = generate_synthetic(cfg, s);
  std::vector<std::string> tickers;
  for (auto& x : secs) tickers.push_back(x.ticker);
  Panel p = build_panel(s, tickers, cfg.tf);
  double from_sum = 0, to_sum = 0;
  for (std::size_t t = cfg.rotation_start + 1; t < p.T(); ++t) {
    for (int k = 0; k < cfg.per_sector; ++k) {
      auto i0 = static_cast<std::size_t>(cfg.rotation_from * cfg.per_sector + k);
      auto i1 = static_cast<std::size_t>(cfg.rotation_to * cfg.per_sector + k);
      from_sum += p.close[p.idx(t, i0)] / p.close[p.idx(t - 1, i0)] - 1;
      to_sum += p.close[p.idx(t, i1)] / p.close[p.idx(t - 1, i1)] - 1;
    }
  }
  CHECK(to_sum > 0);
  CHECK(from_sum < 0);
}

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
