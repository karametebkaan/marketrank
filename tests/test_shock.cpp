#include <doctest/doctest.h>

#include <cmath>
#include <algorithm>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/shock.hpp"
#include "test_util.hpp"

using namespace fx;

namespace {
struct Market {
  std::vector<Security> secs;
  Panel panel;
};

Market make_market() {
  SyntheticConfig cfg;
  BarStore store(test::temp_dir("shock"));
  Market m;
  m.secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : m.secs) tickers.push_back(s.ticker);
  m.panel = build_panel(store, tickers, cfg.tf);
  return m;
}

std::size_t first_in(const Market& m, const std::string& sector) {
  for (std::size_t i = 0; i < m.secs.size(); ++i)
    if (m.secs[i].sector == sector) return i;
  FAIL("sector not found");
  return 0;
}

CoreParams params() {
  CoreParams p;
  p.min_dollar_volume = 0;
  return p;
}

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }
}  // namespace

TEST_CASE("no shocks: base and shocked frames are identical") {
  const Market m = make_market();
  const auto [base, shocked] = run_with_shock(m.panel, params(), {});
  REQUIRE(base.pi.size() == shocked.pi.size());
  for (std::size_t i = 0; i < base.pi.size(); ++i) CHECK(same_bits(base.pi[i], shocked.pi[i]));
  CHECK(test::same_values(base.h, shocked.h));
  REQUIRE(base.forecasts.size() == shocked.forecasts.size());
  for (std::size_t k = 0; k < base.forecasts.size(); ++k) {
    INFO("forecast " << k);
    CHECK(test::same_values(base.forecasts[k].score, shocked.forecasts[k].score));
  }
  const ShockDelta d = shock_response(base, shocked);
  CHECK(d.l1_dpi == 0);
}

TEST_CASE("a sell-off lowers the shocked node's hotness") {
  const Market m = make_market();
  const std::size_t i = first_in(m, "Sector1");
  const auto [base, shocked] = run_with_shock(m.panel, params(), {{i, -10.0}});
  const ShockDelta d = shock_response(base, shocked);
  CHECK(d.dh[i] < 0);
  CHECK(d.l1_dpi > 0);
}

TEST_CASE("a buying surge raises the shocked node's hotness") {
  const Market m = make_market();
  const std::size_t i = first_in(m, "Sector0");
  const auto [base, shocked] = run_with_shock(m.panel, params(), {{i, 10.0}});
  CHECK(shock_response(base, shocked).dh[i] > 0);
}

TEST_CASE("shock runs are deterministic") {
  const Market m = make_market();
  const std::vector<Shock> sh{{first_in(m, "Sector1"), -10.0}};
  const ShockDelta a = [&] { auto [b, s] = run_with_shock(m.panel, params(), sh); return shock_response(b, s); }();
  const ShockDelta b = [&] { auto [x, s] = run_with_shock(m.panel, params(), sh); return shock_response(x, s); }();
  REQUIRE(a.dh.size() == b.dh.size());
  for (std::size_t i = 0; i < a.dh.size(); ++i) {
    CHECK(same_bits(a.dh[i], b.dh[i]));
    CHECK(same_bits(a.dpi[i], b.dpi[i]));
  }
  CHECK(same_bits(a.l1_dpi, b.l1_dpi));
}

TEST_CASE("an unknown or inactive node throws") {
  const Market m = make_market();
  CHECK_THROWS_AS(run_with_shock(m.panel, params(), {{m.panel.N() + 5, -1.0}}), std::invalid_argument);
  CoreParams strict = params();
  strict.min_dollar_volume = 1e18;  // deactivates every node
  CHECK_THROWS(run_with_shock(m.panel, strict, {{0, -1.0}}));
}

namespace {
// n nodes, T bars of noisy returns; node 0 trades 100x the dollar volume and falls on the last bar.
Panel heavy_panel(std::size_t n = 12, std::size_t T = 40) {
  Panel p;
  p.tickers.resize(n);
  for (std::size_t i = 0; i < n; ++i) p.tickers[i] = "T" + std::to_string(i);
  std::mt19937 rng(7);
  std::normal_distribution<double> nd(0.0, 0.01);
  std::vector<double> px(n, 100.0);
  for (std::size_t t = 0; t < T; ++t) {
    p.times.push_back(static_cast<TimePoint>(t) * 86400);
    for (std::size_t i = 0; i < n; ++i) {
      double r = nd(rng);
      if (i == 0 && t + 1 == T) r = -0.03;
      px[i] *= 1.0 + r;
      p.open.push_back(px[i]);
      p.high.push_back(px[i]);
      p.low.push_back(px[i]);
      p.close.push_back(px[i]);
      p.volume.push_back(i == 0 ? 1e6 : 1e4);
      p.vwap.push_back(px[i]);
    }
  }
  return p;
}
}  // namespace

TEST_CASE("a heavy seller gets more selling from a sell shock") {
  const Panel panel = heavy_panel();
  const auto [base, shocked] = run_with_shock(panel, params(), {{0, -10.0}});
  const ShockDelta d = shock_response(base, shocked);
  CHECK(d.dh[0] < 0);
  CHECK(shocked.inflow[0] <= base.inflow[0]);
  CHECK(d.l1_dpi > 0);
}

TEST_CASE("a buy shock raises the node's hotness") {
  const Panel panel = heavy_panel();
  for (std::size_t i : {std::size_t{0}, std::size_t{5}}) {
    const auto [base, shocked] = run_with_shock(panel, params(), {{i, 10.0}});
    CHECK(shock_response(base, shocked).dh[i] > 0);
  }
}

TEST_CASE("duplicate shocks add their sizes") {
  const Panel panel = heavy_panel();
  const auto [b1, s1] = run_with_shock(panel, params(), {{3, -5.0}, {3, -5.0}});
  const auto [b2, s2] = run_with_shock(panel, params(), {{3, -10.0}});
  const ShockDelta d1 = shock_response(b1, s1), d2 = shock_response(b2, s2);
  for (std::size_t i = 0; i < d1.dh.size(); ++i) {
    CHECK(same_bits(d1.dh[i], d2.dh[i]));
    CHECK(same_bits(d1.dpi[i], d2.dpi[i]));
  }
}

TEST_CASE("a vol-scaled shock on a node without return history throws") {
  CoreParams p = params();
  p.vol_scale = true;
  const Panel young = heavy_panel(12, 5);  // only 3 previous returns at the last bar
  try {
    run_with_shock(young, p, {{0, -10.0}});
    FAIL("expected invalid_argument");
  } catch (const std::invalid_argument& e) {
    CHECK(std::string(e.what()).find("has no return history yet") != std::string::npos);
  }
  const Panel grown = heavy_panel();
  const auto [base, shocked] = run_with_shock(grown, p, {{0, -10.0}});
  CHECK(shock_response(base, shocked).l1_dpi > 0);
}

TEST_CASE("only the node below the liquidity floor is rejected as a shock target") {
  Panel panel = heavy_panel();
  for (std::size_t t = 0; t < panel.T(); ++t) panel.volume[panel.idx(t, 3)] = 10;  // ~$1e3 a bar
  CoreParams p = params();
  p.min_dollar_volume = 1e5;  // others trade >= ~$1e6 a bar
  const auto [base, same] = run_with_shock(panel, p, {});
  for (std::size_t i = 0; i < panel.N(); ++i) CHECK(base.active[i] == (i != 3));
  CHECK_THROWS_AS(run_with_shock(panel, p, {{3, -10.0}}), std::invalid_argument);
  const auto [b5, s5] = run_with_shock(panel, p, {{5, -10.0}});
  CHECK(shock_response(b5, s5).l1_dpi > 0);
}

TEST_CASE("a SIZE 10 shock adds exactly 0.10 x vol_term to the node's pressure in each mode") {
  const Panel panel = heavy_panel(6, 10);
  const std::size_t last = panel.T() - 1;
  // Trailing median dollar volume of node 0 after the last step: bars 1..last (9 < adv_window).
  std::vector<double> dv;
  for (std::size_t t = 1; t <= last; ++t) dv.push_back(panel.volume[panel.idx(t, 0)] * panel.vwap[panel.idx(t, 0)]);
  std::sort(dv.begin(), dv.end());
  const double mdv = dv[dv.size() / 2];
  for (PressureMode mode : {PressureMode::Dollar, PressureMode::Sqrt, PressureMode::Relative}) {
    INFO("mode " << to_string(mode));
    CoreParams p = params();
    p.pressure = mode;
    CorePipeline pipe(panel.N(), p);
    for (std::size_t t = 1; t < last; ++t) pipe.step(panel, t);
    CorePipeline copy = pipe;
    pipe.step(panel, last);
    copy.step(panel, last, {{0, 10.0}});
    const double vol_term = mode == PressureMode::Dollar ? mdv : mode == PressureMode::Sqrt ? std::sqrt(mdv) : 1.0;
    const double dp = copy.last_pressure()[0] - pipe.last_pressure()[0];
    CHECK(dp == doctest::Approx(0.10 * vol_term).epsilon(1e-12));
    for (std::size_t i = 1; i < panel.N(); ++i) CHECK(same_bits(copy.last_pressure()[i], pipe.last_pressure()[i]));
  }
}
