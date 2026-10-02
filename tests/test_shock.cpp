#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
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
