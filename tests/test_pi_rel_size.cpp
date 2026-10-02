// Landscape height pi relative to size, the floor tie (CLI, territory medians), and the rank report.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "cli/args.hpp"
#include "cli/rank_report.hpp"
#include "geom/landscape.hpp"
#include "geom/territory.hpp"
#include "graph/hotness.hpp"
#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/core_pipeline.hpp"
#include "test_util.hpp"

using namespace mr;

namespace {
const double kNaN = std::numeric_limits<double>::quiet_NaN();

// Four active nodes; pi {0.4, 0.4, 0.1, 0.1}; trailing median dollar volume {4, 2, 0, 4}, so the size
// shares over the valid nodes are {0.4, 0.2, NaN, 0.4}.
Frame hand_frame() {
  Frame f;
  const std::size_t n = 4;
  f.active.assign(n, true);
  f.pi = {0.4, 0.4, 0.1, 0.1};
  f.h = {0.6, 0.6, -0.6, -0.6};
  f.size_ref = {4, 2, 0, 4};
  f.pulse.assign(n, 0.0);
  f.P.n = n;
  f.P.row_ptr = {0, 1, 2, 3, 4};
  f.P.col = {0, 1, 2, 3};
  f.P.val = {1, 1, 1, 1};
  f.P.raw = {0, 0, 0, 0};
  return f;
}
}  // namespace

TEST_CASE("size shares: ref / sum over valid refs, NaN for zero or non-finite") {
  const auto s = size_shares(std::vector<double>{4, 2, 0, 4, kNaN, -1});
  REQUIRE(s.size() == 6);
  CHECK(s[0] == doctest::Approx(0.4));
  CHECK(s[1] == doctest::Approx(0.2));
  CHECK(std::isnan(s[2]));
  CHECK(s[3] == doctest::Approx(0.4));
  CHECK(std::isnan(s[4]));
  CHECK(std::isnan(s[5]));
  // The same notion of size as HotRef::Size: relative_hotness(pi, ref) = pi / share - 1 where the share is valid.
  const std::vector<double> pi = {0.4, 0.4, 0.1, 0.1};
  const auto h = relative_hotness(pi, std::vector<double>{4, 2, 0, 4});
  CHECK(h[0] == doctest::Approx(pi[0] / s[0] - 1));
  CHECK(h[1] == doctest::Approx(pi[1] / s[1] - 1));
}

TEST_CASE("landscape value pi_rel_size: log(pi / size share)") {
  CHECK(landscape_value(0, 0.4, 4, 0.4, LandscapeValue::PiRelSize, HeightMode::SignedLog) == doctest::Approx(0.0));
  CHECK(landscape_value(0, 0.4, 4, 0.2, LandscapeValue::PiRelSize, HeightMode::SignedLog) == doctest::Approx(std::log(2.0)));
  CHECK(std::isnan(landscape_value(0, 0.4, 4, 0.0, LandscapeValue::PiRelSize, HeightMode::SignedLog)));
  CHECK(std::isnan(landscape_value(0, 0.4, 4, kNaN, LandscapeValue::PiRelSize, HeightMode::SignedLog)));
  CHECK(landscape_value(0, 0.4, 4, kNaN, LandscapeValue::Pi, HeightMode::SignedLog) == doctest::Approx(std::log(1.6)));
  CHECK(parse_landscape_value("pi_rel_size") == LandscapeValue::PiRelSize);
  CHECK(to_string(LandscapeValue::PiRelSize) == "pi_rel_size");

  LandscapeParams p;
  p.value = LandscapeValue::PiRelSize;
  p.territory = TerritoryMode::Sector;
  const LandscapeFrame lf = LandscapeBuilder(4, p).build(hand_frame());
  REQUIRE(lf.nodes.size() == 4);
  CHECK(lf.nodes[0].hdisp == doctest::Approx(0.0));            // pi equals its size share
  CHECK(lf.nodes[1].hdisp == doctest::Approx(std::log(2.0)));  // twice its share
  CHECK(std::isnan(lf.nodes[2].hdisp));                         // zero size: IDW ignores it
  CHECK(lf.nodes[3].hdisp == doctest::Approx(std::log(0.25)));
  CHECK(lf.nodes[1].size_share == doctest::Approx(0.2));
  CHECK(lf.nodes[0].pi == 0.4);  // the truth is unchanged
  // restyle keeps it
  const LandscapeFrame r = restyle(lf, p);
  CHECK(r.nodes[1].hdisp == doctest::Approx(std::log(2.0)));
  CHECK(std::isnan(r.nodes[2].hdisp));
  // and it orders: under PiRelSize the highest log(pi/s) (node 1) takes the centre slot of the mountain,
  // the slot the highest pi takes under Pi.
  LandscapeParams q = p;
  q.value = LandscapeValue::Pi;
  CHECK_FALSE(same_placement(p, q));
}

TEST_CASE("the pipeline frame carries the size reference without changing pi") {
  SyntheticConfig cfg;
  BarStore store(test::temp_dir("pirel"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& x : secs) tickers.push_back(x.ticker);
  const Panel panel = build_panel(store, tickers, cfg.tf);
  const Frame f = run_panel_last(panel, CoreParams::market_rank());
  REQUIRE(f.size_ref.size() == panel.N());
  for (std::size_t i = 0; i < panel.N(); ++i) {
    if (f.active[i]) CHECK(f.size_ref[i] >= CoreParams::market_rank().min_dollar_volume);  // the floor's median
    else CHECK(std::isnan(f.size_ref[i]));
  }
  // pi is the same whatever the landscape does with the size: h = pi*N - 1 under the uniform reference
  const auto n = static_cast<double>(std::count(f.active.begin(), f.active.end(), true));
  for (std::size_t i = 0; i < panel.N(); ++i)
    if (f.active[i]) CHECK(f.h[i] == doctest::Approx(f.pi[i] * n - 1.0));
}

TEST_CASE("territory medians can exclude nodes (floor stocks under value = pi)") {
  const std::size_t na = 12, nb = 12, n = na + nb;
  std::vector<std::uint32_t> group(n);
  std::vector<double> s(n);
  std::vector<bool> active(n, true), floor(n, false);
  for (std::size_t i = 0; i < n; ++i) {
    group[i] = i < na ? 0 : 1;
    if (i < 10) s[i] = -2.0, floor[i] = true;  // group 0: ten floor stocks
    else if (i < na) s[i] = 1.0;               // and two above
    else s[i] = 0.0;
  }
  const LatticeSize sz = lattice_size(n);
  const auto all = territory_layout(active, group, s, sz);
  CHECK_FALSE(all.territories[0].mountain);  // the floor drags its median down
  const auto ex = territory_layout(active, group, s, sz, nullptr, 0.15, floor);
  CHECK(ex.territories[0].mountain);
  CHECK(ex.territories[1].mountain);  // median 0 >= overall median 0
}

TEST_CASE("landscape nodes flag the teleport floor") {
  Frame f = hand_frame();
  f.pi_floor = 0.1;
  const LandscapeFrame lf = LandscapeBuilder(4, LandscapeParams{}).build(f);
  CHECK_FALSE(lf.nodes[0].floor);
  CHECK(lf.nodes[2].floor);
  CHECK(lf.nodes[3].floor);
  CHECK(at_teleport_floor(0.1 * (1 + 1e-7), 0.1));
  CHECK_FALSE(at_teleport_floor(0.1 * (1 + 1e-5), 0.1));
  CHECK_FALSE(at_teleport_floor(0.1, 0.0));  // no floor known
}

TEST_CASE("rank report: pi desc then lower i; the bottom section folds the floor tie") {
  Frame f;
  f.active = {true, true, true, true, true, true, false};
  f.pi = {0.3, 0.1, 0.1, 0.3, 0.05, 0.05, 0.0};
  f.h = {0.8, -0.4, -0.4, 0.8, -0.7, -0.7, kNaN};
  f.pi_floor = 0.05;
  const auto order = rank_order(f, RankBy::Pi);
  CHECK(order == std::vector<std::size_t>{0, 3, 1, 2, 4, 5});
  const BottomSection b = bottom_section(f, order, RankBy::Pi, 10);
  CHECK(b.floor_count == 2);
  CHECK(b.floor_score == doctest::Approx(0.05 * 6));
  CHECK(b.rows == std::vector<std::size_t>{2, 1, 3, 0});  // lowest non-floor first
  CHECK(floor_line(b) == "2 stocks tied at the teleport floor (π·N = 0.3000)");
  const BottomSection b2 = bottom_section(f, order, RankBy::Pi, 1);
  CHECK(b2.rows == std::vector<std::size_t>{2});
  // by hotness: no folding
  const auto ho = rank_order(f, RankBy::Hotness);
  CHECK(ho == std::vector<std::size_t>{0, 3, 1, 2, 4, 5});
  const BottomSection bh = bottom_section(f, ho, RankBy::Hotness, 2);
  CHECK(bh.floor_count == 0);
  CHECK(bh.rows == std::vector<std::size_t>{5, 4});
}

TEST_CASE("pi_rel_size is the landscape default under marketrank") {
  CHECK(default_landscape_value("marketrank") == LandscapeValue::PiRelSize);
  CHECK(default_landscape_value("money-flow") == LandscapeValue::Hotness);
  CHECK(default_landscape_value("legacy") == LandscapeValue::Hotness);
  CHECK(default_landscape_value("defaults") == LandscapeValue::Hotness);
  CHECK(LandscapeParams{}.territory == TerritoryMode::Flux);
  CHECK(LandscapeParams{}.smoother == Smoother::Cvt);
  CHECK(LandscapeParams{}.idw.subdivision == 1);
}
