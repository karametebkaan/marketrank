#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <set>
#include <vector>

#include "geom/landscape.hpp"
#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/core_pipeline.hpp"
#include "test_util.hpp"

using namespace fx;

namespace {
Frame synthetic_frame() {
  SyntheticConfig cfg;
  BarStore store(test::temp_dir("landscape"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  return run_panel_last(build_panel(store, tickers, cfg.tf), CoreParams::money_flow());
}
}  // namespace

TEST_CASE("display height") {
  CHECK(display_height(0.0, HeightMode::SignedLog) == 0.0);
  CHECK(display_height(std::exp(1.0) - 1.0, HeightMode::SignedLog) == doctest::Approx(1.0));
  CHECK(display_height(-(std::exp(2.0) - 1.0), HeightMode::SignedLog) == doctest::Approx(-2.0));
  CHECK(display_height(-0.3, HeightMode::Linear) == -0.3);
  CHECK(parse_height_mode(to_string(HeightMode::Linear)) == HeightMode::Linear);
}

TEST_CASE("landscape build: one cell per active node, raster sized from the lattice, arcs sorted") {
  const Frame f = synthetic_frame();
  LandscapeParams p;
  p.max_arcs = 50;
  LandscapeBuilder b(f.active.size(), p);
  LandscapeFrame lf = b.build(f);
  std::size_t n_active = 0;
  for (bool a : f.active) n_active += a ? 1 : 0;
  CHECK(lf.nodes.size() == n_active);
  CHECK(lf.size == lattice_size(n_active));
  std::set<std::int32_t> cells;
  for (const auto& nd : lf.nodes) {
    CHECK(nd.cell >= 0);
    CHECK(cells.insert(nd.cell).second);
    CHECK(nd.hdisp == doctest::Approx(display_height(nd.h, HeightMode::SignedLog)));
    CHECK(nd.fx >= 0.0f);
    CHECK(nd.fx <= 1.0f);
  }
  CHECK(lf.raster.w == lf.size.cols * static_cast<std::size_t>(p.idw.subdivision));
  CHECK(lf.raster.h == lf.size.rows * static_cast<std::size_t>(p.idw.subdivision));
  REQUIRE(p.idw.subdivision == 1);
  // Mesh vertices at occupied cells carry the node's exact display height.
  for (const auto& nd : lf.nodes)
    CHECK(lf.raster.z[static_cast<std::size_t>(nd.cell)] == doctest::Approx(nd.hdisp).epsilon(1e-6));
  CHECK(lf.arcs.size() <= 50);
  for (std::size_t k = 1; k < lf.arcs.size(); ++k) CHECK(lf.arcs[k - 1].w >= lf.arcs[k].w);
  CHECK(lf.t == f.t);
}

TEST_CASE("rebuilding on the same frame keeps most cells (warm start + hysteresis)") {
  const Frame f = synthetic_frame();
  LandscapeBuilder b(f.active.size(), LandscapeParams{});
  LandscapeFrame a = b.build(f);
  LandscapeFrame c = b.build(f);
  std::size_t same = 0;
  for (std::size_t k = 0; k < a.nodes.size(); ++k) same += a.nodes[k].cell == c.nodes[k].cell ? 1 : 0;
  CHECK(static_cast<double>(same) >= 0.8 * static_cast<double>(a.nodes.size()));
}

TEST_CASE("delta raster uses the base cells") {
  const Frame f = synthetic_frame();
  LandscapeBuilder b(f.active.size(), LandscapeParams{});
  LandscapeFrame lf = b.build(f);
  std::vector<double> delta(f.active.size(), 0.0);
  delta[lf.nodes.front().i] = 1.0;
  Raster r = delta_raster(lf, delta, LandscapeParams{});
  CHECK(r.w == lf.raster.w);
  CHECK(r.zmax > 0.0f);
  CHECK(r.zmin >= 0.0f);
  const auto front_cell = static_cast<std::size_t>(lf.nodes.front().cell);
  CHECK(r.z[front_cell] == doctest::Approx(display_height(1.0, HeightMode::SignedLog)));
  for (const auto& nd : lf.nodes)
    if (nd.cell != lf.nodes.front().cell) CHECK(r.z[static_cast<std::size_t>(nd.cell)] == 0.0f);
  CHECK_THROWS_AS(delta_raster(lf, std::vector<double>(1, 0.0), LandscapeParams{}), std::invalid_argument);
  CHECK_THROWS_AS(delta_raster(lf, std::vector<double>(f.active.size() + 1, 0.0), LandscapeParams{}),
                  std::invalid_argument);
}

TEST_CASE("top_arcs on a hand-built Csr") {
  Csr P;
  P.n = 4;
  // row 0: self(5), ->1 (2), ->2 (2), ->3 (9, inactive)
  // row 1: ->0 (2), ->2 (0, excluded), ->3 (7, inactive)
  // row 2: ->0 (3), ->1 (-1, excluded)
  // row 3 (inactive): ->0 (8)
  P.row_ptr = {0, 4, 7, 9, 10};
  P.col = {0, 1, 2, 3, 0, 2, 3, 0, 1, 0};
  P.raw = {5, 2, 2, 9, 2, 0, 7, 3, -1, 8};
  P.val = P.raw;
  const std::vector<bool> active{true, true, true, false};
  auto arcs = top_arcs(P, active, 100);
  const std::vector<std::array<double, 3>> want_all{{2, 0, 3}, {0, 1, 2}, {0, 2, 2}, {1, 0, 2}};
  // sorted: w=3 (2,0); w=2: (0,1), (0,2), (1,0)
  REQUIRE(arcs.size() == 4);
  for (std::size_t k = 0; k < 4; ++k) {
    CHECK(arcs[k].a == static_cast<std::uint32_t>(want_all[k][0]));
    CHECK(arcs[k].b == static_cast<std::uint32_t>(want_all[k][1]));
    CHECK(arcs[k].w == want_all[k][2]);
  }
  auto top2 = top_arcs(P, active, 2);
  REQUIRE(top2.size() == 2);
  CHECK((top2[0].a == 2 && top2[0].b == 0 && top2[0].w == 3.0));
  CHECK((top2[1].a == 0 && top2[1].b == 1 && top2[1].w == 2.0));
  CHECK_THROWS_AS(top_arcs(P, std::vector<bool>(3, true), 10), std::invalid_argument);
}

TEST_CASE("build validates frame sizes") {
  Frame f = synthetic_frame();
  LandscapeBuilder b(f.active.size(), LandscapeParams{});
  Frame bad = f;
  bad.h.pop_back();
  CHECK_THROWS_AS(b.build(bad), std::invalid_argument);
  bad = f;
  bad.pi.pop_back();
  CHECK_THROWS_AS(b.build(bad), std::invalid_argument);
  bad = f;
  bad.forecasts.front().score.pop_back();
  CHECK_THROWS_AS(b.build(bad), std::invalid_argument);
  bad = f;
  bad.P.n += 1;
  CHECK_THROWS_AS(b.build(bad), std::invalid_argument);
}

TEST_CASE("builder state: lattice size change, unit-range fx/fy, single node") {
  const Frame f = synthetic_frame();
  const std::size_t n = f.active.size();
  Frame small = f;
  std::size_t kept = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (small.active[i] && (kept++ % 2 == 1)) small.active[i] = false;
  }
  std::size_t n_small = 0, n_full = 0;
  for (bool a : small.active) n_small += a ? 1 : 0;
  for (bool a : f.active) n_full += a ? 1 : 0;
  REQUIRE(!(lattice_size(n_small) == lattice_size(n_full)));
  LandscapeBuilder b(n, LandscapeParams{});
  LandscapeFrame a = b.build(small);
  LandscapeFrame c = b.build(f);
  CHECK(a.size == lattice_size(n_small));
  CHECK(c.size == lattice_size(n_full));
  CHECK(c.nodes.size() == n_full);
  std::set<std::int32_t> cells;
  for (const auto& nd : c.nodes) {
    CHECK(nd.cell >= 0);
    CHECK(static_cast<std::size_t>(nd.cell) < c.size.cells());
    CHECK(cells.insert(nd.cell).second);
    CHECK(nd.fx >= 0.0f);
    CHECK(nd.fx <= 1.0f);
    CHECK(nd.fy >= 0.0f);
    CHECK(nd.fy <= 1.0f);
  }
  // Across a size change there is no hysteresis: cells are the plain RCB assignment of the node positions
  // (fx, fy are a monotone per-axis rescale of the embedding, which preserves the RCB sort order).
  std::vector<double> xy(2 * n, 0.0);
  for (const auto& nd : c.nodes) {
    xy[2 * nd.i] = nd.fx;
    xy[2 * nd.i + 1] = nd.fy;
  }
  const auto plain = rcb_assign(xy, f.active, c.size);
  for (const auto& nd : c.nodes) CHECK(nd.cell == plain[nd.i]);

  Frame one = f;
  std::fill(one.active.begin(), one.active.end(), false);
  one.active[0] = true;
  LandscapeBuilder b1(n, LandscapeParams{});
  LandscapeFrame s = b1.build(one);
  REQUIRE(s.nodes.size() == 1);
  CHECK(s.nodes[0].fx == 0.0f);
  CHECK(s.nodes[0].fy == 0.0f);
  CHECK(std::isfinite(s.nodes[0].hdisp));
  for (float z : s.raster.z) CHECK(std::isfinite(z));
}
