#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "geom/landscape.hpp"
#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/core_pipeline.hpp"
#include "test_util.hpp"

using namespace fx;

namespace {
Frame synthetic_frame(std::uint64_t seed = 42) {
  SyntheticConfig cfg;
  cfg.seed = seed;
  BarStore store(test::temp_dir("landscape"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  return run_panel_last(build_panel(store, tickers, cfg.tf), CoreParams::money_flow());
}
}  // namespace

// The synthetic market has 5 sectors of 10 consecutive stocks.
static std::vector<std::uint32_t> synthetic_groups(std::size_t n) {
  std::vector<std::uint32_t> g(n);
  for (std::size_t i = 0; i < n; ++i) g[i] = static_cast<std::uint32_t>(i / 10);
  return g;
}

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
  p.smooth = 0;  // exactness of the IDW mesh vertices is checked without display smoothing
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

TEST_CASE("rebuilding on the same frame keeps most cells (smoothed ranking)") {
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
  LandscapeParams exact;
  exact.smooth = 0;  // unsmoothed: exact at the base cells
  Raster r = delta_raster(lf, delta, exact);
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
  Frame one = f;
  std::fill(one.active.begin(), one.active.end(), false);
  one.active[0] = true;
  LandscapeBuilder b1(n, LandscapeParams{});
  LandscapeFrame s = b1.build(one);
  REQUIRE(s.nodes.size() == 1);
  CHECK(s.nodes[0].fx == 0.5f);
  CHECK(s.nodes[0].fy == 0.5f);
  CHECK(std::isfinite(s.nodes[0].hdisp));
  for (float z : s.raster.z) CHECK(std::isfinite(z));
}

namespace {
// Pearson correlation of the display heights of horizontally and vertically adjacent occupied cells.
double neighbour_corr(const LatticeSize& size, const std::vector<std::int32_t>& cells, const std::vector<double>& v) {
  const std::size_t W = size.cols, H = size.rows;
  std::vector<double> hd(W * H, std::numeric_limits<double>::quiet_NaN());
  for (std::size_t k = 0; k < cells.size(); ++k) hd[static_cast<std::size_t>(cells[k])] = v[k];
  std::vector<double> xs, ys;
  for (std::size_t r = 0; r < H; ++r)
    for (std::size_t c = 0; c < W; ++c) {
      const double a = hd[r * W + c];
      if (std::isnan(a)) continue;
      if (c + 1 < W && !std::isnan(hd[r * W + c + 1])) {
        xs.push_back(a);
        ys.push_back(hd[r * W + c + 1]);
      }
      if (r + 1 < H && !std::isnan(hd[(r + 1) * W + c])) {
        xs.push_back(a);
        ys.push_back(hd[(r + 1) * W + c]);
      }
    }
  double mx = 0, my = 0;
  for (std::size_t k = 0; k < xs.size(); ++k) mx += xs[k], my += ys[k];
  mx /= static_cast<double>(xs.size());
  my /= static_cast<double>(xs.size());
  double sxy = 0, sxx = 0, syy = 0;
  for (std::size_t k = 0; k < xs.size(); ++k) {
    sxy += (xs[k] - mx) * (ys[k] - my);
    sxx += (xs[k] - mx) * (xs[k] - mx);
    syy += (ys[k] - my) * (ys[k] - my);
  }
  return sxy / std::sqrt(sxx * syy);
}

double median_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
}  // namespace

TEST_CASE("lattice neighbours have correlated heights: territories beat a shuffled-cell baseline over seeds 1..9") {
  std::vector<double> flux, sector, shuffled;
  for (std::uint64_t seed = 1; seed <= 9; ++seed) {
    const Frame f = synthetic_frame(seed);
    const std::size_t n = f.active.size();
    for (TerritoryMode mode : {TerritoryMode::Flux, TerritoryMode::Sector}) {
      LandscapeParams p;
      p.territory = mode;
      p.smooth = 0;
      p.idw.subdivision = 1;
      LandscapeBuilder b(n, p, synthetic_groups(n));
      LandscapeFrame lf = b.build(f);
      std::vector<std::int32_t> cells;
      std::vector<double> hd;
      for (const auto& nd : lf.nodes) {
        cells.push_back(nd.cell);
        hd.push_back(nd.hdisp);
      }
      (mode == TerritoryMode::Flux ? flux : sector).push_back(neighbour_corr(lf.size, cells, hd));
      if (mode == TerritoryMode::Flux) {  // same cells, node values permuted (fixed Fisher-Yates on mt19937_64)
        std::mt19937_64 rng(1000 + seed);
        for (std::size_t k = hd.size(); k > 1; --k) std::swap(hd[k - 1], hd[rng() % k]);
        shuffled.push_back(neighbour_corr(lf.size, cells, hd));
      }
    }
  }
  const double mf = median_of(flux), ms = median_of(sector), mb = median_of(shuffled);
  MESSAGE("median neighbour correlation over seeds 1..9: flux " << mf << ", sector " << ms << ", shuffled " << mb);
  CHECK(mf - mb >= 0.25);
  CHECK(ms - mb >= 0.25);
  CHECK(mf >= 0.3);
}

TEST_CASE("territory placement is stable: identical rebuilds, and 1% hotness noise moves few nodes") {
  const Frame f = synthetic_frame();
  const std::size_t n = f.active.size();
  const auto groups = synthetic_groups(n);
  LandscapeBuilder b1(n, LandscapeParams{}, groups), b2(n, LandscapeParams{}, groups);
  auto a = b1.build(f), c = b2.build(f);
  REQUIRE(a.nodes.size() == c.nodes.size());
  for (std::size_t k = 0; k < a.nodes.size(); ++k) CHECK(a.nodes[k].cell == c.nodes[k].cell);
  Frame g = f;
  for (std::size_t i = 0; i < n; ++i)
    if (g.active[i]) g.h[i] *= 1.0 + 0.01 * ((i % 3 == 0) ? 1.0 : -1.0);
  auto d = b1.build(g);
  std::size_t moved = 0;
  for (std::size_t k = 0; k < a.nodes.size(); ++k) moved += a.nodes[k].cell != d.nodes[k].cell ? 1 : 0;
  CHECK(static_cast<double>(moved) <= 0.1 * static_cast<double>(a.nodes.size()));
  CHECK_THROWS_AS(LandscapeBuilder(n, LandscapeParams{}, std::vector<std::uint32_t>(3, 0)), std::invalid_argument);
}

TEST_CASE("display smoothing: base and delta rasters are Gaussian-smoothed, node values stay exact") {
  const Frame f = synthetic_frame();
  LandscapeParams p0, p1;
  p0.smooth = 0;
  CHECK(LandscapeParams{}.smooth == 1.0);
  const auto groups = synthetic_groups(f.active.size());
  LandscapeBuilder b0(f.active.size(), p0, groups), b1(f.active.size(), p1, groups);
  const LandscapeFrame a = b0.build(f), s = b1.build(f);
  REQUIRE(a.nodes.size() == s.nodes.size());
  for (std::size_t k = 0; k < a.nodes.size(); ++k) {
    CHECK(a.nodes[k].cell == s.nodes[k].cell);
    CHECK(a.nodes[k].h == s.nodes[k].h);
    CHECK(a.nodes[k].hdisp == s.nodes[k].hdisp);
  }
  CHECK(s.raster.z == smooth_raster(a.raster, 1.0, p1.idw.subdivision).z);
  CHECK(s.raster.z != a.raster.z);
  std::vector<double> delta(f.active.size(), 0.0);
  delta[a.nodes.front().i] = 1.0;
  const Raster d0 = delta_raster(a, delta, p0), d1 = delta_raster(a, delta, p1);
  CHECK(d1.z == smooth_raster(d0, 1.0, p1.idw.subdivision).z);
  CHECK(d1.zmax < d0.zmax);
  LandscapeParams bad;
  bad.smooth = -1;
  CHECK_THROWS_AS(LandscapeBuilder(f.active.size(), bad), std::invalid_argument);
}

TEST_CASE("flux territories: contiguous groups, loose last, tracker state survives frames") {
  const Frame f = synthetic_frame();
  const std::size_t n = f.active.size();
  LandscapeParams p;
  LandscapeBuilder b(n, p, synthetic_groups(n));
  auto a = b.build(f);
  CHECK(a.communities >= 1);
  std::map<std::int32_t, std::vector<std::int32_t>> by_group;
  for (const auto& nd : a.nodes) by_group[nd.group].push_back(nd.cell);
  const auto W = static_cast<std::int32_t>(a.size.cols);
  for (auto& [g, cells] : by_group) {
    std::set<std::int32_t> rest(cells.begin(), cells.end());
    std::vector<std::int32_t> st{*rest.begin()};
    rest.erase(rest.begin());
    while (!st.empty()) {
      auto c = st.back();
      st.pop_back();
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const std::int32_t nx = c % W + dx;
          if (nx < 0 || nx >= W) continue;
          auto it = rest.find((c / W + dy) * W + nx);
          if (it != rest.end()) {
            st.push_back(*it);
            rest.erase(it);
          }
        }
    }
    // loose stocks (group -1) are a pool and need not be contiguous
    if (g >= 0) CHECK(rest.empty());
  }
  // a re-cluster on a frame with 1% perturbed fluxes keeps >= 90% of stocks in their old community
  LandscapeParams q;
  q.recluster_bars = 1;
  LandscapeBuilder b2(n, q);
  auto first = b2.build(f);
  Frame g = f;
  for (std::size_t k = 0; k < g.P.raw.size(); ++k) g.P.raw[k] *= 1.0 + 0.01 * ((k % 2) ? 1.0 : -1.0);
  auto second = b2.build(g);
  std::size_t same = 0;
  for (std::size_t k = 0; k < first.nodes.size(); ++k) same += first.nodes[k].group == second.nodes[k].group ? 1 : 0;
  CHECK(static_cast<double>(same) >= 0.9 * static_cast<double>(first.nodes.size()));
  // identical rebuilds give identical cells
  LandscapeBuilder b3(n, p), b4(n, p);
  auto c3 = b3.build(f), c4 = b4.build(f);
  for (std::size_t k = 0; k < c3.nodes.size(); ++k) CHECK(c3.nodes[k].cell == c4.nodes[k].cell);
}

TEST_CASE("territory mode parsing") {
  CHECK(parse_territory_mode("flux") == TerritoryMode::Flux);
  CHECK(parse_territory_mode("sector") == TerritoryMode::Sector);
  CHECK_THROWS_AS(parse_territory_mode("hex"), std::invalid_argument);
  CHECK(to_string(TerritoryMode::Sector) == "sector");
}
