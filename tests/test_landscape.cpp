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

using namespace mr;

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
  p.smoother = Smoother::None;  // exactness of the IDW mesh vertices is checked without display smoothing
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
  exact.smoother = Smoother::None;  // unsmoothed: exact at the base cells
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
      p.smoother = Smoother::None;
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
  p0.smoother = Smoother::None;
  p1.smoother = Smoother::Gaussian;
  CHECK(LandscapeParams{}.smooth == 1.0);
  CHECK(LandscapeParams{}.smoother == Smoother::Cvt);
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

namespace {
// A frame over n nodes with the given hotness and no flux edges (enough for sector-mode placement).
Frame plain_frame(const std::vector<double>& h, TimePoint t) {
  Frame f;
  f.t = t;
  f.active.assign(h.size(), true);
  f.h = h;
  f.pi.assign(h.size(), 1.0 / static_cast<double>(h.size()));
  f.P.n = h.size();
  f.P.row_ptr.assign(h.size() + 1, 0);
  return f;
}

double spearman_rho(const std::vector<double>& a, const std::vector<double>& b) {
  auto ranks = [](const std::vector<double>& v) {
    std::vector<std::size_t> o(v.size());
    for (std::size_t k = 0; k < o.size(); ++k) o[k] = k;
    std::sort(o.begin(), o.end(), [&](auto x, auto y) { return v[x] < v[y]; });
    std::vector<double> r(v.size());
    for (std::size_t i = 0; i < o.size();) {
      std::size_t j = i;
      while (j + 1 < o.size() && v[o[j + 1]] == v[o[i]]) ++j;
      for (std::size_t k = i; k <= j; ++k) r[o[k]] = 0.5 * static_cast<double>(i + j);
      i = j + 1;
    }
    return r;
  };
  const auto ra = ranks(a), rb = ranks(b);
  double ma = 0, mb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) ma += ra[i], mb += rb[i];
  ma /= static_cast<double>(a.size());
  mb /= static_cast<double>(a.size());
  double sab = 0, saa = 0, sbb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sab += (ra[i] - ma) * (rb[i] - mb);
    saa += (ra[i] - ma) * (ra[i] - ma);
    sbb += (rb[i] - mb) * (rb[i] - mb);
  }
  return sab / std::sqrt(saa * sbb);
}
}  // namespace

TEST_CASE("stocks keep their cells while hotness drifts 2% per frame, and territories stay mountains or craters") {
  // Real-shaped: 2000 stocks in 5 groups of unequal size, heavy-tailed hotness with hot and cold groups. As on real
  // data the active set churns (10 stocks leave and 10 join every frame), so the territory ranges shift a little.
  const std::vector<std::size_t> sizes = {800, 500, 350, 230, 120};
  const std::vector<double> offset = {0.6, -0.5, 0.4, -0.8, 1.0};
  std::vector<std::uint32_t> group;
  for (std::size_t g = 0; g < sizes.size(); ++g) group.insert(group.end(), sizes[g], static_cast<std::uint32_t>(g));
  const std::size_t n = group.size();
  std::mt19937_64 rng(2026);
  std::normal_distribution<double> N01(0.0, 1.0);
  std::vector<double> z(n);
  for (std::size_t i = 0; i < n; ++i) z[i] = offset[group[i]] + 0.7 * N01(rng);
  std::vector<bool> active(n, true);
  for (std::size_t i = 0; i < 10; ++i) active[i * 197] = false;
  auto frame_at = [&](TimePoint t) {
    Frame f = plain_frame(std::vector<double>(n), t);
    f.active = active;
    for (std::size_t i = 0; i < n; ++i) f.h[i] = active[i] ? std::exp(z[i]) - 1.0 : std::nan("");
    return f;
  };
  LandscapeParams p;
  p.territory = TerritoryMode::Sector;
  LandscapeBuilder b(n, p, group);
  LandscapeFrame prev = b.build(frame_at(0));
  const auto W = static_cast<std::int32_t>(prev.size.cols);
  for (int frame = 1; frame <= 10; ++frame) {
    for (auto& v : z) v += 0.02 * N01(rng);  // ~2% drift of 1 + h per frame
    std::vector<std::size_t> on, off;
    for (std::size_t i = 0; i < n; ++i) (active[i] ? on : off).push_back(i);
    std::shuffle(on.begin(), on.end(), rng);
    for (std::size_t k = 0; k < 10; ++k) active[on[k]] = false;
    for (auto i : off) active[i] = true;
    const LandscapeFrame cur = b.build(frame_at(frame));
    REQUIRE(cur.size == prev.size);
    std::map<std::uint32_t, std::int32_t> before;
    for (const auto& nd : prev.nodes) before[nd.i] = nd.cell;
    std::vector<double> d;
    std::size_t kept = 0;
    for (const auto& nd : cur.nodes) {
      auto it = before.find(nd.i);
      if (it == before.end()) continue;
      const auto a = it->second, c = nd.cell;
      d.push_back(std::hypot(c % W - a % W, c / W - a / W));
      kept += a == c ? 1 : 0;
    }
    std::nth_element(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(d.size() / 2), d.end());
    const double med = d[d.size() / 2], keep = static_cast<double>(kept) / static_cast<double>(d.size());
    INFO("frame " << frame << ": median displacement " << med << " cells, " << 100 * keep << "% kept");
    CHECK(med <= 3.0);
    CHECK(keep >= 0.6);
    // Territory ranges depend only on the group counts and the lattice, so a fresh layout gives the centres.
    const auto terr = territory_layout(active, group, std::vector<double>(n, 0.0), cur.size).territories;
    std::vector<double> all;
    for (const auto& nd : cur.nodes) all.push_back(nd.hdisp);
    std::sort(all.begin(), all.end());
    for (const auto& t : terr) {
      std::vector<double> hv, dist;
      for (const auto& nd : cur.nodes)
        if (group[nd.i] == t.group) {
          hv.push_back(nd.hdisp);
          dist.push_back(std::hypot(nd.cell % W + 0.5 - t.cx, nd.cell / W + 0.5 - t.cy));
        }
      const double rho = spearman_rho(hv, dist);
      std::vector<double> sorted = hv;
      std::sort(sorted.begin(), sorted.end());
      const bool mountain = sorted[sorted.size() / 2] >= all[all.size() / 2];
      INFO("group " << t.group << (mountain ? " mountain" : " crater") << " rho " << rho);
      if (mountain)
        CHECK(rho <= -0.8);
      else
        CHECK(rho >= 0.8);
    }
    if (frame == 10) MESSAGE("frame 10: median displacement " << med << " cells, " << 100 * keep << "% kept");
    prev = cur;
  }
}

TEST_CASE("planted flux communities A-B-C: each one contiguous, trading partners adjacent in the territory order") {
  // Three communities of 40 that trade densely inside; A trades with B and B with C, never A with C. Nodes are
  // interleaved (community = i % 3) so index order says nothing about the layout.
  const std::size_t n = 120;
  auto comm = [](std::size_t i) { return i % 3; };
  std::mt19937_64 rng(5);
  std::vector<std::vector<std::pair<std::uint32_t, double>>> rows(n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) continue;
      const std::size_t a = comm(i), b = comm(j);
      if (a == b && rng() % 2) rows[i].push_back({static_cast<std::uint32_t>(j), 1.0 + static_cast<double>(rng() % 100) / 100.0});
      else if (a != b && (a + b == 1 || a + b == 3) && rng() % 20 == 0) rows[i].push_back({static_cast<std::uint32_t>(j), 0.2});
    }
  std::vector<double> h(n);
  for (std::size_t i = 0; i < n; ++i) h[i] = std::sin(0.7 * static_cast<double>(i));
  Frame f = plain_frame(h, 1);
  f.P.row_ptr.assign(1, 0);
  for (auto& r : rows) {
    for (auto& e : r) {
      f.P.col.push_back(e.first);
      f.P.raw.push_back(e.second);
      f.P.val.push_back(e.second);
    }
    f.P.row_ptr.push_back(f.P.col.size());
  }
  LandscapeBuilder b(n, LandscapeParams{});
  const LandscapeFrame lf = b.build(f);
  REQUIRE(lf.nodes.size() == n);
  // Louvain recovers the planted communities: one label each, no loose stocks.
  std::map<std::size_t, std::set<std::int32_t>> labels;
  for (const auto& nd : lf.nodes) labels[comm(nd.i)].insert(nd.group);
  for (std::size_t c = 0; c < 3; ++c) {
    REQUIRE(labels[c].size() == 1);
    CHECK(*labels[c].begin() >= 0);
  }
  CHECK(lf.communities == 3);
  CHECK(lf.loose == 0);
  // Contiguity (8-connected) of each community's cells.
  const auto W = static_cast<std::int32_t>(lf.size.cols);
  for (std::size_t c = 0; c < 3; ++c) {
    std::set<std::int32_t> rest;
    for (const auto& nd : lf.nodes)
      if (comm(nd.i) == c) rest.insert(nd.cell);
    std::vector<std::int32_t> st{*rest.begin()};
    rest.erase(rest.begin());
    while (!st.empty()) {
      const auto cell = st.back();
      st.pop_back();
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const std::int32_t x = cell % W + dx;
          if (x < 0 || x >= W) continue;
          auto it = rest.find((cell / W + dy) * W + x);
          if (it != rest.end()) {
            st.push_back(*it);
            rest.erase(it);
          }
        }
    }
    CHECK(rest.empty());
  }
  // Territory order along the curve: B (the only community trading with both) sits between A and C.
  const auto order = gilbert_order(lf.size);
  std::vector<std::size_t> at(lf.size.cells());
  for (std::size_t k = 0; k < order.size(); ++k) at[static_cast<std::size_t>(order[k])] = k;
  std::vector<std::size_t> first(3, order.size());
  for (const auto& nd : lf.nodes) first[comm(nd.i)] = std::min(first[comm(nd.i)], at[static_cast<std::size_t>(nd.cell)]);
  const bool abc = first[0] < first[1] && first[1] < first[2], cba = first[2] < first[1] && first[1] < first[0];
  CHECK((abc || cba));
}

TEST_CASE("smoother: default is CVT, none/gaussian/cvt differ, delta raster follows, mass change is reported") {
  const Frame f = synthetic_frame();
  const auto groups = synthetic_groups(f.active.size());
  LandscapeParams pn, pg, pc;
  pn.smoother = Smoother::None;
  pg.smoother = Smoother::Gaussian;
  CHECK(pc.smoother == Smoother::Cvt);
  CHECK(pc.cvt.iterations == 12);
  CHECK(pc.cvt.lambda == 0.6);
  CHECK(pc.cvt.eps_frac == 0.1);
  LandscapeBuilder bn(f.active.size(), pn, groups), bc(f.active.size(), pc, groups);
  const LandscapeFrame n = bn.build(f), c = bc.build(f);
  Raster expect = n.raster;
  cvt_smooth(expect, pc.cvt);
  CHECK(c.raster.z == expect.z);
  for (std::size_t k = 0; k < n.nodes.size(); ++k) CHECK(c.nodes[k].hdisp == n.nodes[k].hdisp);  // node values stay exact
  std::vector<double> delta(f.active.size(), 0.0);
  delta[n.nodes.front().i] = 1.0;
  Raster d = delta_raster(n, delta, pn);
  cvt_smooth(d, pc.cvt);
  CHECK(delta_raster(n, delta, pc).z == d.z);
  auto mean = [](const Raster& r) { double s = 0; for (float z : r.z) s += z; return s / static_cast<double>(r.z.size()); };
  MESSAGE("synthetic frame mean(z): none " << mean(n.raster) << ", cvt " << mean(c.raster));
  CHECK(parse_smoother("cvt") == Smoother::Cvt);
  CHECK(parse_smoother("gaussian") == Smoother::Gaussian);
  CHECK(parse_smoother("none") == Smoother::None);
  CHECK_THROWS_AS(parse_smoother("x"), std::invalid_argument);
  LandscapeParams bad;
  bad.cvt.lambda = 0;
  CHECK_THROWS_AS(LandscapeBuilder(f.active.size(), bad), std::invalid_argument);
}

TEST_CASE("pinned nodes: the CVT surface passes exactly through them in build, restyle and the delta raster") {
  const Frame f = synthetic_frame();
  LandscapeParams p;  // CVT default
  LandscapeBuilder b0(f.active.size(), p);
  const LandscapeFrame free_ = b0.build(f);
  REQUIRE(free_.nodes.size() > 5);
  LandscapeParams pp = p;
  pp.pinned = {free_.nodes[0].i, free_.nodes[3].i, static_cast<std::uint32_t>(f.active.size() + 7)};  // last: not in the universe, ignored
  LandscapeBuilder b1(f.active.size(), pp);
  const LandscapeFrame pin = b1.build(f);
  REQUIRE(pin.nodes.size() == free_.nodes.size());
  const auto at = [](const LandscapeFrame& lf, std::size_t k) { return lf.raster.z[static_cast<std::size_t>(lf.nodes[k].cell)]; };
  for (std::size_t k : {std::size_t(0), std::size_t(3)}) {
    CHECK(pin.nodes[k].cell == free_.nodes[k].cell);  // pinning is display-only
    CHECK(at(pin, k) == static_cast<float>(pin.nodes[k].hdisp));
  }
  CHECK(at(free_, 0) != static_cast<float>(free_.nodes[0].hdisp));  // without the pin CVT moves it
  CHECK(same_placement(p, pp));
  const LandscapeFrame re = restyle(free_, pp);
  CHECK(at(re, 0) == static_cast<float>(re.nodes[0].hdisp));
  std::vector<double> delta(f.active.size(), 0.0);
  delta[pin.nodes[1].i] = 1.0;
  delta[pin.nodes[0].i] = -0.5;
  const Raster d = delta_raster(pin, delta, pp);
  CHECK(d.z[static_cast<std::size_t>(pin.nodes[0].cell)] == static_cast<float>(display_height(-0.5, pp.height)));
  CHECK(d.z[static_cast<std::size_t>(pin.nodes[3].cell)] == 0.0f);
  // Subdivision 3: the pinned node's cell-centre pixel carries its exact value.
  LandscapeParams s3 = pp;
  s3.idw.subdivision = 3;
  const LandscapeFrame r3 = restyle(free_, s3);
  const auto& n0 = r3.nodes[0];
  const std::size_t cols = r3.size.cols, px = (static_cast<std::size_t>(n0.cell) % cols) * 3 + 1,
                    py = (static_cast<std::size_t>(n0.cell) / cols) * 3 + 1;
  CHECK(r3.raster.z[py * r3.raster.w + px] == static_cast<float>(n0.hdisp));
}

TEST_CASE("exclude_etf: excluded nodes stay in the frame without a cell, the others are placed, pi is untouched") {
  const Frame f = synthetic_frame();
  const std::size_t n = f.active.size();
  std::vector<char> ex(n, 0);
  for (std::size_t i = n - 7; i < n; ++i) ex[i] = 1;
  LandscapeParams p;
  LandscapeBuilder all(n, p, synthetic_groups(n));
  LandscapeBuilder some(n, p, synthetic_groups(n));
  some.set_excluded(ex);
  const LandscapeFrame a = all.build(f), b = some.build(f);
  REQUIRE(a.nodes.size() == b.nodes.size());
  std::set<std::int32_t> cells;
  std::size_t placed = 0;
  for (std::size_t k = 0; k < b.nodes.size(); ++k) {
    const auto& nd = b.nodes[k];
    CHECK(nd.pi == a.nodes[k].pi);  // the truth is the same
    CHECK(nd.h == a.nodes[k].h);
    if (ex[nd.i]) {
      CHECK(nd.cell < 0);
    } else {
      CHECK(nd.cell >= 0);
      CHECK(cells.insert(nd.cell).second);
      ++placed;
    }
  }
  CHECK(b.size == lattice_size(placed));
  for (const auto& ar : b.arcs) CHECK((!ex[ar.a] && !ex[ar.b]));
}

TEST_CASE("exclude_etf is a placement parameter and defaults to true") {
  LandscapeParams a, b;
  CHECK(a.exclude_etf);
  CHECK(same_placement(a, b));
  b.exclude_etf = false;
  CHECK_FALSE(same_placement(a, b));
}
