#include <doctest/doctest.h>

#include <cmath>
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
}
