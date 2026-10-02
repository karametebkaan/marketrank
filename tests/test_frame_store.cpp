#include <doctest/doctest.h>

#include <chrono>
#include <thread>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "server/frame_store.hpp"
#include "test_util.hpp"

using namespace fx;
using namespace std::chrono_literals;

namespace {
struct Market {
  Panel panel;
  std::vector<Security> secs;
};
Market market() {
  SyntheticConfig cfg;
  cfg.bars = 120;
  cfg.rotation_start = 60;
  BarStore store(test::temp_dir("frame_store"));
  Market m;
  m.secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : m.secs) tickers.push_back(s.ticker);
  m.panel = build_panel(store, tickers, cfg.tf);
  return m;
}
void wait_ready(const FrameStore& fs) {
  for (int k = 0; k < 600 && !fs.status().ready; ++k) {
    REQUIRE(fs.status().error.empty());
    std::this_thread::sleep_for(50ms);
  }
  REQUIRE(fs.status().ready);
}
}  // namespace

TEST_CASE("frame store computes landscapes for the last max_frames bars") {
  Market m = market();
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 30);
  fs.start();
  wait_ready(fs);
  auto st = fs.status();
  CHECK(st.computed == st.total);
  CHECK(st.total == m.panel.T() - 1);
  auto times = fs.times();
  REQUIRE(times.size() == 30);
  CHECK(times.back() == m.panel.times.back());
  auto latest = fs.landscape(std::nullopt);
  REQUIRE(latest);
  CHECK(latest->t == m.panel.times.back());
  CHECK(fs.landscape(times.front()));
  CHECK_FALSE(fs.landscape(m.panel.times.front()));
}

TEST_CASE("frame store shock gives deltas and a delta raster on the latest landscape") {
  Market m = market();
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  fs.start();
  wait_ready(fs);
  auto latest = fs.landscape(std::nullopt);
  const std::size_t node = latest->nodes.front().i;
  auto r = fs.shock({{node, -10.0}});
  CHECK(r.t == m.panel.times.back());
  CHECK(r.delta.dh.size() == m.secs.size());
  CHECK(r.delta.l1_dpi > 0);
  CHECK(r.delta.dh[node] < 0);
  CHECK(r.raster.w == latest->raster.w);
}

TEST_CASE("frame store restarts on new parameters and is deterministic") {
  Market m = market();
  FrameStore a(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  FrameStore b(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  a.start();
  b.start();
  wait_ready(a);
  wait_ready(b);
  CHECK(a.landscape(std::nullopt)->raster.z == b.landscape(std::nullopt)->raster.z);
  const auto g = a.status().generation;
  a.set_params(CoreParams::legacy(), LandscapeParams{});
  wait_ready(a);
  CHECK(a.status().generation > g);
  CHECK(a.times().size() == 10);
}

TEST_CASE("wait_for_change returns when progress happens") {
  Market m = market();
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  const auto v0 = fs.wait_for_change(0, 1ms);
  fs.start();
  const auto v1 = fs.wait_for_change(v0, 5000ms);
  CHECK(v1 > v0);
  wait_ready(fs);
}

TEST_CASE("shock before ready throws") {
  Market m = market();
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  CHECK_THROWS_AS(fs.shock({{0, -1.0}}), std::runtime_error);
}
