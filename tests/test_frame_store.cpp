#include <doctest/doctest.h>

#include <chrono>
#include <atomic>
#include <cmath>
#include <thread>
#include <omp.h>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/shock.hpp"
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

TEST_CASE("frame store survives concurrent set_params") {
  Market m = market();
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  fs.start();
  auto hammer = [&](bool legacy) {
    for (int k = 0; k < 50; ++k)
      fs.set_params(legacy ? CoreParams::legacy() : CoreParams::money_flow(), LandscapeParams{});
  };
  std::thread a(hammer, true), b(hammer, false);
  a.join();
  b.join();
  wait_ready(fs);
}

TEST_CASE("frame store destruction does not hang mid-compute") {
  Market m = market();
  {
    FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  }
  {
    FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
    fs.start();
    std::this_thread::sleep_for(5ms);
  }
  {
    FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
    fs.start();
    fs.set_params(CoreParams::legacy(), LandscapeParams{});
    fs.set_params(CoreParams::money_flow(), LandscapeParams{});
  }
  CHECK(true);
}

TEST_CASE("frame store shock matches run_with_shock and empty shock is neutral") {
  Market m = market();
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  fs.start();
  wait_ready(fs);
  auto z = fs.shock({});
  CHECK(z.delta.l1_dpi == 0.0);
  for (double d : z.delta.dh)
    if (!std::isnan(d)) CHECK(d == 0.0);
  const std::size_t node = fs.landscape(std::nullopt)->nodes.front().i;
  auto ref = run_with_shock(m.panel, CoreParams::money_flow(), {{node, -10.0}});
  auto rd = shock_response(ref.first, ref.second);
  auto r = fs.shock({{node, -10.0}});
  CHECK(test::same_values(rd.dh, r.delta.dh));
  CHECK(test::same_values(rd.dpi, r.delta.dpi));
}

TEST_CASE("frame store reports worker errors and stays not ready") {
  SyntheticConfig cfg;
  cfg.bars = 2;
  cfg.rotation_start = 1;
  BarStore store(test::temp_dir("frame_store_short"));
  Market m;
  m.secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : m.secs) tickers.push_back(s.ticker);
  m.panel = build_panel(store, tickers, cfg.tf);
  REQUIRE(m.panel.T() < 3);
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  fs.start();
  for (int k = 0; k < 200 && fs.status().error.empty(); ++k) std::this_thread::sleep_for(10ms);
  auto st = fs.status();
  CHECK_FALSE(st.error.empty());
  CHECK_FALSE(st.ready);
  CHECK_FALSE(st.running);
}

TEST_CASE("frame store concurrent shocks agree") {
  Market m = market();
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  fs.start();
  wait_ready(fs);
  const std::size_t node = fs.landscape(std::nullopt)->nodes.front().i;
  auto ref = fs.shock({{node, -10.0}});
  std::vector<FrameStore::ShockResult> rs(4);
  std::vector<std::thread> th;
  for (int k = 0; k < 4; ++k)
    th.emplace_back([&, k] {
      for (int i = 0; i < 3; ++i) rs[k] = fs.shock({{node, -10.0}});
    });
  for (auto& t : th) t.join();
  for (auto& r : rs) {
    CHECK(test::same_values(r.delta.dh, ref.delta.dh));
    CHECK(r.raster.z == ref.raster.z);
  }
}

TEST_CASE("frame store honours the caller's OpenMP thread count and is bit-identical across counts") {
  Market m = market();
  const int saved = omp_get_max_threads();
  struct Out {
    std::shared_ptr<const LandscapeFrame> latest;
    FrameStore::ShockResult shock;
    int threads;
  };
  auto run = [&](int threads) {
    omp_set_num_threads(threads);
    FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
    fs.start();
    wait_ready(fs);
    Out o;
    o.latest = fs.landscape(std::nullopt);
    o.shock = fs.shock({{o.latest->nodes.front().i, -10.0}});
    o.threads = fs.status().threads;
    return o;
  };
  auto one = run(1);
  auto eight = run(8);
  omp_set_num_threads(saved);
  CHECK(one.threads == 1);
  CHECK(eight.threads == 8);
  CHECK(one.latest->raster.z == eight.latest->raster.z);
  CHECK(test::same_values(one.shock.delta.dh, eight.shock.delta.dh));
  CHECK(test::same_values(one.shock.delta.dpi, eight.shock.delta.dpi));
  CHECK(one.shock.raster.z == eight.shock.raster.z);
}

TEST_CASE("frame store rejects zero max_frames") {
  Market m = market();
  CHECK_THROWS_AS(FrameStore(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 0),
                  std::invalid_argument);
}

TEST_CASE("frame store recent() returns the cached frames ending at t, oldest first") {
  Market m = market();
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  CHECK(fs.recent(std::nullopt, 3).empty());
  fs.start();
  wait_ready(fs);
  const auto times = fs.times();
  auto last3 = fs.recent(std::nullopt, 3);
  REQUIRE(last3.size() == 3);
  CHECK(last3[0]->t == times[7]);
  CHECK(last3[2]->t == times[9]);
  auto early = fs.recent(times[1], 5);
  REQUIRE(early.size() == 2);
  CHECK(early[0]->t == times[0]);
  CHECK(early[1]->t == times[1]);
  CHECK(fs.recent(m.panel.times.front(), 5).empty());
  CHECK(fs.recent(std::nullopt, 0).empty());
}

TEST_CASE("frame store skips the warm-up bars: no landscape and no clustering before bar warmup_bars") {
  Market m = market();
  REQUIRE(m.panel.T() == 120);
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 300);
  fs.start();
  wait_ready(fs);
  const auto times = fs.times();
  CHECK(LandscapeParams{}.warmup_bars == 5);
  REQUIRE(times.size() == 120 - 5);
  CHECK(times.front() == m.panel.times[5]);
  CHECK(fs.landscape(times.front())->reclustered);  // the first landscape clusters
  CHECK_FALSE(fs.landscape(times[1])->reclustered);
  CHECK(fs.landscape(times[5])->reclustered);  // then every recluster_bars frames
  // a short panel still yields its last bar
  LandscapeParams lp;
  lp.warmup_bars = 500;
  FrameStore late(m.panel, m.secs, CoreParams::money_flow(), lp, 300);
  late.start();
  wait_ready(late);
  REQUIRE(late.times().size() == 1);
  CHECK(late.times().back() == m.panel.times.back());
}
