#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/core_pipeline.hpp"
#include "test_util.hpp"

using namespace mr;

namespace {
// The small hand-built panels trade a few thousand dollars per bar; the $1M liquidity floor would
// deactivate every node, and these tests are about other behaviour.
CoreParams no_floor() {
  CoreParams p;
  p.min_dollar_volume = 0;
  return p;
}
}  // namespace

namespace {
void check_rotation(const CoreParams& params) {
  SyntheticConfig cfg;
  BarStore store(test::temp_dir("pipeline"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  Panel panel = build_panel(store, tickers, cfg.tf);
  Frame f = run_panel_last(panel, params);
  CHECK(f.solve.converged);
  REQUIRE(f.h.size() == 50);
  REQUIRE(f.forecasts.size() == 3);
  CHECK(f.forecasts[2].k == 8);
  CHECK(f.t == panel.times.back());
  std::map<std::string, double> sector_mean;
  for (std::size_t i = 0; i < secs.size(); ++i) sector_mean[secs[i].sector] += f.h[i] / 10.0;
  for (const auto& [sector, mean] : sector_mean) {
    INFO(sector << " mean h " << mean);
    if (sector != "Sector1") CHECK(sector_mean["Sector1"] > mean);
  }
  CHECK(sector_mean["Sector1"] > 0);
  CHECK(sector_mean["Sector0"] < sector_mean["Sector1"]);
}
}  // namespace

TEST_CASE("planted rotation makes the receiving sector the top hill (defaults)") {
  check_rotation(CoreParams{});
}

TEST_CASE("planted rotation makes the receiving sector the top hill (legacy)") {
  check_rotation(CoreParams::legacy());
}

TEST_CASE("net-flow hotness on the planted rotation is bounded and favours the receiving sector") {
  SyntheticConfig cfg;
  BarStore store(test::temp_dir("pipeline_netflow"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  Panel panel = build_panel(store, tickers, cfg.tf);
  CoreParams params;
  params.h_ref = HotRef::NetFlow;
  Frame f = run_panel_last(panel, params);
  REQUIRE(f.h.size() == secs.size());
  std::map<std::string, double> sector_mean;
  for (std::size_t i = 0; i < secs.size(); ++i) {
    if (!f.active[i]) continue;
    CHECK(std::isfinite(f.h[i]));
    CHECK(f.h[i] > -1.0);
    CHECK(f.h[i] < 1.0);
    sector_mean[secs[i].sector] += f.h[i] / 10.0;
  }
  for (const auto& [sector, mean] : sector_mean) INFO(sector << " mean h " << mean);
  CHECK(sector_mean["Sector1"] > sector_mean["Sector0"]);
}

TEST_CASE("pipeline requires at least two bars") {
  Panel p;
  p.times = {100};
  p.tickers = {"A"};
  p.close = {1};
  p.volume = {1};
  p.vwap = {1};
  CHECK_THROWS_AS(run_panel_last(p, CoreParams{}), std::runtime_error);
}

namespace {
Panel small_panel(bool dead_node) {
  Panel p;
  const std::size_t T = 6, N = 4;
  p.tickers = {"A", "B", "DEAD", "C"};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t t = 0; t < T; ++t) {
    p.times.push_back(static_cast<TimePoint>(100 + t));
    for (std::size_t i = 0; i < N; ++i) {
      const bool dead = dead_node && i == 2;
      const double c = 10.0 + static_cast<double>(i) + 0.5 * static_cast<double>(t) *
                                                           (i % 2 == 0 ? 1.0 : -0.3);
      p.close.push_back(dead ? nan : c);
      p.volume.push_back(dead ? nan : 1000.0 * (1.0 + static_cast<double>(i)));
      p.vwap.push_back(dead ? nan : c);
    }
  }
  return p;
}
}  // namespace

TEST_CASE("a ticker with no bars is inactive and takes no part in the solve") {
  Panel p = small_panel(true);
  Frame f = run_panel_last(p, no_floor());
  REQUIRE(f.active.size() == 4);
  CHECK(f.active == std::vector<bool>{true, true, false, true});
  CHECK(f.pi[2] == 0.0);
  CHECK(std::isnan(f.h[2]));
  for (const auto& fc : f.forecasts) CHECK(std::isnan(fc.score[2]));
  double sum = 0;
  for (std::size_t i = 0; i < 4; ++i)
    if (i != 2) sum += f.pi[i];
  CHECK(sum == doctest::Approx(1.0));
  double hsum = 0;
  for (std::size_t i = 0; i < 4; ++i)
    if (i != 2) hsum += f.h[i];
  CHECK(hsum == doctest::Approx(0.0).epsilon(1e-9));  // h = N_active * pi - 1
  // dead node: self-loop only, nothing flows into it
  CHECK(f.P.row_ptr[3] - f.P.row_ptr[2] == 1);
  CHECK(f.P.raw[f.P.row_ptr[2]] == 0.0);
  for (std::size_t e = 0; e < f.P.col.size(); ++e)
    if (f.P.col[e] == 2) CHECK(e == f.P.row_ptr[2]);
}

TEST_CASE("frame carries raw pruned weights and the fast matrix") {
  Panel p = small_panel(false);
  Frame f = run_panel_last(p, CoreParams::legacy());
  REQUIRE(f.P.raw.size() == f.P.val.size());
  CHECK(f.P_fast.n == 4);
  CHECK(f.P_fast.raw.size() == f.P_fast.val.size());
  CHECK(f.active == std::vector<bool>(4, true));
  for (std::size_t i = 0; i < 4; ++i) {
    double s = 0;
    for (auto e = f.P.row_ptr[i]; e < f.P.row_ptr[i + 1]; ++e) s += f.P.raw[e];
    if (s > 0)
      for (auto e = f.P.row_ptr[i]; e < f.P.row_ptr[i + 1]; ++e)
        CHECK(f.P.val[e] == doctest::Approx(f.P.raw[e] / s));
  }
}

TEST_CASE("step rejects out-of-range t and mismatched panels") {
  Panel p = small_panel(false);
  CorePipeline pipe(4, no_floor());
  CHECK_THROWS_AS(pipe.step(p, 0), std::invalid_argument);
  CHECK_THROWS_AS(pipe.step(p, p.T()), std::invalid_argument);
  CorePipeline wrong(5, no_floor());
  CHECK_THROWS_AS(wrong.step(p, 1), std::invalid_argument);
  CHECK_NOTHROW(pipe.step(p, 1));
}

TEST_CASE("hotness references size and longrun give finite relative hotness") {
  for (HotRef ref : {HotRef::Size, HotRef::LongRun}) {
    CoreParams p = no_floor();
    p.h_ref = ref;
    Frame f = run_panel_last(small_panel(false), p);
    CHECK(f.solve.converged);
    for (double h : f.h) CHECK(std::isfinite(h));
  }
}

TEST_CASE("CoreParams::validate rejects bad settings") {
  CoreParams a;
  a.alpha = 0;
  CHECK_THROWS_AS(a.validate(), std::invalid_argument);
  CoreParams b;
  b.flux.lambda = 1.5;
  CHECK_THROWS_AS(b.validate(), std::invalid_argument);
  CoreParams c;
  c.horizons.clear();
  CHECK_THROWS_AS(c.validate(), std::invalid_argument);
  CoreParams d;
  d.transition.k_out = 0;
  CHECK_THROWS_AS(d.validate(), std::invalid_argument);
  CoreParams e;
  e.flux.sink_candidates = 10;
  e.flux.sinks_per_source = 20;
  CHECK_THROWS_AS(e.validate(), std::invalid_argument);
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CoreParams f;
  f.transition.retention = inf;
  CHECK_THROWS_AS(f.validate(), std::invalid_argument);
  CoreParams g;
  g.halflife_slow = nan;
  CHECK_THROWS_AS(g.validate(), std::invalid_argument);
  CoreParams h;
  h.alpha = nan;
  CHECK_THROWS_AS(h.validate(), std::invalid_argument);
  CoreParams l;
  l.flux.lambda = nan;
  CHECK_THROWS_AS(l.validate(), std::invalid_argument);
  CoreParams m;
  m.halflife_fast = -inf;
  CHECK_THROWS_AS(m.validate(), std::invalid_argument);
  CoreParams ok;
  ok.halflife_long = inf;  // +infinity = no decay is allowed
  CHECK_NOTHROW(ok.validate());
  CHECK_NOTHROW(CoreParams{}.validate());
  CHECK_NOTHROW(CoreParams::legacy().validate());
  CHECK_THROWS_AS(CorePipeline(4, a), std::invalid_argument);
}

TEST_CASE("size reference is neutral for a ticker with no volume history yet") {
  Panel p;
  const std::size_t T = 8, N = 4;
  p.tickers = {"A", "B", "C", "D"};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t t = 0; t < T; ++t) {
    p.times.push_back(static_cast<TimePoint>(100 + t));
    for (std::size_t i = 0; i < N; ++i) {
      const bool missing = i == 3 && t < 6;
      const double c = 10.0 + static_cast<double>(i) +
                       0.5 * static_cast<double>(t) * (i % 2 == 0 ? 1.0 : -0.3);
      p.close.push_back(missing ? nan : c);
      p.volume.push_back(missing ? nan : (i == 3 ? 0.0 : 1000.0 * (1.0 + static_cast<double>(i))));
      p.vwap.push_back(missing ? nan : c);
    }
  }
  CoreParams params = no_floor();
  params.h_ref = HotRef::Size;
  Frame f = run_panel_last(p, params);
  CHECK(f.active[3]);
  for (double h : f.h) CHECK(std::isfinite(h));
  CHECK(std::abs(f.h[3]) < 50);
}

namespace {
// N tickers, T bars; present(t, i) says whether ticker i has a bar at t.
template <class Present>
Panel presence_panel(std::size_t T, std::size_t N, Present present) {
  Panel p;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t i = 0; i < N; ++i) p.tickers.push_back("T" + std::to_string(i));
  for (std::size_t t = 0; t < T; ++t) {
    p.times.push_back(static_cast<TimePoint>(100 + t));
    for (std::size_t i = 0; i < N; ++i) {
      const bool on = present(t, i);
      const double c = 10.0 + static_cast<double>(i) +
                       0.5 * static_cast<double>(t) * (i % 2 == 0 ? 1.0 : -0.3) +
                       0.2 * static_cast<double>((t * 7 + i * 3) % 5);
      p.close.push_back(on ? c : nan);
      p.volume.push_back(on ? 1000.0 * (1.0 + static_cast<double>(i)) : nan);
      p.vwap.push_back(on ? c : nan);
    }
  }
  return p;
}
}  // namespace

TEST_CASE("frames are causal") {
  SyntheticConfig cfg;
  cfg.bars = 80;
  cfg.rotation_start = 20;
  BarStore store(test::temp_dir("causal"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  Panel panel = build_panel(store, tickers, cfg.tf);
  const std::size_t k = 40;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  // A late listing (ticker 6 starts trading after bar k) and a delisting (ticker 9 stops at k+2).
  for (std::size_t t = 0; t < panel.T(); ++t)
    for (std::size_t i : {std::size_t{6}, std::size_t{9}}) {
      if ((i == 6 && t <= k + 5) || (i == 9 && t > k + 2)) {
        panel.close[panel.idx(t, i)] = nan;
        panel.volume[panel.idx(t, i)] = nan;
        panel.vwap[panel.idx(t, i)] = nan;
      }
    }
  Panel future = panel;
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> u(0.5, 2.0);
  for (std::size_t t = k + 1; t < future.T(); ++t)
    for (std::size_t i = 0; i < future.N(); ++i) {
      const std::size_t c = future.idx(t, i);
      if (i % 3 == 0) {
        future.close[c] = future.volume[c] = future.vwap[c] = nan;
      } else {
        future.close[c] *= u(rng);
        future.volume[c] *= u(rng);
        future.vwap[c] = future.close[c];
      }
    }
  for (const CoreParams& params : {CoreParams{}, CoreParams::legacy()}) {
    CoreParams lr = params;
    lr.h_ref = HotRef::LongRun;
    for (const CoreParams& p : {params, lr}) {
      CorePipeline a(panel.N(), p), b(future.N(), p);
      Frame fa, fb;
      for (std::size_t t = 1; t <= k; ++t) {
        fa = a.step(panel, t);
        fb = b.step(future, t);
      }
      CHECK(fa.active == fb.active);
      CHECK(test::same_values(fa.pi, fb.pi));
      CHECK(test::same_values(fa.h, fb.h));
      CHECK(test::same_values(fa.forecasts.front().score, fb.forecasts.front().score));
      CHECK_FALSE(fa.active[6]);
    }
  }
}

TEST_CASE("a late listing becomes active only once it trades") {
  // Ticker 3 trades from bar 6; ticker 2 never trades.
  Panel p = presence_panel(12, 5, [](std::size_t t, std::size_t i) {
    return i == 2 ? false : (i != 3 || t >= 6);
  });
  CorePipeline pipe(p.N(), no_floor());
  for (std::size_t t = 1; t < p.T(); ++t) {
    const Frame f = pipe.step(p, t);
    INFO("t = " << t);
    CHECK(f.active[3] == (t >= 6));
    CHECK_FALSE(f.active[2]);
    CHECK(f.active[0]);
    CHECK(f.solve.converged);
    double sum = 0;
    for (double x : f.pi) sum += x;
    CHECK(sum == doctest::Approx(1.0));
    if (!f.active[3]) {
      CHECK(f.pi[3] == 0.0);
      CHECK(std::isnan(f.h[3]));
    }
  }
}

TEST_CASE("a stale ticker drops out after stale_bars") {
  // Ticker 1 trades at bars 0..3 only.
  Panel p = presence_panel(10, 4, [](std::size_t t, std::size_t i) { return i != 1 || t <= 3; });
  CoreParams params = no_floor();
  params.stale_bars = 2;
  CorePipeline pipe(p.N(), params);
  for (std::size_t t = 1; t < p.T(); ++t) {
    const Frame f = pipe.step(p, t);
    INFO("t = " << t);
    CHECK(f.active[1] == (t <= 5));
    double sum = 0;
    for (double x : f.pi) sum += x;
    CHECK(sum == doctest::Approx(1.0));
  }
  CoreParams bad;
  bad.stale_bars = 0;
  CHECK_THROWS_AS(bad.validate(), std::invalid_argument);
}

TEST_CASE("affinity at bar t uses only the returns before t") {
  // At t = 2 the window before the bar holds one return, too few for a correlation, so the
  // affinity must be neutral and the frame must equal the lambda = 0 frame exactly.
  Panel p = presence_panel(6, 6, [](std::size_t, std::size_t) { return true; });
  CoreParams with = no_floor();
  with.flux.lambda = 1.0;
  CoreParams without = with;
  without.flux.lambda = 0.0;
  CorePipeline a(p.N(), with), b(p.N(), without);
  Frame fa, fb;
  for (std::size_t t = 1; t <= 2; ++t) {
    fa = a.step(p, t);
    fb = b.step(p, t);
  }
  CHECK(fa.P.col == fb.P.col);
  CHECK(fa.P.raw == fb.P.raw);
  CHECK(test::same_values(fa.pi, fb.pi));
  // One bar later the window holds two returns and the affinity takes effect.
  fa = a.step(p, 3);
  fb = b.step(p, 3);
  CHECK_FALSE(fa.P.raw == fb.P.raw);
}

TEST_CASE("the frame surfaces the long-run solve") {
  CoreParams p = no_floor();
  p.h_ref = HotRef::LongRun;
  const Frame f = run_panel_last(small_panel(false), p);
  CHECK(f.solve_long.converged);
  CHECK(f.solve_long.iterations > 0);
  const Frame u = run_panel_last(small_panel(false), no_floor());
  CHECK_FALSE(u.solve_long.converged);
  CHECK(u.solve_long.pi.empty());
}

namespace {
// Node 0 trades about `low` dollars per bar until bar `rise`, then `high`; nodes 1.. trade ~1e8.
Panel liquidity_panel(std::size_t T, std::size_t N, double low, double high, std::size_t rise) {
  Panel p = presence_panel(T, N, [](std::size_t, std::size_t) { return true; });
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t i = 0; i < N; ++i) {
      const double c = p.close[p.idx(t, i)];
      const double dv = i == 0 ? (t < rise ? low : high) : 1e8;
      p.volume[p.idx(t, i)] = dv / c;
      p.vwap[p.idx(t, i)] = c;
    }
  return p;
}
}  // namespace

TEST_CASE("liquidity floor deactivates an illiquid node; min_dollar_volume = 0 keeps it") {
  Panel p = liquidity_panel(30, 5, 1e4, 1e4, 1000);
  CoreParams off;
  off.min_dollar_volume = 0;
  const Frame fd = run_panel_last(p, CoreParams{});
  const Frame fo = run_panel_last(p, off);
  CHECK_FALSE(fd.active[0]);
  CHECK(fd.pi[0] == 0.0);
  CHECK(std::isnan(fd.h[0]));
  CHECK(fd.active[1]);
  CHECK(fo.active[0]);
}

TEST_CASE("liquidity floor is causal: a node is inactive until its median clears the floor") {
  // Dollar volume jumps from 1e4 to 1e8 at bar 20. The trailing 20-bar median (incl. bar t) only
  // exceeds 1e6 once half the window is high: first at t = 29
  // (10 low + 10 high bars: the even-count median is their mean).
  Panel p = liquidity_panel(40, 5, 1e4, 1e8, 20);
  CorePipeline pipe(p.N(), CoreParams{});
  for (std::size_t t = 1; t < p.T(); ++t) {
    const Frame f = pipe.step(p, t);
    INFO("t = " << t);
    CHECK(f.active[0] == (t >= 29));
  }
  // Frames before the jump do not depend on what happens after it.
  Panel q = liquidity_panel(40, 5, 1e4, 1e4, 20);
  CorePipeline pa(p.N(), CoreParams{}), pb(q.N(), CoreParams{});
  for (std::size_t t = 1; t < 20; ++t) {
    const Frame fa = pa.step(p, t), fb = pb.step(q, t);
    INFO("t = " << t);
    CHECK(fa.active == fb.active);
    CHECK(test::same_values(fa.pi, fb.pi));
    CHECK(test::same_values(fa.h, fb.h));
  }
}

TEST_CASE("a node below the liquidity floor leaves the active nodes' frame untouched") {
  const std::size_t T = 30;
  Panel base = liquidity_panel(T, 5, 1e8, 1e8, 1000);
  Panel ext = liquidity_panel(T, 6, 1e8, 1e8, 1000);
  // Node 5: ~1e4 dollars per bar, large volatile returns.
  for (std::size_t t = 0; t < T; ++t) {
    const double c = 10.0 * (1.0 + 0.3 * ((t % 2) ? 1.0 : -1.0)) + static_cast<double>(t % 3);
    ext.close[ext.idx(t, 5)] = c;
    ext.vwap[ext.idx(t, 5)] = c;
    ext.volume[ext.idx(t, 5)] = 1e4 / c;
  }
  CorePipeline pa(base.N(), CoreParams{}), pb(ext.N(), CoreParams{});
  for (std::size_t t = 1; t < T; ++t) {
    const Frame fa = pa.step(base, t), fb = pb.step(ext, t);
    INFO("t = " << t);
    CHECK_FALSE(fb.active[5]);
    CHECK(test::same_values(fa.pi, std::vector<double>(fb.pi.begin(), fb.pi.begin() + 5)));
    CHECK(test::same_values(fa.h, std::vector<double>(fb.h.begin(), fb.h.begin() + 5)));
  }
}

TEST_CASE("validate rejects bad liquidity floor and volume cap") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (double bad : {-1.0, nan, inf}) {
    CoreParams a;
    a.min_dollar_volume = bad;
    CHECK_THROWS_AS(a.validate(), std::invalid_argument);
    CoreParams b;
    b.max_volume_ratio = bad;
    CHECK_THROWS_AS(b.validate(), std::invalid_argument);
  }
  CHECK(CoreParams{}.pressure == PressureMode::Sqrt);
  CHECK(CoreParams{}.min_dollar_volume == 1e6);
  CHECK(CoreParams{}.max_volume_ratio == 5.0);
  CHECK(CoreParams::legacy().min_dollar_volume == 0);
  CHECK(CoreParams::legacy().max_volume_ratio == 0);
}
