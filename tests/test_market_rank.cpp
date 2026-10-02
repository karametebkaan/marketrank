// MarketRank concept model: the market_rank() preset, the slide-3 worked example (golden), the
// MarketRank score π·N and its heartbeat Δlog π, and the π-valued landscape.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cli/args.hpp"
#include "flux_test_util.hpp"
#include "geom/landscape.hpp"
#include "graph/markov_solver.hpp"
#include "graph/sparse_flux.hpp"
#include "graph/transition.hpp"
#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/core_pipeline.hpp"
#include "pipeline/evaluation.hpp"
#include "server/top_list.hpp"
#include "test_util.hpp"

using namespace mr;

namespace {
// Slide 3: stocks A, B, C (indices 0, 1, 2), T in dollars, row = from, column = to.
const std::vector<double> kSlide3T = {0,       500'000, 100'000,    //
                                      200'000, 0,       1'000'000,  //
                                      750'000, 400'000, 0};

// P by row-normalizing T, as a dense-to-Csr matrix (no diagonal).
Csr slide3_P() {
  Csr P;
  P.n = 3;
  P.row_ptr.push_back(0);
  for (std::size_t i = 0; i < 3; ++i) {
    double s = 0;
    for (std::size_t j = 0; j < 3; ++j) s += kSlide3T[i * 3 + j];
    for (std::size_t j = 0; j < 3; ++j)
      if (kSlide3T[i * 3 + j] != 0) {
        P.col.push_back(static_cast<std::uint32_t>(j));
        P.val.push_back(kSlide3T[i * 3 + j] / s);
        P.raw.push_back(kSlide3T[i * 3 + j]);
      }
    P.row_ptr.push_back(P.col.size());
  }
  return P;
}

// The accumulator a pipeline would hold after one bar carrying exactly T, with market_rank()'s
// slow half-life.
FluxAccumulator slide3_acc() {
  const CoreParams mr = CoreParams::market_rank();
  BarFlux bar;
  bar.rows.resize(3);
  bar.out.assign(3, 0.0);
  bar.in.assign(3, 0.0);
  for (std::uint32_t i = 0; i < 3; ++i)
    for (std::uint32_t j = 0; j < 3; ++j) {
      const double w = kSlide3T[i * 3 + j];
      if (w == 0) continue;
      bar.rows[i].push_back({j, w});
      bar.out[i] += w;
      bar.in[j] += w;
    }
  FluxAccumulator acc(3, mr.halflife_slow, mr.row_cap);
  acc.add(bar);
  return acc;
}

void check_slide3(const std::vector<double>& pi7, const std::vector<double>& pi) {
  REQUIRE(pi7.size() == 3);
  REQUIRE(pi.size() == 3);
  // (a) after exactly 7 damped iterations from the uniform start: the published values
  CHECK(std::abs(pi7[1] - 0.35980) < 5e-5);  // B
  CHECK(std::abs(pi7[2] - 0.34770) < 5e-5);  // C
  CHECK(std::abs(pi7[0] - 0.29248) < 5e-5);  // A
  // (b) converged
  CHECK(std::abs(pi[1] - 0.36016) < 1e-5);
  CHECK(std::abs(pi[2] - 0.34665) < 1e-5);
  CHECK(std::abs(pi[0] - 0.29319) < 1e-5);
  CHECK(pi[1] > pi[2]);
  CHECK(pi[2] > pi[0]);
}

Panel synthetic_panel(std::vector<Security>* secs_out = nullptr) {
  SyntheticConfig cfg;
  BarStore store(test::temp_dir("market_rank"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  if (secs_out) *secs_out = secs;
  return build_panel(store, tickers, cfg.tf);
}
}  // namespace

TEST_CASE("market_rank(): the concept model preset") {
  const CoreParams p = CoreParams::market_rank();
  CHECK(p.pressure == PressureMode::Dollar);
  CHECK(p.transition.lift == LiftMode::Off);
  CHECK(p.transition.retention == 0.0);
  CHECK(p.transition.dangling == DanglingMode::Teleport);
  CHECK(p.transition.k_out == 20);
  CHECK(p.transition.k_in == 10);
  CHECK(p.alpha == 0.85);
  CHECK(p.h_ref == HotRef::Uniform);
  CHECK(p.max_volume_ratio == 0.0);
  CHECK_FALSE(p.vol_scale);
  CHECK(p.halflife_slow >= 1e9);  // effectively cumulative over the data window
  CHECK(p.halflife_fast == 3.0);
  CHECK(p.min_dollar_volume == CoreParams{}.min_dollar_volume);
  CHECK(p.stale_bars == CoreParams{}.stale_bars);
  CHECK_NOTHROW(p.validate());
  // CoreParams{} is unchanged
  const CoreParams d;
  CHECK(d.pressure == PressureMode::Sqrt);
  CHECK(d.transition.lift == LiftMode::Excess);
  CHECK(d.transition.retention == 1.0);
  CHECK(d.transition.dangling == DanglingMode::SelfLoop);
  CHECK(d.halflife_slow == 20.0);
  CHECK(d.max_volume_ratio == 5.0);
  CHECK(CoreParams::legacy().transition.dangling == DanglingMode::SelfLoop);
  CHECK(CoreParams::money_flow().transition.dangling == DanglingMode::SelfLoop);
}

TEST_CASE("slide 3 (golden): the markov solver reproduces the A/B/C example") {
  const Csr P = slide3_P();
  const SolveResult seven = stationary(P, 0.85, {}, 0.0, 7);
  CHECK_FALSE(seven.converged);
  CHECK(seven.iterations == 7);
  const std::vector<double> uniform(3, 1.0 / 3.0);
  const std::vector<double> prop7 = propagate(P, 0.85, uniform, 7);
  for (int i = 0; i < 3; ++i) CHECK(prop7[i] == doctest::Approx(seven.pi[i]).epsilon(1e-14));
  const SolveResult conv = stationary(P, 0.85, {});
  CHECK(conv.converged);
  check_slide3(seven.pi, conv.pi);
}

TEST_CASE("slide 3 (golden): the pipeline path under market_rank() builds the same P") {
  const CoreParams mr = CoreParams::market_rank();
  const FluxAccumulator acc = slide3_acc();
  const Csr P = build_transition(acc, mr.transition);
  const Csr R = slide3_P();
  REQUIRE(P.n == 3);
  REQUIRE(P.row_ptr == R.row_ptr);
  CHECK(P.col == R.col);  // no diagonal: retention 0 emits no self-loops
  for (std::size_t e = 0; e < R.val.size(); ++e) CHECK(P.val[e] == doctest::Approx(R.val[e]).epsilon(1e-15));
  for (std::size_t e = 0; e < R.raw.size(); ++e) CHECK(P.raw[e] == doctest::Approx(R.raw[e]).epsilon(1e-12));
  check_slide3(stationary(P, mr.alpha, {}, 0.0, 7).pi, stationary(P, mr.alpha, {}).pi);
  // The dense test helper (no decay at all) gives the identical chain.
  const Csr Q = build_transition(test::acc_from_dense(kSlide3T, 3), mr.transition);
  CHECK(Q.col == R.col);
  for (std::size_t e = 0; e < R.val.size(); ++e) CHECK(Q.val[e] == doctest::Approx(R.val[e]).epsilon(1e-15));
}

TEST_CASE("retention 0 emits no diagonal; an empty active row teleports under DanglingMode::Teleport") {
  // Node 2 only receives (no outflow): a dangling node.
  const std::vector<double> F = {0, 4, 1,  //
                                 2, 0, 2,  //
                                 0, 0, 0};
  TransitionParams tp = CoreParams::market_rank().transition;
  const Csr P = build_transition(test::acc_from_dense(F, 3), tp);
  for (std::size_t i = 0; i < 3; ++i)
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) CHECK(P.col[e] != i);
  CHECK(P.row_ptr[3] == P.row_ptr[2]);  // row 2 is empty
  // An inactive row stays a self-loop (it is compacted out before the solve anyway).
  const Csr Pi = build_transition(test::acc_from_dense(F, 3), tp, {true, false, true});
  REQUIRE(Pi.row_ptr[2] - Pi.row_ptr[1] == 1);
  CHECK(Pi.col[Pi.row_ptr[1]] == 1);
  CHECK(Pi.row_ptr[3] == Pi.row_ptr[2]);  // row 2 stays empty (active, no outflow)
  // SelfLoop keeps the old behaviour for an empty row.
  tp.dangling = DanglingMode::SelfLoop;
  const Csr S = build_transition(test::acc_from_dense(F, 3), tp);
  REQUIRE(S.row_ptr[3] - S.row_ptr[2] == 1);
  CHECK(S.col[S.row_ptr[2]] == 2);
  CHECK(S.val[S.row_ptr[2]] == 1.0);
}

TEST_CASE("the solver teleports a dangling node's mass uniformly (standard PageRank)") {
  Csr P;
  P.n = 3;
  P.row_ptr = {0, 1, 3, 3};  // 0 -> 1; 1 -> 0, 2 (half each); 2 has no out-edges
  P.col = {1, 0, 2};
  P.val = {1.0, 0.5, 0.5};
  P.raw = {1, 1, 1};
  const SolveResult r = stationary(P, 0.85, {});
  CHECK(r.converged);
  CHECK(r.pi[0] == doctest::Approx(0.30319148936170215).epsilon(1e-9));
  CHECK(r.pi[1] == doctest::Approx(0.39361702127659576).epsilon(1e-9));
  CHECK(r.pi[2] == doctest::Approx(0.30319148936170215).epsilon(1e-9));
  const auto one = damped_step(P, 0.85, std::vector<double>{0, 0, 1});
  for (double x : one) CHECK(x == doctest::Approx(1.0 / 3.0));
  const auto prop = propagate(P, 0.85, std::vector<double>{0, 0, 1}, 1);
  for (double x : prop) CHECK(x == doctest::Approx(1.0 / 3.0));
}

TEST_CASE("frames carry the heartbeat: pulse = Δlog π against the previous bar's frame") {
  const Panel panel = synthetic_panel();
  CoreParams p = CoreParams::market_rank();
  CorePipeline pipe(panel.N(), p);
  const Frame f1 = pipe.step(panel, 1);
  REQUIRE(f1.pulse.size() == panel.N());
  for (double x : f1.pulse) CHECK(std::isnan(x));  // first frame: no previous π
  const Frame f2 = pipe.step(panel, 2);
  const Frame f3 = pipe.step(panel, 3);
  REQUIRE(f3.pulse.size() == panel.N());
  std::size_t finite = 0;
  for (std::size_t i = 0; i < panel.N(); ++i) {
    if (f3.active[i] && f2.active[i]) {
      REQUIRE(std::isfinite(f3.pulse[i]));
      CHECK(f3.pulse[i] == doctest::Approx(std::log(f3.pi[i]) - std::log(f2.pi[i])).epsilon(1e-12));
      ++finite;
    } else {
      CHECK(std::isnan(f3.pulse[i]));
    }
  }
  CHECK(finite > 0);
  // The score is π·N_active: 1 on average over the active nodes.
  std::size_t n_active = 0;
  double sum = 0;
  for (std::size_t i = 0; i < panel.N(); ++i)
    if (f3.active[i]) ++n_active, sum += f3.pi[i] * 1.0;
  CHECK(n_active > 0);
  CHECK(sum * static_cast<double>(n_active) / static_cast<double>(n_active) == doctest::Approx(1.0));
}

TEST_CASE("market_rank() on the planted rotation: the receiving sector has the highest MarketRank") {
  std::vector<Security> secs;
  const Panel panel = synthetic_panel(&secs);
  const Frame f = run_panel_last(panel, CoreParams::market_rank());
  CHECK(f.solve.converged);
  std::map<std::string, double> mean;
  for (std::size_t i = 0; i < secs.size(); ++i) mean[secs[i].sector] += f.pi[i] * 50.0 / 10.0;
  for (const auto& [s, m] : mean)
    if (s != "Sector1") CHECK(mean["Sector1"] > m);
  CHECK(mean["Sector1"] > 1.0);
}

TEST_CASE("landscape value Pi: height log(π·N), and π (not h) drives ordering and mountains") {
  const Panel panel = synthetic_panel();
  Frame f = run_panel_last(panel, CoreParams::money_flow());
  Frame g = f;  // same π, hotness reversed
  for (std::size_t i = 0; i < g.h.size(); ++i)
    if (std::isfinite(g.h[i])) g.h[i] = -g.h[i];
  LandscapeParams pi_p;
  pi_p.value = LandscapeValue::Pi;
  pi_p.territory = TerritoryMode::Sector;
  LandscapeParams h_p = pi_p;
  h_p.value = LandscapeValue::Hotness;
  CHECK_FALSE(same_placement(pi_p, h_p));  // the value drives ordering: placement-affecting
  const LandscapeFrame a = LandscapeBuilder(panel.N(), pi_p).build(f);
  const LandscapeFrame b = LandscapeBuilder(panel.N(), pi_p).build(g);
  const LandscapeFrame c = LandscapeBuilder(panel.N(), h_p).build(f);
  const LandscapeFrame d = LandscapeBuilder(panel.N(), h_p).build(g);
  REQUIRE(a.nodes.size() == b.nodes.size());
  const double n = static_cast<double>(a.nodes.size());
  bool any_cell_differs = false;
  for (std::size_t k = 0; k < a.nodes.size(); ++k) {
    CHECK(a.nodes[k].cell == b.nodes[k].cell);  // h is ignored under Pi
    CHECK(a.nodes[k].hdisp == doctest::Approx(std::log(a.nodes[k].pi * n)));
    CHECK(c.nodes[k].hdisp == doctest::Approx(display_height(c.nodes[k].h, HeightMode::SignedLog)));
    if (c.nodes[k].cell != d.nodes[k].cell) any_cell_differs = true;
  }
  CHECK(any_cell_differs);  // and h drives ordering under Hotness
  // One territory (no groups): a mountain, so the highest π takes the centre slot, the same slot the
  // highest h takes under Hotness.
  auto argmax = [](const LandscapeFrame& lf, auto key) {
    std::size_t best = 0;
    for (std::size_t k = 1; k < lf.nodes.size(); ++k)
      if (key(lf.nodes[k]) > key(lf.nodes[best])) best = k;
    return best;
  };
  const auto top_pi = argmax(a, [](const LandscapeNode& x) { return x.pi; });
  const auto top_h = argmax(c, [](const LandscapeNode& x) { return x.h; });
  CHECK(a.nodes[top_pi].cell == c.nodes[top_h].cell);
  // restyle keeps the value mode
  LandscapeParams pi_lin = pi_p;
  pi_lin.height = HeightMode::Linear;
  const LandscapeFrame r = restyle(a, pi_lin);
  for (std::size_t k = 0; k < a.nodes.size(); ++k) CHECK(r.nodes[k].hdisp == doctest::Approx(a.nodes[k].hdisp));
  CHECK(parse_landscape_value("pi") == LandscapeValue::Pi);
  CHECK(parse_landscape_value("hotness") == LandscapeValue::Hotness);
  CHECK(to_string(LandscapeValue::Pi) == "pi");
  CHECK_THROWS_AS(parse_landscape_value("h"), std::invalid_argument);
  CHECK(LandscapeParams{}.value == LandscapeValue::Hotness);
}

TEST_CASE("landscape nodes carry the pulse") {
  const Panel panel = synthetic_panel();
  const Frame f = run_panel_last(panel, CoreParams::market_rank());
  const LandscapeFrame lf = LandscapeBuilder(panel.N(), LandscapeParams{}).build(f);
  for (const auto& nd : lf.nodes) {
    if (std::isnan(f.pulse[nd.i])) CHECK(std::isnan(nd.pulse));
    else CHECK(nd.pulse == f.pulse[nd.i]);
  }
}

TEST_CASE("top list by π: descending π, mr = π·N, series of π·N, pulse carried") {
  auto mk = [](TimePoint t, std::vector<double> pi, std::vector<double> h) {
    auto f = std::make_shared<LandscapeFrame>();
    f->t = t;
    for (std::uint32_t i = 0; i < pi.size(); ++i) {
      LandscapeNode nd{i, static_cast<std::int32_t>(i), 0.f, 0.f, h[i], h[i], pi[i], 0.0};
      nd.pulse = 0.01 * i;
      f->nodes.push_back(nd);
    }
    return std::shared_ptr<const LandscapeFrame>(f);
  };
  auto prev = mk(1, {0.1, 0.2, 0.3, 0.4}, {3, 2, 1, 0});
  auto cur = mk(2, {0.4, 0.1, 0.2, 0.3}, {0, 3, 2, 1});
  const auto rows = top_hot({prev, cur}, 3, 2, TopBy::Pi);
  REQUIRE(rows.size() == 3);
  CHECK(rows[0].i == 0);
  CHECK(rows[1].i == 3);
  CHECK(rows[2].i == 2);
  CHECK(rows[0].mr == doctest::Approx(1.6));
  CHECK(rows[0].pulse == doctest::Approx(0.0));
  CHECK(rows[1].pulse == doctest::Approx(0.03));
  CHECK(rows[0].prev_rank == 4u);  // π 0.1 was last at the previous bar
  REQUIRE(rows[0].series.size() == 2);
  CHECK(rows[0].series[0] == doctest::Approx(0.4));
  CHECK(rows[0].series[1] == doctest::Approx(1.6));
  const auto hrows = top_hot({prev, cur}, 3, 2, TopBy::Hotness);
  CHECK(hrows[0].i == 1);
  CHECK(hrows[0].series[1] == 3.0);
  CHECK(hrows[0].mr == doctest::Approx(0.4));
  CHECK(top_hot({prev, cur}, 3, 2).front().i == 1);  // default stays hotness for the function
}

TEST_CASE("cli: market_rank() is the default for rank and serve; preset flags are exclusive") {
  CHECK(parse_cli({}).params == CoreParams::market_rank());
  CHECK(parse_cli({}).preset == "marketrank");
  CHECK(parse_cli({"--serve"}).params == CoreParams::market_rank());
  CHECK(parse_cli({"--marketrank"}).params == CoreParams::market_rank());
  CHECK(parse_cli({"--money-flow"}).params == CoreParams::money_flow());
  CHECK(parse_cli({"--money-flow"}).preset == "money-flow");
  CHECK(parse_cli({"--serve", "--money-flow"}).params == CoreParams::money_flow());
  CHECK(parse_cli({"--legacy"}).params == CoreParams::legacy());
  CHECK(parse_cli({"--serve", "--legacy"}).params == CoreParams::legacy());
  CHECK_THROWS_AS(parse_cli({"--marketrank", "--legacy"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--marketrank", "--money-flow"}), std::invalid_argument);
  const CliArgs k = parse_cli({"--marketrank", "--k-in", "5"});
  CHECK(k.params.transition.k_in == 5);
  CHECK(k.params.transition.retention == 0.0);
  CHECK(parse_cli({}).rank_by == RankBy::Pi);
  CHECK(parse_cli({"--rank-by", "hotness"}).rank_by == RankBy::Hotness);
  CHECK(parse_cli({"--rank-by", "pi"}).rank_by == RankBy::Pi);
  CHECK_THROWS_AS(parse_cli({"--rank-by", "h"}), std::invalid_argument);
  const std::string u = cli_usage();
  CHECK(u.find("--marketrank") != std::string::npos);
  CHECK(u.find("--rank-by pi|hotness") != std::string::npos);
}

TEST_CASE("evaluation grid includes the marketrank preset") {
  const auto g = evaluation_grid();
  auto it = std::find_if(g.begin(), g.end(), [](const EvalConfig& c) { return c.name == "marketrank"; });
  REQUIRE(it != g.end());
  CHECK(it->params == CoreParams::market_rank());
}

TEST_CASE("floor share counts the teleport floor including the dangling nodes' uniform share") {
  Frame f;
  f.active = {true, true, true, true};
  f.P.n = 4;
  f.P.row_ptr = {0, 1, 2, 2, 3};  // 0 -> 1, 1 -> 0, 2 dangling, 3 -> 1; nobody sends to 2 or 3
  f.P.col = {1, 0, 1};
  f.P.val = {1, 1, 1};
  f.P.raw = {1, 1, 1};
  f.pi = stationary(f.P, 0.85, {}).pi;
  CHECK(f.pi[2] == doctest::Approx(f.pi[3]));
  CHECK(f.pi[2] > 0.15 / 4 * (1 + 1e-3));  // above the plain (1 - alpha)/N floor
  CHECK(floor_share(f, 0.85) == doctest::Approx(0.5));
}
