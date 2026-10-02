#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "walkforward/backtest.hpp"
using namespace mr;

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::size_t kT = 60;

Panel blank_panel(std::vector<std::string> tickers) {
  Panel p;
  p.tickers = std::move(tickers);
  for (std::size_t t = 0; t < kT; ++t) p.times.push_back(static_cast<TimePoint>(19000 + t) * 86400);
  for (auto* v : {&p.open, &p.high, &p.low, &p.close, &p.volume, &p.vwap}) v->assign(kT * p.N(), 1.0);
  return p;
}

// A: open 100+t, close open+0.5. B: open 50*1.01^t, close open*1.002. C: open=close 20+0.1t. D: flat 10.
Panel make_panel() {
  Panel p = blank_panel({"A", "B", "C", "D"});
  for (std::size_t t = 0; t < kT; ++t) {
    const double td = static_cast<double>(t);
    const double oa = 100.0 + td, ob = 50.0 * std::pow(1.01, td), oc = 20.0 + 0.1 * td;
    p.open[p.idx(t, 0)] = oa;
    p.close[p.idx(t, 0)] = oa + 0.5;
    p.open[p.idx(t, 1)] = ob;
    p.close[p.idx(t, 1)] = ob * 1.002;
    p.open[p.idx(t, 2)] = oc;
    p.close[p.idx(t, 2)] = oc;
    p.open[p.idx(t, 3)] = 10.0;
    p.close[p.idx(t, 3)] = 10.0;
  }
  return p;
}

// X: open = close = 100*1.01^t (+1% open to open); Y, Z, W flat at 10.
Panel make_edge_panel() {
  Panel p = blank_panel({"X", "Y", "Z", "W"});
  for (std::size_t t = 0; t < kT; ++t) {
    const double ox = 100.0 * std::pow(1.01, static_cast<double>(t));
    p.open[p.idx(t, 0)] = ox;
    p.close[p.idx(t, 0)] = ox;
    for (std::size_t i = 1; i < 4; ++i) p.open[p.idx(t, i)] = p.close[p.idx(t, i)] = 10.0;
  }
  return p;
}

double O(const Panel& p, std::size_t t, std::size_t i) { return p.open[p.idx(t, i)]; }
double C(const Panel& p, std::size_t t, std::size_t i) { return p.close[p.idx(t, i)]; }

void check_rel(double got, double want, double tol = 1e-12) {
  CHECK(std::abs(got - want) <= tol * std::max(1.0, std::abs(want)));
}

// Strategy that starts 100% in A (empty score at date 0) and moves 100% into B at date 10.
BacktestParams a_to_b_params() {
  BacktestParams bp;
  bp.base = {{"A", 1.0}};
  bp.tilt = 1.0;
  bp.k = 1;
  bp.max_name_tilt = 1.0;
  bp.cost_bps = 10;
  bp.max_turnover = 1.0;
  return bp;
}
std::vector<Decision> a_to_b_decisions(std::size_t target_index = 1) {
  std::vector<double> s(4, kNaN);
  s[target_index] = 1.0;
  return {Decision{0, {}, {}}, Decision{10, s, std::vector<bool>(4, true)}};
}
}  // namespace

TEST_CASE("wf backtest: target weights") {
  const Panel p = make_panel();
  BacktestParams bp;
  bp.base = {{"A", 0.5}, {"B", 0.3}};
  bp.tilt = 0.2;
  bp.k = 2;
  bp.max_name_tilt = 0.05;  // per name min(1/2, 0.05/0.2) = 0.25 of the tilt = 0.05
  const Decision d{5, {3.0, 3.0, kNaN, 9.0}, {true, true, true, false}};  // D ineligible, C no score
  auto w = target_weights(p, bp, d);
  REQUIRE(w.size() == 4);
  check_rel(w[0], 0.8 * 0.5 + 0.05);
  check_rel(w[1], 0.8 * 0.3 + 0.05);
  CHECK(w[2] == 0.0);
  CHECK(w[3] == 0.0);
  bp.k = 1;  // tie on score: lower index wins
  w = target_weights(p, bp, d);
  check_rel(w[0], 0.8 * 0.5 + 0.2 * 0.25);
  check_rel(w[1], 0.8 * 0.3);
  w = target_weights(p, bp, Decision{5, {}, {}});
  CHECK(w[0] == 0.5);
  CHECK(w[1] == 0.3);
  bp.base = {{"ZZZ", 1.0}};
  CHECK_THROWS_AS(target_weights(p, bp, d), std::invalid_argument);
}

TEST_CASE("wf backtest (a): buy-and-hold equals the analytic value") {
  const Panel p = make_panel();
  BacktestParams bp;
  bp.base = {{"A", 0.5}, {"B", 0.5}};
  const std::vector<Decision> ds = {{0, {}, {}}, {20, {0, 1, 2, 3}, {}}, {40, {}, {}}};
  for (double bps : {0.0, 10.0}) {
    bp.cost_bps = bps;
    const EquityCurve c = simulate(p, bp, ds, BenchKind::BuyHoldBase);
    REQUIRE(c.t.size() == kT - 1);
    CHECK(c.t.front() == p.times[1]);
    REQUIRE(c.value.size() == kT - 1);
    const double keep = 1.0 - bps * 1e-4;  // initial funding trades notional 1.0
    const double sa = 0.5 * keep / O(p, 1, 0), sb = 0.5 * keep / O(p, 1, 1);
    for (std::size_t t = 1; t < kT; ++t) check_rel(c.value[t - 1], sa * C(p, t, 0) + sb * C(p, t, 1));
    check_rel(c.costs, bps * 1e-4);
    CHECK(c.turnover == 0.0);
    CHECK(c.trades_csv.size() == 2);
  }
}

TEST_CASE("wf backtest (b): cost of a full A->B move") {
  const Panel p = make_panel();
  const EquityCurve c = simulate(p, a_to_b_params(), a_to_b_decisions());
  const double sa = 0.999 / O(p, 1, 0);
  const double V = sa * O(p, 11, 0);  // value at the open of the execution day
  const double cost = 0.002 * V;      // 10e-4 * (|dA| + |dB|) = 10e-4 * 2V
  check_rel(c.costs - 0.001, cost);
  check_rel(c.turnover, 1.0);
  const double sb = (V - cost) / O(p, 11, 1);  // post-trade value V - cost, all in B, cash 0
  check_rel(sb * O(p, 11, 1), V - cost);
  check_rel(c.value[10], sb * C(p, 11, 1));  // close of bar 11
  check_rel(c.value.back(), sb * C(p, kT - 1, 1));
  CHECK(c.trades_csv.size() == 3);
}

TEST_CASE("wf backtest (c): turnover cap halves the move") {
  const Panel p = make_panel();
  BacktestParams bp = a_to_b_params();
  bp.max_turnover = 0.5;
  const EquityCurve c = simulate(p, bp, a_to_b_decisions());
  const double sa0 = 0.999 / O(p, 1, 0);
  const double V = sa0 * O(p, 11, 0);
  const double cost = 0.001 * V;  // half the notional of (b)
  check_rel(c.costs - 0.001, cost);
  check_rel(c.turnover, 0.5);
  const double f = (V - cost) / V;
  const double sa = 0.5 * V * f / O(p, 11, 0), sb = 0.5 * V * f / O(p, 11, 1);
  check_rel(c.value[10], sa * C(p, 11, 0) + sb * C(p, 11, 1));
  check_rel(c.value.back(), sa * C(p, kT - 1, 0) + sb * C(p, kT - 1, 1));
}

TEST_CASE("wf backtest (d): a stock with no open keeps its shares") {
  Panel p = make_panel();
  p.open[p.idx(11, 0)] = kNaN;  // A cannot trade on the execution day
  BacktestParams bp = a_to_b_params();
  bp.base = {{"A", 0.5}, {"B", 0.5}};
  const EquityCurve c = simulate(p, bp, a_to_b_decisions(2));  // target 100% C
  const double sa = 0.5 * 0.999 / O(p, 1, 0), sb = 0.5 * 0.999 / O(p, 1, 1);
  const double V = sa * C(p, 10, 0) + sb * O(p, 11, 1);  // A valued at its last finite price
  const double sold = sb * O(p, 11, 1);                   // only B's proceeds can fund C
  const double cost = 0.001 * 2 * sold;
  check_rel(c.costs - 0.001, cost);
  check_rel(c.turnover, sold / V);
  const double sc = (sold - cost) / O(p, 11, 2);
  check_rel(c.value[10], sa * C(p, 11, 0) + sc * C(p, 11, 2));
  check_rel(c.value.back(), sa * C(p, kT - 1, 0) + sc * C(p, kT - 1, 2));
  for (const auto& row : c.trades_csv) CHECK(row.rfind(std::to_string(p.times[11]) + ",A,", 0) != 0);
  CHECK(c.trades_csv.size() == 4);  // initial A, B; then B sell, C buy
}

TEST_CASE("wf backtest (e): planted edge beats buy-and-hold; random scores do not") {
  const Panel p = make_edge_panel();
  BacktestParams bp;
  bp.base = {{"Y", 1.0}};
  bp.k = 1;
  std::vector<Decision> edge, rnd;
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (std::size_t d = 0; d + 1 < kT; d += 10) {
    edge.push_back({d, {4.0, 1.0, 2.0, 3.0}, std::vector<bool>(4, true)});
    rnd.push_back({d, {kNaN, u(rng), u(rng), u(rng)}, std::vector<bool>(4, true)});  // X carries the edge
  }
  const EquityCurve bench = simulate(p, bp, edge, BenchKind::BuyHoldBase);
  check_rel(bench.value.back(), 0.999);
  const EquityCurve se = simulate(p, bp, edge);
  CHECK(se.value.back() > bench.value.back());
  const EquityCurve sr = simulate(p, bp, rnd);
  CHECK(sr.value.back() - bench.value.back() <= sr.costs + 1e-12);
  CHECK(sr.value.back() <= 1.0 - sr.costs + 1e-12);  // flat names: no free lunch
}

TEST_CASE("wf backtest (f): deterministic") {
  const Panel p = make_edge_panel();
  BacktestParams bp;
  bp.base = {{"Y", 0.6}, {"Z", 0.4}};
  std::vector<Decision> ds;
  for (std::size_t d = 3; d + 1 < kT; d += 7) ds.push_back({d, {4.0, 1.0, 2.0, 3.0}, {}});
  const EquityCurve a = simulate(p, bp, ds), b = simulate(p, bp, ds);
  CHECK(a.t == b.t);
  CHECK(a.value == b.value);
  CHECK(a.costs == b.costs);
  CHECK(a.turnover == b.turnover);
  CHECK(a.trades_csv == b.trades_csv);
}

TEST_CASE("wf backtest: rebalanced base and single benchmarks") {
  const Panel p = make_panel();
  BacktestParams bp;
  bp.base = {{"A", 0.5}, {"B", 0.5}};
  bp.cost_bps = 0;
  const std::vector<Decision> ds = {{0, {}, {}}, {20, {}, {}}};
  const EquityCurve r = simulate(p, bp, ds, BenchKind::RebalancedBase);
  const double sa = 0.5 / O(p, 1, 0), sb = 0.5 / O(p, 1, 1);
  const double V = sa * O(p, 21, 0) + sb * O(p, 21, 1);
  const double ra = 0.5 * V / O(p, 21, 0), rb = 0.5 * V / O(p, 21, 1);
  check_rel(r.value[19], sa * C(p, 20, 0) + sb * C(p, 20, 1));
  check_rel(r.value.back(), ra * C(p, kT - 1, 0) + rb * C(p, kT - 1, 1));
  CHECK(r.trades_csv.size() == 4);
  bp.cost_bps = 10;
  const EquityCurve s = simulate(p, bp, ds, BenchKind::Single, "D");
  for (double v : s.value) check_rel(v, 0.999);
  CHECK(s.trades_csv.size() == 1);
  CHECK_THROWS_AS(simulate(p, bp, ds, BenchKind::Single, "nope"), std::invalid_argument);
  CHECK(simulate(p, bp, {}).value.empty());
}

TEST_CASE("wf backtest: rebalanced base with costs and a binding turnover cap") {
  const Panel p = make_panel();
  BacktestParams bp;
  bp.base = {{"A", 0.5}, {"B", 0.5}};
  bp.cost_bps = 10;
  bp.max_turnover = 0.001;  // binds: the drift from 50/50 by bar 21 is larger
  const std::vector<Decision> ds = {{0, {}, {}}, {20, {}, {}}};
  const EquityCurve r = simulate(p, bp, ds, BenchKind::RebalancedBase);
  // Initial funding (exempt from the cap): gross 1, cost 0.001, positions scaled by 0.999.
  const double sa = 0.5 * 0.999 / O(p, 1, 0), sb = 0.5 * 0.999 / O(p, 1, 1);
  const double V = sa * O(p, 21, 0) + sb * O(p, 21, 1);
  const double da = 0.5 * V - sa * O(p, 21, 0), db = 0.5 * V - sb * O(p, 21, 1);
  const double raw = (std::abs(da) + std::abs(db)) / (2 * V);
  REQUIRE(raw > bp.max_turnover);
  const double s = bp.max_turnover / raw;
  const double gross = s * (std::abs(da) + std::abs(db)), cost = 1e-3 * gross;
  check_rel(r.costs, 0.001 + cost);
  check_rel(r.turnover, bp.max_turnover);
  const double f = (V - cost) / V;
  const double na = (sa * O(p, 21, 0) + s * da) * f, nb = (sb * O(p, 21, 1) + s * db) * f;
  check_rel(r.value[19], sa * C(p, 20, 0) + sb * C(p, 20, 1));
  check_rel(r.value[20], na / O(p, 21, 0) * C(p, 21, 0) + nb / O(p, 21, 1) * C(p, 21, 1));
  check_rel(r.value.back(), na / O(p, 21, 0) * C(p, kT - 1, 0) + nb / O(p, 21, 1) * C(p, kT - 1, 1));
}

TEST_CASE("wf backtest: no look-ahead - bars after an execution open do not change the curve or trades before it") {
  const Panel p = make_panel();
  BacktestParams bp;
  bp.base = {{"A", 0.4}, {"D", 0.4}};
  bp.k = 2;
  bp.max_turnover = 0.3;
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<Decision> ds;
  for (std::size_t d = 2; d + 1 < kT; d += 5) ds.push_back({d, {u(rng), u(rng), u(rng), u(rng)}, {}});
  const EquityCurve base = simulate(p, bp, ds);
  const std::size_t first = ds.front().date + 1;
  auto trades_through = [&](const EquityCurve& c, std::size_t bar) {
    std::vector<std::string> out;
    for (const auto& row : c.trades_csv)
      if (std::stoll(row.substr(0, row.find(','))) <= p.times[bar]) out.push_back(row);
    return out;
  };
  for (const Decision& d : ds) {
    const std::size_t e = d.date + 1;  // execution bar of this decision
    // (a) every bar after e perturbed: the curve through e's close and every trade through e are unchanged.
    Panel q = p;
    for (std::size_t t = e + 1; t < kT; ++t)
      for (std::size_t i = 0; i < q.N(); ++i)
        for (auto* v : {&q.open, &q.high, &q.low, &q.close, &q.volume, &q.vwap})
          (*v)[q.idx(t, i)] *= 1.0 + 0.3 * u(rng);
    const EquityCurve a = simulate(q, bp, ds);
    for (std::size_t t = first; t <= e; ++t) CHECK(a.value[t - first] == base.value[t - first]);
    CHECK(trades_through(a, e) == trades_through(base, e));
    // (b) also e's close (everything after its open): the trades at e's open are still unchanged.
    for (std::size_t i = 0; i < q.N(); ++i) q.close[q.idx(e, i)] *= 1.25;
    const EquityCurve b = simulate(q, bp, ds);
    CHECK(trades_through(b, e) == trades_through(base, e));
    for (std::size_t t = first; t < e; ++t) CHECK(b.value[t - first] == base.value[t - first]);
  }
}
