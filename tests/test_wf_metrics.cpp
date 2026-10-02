#include <doctest/doctest.h>
#include <cmath>
#include <vector>
#include "core/time.hpp"
#include "walkforward/metrics.hpp"
using namespace mr;

static EquityCurve make_curve(TimePoint t0, const std::vector<double>& rets) {
  EquityCurve c;
  double v = 1.0;
  for (std::size_t i = 0; i < rets.size(); ++i) {
    v *= 1.0 + rets[i];
    c.t.push_back(t0 + static_cast<TimePoint>(i) * 86400);
    c.value.push_back(v);
  }
  return c;
}

TEST_CASE("performance on analytic curves") {
  const TimePoint t0 = utc_seconds(2020, 1, 1);
  auto s = make_curve(t0, std::vector<double>(252, 0.001));
  auto b = make_curve(t0, std::vector<double>(252, 0.0));
  Perf p = performance(s, b);
  CHECK(p.ann_excess == doctest::Approx(0.001 * 252));
  CHECK(p.max_drawdown == doctest::Approx(0.0).epsilon(1e-12));
  CHECK(p.cum_return == doctest::Approx(std::pow(1.001, 252) - 1));
  CHECK(p.ann_return == doctest::Approx(std::pow(1.001, 252) - 1));

  std::vector<double> r(10, 0.0);
  r[3] = -0.2; r[6] = 0.25;  // 1 -> 0.8 -> 1.0
  Perf d = performance(make_curve(t0, r), make_curve(t0, std::vector<double>(10, 0.0)));
  CHECK(d.max_drawdown == doctest::Approx(0.2));
  // drawdown from the 1.0 starting capital on the very first day
  Perf f = performance(make_curve(t0, {-0.1, 0.0}), make_curve(t0, {0.0, 0.0}));
  CHECK(f.max_drawdown == doctest::Approx(0.1));
}

TEST_CASE("year hit rate on two synthetic years") {
  std::vector<double> rs(130, 0.001), rn(130, -0.001);
  auto a = make_curve(utc_seconds(2020, 1, 1), rs);
  auto c = make_curve(utc_seconds(2021, 1, 1), rn);
  EquityCurve s = a, b;
  double last = a.value.back();
  for (std::size_t i = 0; i < c.t.size(); ++i) { s.t.push_back(c.t[i]); s.value.push_back(last * c.value[i]); }
  b.t = s.t; b.value.assign(s.t.size(), 1.0);
  Perf p = performance(s, b);
  CHECK(p.years == 2);
  CHECK(p.year_hit_rate == doctest::Approx(0.5));
  // short years are not counted
  auto s2 = make_curve(utc_seconds(2020, 1, 1), std::vector<double>(50, 0.001));
  Perf q = performance(s2, make_curve(utc_seconds(2020, 1, 1), std::vector<double>(50, 0.0)));
  CHECK(q.years == 0);
  CHECK(q.year_hit_rate == 0.0);
}

TEST_CASE("alignment on common dates") {
  const TimePoint t0 = utc_seconds(2020, 1, 1);
  auto s = make_curve(t0, std::vector<double>(10, 0.001));
  auto b = make_curve(t0 + 5 * 86400, std::vector<double>(10, 0.0));
  Perf p = performance(s, b);  // common: days 5..9 -> 5 days
  CHECK(p.cum_return == doctest::Approx(std::pow(1.001, 5) - 1));  // 5 common days, not the earlier history
  CHECK(p.ann_excess == doctest::Approx(0.001 * 252));
  // both start together: first return is vs 1.0
  Perf q = performance(make_curve(t0, {0.05, 0.0}), make_curve(t0, {0.0, 0.0}));
  CHECK(q.cum_return == doctest::Approx(0.05));
  CHECK(q.ann_excess == doctest::Approx(0.025 * 252));
}

TEST_CASE("deflated sharpe") {
  double expect = norm_cdf(0.1 * std::sqrt(999.0) / std::sqrt(1 + 0.5 * 0.01));
  CHECK(deflated_sharpe(0.1, 1000, 0, 3, 0.0004, 1) == doctest::Approx(expect));
  double d10 = deflated_sharpe(0.1, 1000, 0, 3, 0.0004, 10);
  double d100 = deflated_sharpe(0.1, 1000, 0, 3, 0.0004, 100);
  CHECK(d10 < expect);
  CHECK(d100 < d10);
}

TEST_CASE("decision gate") {
  Perf p;
  p.ann_excess = 0.02; p.excess_ci95 = {0.005, 0.03}; p.year_hit_rate = 0.7; p.max_drawdown = 0.30;
  GateResult g = decision_gate(p, 0.28, 0.97, true);
  CHECK(g.pass);
  CHECK(g.c1); CHECK(g.c2); CHECK(g.c3); CHECK(g.c4); CHECK(g.c5);
  CHECK(g.reason.empty());

  Perf a = p; a.excess_ci95.lo = -0.001;
  g = decision_gate(a, 0.28, 0.97, true);
  CHECK(!g.c1); CHECK(!g.pass); CHECK(g.reason.find("c1") != std::string::npos);
  a = p; a.ann_excess = -0.01;
  g = decision_gate(a, 0.28, 0.97, true);
  CHECK(!g.c1); CHECK(g.reason.find("c1") != std::string::npos);
  a = p; a.year_hit_rate = 0.5;
  g = decision_gate(a, 0.28, 0.97, true);
  CHECK(!g.c2); CHECK(g.c1); CHECK(g.reason.find("c2") != std::string::npos);
  g = decision_gate(p, 0.28, 0.95, true);
  CHECK(!g.c3); CHECK(g.reason.find("c3") != std::string::npos);
  g = decision_gate(p, 0.20, 0.97, true);
  CHECK(!g.c4); CHECK(g.reason.find("c4") != std::string::npos);
  g = decision_gate(p, 0.28, 0.97, false);
  CHECK(!g.c5); CHECK(g.reason.find("c5") != std::string::npos);
  g = decision_gate(p, 0.20, 0.5, false);
  CHECK(g.reason.find("c3") != std::string::npos);
  CHECK(g.reason.find("c4") != std::string::npos);
  CHECK(g.reason.find("c5") != std::string::npos);
  CHECK(!g.pass);
}

TEST_CASE("alternating returns give analytic moments") {
  const double a = 0.01;
  const std::size_t T = 1000;
  std::vector<double> r;
  for (std::size_t i = 0; i < T; ++i) r.push_back(i % 2 == 0 ? a : -a);
  // values: multiplicative returns +a/-a; use log-free check via mean of simple returns
  const TimePoint t0 = utc_seconds(2020, 1, 1);
  Perf p = performance(make_curve(t0, r), make_curve(t0, std::vector<double>(T, 0.0)));
  double mean = 0;
  for (double v : r) mean += v;
  mean /= T;
  CHECK(mean == doctest::Approx(0.0).epsilon(1e-9));
  const double sd = a * std::sqrt(double(T) / (T - 1));
  CHECK(p.ann_vol == doctest::Approx(sd * std::sqrt(252.0)).epsilon(1e-6));
  CHECK(p.skew == doctest::Approx(0.0).scale(1.0).epsilon(1e-6));
  CHECK(p.kurt == doctest::Approx(1.0).epsilon(1e-6));
  CHECK(p.ir == doctest::Approx(p.sharpe));
  CHECK(p.sharpe_daily * std::sqrt(252.0) == doctest::Approx(p.sharpe));
}

TEST_CASE("constant series: CI and degenerate sd") {
  const TimePoint t0 = utc_seconds(2020, 1, 1);
  Perf p = performance(make_curve(t0, std::vector<double>(300, 0.001)),
                       make_curve(t0, std::vector<double>(300, 0.0)));
  CHECK(p.excess_ci95.lo <= p.ann_excess + 1e-12);
  CHECK(p.excess_ci95.hi >= p.ann_excess - 1e-12);
  CHECK(p.excess_ci95.lo == doctest::Approx(0.001 * 252));
  CHECK(p.excess_ci95.hi == doctest::Approx(0.001 * 252));
  CHECK(std::isnan(p.sharpe));
  CHECK(std::isnan(p.ir));
}

TEST_CASE("gate reason for undefined DSR") {
  Perf p;
  p.ann_excess = 0.02; p.excess_ci95 = {0.005, 0.03}; p.year_hit_rate = 0.7; p.max_drawdown = 0.2;
  GateResult g = decision_gate(p, 0.28, std::nan(""), true);
  CHECK(!g.c3);
  CHECK(g.reason.find("c3: deflated Sharpe undefined (non-positive variance term)") != std::string::npos);
}
