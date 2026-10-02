#include <doctest/doctest.h>
#include <cmath>
#include <vector>
#include "core/time.hpp"
#include "walkforward/universe.hpp"
using namespace mr;

namespace {
// 3 stocks, weekday daily bars 2024-01-29 .. 2024-03-05. Stock 0: 100M/day, 1: 10M/day, 2: 60M/day.
Panel make_panel() {
  Panel p;
  p.tickers = {"A", "B", "C"};
  const std::int64_t d0 = days_from_civil(2024, 1, 29);
  const std::int64_t d1 = days_from_civil(2024, 3, 5);
  for (std::int64_t d = d0; d <= d1; ++d) {
    const unsigned wd = weekday_from_days(d);
    if (wd == 0 || wd == 6) continue;
    p.times.push_back(d * 86400);
  }
  const std::size_t T = p.T(), N = 3;
  for (auto* v : {&p.open, &p.high, &p.low, &p.close, &p.volume, &p.vwap}) v->assign(T * N, 1.0);
  const double dv[3] = {100e6, 10e6, 60e6};
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t i = 0; i < N; ++i) {
      p.vwap[p.idx(t, i)] = 10.0;
      p.volume[p.idx(t, i)] = dv[i] / 10.0;
      p.open[p.idx(t, i)] = 100.0 + static_cast<double>(t) * (1.0 + static_cast<double>(i));
    }
  return p;
}
std::size_t index_of(const Panel& p, int y, unsigned m, unsigned d) {
  const TimePoint want = days_from_civil(y, m, d) * 86400;
  for (std::size_t t = 0; t < p.T(); ++t)
    if (p.times[t] == want) return t;
  FAIL("date not found");
  return 0;
}
}  // namespace

TEST_CASE("parse_rebalance") {
  CHECK(parse_rebalance("monthly") == Rebalance::Monthly);
  CHECK(parse_rebalance("weekly") == Rebalance::Weekly);
  CHECK_THROWS_AS(parse_rebalance("daily"), std::invalid_argument);
}

TEST_CASE("rebalance_dates monthly and weekly") {
  const Panel p = make_panel();
  const auto m = rebalance_dates(p, Rebalance::Monthly, 0);
  REQUIRE(m.size() == 2);
  CHECK(m[0] == index_of(p, 2024, 1, 31));
  CHECK(m[1] == index_of(p, 2024, 2, 29));
  const auto w = rebalance_dates(p, Rebalance::Weekly, 0);
  for (std::size_t t : w) CHECK(weekday_from_days(p.times[t] / 86400) == 5);
  CHECK(w.front() == index_of(p, 2024, 2, 2));
  CHECK(w.back() == index_of(p, 2024, 3, 1));  // Fri 3/1; the 3/5 week is incomplete / no next open
  CHECK(w.size() == 5);
  const auto mw = rebalance_dates(p, Rebalance::Monthly, index_of(p, 2024, 2, 1));
  REQUIRE(mw.size() == 1);
  CHECK(mw[0] == index_of(p, 2024, 2, 29));
}

TEST_CASE("eligible_at thresholds, top_n, causality") {
  Panel p = make_panel();
  const std::size_t t = 10;
  auto e = eligible_at(p, t, 3, 50e6, 0);
  CHECK(e == std::vector<bool>{true, false, true});
  e = eligible_at(p, t, 3, 50e6, 1);
  CHECK(e == std::vector<bool>{true, false, false});
  e = eligible_at(p, t, 3, 0.0, 2);
  CHECK(e == std::vector<bool>{true, false, true});
  // causal: change bars after t
  for (std::size_t u = t + 1; u < p.T(); ++u)
    for (std::size_t i = 0; i < 3; ++i) p.volume[p.idx(u, i)] = (i == 1) ? 1e12 : 0.0;
  CHECK(eligible_at(p, t, 3, 50e6, 0) == std::vector<bool>{true, false, true});
  // too few finite values
  p.vwap[p.idx(t, 0)] = NAN;
  p.vwap[p.idx(t - 1, 0)] = NAN;
  p.vwap[p.idx(t - 2, 0)] = NAN;
  CHECK(eligible_at(p, t, 4, 50e6, 0)[0] == false);  // 1 finite < window/2 = 2
  CHECK(eligible_at(p, t, 4, 50e6, 0)[2] == true);
  // ties broken by lower index
  Panel q = make_panel();
  for (std::size_t u = 0; u < q.T(); ++u) q.volume[q.idx(u, 2)] = q.volume[q.idx(u, 0)];
  CHECK(eligible_at(q, t, 3, 0.0, 1) == std::vector<bool>{true, false, false});
}

TEST_CASE("forward_oo_return") {
  Panel p = make_panel();
  const std::size_t t = 5;
  const auto r = forward_oo_return(p, t, 2);
  for (std::size_t i = 0; i < 3; ++i) CHECK(r[i] == p.open[p.idx(t + 3, i)] / p.open[p.idx(t + 1, i)] - 1.0);
  p.open[p.idx(t + 3, 1)] = NAN;
  CHECK(std::isnan(forward_oo_return(p, t, 2)[1]));
  CHECK(std::isnan(forward_oo_return(p, p.T() - 2, 2)[0]));
}
