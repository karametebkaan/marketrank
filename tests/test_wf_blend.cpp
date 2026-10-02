#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "walkforward/blend.hpp"
using namespace mr;

namespace {
constexpr std::size_t kN = 200;

// planted: signal 0 = label + noise(0.5); others noise.
std::vector<MonthRecord> make_months(std::size_t M, bool planted, unsigned seed) {
  std::mt19937_64 g(seed);
  std::normal_distribution<double> nd(0, 1);
  std::vector<MonthRecord> ms(M);
  for (auto& r : ms) {
    r.label.resize(kN);
    for (auto& v : r.label) v = nd(g);
    for (std::size_t s = 0; s < kSignals; ++s) {
      r.z[s].resize(kN);
      for (std::size_t i = 0; i < kN; ++i)
        r.z[s][i] = (planted && s == 0) ? r.label[i] + 0.5 * nd(g) : nd(g);
    }
  }
  return ms;
}
const BlendParams kMonthly{36, 1, 24, 12, 2.0};
bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }
}  // namespace

TEST_CASE("blend: planted signal gets the weight and opens the gate (monthly-style params)") {
  const auto ms = make_months(80, true, 1);
  const auto r = run_blend(ms, kMonthly);
  REQUIRE(r.size() == 80);
  CHECK(r[60].w[0] > 0.8);
  bool opened = false;
  for (std::size_t m = 0; m <= 36 + 1 + 12; ++m) opened = opened || r[m].gate_open;
  CHECK(opened);
  CHECK(r[79].gate_open);
  CHECK(r[79].score.size() == kN);
}

TEST_CASE("blend: planted signal opens the gate under default weekly params") {
  const BlendParams d;
  const auto ms = make_months(400, true, 2);
  const auto r = run_blend(ms, d);
  bool opened = false;
  for (std::size_t m = 0; m <= d.train_months + d.embargo + d.gate_min; ++m) opened = opened || r[m].gate_open;
  CHECK(opened);
  CHECK(r[399].w[0] > 0.8);
}

TEST_CASE("blend: pure noise keeps the gate closed (mean over 20 seeds)") {
  double frac_sum = 0;
  for (unsigned seed = 100; seed < 120; ++seed) {
    const auto r = run_blend(make_months(200, false, seed), kMonthly);
    std::size_t open = 0, tot = 0;
    for (std::size_t m = 36 + 1 + 12; m < r.size(); ++m) { ++tot; open += r[m].gate_open; }
    frac_sum += static_cast<double>(open) / static_cast<double>(tot);
  }
  CHECK(frac_sum / 20.0 <= 0.05);
}

TEST_CASE("blend: causality - labels at periods >= m-1 do not affect step m (every m)") {
  const std::size_t M = 80;
  const auto ms = make_months(M, true, 4);
  const auto base = run_blend(ms, kMonthly);
  for (std::size_t m = 8; m < M; ++m) {
    auto ms2 = ms;
    for (std::size_t j = m - 1; j < M; ++j)
      for (auto& v : ms2[j].label) v = std::numeric_limits<double>::quiet_NaN();  // a leaked oos_ic(m-1) would become NaN and shrink the gate sample n
    const auto alt = run_blend(ms2, kMonthly);
    for (std::size_t s = 0; s < kSignals; ++s) CHECK(same_bits(base[m].w[s], alt[m].w[s]));
    CHECK(base[m].gate_open == alt[m].gate_open);
    REQUIRE(base[m].score.size() == alt[m].score.size());
    for (std::size_t i = 0; i < base[m].score.size(); ++i) CHECK(same_bits(base[m].score[i], alt[m].score[i]));
    for (std::size_t j = 0; j + 2 <= m; ++j) CHECK(same_bits(base[j].oos_ic, alt[j].oos_ic));
  }
}

TEST_CASE("blend: fewer than 6 training periods gives no weights") {
  const auto r = run_blend(make_months(20, true, 6), kMonthly);
  for (std::size_t m = 0; m <= 6; ++m) {
    for (double w : r[m].w) CHECK(w == 0.0);
    CHECK(std::isnan(r[m].oos_ic));
  }
  double sum = 0;
  for (double w : r[7].w) sum += w;  // training window [0,5]: 6 periods
  CHECK(sum > 0);
  CHECK(r[7].w[0] > 0);
}

TEST_CASE("blend: weights are non-negative and sum to 1 or 0") {
  for (bool planted : {false, true}) {
    const auto r = run_blend(make_months(100, planted, 5), kMonthly);
    for (const auto& st : r) {
      double sum = 0;
      for (double w : st.w) { CHECK(w >= 0); sum += w; }
      CHECK((sum == 0 || std::abs(sum - 1) < 1e-12));
    }
  }
}
