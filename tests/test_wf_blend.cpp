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

TEST_CASE("blend: pure noise keeps the gate closed") {
  const auto ms = make_months(200, false, 3);
  const auto r = run_blend(ms, kMonthly);
  std::size_t open = 0, tot = 0;
  for (std::size_t m = 36 + 1 + 12; m < r.size(); ++m) { ++tot; open += r[m].gate_open; }
  CHECK(static_cast<double>(open) <= 0.05 * static_cast<double>(tot));
}

TEST_CASE("blend: causality - labels at months >= m-1 do not affect step m") {
  const auto ms = make_months(80, true, 4);
  const auto base = run_blend(ms, kMonthly);
  const std::size_t m = 60;
  auto ms2 = ms;
  std::mt19937_64 g(99);
  std::normal_distribution<double> nd(0, 1);
  for (std::size_t j = m - 1; j < ms2.size(); ++j)
    for (auto& v : ms2[j].label) v = nd(g);
  const auto alt = run_blend(ms2, kMonthly);
  for (std::size_t s = 0; s < kSignals; ++s) CHECK(same_bits(base[m].w[s], alt[m].w[s]));
  CHECK(base[m].gate_open == alt[m].gate_open);
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
  // fewer than 6 training periods -> no weights, no score
  const auto r = run_blend(make_months(8, true, 6), kMonthly);
  for (const auto& st : r) { CHECK(!st.gate_open); CHECK(st.score.empty()); }
}
