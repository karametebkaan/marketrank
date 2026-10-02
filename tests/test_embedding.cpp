#include <doctest/doctest.h>
#include <omp.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "geom/embedding.hpp"
#include "geom/landscape.hpp"

using namespace fx;

namespace {
// Two groups of 20: A (h +5, pi 0.04) and B (h -0.8, pi 0.01), small deterministic jitter, one forecast with score = h.
Frame two_group_frame() {
  const std::size_t n = 40;
  Frame f;
  f.active.assign(n, true);
  f.h.resize(n);
  f.pi.resize(n);
  Forecast fc;
  fc.k = 1;
  fc.score.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double jit = 0.01 * static_cast<double>(i) / 40.0 * (i % 2 ? 1.0 : -1.0);
    const bool a = i < 20;
    f.h[i] = (a ? 5.0 : -0.8) + jit;
    f.pi[i] = (a ? 0.04 : 0.01) * (1.0 + jit);
    fc.score[i] = f.h[i];
  }
  f.forecasts.push_back(fc);
  return f;
}

double dist(const std::vector<double>& xy, std::size_t a, std::size_t b) {
  return std::hypot(xy[2 * a] - xy[2 * b], xy[2 * a + 1] - xy[2 * b + 1]);
}

// Mean intra-group distance < 0.25 x centroid distance, over nodes [first, 40).
void check_split(const std::vector<double>& xy, std::size_t first) {
  auto within = [&](std::size_t lo, std::size_t hi) {
    double s = 0;
    int c = 0;
    for (std::size_t a = lo; a < hi; ++a)
      for (std::size_t b = a + 1; b < hi; ++b) {
        s += dist(xy, a, b);
        ++c;
      }
    return s / c;
  };
  auto centroid = [&](std::size_t lo, std::size_t hi) {
    double x = 0, y = 0;
    for (std::size_t a = lo; a < hi; ++a) {
      x += xy[2 * a];
      y += xy[2 * a + 1];
    }
    const double c = static_cast<double>(hi - lo);
    return std::pair<double, double>{x / c, y / c};
  };
  auto ca = centroid(first, 20), cb = centroid(20, 40);
  const double between = std::hypot(ca.first - cb.first, ca.second - cb.second);
  CHECK(between > 0.1);
  CHECK(within(first, 20) < 0.25 * between);
  CHECK(within(20, 40) < 0.25 * between);
}
}  // namespace

TEST_CASE("stocks with similar solver outputs are placed together") {
  Frame f = two_group_frame();
  SolveEmbedding e(40);
  auto xy = e.positions(f);
  REQUIRE(xy.size() == 80);
  check_split(xy, 0);
}

TEST_CASE("one extreme outlier does not collapse the rest") {
  Frame f = two_group_frame();
  f.h[0] = 1e6;
  f.pi[0] = 0.9;
  f.forecasts[0].score[0] = 1e6;
  SolveEmbedding e(40);
  auto xy = e.positions(f);
  check_split(xy, 1);
}

TEST_CASE("feature standardization is robust and handles degenerate columns") {
  Frame f = two_group_frame();
  EmbeddingParams p;
  std::size_t F = 0;
  {
    Frame g = f;
    for (double& v : g.pi) v = 0.02;  // all equal: MAD 0
    g.h[3] = std::numeric_limits<double>::quiet_NaN();
    auto Y = solve_features(g, p, F);
    REQUIRE(F == 3);
    REQUIRE(Y.size() == 40 * 3);
    for (std::size_t i = 0; i < 40; ++i) CHECK(Y[i * 3 + 1] == 0.0);
    CHECK(Y[3 * 3 + 0] == 0.0);
    for (double v : Y) CHECK(std::isfinite(v));
  }
  {
    Frame g = f;
    g.h[0] = 1e6;
    auto Y = solve_features(g, p, F);
    double mx = 0;
    for (double v : Y) mx = std::max(mx, std::fabs(v));
    CHECK(mx <= p.clip + 1e-12);
    CHECK(Y[0] == doctest::Approx(p.clip));
  }
  {
    Frame g = f;
    g.forecasts.clear();
    solve_features(g, p, F);
    CHECK(F == 2);
  }
  {
    Frame g = f;
    g.active[5] = false;
    auto Y = solve_features(g, p, F);
    for (std::size_t d = 0; d < F; ++d) CHECK(Y[5 * F + d] == 0.0);
  }
  {
    Frame g = f;
    g.pi.pop_back();
    CHECK_THROWS_AS(solve_features(g, p, F), std::invalid_argument);
    Frame k = f;
    k.forecasts[0].score.pop_back();
    CHECK_THROWS_AS(solve_features(k, p, F), std::invalid_argument);
  }
}

TEST_CASE("pca2 on rank-1 data puts everything on the first axis") {
  const std::size_t n = 20, D = 16;
  std::vector<double> Y(n * D);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t d = 0; d < D; ++d) Y[i * D + d] = (static_cast<double>(i) - 7.5) * (1.0 + 0.1 * static_cast<double>(d));
  std::vector<bool> active(n, true);
  auto xy = pca2(Y, D, active);
  double ss = 0;
  for (std::size_t i = 0; i < n; ++i) {
    CHECK(std::abs(xy[2 * i + 1]) <= 1e-9);
    ss += xy[2 * i] * xy[2 * i];
  }
  CHECK(std::sqrt(ss / static_cast<double>(n)) == doctest::Approx(1.0));
}

TEST_CASE("align_to recovers a rotation and a reflection") {
  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> u(-1, 1);
  const std::size_t n = 50;
  std::vector<double> X(2 * n);
  for (auto& v : X) v = u(rng);
  std::vector<bool> all(n, true);
  const double c = std::cos(1.1), s = std::sin(1.1);
  std::vector<double> R(2 * n), M(2 * n);
  for (std::size_t i = 0; i < n; ++i) {
    R[2 * i] = c * X[2 * i] - s * X[2 * i + 1];
    R[2 * i + 1] = s * X[2 * i] + c * X[2 * i + 1];
    M[2 * i] = X[2 * i];
    M[2 * i + 1] = -X[2 * i + 1];
  }
  align_to(R, X, all);
  align_to(M, X, all);
  for (std::size_t i = 0; i < 2 * n; ++i) {
    CHECK(R[i] == doctest::Approx(X[i]).epsilon(1e-9).scale(1.0));
    CHECK(M[i] == doctest::Approx(X[i]).epsilon(1e-9).scale(1.0));
  }
}

TEST_CASE("embedding is deterministic across thread counts") {
  const std::size_t n = 2000;
  std::mt19937_64 rng(5);
  std::normal_distribution<double> g(0, 1);
  Frame f;
  f.active.assign(n, true);
  f.h.resize(n);
  f.pi.resize(n);
  Forecast fc;
  fc.score.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    f.h[i] = g(rng);
    f.pi[i] = std::exp(g(rng) - 6.0);
    fc.score[i] = g(rng);
  }
  f.forecasts.push_back(fc);
  const int saved_threads = omp_get_max_threads();
  omp_set_num_threads(1);
  SolveEmbedding e1(n);
  auto a = e1.positions(f);
  omp_set_num_threads(8);
  SolveEmbedding e2(n);
  auto b = e2.positions(f);
  omp_set_num_threads(saved_threads);
  REQUIRE(a.size() == b.size());
  bool same = true;
  for (std::size_t i = 0; i < a.size(); ++i) same = same && (a[i] == b[i]);
  CHECK(same);
}

TEST_CASE("positions are stable on a repeated frame, and finite with non-finite inputs") {
  Frame f = two_group_frame();
  SolveEmbedding e(40);
  auto a = e.positions(f);
  auto b = e.positions(f);
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) CHECK(std::abs(a[i] - b[i]) < 1e-9);
  f.h[7] = std::numeric_limits<double>::infinity();
  f.pi[9] = std::numeric_limits<double>::quiet_NaN();
  auto c = e.positions(f);
  for (double v : c) CHECK(std::isfinite(v));
}
