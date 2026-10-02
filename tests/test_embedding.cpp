#include <doctest/doctest.h>
#include <omp.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "geom/embedding.hpp"

using namespace fx;

namespace {
// Build a Csr from per-row (col, weight) lists; rows are renormalized so val sums to 1.
Csr make_csr(const std::vector<std::vector<std::pair<std::uint32_t, double>>>& rows) {
  Csr P;
  P.n = rows.size();
  P.row_ptr.push_back(0);
  for (const auto& r : rows) {
    double s = 0;
    for (const auto& e : r) s += e.second;
    for (const auto& e : r) {
      P.col.push_back(e.first);
      P.val.push_back(e.second / s);
    }
    P.row_ptr.push_back(P.col.size());
  }
  P.raw = P.val;
  return P;
}

Csr cluster_csr() {
  std::vector<std::vector<std::pair<std::uint32_t, double>>> rows(16);
  for (std::uint32_t i = 0; i < 10; ++i) {
    std::uint32_t base = i < 5 ? 10 : 13;
    rows[i].push_back({i, 0.3});
    for (std::uint32_t s = 0; s < 3; ++s) rows[i].push_back({base + s, 0.7 / 3});
  }
  for (std::uint32_t i = 10; i < 16; ++i) {
    std::uint32_t base = i < 13 ? 10 : 13;
    rows[i].push_back({i, 0.5});
    for (std::uint32_t s = 0; s < 3; ++s)
      if (base + s != i) rows[i].push_back({base + s, 0.25});
  }
  return make_csr(rows);
}

double dist(const std::vector<double>& xy, std::size_t a, std::size_t b) {
  return std::hypot(xy[2 * a] - xy[2 * b], xy[2 * a + 1] - xy[2 * b + 1]);
}

void check_clusters(const std::vector<double>& xy) {
  auto mean_within = [&](std::size_t lo, std::size_t hi, std::size_t skip) {
    double s = 0;
    int c = 0;
    for (std::size_t a = lo; a < hi; ++a)
      for (std::size_t b = a + 1; b < hi; ++b) {
        if (a == skip || b == skip) continue;
        s += dist(xy, a, b);
        ++c;
      }
    return s / c;
  };
  auto centroid = [&](std::size_t lo, std::size_t hi, std::size_t skip) {
    double x = 0, y = 0;
    int c = 0;
    for (std::size_t a = lo; a < hi; ++a) {
      if (a == skip) continue;
      x += xy[2 * a];
      y += xy[2 * a + 1];
      ++c;
    }
    return std::pair<double, double>{x / c, y / c};
  };
  std::size_t skip = 1000;
  // skip node 3 if inactive (xy exactly zero there while others are not)
  if (xy[6] == 0 && xy[7] == 0) skip = 3;
  auto ca = centroid(0, 5, skip), cb = centroid(5, 10, skip);
  double between = std::hypot(ca.first - cb.first, ca.second - cb.second);
  CHECK(between > 0.1);
  CHECK(mean_within(0, 5, skip) < 0.25 * between);
  CHECK(mean_within(5, 10, skip) < 0.25 * between);
}
}  // namespace

TEST_CASE("stocks sending money to the same destinations cluster even without edges between them") {
  Csr P = cluster_csr();
  std::vector<bool> active(16, true);
  EmbeddingParams p;
  auto Y = destination_signatures(P, active, p);
  REQUIRE(Y.size() == 16u * static_cast<std::size_t>(p.dims));
  auto xy = pca2(Y, static_cast<std::size_t>(p.dims), active);
  REQUIRE(xy.size() == 32);
  check_clusters(xy);
}

TEST_CASE("isolated active nodes get distinct finite positions") {
  std::vector<std::vector<std::pair<std::uint32_t, double>>> rows(6);
  for (std::uint32_t i = 0; i < 6; ++i) rows[i].push_back({i, 1.0});
  Csr P = make_csr(rows);
  std::vector<bool> active(6, true);
  EmbeddingParams p;
  auto xy = pca2(destination_signatures(P, active, p), static_cast<std::size_t>(p.dims), active);
  for (double v : xy) CHECK(std::isfinite(v));
  for (std::size_t a = 0; a < 6; ++a)
    for (std::size_t b = a + 1; b < 6; ++b) CHECK(dist(xy, a, b) > 1e-9);
}

TEST_CASE("inactive rows are zero and ignored") {
  Csr P = cluster_csr();
  std::vector<bool> active(16, true);
  active[3] = false;
  EmbeddingParams p;
  auto Y = destination_signatures(P, active, p);
  for (int d = 0; d < p.dims; ++d) CHECK(Y[3 * static_cast<std::size_t>(p.dims) + static_cast<std::size_t>(d)] == 0.0);
  auto xy = pca2(Y, static_cast<std::size_t>(p.dims), active);
  CHECK(xy[6] == 0.0);
  CHECK(xy[7] == 0.0);
  check_clusters(xy);
}

TEST_CASE("embedding is deterministic across thread counts") {
  const std::size_t n = 2000;
  std::mt19937_64 rng(5);
  std::vector<std::vector<std::pair<std::uint32_t, double>>> rows(n);
  for (std::size_t i = 0; i < n; ++i) {
    rows[i].push_back({static_cast<std::uint32_t>(i), 0.5 + (rng() % 100) / 200.0});
    for (int k = 0; k < 5; ++k)
      rows[i].push_back({static_cast<std::uint32_t>((i + 1 + rng() % (n - 1)) % n), 0.1 + (rng() % 1000) / 1000.0});
  }
  Csr P = make_csr(rows);
  std::vector<bool> active(n, true);
  omp_set_num_threads(1);
  SolveEmbedding e1(n);
  auto a = e1.positions(P, active);
  omp_set_num_threads(8);
  SolveEmbedding e2(n);
  auto b = e2.positions(P, active);
  omp_set_num_threads(omp_get_max_threads());
  REQUIRE(a.size() == b.size());
  bool same = true;
  for (std::size_t i = 0; i < a.size(); ++i) same = same && (a[i] == b[i]);
  CHECK(same);
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

TEST_CASE("positions are stable on a repeated frame") {
  Csr P = cluster_csr();
  std::vector<bool> active(16, true);
  SolveEmbedding e(16);
  auto a = e.positions(P, active);
  auto b = e.positions(P, active);
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) CHECK(std::abs(a[i] - b[i]) < 1e-9);
}
