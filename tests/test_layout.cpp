#include <doctest/doctest.h>
#include <omp.h>

#include <cmath>
#include <random>
#include <vector>

#include "geom/layout.hpp"

using namespace fx;

namespace {
std::vector<double> random_xy(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> xy(2 * n);
  for (auto& v : xy) v = u(rng);
  return xy;
}
}  // namespace

TEST_CASE("Barnes-Hut repulsion approximates the exact sum") {
  const std::size_t n = 400;
  auto xy = random_xy(n, 3);
  std::vector<bool> active(n, true);
  active[7] = false;
  auto exact = exact_repulsion(xy, active);
  auto bh = barnes_hut_repulsion(xy, active, 0.5);
  double num = 0, den = 0;
  for (std::size_t k = 0; k < 2 * n; ++k) {
    num += (bh[k] - exact[k]) * (bh[k] - exact[k]);
    den += exact[k] * exact[k];
  }
  CHECK(std::sqrt(num / den) < 0.02);
  CHECK(bh[14] == 0.0);
  CHECK(bh[15] == 0.0);
  auto bh0 = barnes_hut_repulsion(xy, active, 0.0);  // never approximates
  for (std::size_t k = 0; k < 2 * n; ++k) CHECK(bh0[k] == doctest::Approx(exact[k]).epsilon(1e-9));
}

TEST_CASE("layout pulls strongly connected clusters together") {
  const std::size_t n = 40;
  std::vector<bool> active(n, true);
  std::vector<LayoutEdge> edges;
  for (std::uint32_t a = 0; a < 20; ++a)
    for (std::uint32_t b = a + 1; b < 20; ++b) {
      edges.push_back({a, b, 1.0});
      edges.push_back({a + 20, b + 20, 1.0});
    }
  auto xy = initial_positions(n, 7);
  LayoutParams p;
  run_layout(xy, active, edges, p, 300);
  auto dist = [&](std::size_t i, std::size_t j) { return std::hypot(xy[2 * i] - xy[2 * j], xy[2 * i + 1] - xy[2 * j + 1]); };
  double intra = 0, inter = 0;
  int ni = 0, nx = 0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j) {
      if ((i < 20) == (j < 20)) intra += dist(i, j), ++ni;
      else inter += dist(i, j), ++nx;
    }
  CHECK(intra / ni < 0.5 * (inter / nx));
}

TEST_CASE("layout is deterministic across thread counts") {
  const std::size_t n = 600;
  std::vector<bool> active(n, true);
  std::vector<LayoutEdge> edges;
  for (std::uint32_t i = 0; i + 1 < n; ++i) edges.push_back({i, i + 1, 1.0 + (i % 7)});
  auto a = initial_positions(n, 7), b = a;
  const int saved = omp_get_max_threads();
  omp_set_num_threads(1);
  run_layout(a, active, edges, LayoutParams{}, 30);
  omp_set_num_threads(std::max(saved, 4));
  run_layout(b, active, edges, LayoutParams{}, 30);
  omp_set_num_threads(saved);
  CHECK(a == b);
}

TEST_CASE("layout_edges keeps the top raw edges per active row, off-diagonal") {
  Csr P;
  P.n = 3;
  P.row_ptr = {0, 3, 4, 5};
  P.col = {0, 1, 2, 2, 1};
  P.val = {0.2, 0.5, 0.3, 1.0, 1.0};
  P.raw = {9.0, 5.0, 3.0, 1.0, 2.0};
  auto e = layout_edges(P, {true, true, true}, 1);
  REQUIRE(e.size() == 3);
  CHECK(e[0].a == 0);
  CHECK(e[0].b == 1);   // self edge (raw 9) skipped
  CHECK(e[0].w == 5.0);
  CHECK(e[1].a == 1);
  CHECK(e[2].a == 2);
  CHECK(layout_edges(P, {true, false, true}, 5).size() == 1);  // 0->2 only (1 inactive; 2->1 dropped)
}

TEST_CASE("initial positions are deterministic and in range") {
  auto a = initial_positions(50, 7), b = initial_positions(50, 7);
  CHECK(a == b);
  for (double v : a) CHECK(std::abs(v) <= 1.0);
}
