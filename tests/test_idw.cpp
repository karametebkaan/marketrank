#include <doctest/doctest.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "geom/idw.hpp"

using namespace mr;

TEST_CASE("IDW with subdivision 1 is exact at the nodes") {
  LatticeSize s{3, 2};
  std::vector<std::int32_t> cell = {0, 1, 2, 3, 4, 5};
  std::vector<double> v = {1, 2, 3, 4, 5, 6};
  IdwParams p;
  p.subdivision = 1;
  Raster r = idw_raster(cell, v, s, p);
  REQUIRE(r.w == 3);
  REQUIRE(r.h == 2);
  for (int k = 0; k < 6; ++k) CHECK(r.z[static_cast<std::size_t>(k)] == doctest::Approx(v[static_cast<std::size_t>(k)]));
  CHECK(r.zmin == doctest::Approx(1));
  CHECK(r.zmax == doctest::Approx(6));
}

TEST_CASE("IDW stays within the node value range and uses the nearest node beyond the radius") {
  LatticeSize s{20, 20};
  std::vector<std::int32_t> cell = {0, 399};
  std::vector<double> v = {-2.0, 5.0};
  IdwParams p;
  p.radius_cells = 2;
  Raster r = idw_raster(cell, v, s, p);
  REQUIRE(r.z.size() == 80 * 80);
  for (float z : r.z) {
    CHECK(z >= -2.0f - 1e-5f);
    CHECK(z <= 5.0f + 1e-5f);
  }
  CHECK(r.z[0] == doctest::Approx(-2.0));                  // next to node 0
  CHECK(r.z[80 * 79 + 79] == doctest::Approx(5.0));       // next to node 399
  CHECK(r.z[40 * 80 + 10] == doctest::Approx(-2.0));      // closer to node 0, outside both radii
}

TEST_CASE("IDW ignores non-finite values and is deterministic across threads") {
  LatticeSize s{10, 10};
  std::vector<std::int32_t> cell;
  std::vector<double> v;
  for (int i = 0; i < 100; ++i) {
    cell.push_back(i);
    v.push_back(i == 55 ? std::numeric_limits<double>::quiet_NaN() : std::sin(i * 0.3));
  }
  const int saved = omp_get_max_threads();
  omp_set_num_threads(1);
  Raster a = idw_raster(cell, v, s, IdwParams{});
  omp_set_num_threads(std::max(saved, 4));
  Raster b = idw_raster(cell, v, s, IdwParams{});
  omp_set_num_threads(saved);
  CHECK(a.z == b.z);
  for (float z : a.z) CHECK(std::isfinite(z));
}

TEST_CASE("IDW sparse lattice uses an O(cells) nearest-node fallback") {
  LatticeSize s{78, 77};
  std::vector<std::int32_t> cell = {0};
  std::vector<double> v = {3.5};
  const auto t0 = std::chrono::steady_clock::now();
  Raster r = idw_raster(cell, v, s, IdwParams{});
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
  MESSAGE("IDW fallback on a 78x77 lattice with one node: " << ms << " ms");
#if defined(NDEBUG) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
  CHECK(ms < 300);  // optimized builds only; the time is printed above either way
#endif
  REQUIRE(r.z.size() == 78 * 4 * 77 * 4);
  for (float z : r.z) CHECK(z == doctest::Approx(3.5));
}

TEST_CASE("IDW weight underflow falls back to the nearest in-window node") {
  LatticeSize s{4, 1};
  std::vector<std::int32_t> cell = {0, 2};
  std::vector<double> v = {1.0, 9.0};
  IdwParams p;
  p.power = 2000;
  p.radius_cells = 3;
  Raster r = idw_raster(cell, v, s, p);
  // pixel centre u = 1.125 (px 4): nearest node is cell 0 at 0.5 (d 0.625) vs cell 2 at 2.5 (d 1.375)
  CHECK(r.z[4] == doctest::Approx(1.0));
  // pixel centre u = 1.875 (px 7): nearest is cell 2
  CHECK(r.z[7] == doctest::Approx(9.0));
}

TEST_CASE("IDW weighted value between two nodes") {
  LatticeSize s{3, 1};
  std::vector<std::int32_t> cell = {0, 2};
  std::vector<double> v = {0.0, 2.0};
  IdwParams p;
  p.subdivision = 1;
  Raster r = idw_raster(cell, v, s, p);
  CHECK(r.z[1] == doctest::Approx(1.0));
}

namespace {
Raster make_raster(std::size_t w, std::size_t h, double (*f)(std::size_t, std::size_t)) {
  Raster r;
  r.w = w;
  r.h = h;
  r.z.resize(w * h);
  for (std::size_t y = 0; y < h; ++y)
    for (std::size_t x = 0; x < w; ++x) r.z[y * w + x] = static_cast<float>(f(x, y));
  return r;
}
double neighbour_diff_variance(const Raster& r) {
  std::vector<double> d;
  for (std::size_t y = 0; y < r.h; ++y)
    for (std::size_t x = 0; x + 1 < r.w; ++x) d.push_back(double(r.z[y * r.w + x + 1]) - double(r.z[y * r.w + x]));
  double m = 0;
  for (double v : d) m += v;
  m /= static_cast<double>(d.size());
  double s = 0;
  for (double v : d) s += (v - m) * (v - m);
  return s / static_cast<double>(d.size());
}
}  // namespace

TEST_CASE("smoothing: sigma 0 is the identity, a constant field stays constant, bad sigma throws") {
  Raster wave = make_raster(13, 9, [](std::size_t x, std::size_t y) { return std::sin(0.7 * double(x)) + 0.3 * double(y); });
  Raster same = smooth_raster(wave, 0.0, 1);
  CHECK(same.z == wave.z);
  CHECK(same.zmin == wave.zmin);
  Raster c = make_raster(11, 7, [](std::size_t, std::size_t) { return 2.5; });
  for (double sigma : {0.5, 1.0, 4.0}) {
    Raster s = smooth_raster(c, sigma, 2);
    REQUIRE(s.z.size() == c.z.size());
    for (float z : s.z) CHECK(z == doctest::Approx(2.5).epsilon(1e-6));  // edge renormalization keeps the mean
    CHECK(s.zmin == doctest::Approx(2.5));
    CHECK(s.zmax == doctest::Approx(2.5));
  }
  CHECK_THROWS_AS(smooth_raster(c, -1.0, 1), std::invalid_argument);
  CHECK_THROWS_AS(smooth_raster(c, std::numeric_limits<double>::quiet_NaN(), 1), std::invalid_argument);
}

TEST_CASE("smoothing is deterministic across thread counts and recomputes zmin/zmax") {
  Raster r = make_raster(80, 64, [](std::size_t x, std::size_t y) { return std::sin(double(x * 7 + y * 13)); });
  const int saved = omp_get_max_threads();
  omp_set_num_threads(1);
  Raster a = smooth_raster(r, 1.5, 2);
  omp_set_num_threads(8);
  Raster b = smooth_raster(r, 1.5, 2);
  omp_set_num_threads(saved);
  CHECK(a.z == b.z);
  CHECK(a.zmin == *std::min_element(a.z.begin(), a.z.end()));
  CHECK(a.zmax == *std::max_element(a.z.begin(), a.z.end()));
  CHECK(a.zmax < *std::max_element(r.z.begin(), r.z.end()));
}

TEST_CASE("smoothing reduces neighbour-difference variance on a checkerboard") {
  Raster cb = make_raster(20, 20, [](std::size_t x, std::size_t y) { return (x + y) % 2 ? 1.0 : -1.0; });
  cb.zmin = -1;
  cb.zmax = 1;
  Raster s = smooth_raster(cb, 1.0, 1);
  const double before = neighbour_diff_variance(cb), after = neighbour_diff_variance(s);
  MESSAGE("checkerboard neighbour-diff variance " << before << " -> " << after);
  CHECK(after < 0.05 * before);
}
