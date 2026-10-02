#include <doctest/doctest.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "geom/idw.hpp"

using namespace fx;

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
