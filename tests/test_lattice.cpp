#include <doctest/doctest.h>

#include "geom/lattice.hpp"

using namespace mr;

TEST_CASE("lattice size is near-square and fits all active nodes") {
  CHECK(lattice_size(0).cells() == 0);
  auto s = lattice_size(6000);
  CHECK(s.cols == 78);
  CHECK(s.rows == 77);
  CHECK(s.cells() >= 6000);
  CHECK(lattice_size(1).cells() == 1);
}
