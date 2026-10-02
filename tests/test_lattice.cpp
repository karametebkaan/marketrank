#include <doctest/doctest.h>

#include <algorithm>
#include <random>
#include <set>
#include <vector>

#include "geom/lattice.hpp"

using namespace fx;

namespace {
void check_bijection(const std::vector<std::int32_t>& cell, const std::vector<bool>& active, LatticeSize s) {
  std::set<std::int32_t> seen;
  for (std::size_t i = 0; i < cell.size(); ++i) {
    if (!active[i]) {
      CHECK(cell[i] == -1);
      continue;
    }
    REQUIRE(cell[i] >= 0);
    CHECK(static_cast<std::size_t>(cell[i]) < s.cells());
    CHECK(seen.insert(cell[i]).second);
  }
}
}  // namespace

TEST_CASE("lattice size is near-square and fits all active nodes") {
  CHECK(lattice_size(0).cells() == 0);
  auto s = lattice_size(6000);
  CHECK(s.cols == 78);
  CHECK(s.rows == 77);
  CHECK(s.cells() >= 6000);
  CHECK(lattice_size(1).cells() == 1);
}

TEST_CASE("rcb assigns one active node per cell and preserves x order across columns") {
  const std::size_t n = 103;
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> u(-1, 1);
  std::vector<double> xy(2 * n);
  for (auto& v : xy) v = u(rng);
  std::vector<bool> active(n, true);
  active[3] = active[50] = false;
  auto s = lattice_size(101);
  auto cell = rcb_assign(xy, active, s);
  check_bijection(cell, active, s);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      if (active[i] && active[j] && cell[i] % static_cast<int>(s.cols) < cell[j] % static_cast<int>(s.cols))
        CHECK(xy[2 * i] <= xy[2 * j]);
}

TEST_CASE("hysteresis keeps nearby previous cells and stays a bijection") {
  LatticeSize s{4, 4};
  std::vector<bool> active(5, true);
  std::vector<std::int32_t> prev = {0, 5, 14, 15, -1};
  std::vector<std::int32_t> next = {1, 5, 0, 3, 2};  // node 0 moved 1 cell; node 2 moved far
  auto out = apply_hysteresis(prev, next, s, 2);
  check_bijection(out, active, s);
  CHECK(out[0] == 0);   // kept (shift 1)
  CHECK(out[1] == 5);
  CHECK(out[2] != 14);  // shift 3 > 2: leaves its old cell; its new cell 0 is taken by node 0, so it takes the nearest free cell
  CHECK(out[3] == 3);   // prev 15 is 3 rows away -> new cell
  CHECK(out[4] == 2);   // new node
}

TEST_CASE("hysteresis resolves conflicts deterministically") {
  LatticeSize s{3, 3};
  std::vector<bool> active(3, true);
  std::vector<std::int32_t> prev = {4, 4, 4};  // impossible prior state, still must yield a bijection
  std::vector<std::int32_t> next = {0, 1, 2};
  auto out = apply_hysteresis(prev, next, s, 2);
  check_bijection(out, active, s);
  CHECK(out[0] == 4);
  CHECK(out == apply_hysteresis(prev, next, s, 2));
}
