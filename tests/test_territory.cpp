#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <numeric>
#include <set>
#include <vector>

#include "geom/territory.hpp"

using namespace mr;

namespace {
bool connected8(const std::vector<std::int32_t>& cells, std::size_t cols) {
  if (cells.empty()) return true;
  std::set<std::int32_t> rest(cells.begin(), cells.end());
  std::vector<std::int32_t> stack{*rest.begin()};
  rest.erase(rest.begin());
  const auto W = static_cast<std::int32_t>(cols);
  while (!stack.empty()) {
    auto c = stack.back();
    stack.pop_back();
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        if (!dx && !dy) continue;
        const std::int32_t cx = c % W + dx, cy = c / W + dy;
        if (cx < 0 || cx >= W || cy < 0) continue;
        auto it = rest.find(cy * W + cx);
        if (it != rest.end()) {
          stack.push_back(*it);
          rest.erase(it);
        }
      }
  }
  return rest.empty();
}

double spearman(std::vector<double> a, std::vector<double> b) {
  auto ranks = [](const std::vector<double>& v) {
    std::vector<std::size_t> o(v.size());
    std::iota(o.begin(), o.end(), 0);
    std::sort(o.begin(), o.end(), [&](auto x, auto y) { return v[x] < v[y]; });
    std::vector<double> r(v.size());
    for (std::size_t i = 0; i < o.size();) {
      std::size_t j = i;
      while (j + 1 < o.size() && v[o[j + 1]] == v[o[i]]) ++j;
      for (std::size_t k = i; k <= j; ++k) r[o[k]] = 0.5 * static_cast<double>(i + j);
      i = j + 1;
    }
    return r;
  };
  auto ra = ranks(a), rb = ranks(b);
  const double n = static_cast<double>(a.size());
  double ma = 0, mb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) ma += ra[i], mb += rb[i];
  ma /= n;
  mb /= n;
  double sab = 0, saa = 0, sbb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sab += (ra[i] - ma) * (rb[i] - mb);
    saa += (ra[i] - ma) * (ra[i] - ma);
    sbb += (rb[i] - mb) * (rb[i] - mb);
  }
  return sab / std::sqrt(saa * sbb);
}
}  // namespace

TEST_CASE("gilbert_order visits every cell once with unit steps") {
  for (auto sz : {LatticeSize{1, 1}, LatticeSize{2, 3}, LatticeSize{7, 5}, LatticeSize{79, 79}, LatticeSize{64, 64},
                  LatticeSize{100, 37}, LatticeSize{3, 40}}) {
    auto o = gilbert_order(sz);
    REQUIRE(o.size() == sz.cells());
    CHECK(o[0] == 0);
    std::vector<char> seen(sz.cells(), 0);
    bool perm = true, step = true;
    for (std::size_t k = 0; k < o.size(); ++k) {
      if (o[k] < 0 || static_cast<std::size_t>(o[k]) >= sz.cells() || seen[static_cast<std::size_t>(o[k])]) {
        perm = false;
        break;
      }
      seen[static_cast<std::size_t>(o[k])] = 1;
      if (k) {
        const long c = static_cast<long>(sz.cols);
        const long dx = std::labs(o[k] % c - o[k - 1] % c), dy = std::labs(o[k] / c - o[k - 1] / c);
        step = step && std::max(dx, dy) == 1;
      }
    }
    CHECK(perm);
    CHECK(step);
    CHECK(gilbert_order(sz) == o);
  }
}

TEST_CASE("territories are contiguous and every node gets a unique cell") {
  const std::vector<std::size_t> sizes = {1, 7, 30, 200, 2000};
  std::vector<std::uint32_t> group;
  for (std::size_t g = 0; g < sizes.size(); ++g) group.insert(group.end(), sizes[g], static_cast<std::uint32_t>(g));
  const std::size_t n = group.size();
  REQUIRE(n == 2238);
  std::vector<bool> active(n, true);
  std::vector<double> s(n);
  for (std::size_t i = 0; i < n; ++i) s[i] = std::sin(static_cast<double>(i) * 0.37);
  const LatticeSize sz = lattice_size(n);
  auto L = territory_layout(active, group, s, sz);
  std::set<std::int32_t> uniq;
  for (auto c : L.cell) {
    CHECK(c >= 0);
    CHECK(uniq.insert(c).second);
  }
  REQUIRE(L.territories.size() == sizes.size());
  const auto order = gilbert_order(sz);
  for (std::size_t g = 0; g < sizes.size(); ++g) {
    std::vector<std::int32_t> cells;
    for (std::size_t i = 0; i < n; ++i)
      if (group[i] == g) cells.push_back(L.cell[i]);
    CHECK(cells.size() == sizes[g]);
    CHECK(connected8(cells, sz.cols));
    const auto& t = L.territories[g];
    CHECK(connected8(std::vector<std::int32_t>(order.begin() + static_cast<std::ptrdiff_t>(t.begin),
                                               order.begin() + static_cast<std::ptrdiff_t>(t.end)), sz.cols));
    CHECK(t.end - t.begin >= sizes[g]);
  }
}

TEST_CASE("high groups are mountains and low groups craters") {
  const std::size_t na = 200, nb = 200, nc = 100;
  const std::size_t n = na + nb + nc;
  std::vector<std::uint32_t> group(n);
  std::vector<double> s(n);
  std::vector<bool> active(n, true);
  for (std::size_t i = 0; i < n; ++i) {
    group[i] = i < na ? 0 : (i < na + nb ? 1 : 2);
    s[i] = i < na ? 5.0 + 0.01 * static_cast<double>((i * 37) % na) + 1e-5 * static_cast<double>(i)
                  : (i < na + nb ? -5.0 - 0.01 * static_cast<double>((i * 53) % nb) - 1e-5 * static_cast<double>(i)
                                 : 0.001 * static_cast<double>(i));
  }
  const LatticeSize sz = lattice_size(n);
  auto L = territory_layout(active, group, s, sz);
  REQUIRE(L.territories.size() == 3);
  CHECK(L.territories[0].mountain);
  CHECK_FALSE(L.territories[1].mountain);
  for (std::size_t g = 0; g < 2; ++g) {
    const auto& t = L.territories[g];
    std::vector<double> sv, dv;
    std::size_t best = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (group[i] != g) continue;
      const double dx = L.cell[i] % static_cast<std::int32_t>(sz.cols) + 0.5 - t.cx;
      const double dy = L.cell[i] / static_cast<std::int32_t>(sz.cols) + 0.5 - t.cy;
      sv.push_back(s[i]);
      dv.push_back(std::hypot(dx, dy));
      if (!best || (g == 0 ? s[i] > s[best] : s[i] < s[best])) best = i;
    }
    const double rho = spearman(sv, dv);
    if (g == 0)
      CHECK(rho <= -0.9);
    else
      CHECK(rho >= 0.9);
    // the extreme node sits in the innermost ring (distance < 1 cell band of the nearest cell)
    const double bd = std::hypot(L.cell[best] % static_cast<std::int32_t>(sz.cols) + 0.5 - t.cx,
                                 L.cell[best] / static_cast<std::int32_t>(sz.cols) + 0.5 - t.cy);
    CHECK(std::floor(bd) == std::floor(*std::min_element(dv.begin(), dv.end())));
  }
}

TEST_CASE("territory edge cases: one group, empty groups, a single node, bad sizes") {
  const std::size_t n = 30;
  std::vector<bool> active(n, true);
  std::vector<double> s(n);
  for (std::size_t i = 0; i < n; ++i) s[i] = static_cast<double>(i);
  auto L = territory_layout(active, {}, s, lattice_size(n));
  CHECK(L.territories.size() == 1);
  std::set<std::int32_t> u(L.cell.begin(), L.cell.end());
  CHECK(u.size() == n);
  // group ids 0 and 5 only; the groups in between have no active nodes and no territory
  std::vector<std::uint32_t> g(n, 0);
  for (std::size_t i = 15; i < n; ++i) g[i] = 5;
  auto M = territory_layout(active, g, s, lattice_size(n));
  CHECK(M.territories.size() == 2);
  CHECK(M.territories[1].group == 5);
  // one active node; non-finite s
  std::vector<bool> one(n, false);
  one[4] = true;
  s[4] = std::numeric_limits<double>::quiet_NaN();
  auto O = territory_layout(one, {}, s, lattice_size(1));
  CHECK(O.cell[4] == 0);
  CHECK(O.cell[3] == -1);
  CHECK_THROWS(territory_layout(active, std::vector<std::uint32_t>(3, 0), s, lattice_size(n)));
  CHECK_THROWS(territory_layout(active, {}, s, LatticeSize{2, 2}));
}

namespace {
double ring_of(std::int32_t c, std::size_t cols, double cx, double cy) {
  const double dx = c % static_cast<std::int32_t>(cols) + 0.5 - cx, dy = c / static_cast<std::int32_t>(cols) + 0.5 - cy;
  return std::floor(std::sqrt(dx * dx + dy * dy));
}
double angle_of(std::int32_t c, std::size_t cols, double cx, double cy) {
  const double dx = c % static_cast<std::int32_t>(cols) + 0.5 - cx, dy = c / static_cast<std::int32_t>(cols) + 0.5 - cy;
  const double a = std::atan2(dy, dx);
  return a == std::numbers::pi ? -std::numbers::pi : a;
}
}  // namespace

TEST_CASE("inside a territory the cells form a spiral: ring by ring, each ring by ascending angle from -pi") {
  const std::size_t n = 1000;
  std::vector<bool> active(n, true);
  std::vector<double> s(n);
  for (std::size_t i = 0; i < n; ++i) s[i] = static_cast<double>(n - i);  // node i takes slot i (mountain)
  const LatticeSize sz = lattice_size(n);
  auto L = territory_layout(active, {}, s, sz);
  REQUIRE(L.territories.size() == 1);
  const auto& t = L.territories[0];
  double step = 0;
  for (std::size_t i = 1; i < n; ++i) {
    const double r0 = ring_of(L.cell[i - 1], sz.cols, t.cx, t.cy), r1 = ring_of(L.cell[i], sz.cols, t.cx, t.cy);
    CHECK(r0 <= r1);
    if (r0 == r1) {
      const double a0 = angle_of(L.cell[i - 1], sz.cols, t.cx, t.cy), a1 = angle_of(L.cell[i], sz.cols, t.cx, t.cy);
      CHECK((a0 < a1 || (a0 == a1 && L.cell[i - 1] < L.cell[i])));
    }
    const auto W = static_cast<std::int32_t>(sz.cols);
    step += std::hypot(L.cell[i] % W - L.cell[i - 1] % W, L.cell[i] / W - L.cell[i - 1] / W);
  }
  MESSAGE("mean distance between consecutive slots: " << step / static_cast<double>(n - 1));
  CHECK(step / static_cast<double>(n - 1) <= 1.6);  // consecutive ranks are neighbours, not opposite sides of a ring
}

namespace {
PlacementMemory memory_of(const TerritoryLayout& L, LatticeSize sz) { return PlacementMemory{sz, L.cell}; }
}  // namespace

TEST_CASE("placement hysteresis keeps cells for small rank changes and stays a bijection") {
  const std::size_t n = 390;  // 20 x 20 lattice: 10 spare cells
  std::vector<bool> active(n, true);
  std::vector<std::uint32_t> group(n);
  std::vector<double> s(n);
  for (std::size_t i = 0; i < n; ++i) {
    group[i] = i < 300 ? 0 : 1;
    s[i] = static_cast<double>(n - i);
  }
  const LatticeSize sz = lattice_size(n);
  const auto L0 = territory_layout(active, group, s, sz);
  const auto mem = memory_of(L0, sz);
  // identical input: identical cells
  CHECK(territory_layout(active, group, s, sz, &mem).cell == L0.cell);
  // swap the ranks of nodes 10 and 12 (2 slots apart, within the tolerance): both keep their cells
  auto s2 = s;
  std::swap(s2[10], s2[12]);
  auto L1 = territory_layout(active, group, s2, sz, &mem);
  CHECK(L1.cell == L0.cell);
  CHECK(territory_layout(active, group, s2, sz).cell != L0.cell);  // without memory they trade places
  // node 5 drops far down the ranking (beyond 0.15 * territory size): it moves; everything stays a bijection
  auto s3 = s;
  s3[5] = -1000;
  auto L2 = territory_layout(active, group, s3, sz, &mem);
  CHECK(L2.cell[5] != L0.cell[5]);
  std::set<std::int32_t> u(L2.cell.begin(), L2.cell.end());
  CHECK(u.size() == n);
  std::size_t kept = 0;
  for (std::size_t i = 0; i < n; ++i) kept += L2.cell[i] == L0.cell[i] ? 1 : 0;
  CHECK(kept >= n - 10);
  // nodes whose memory is withheld (-1) or whose previous cell lies outside their territory are placed afresh
  // (10 and 12 trade places)
  const auto fresh = territory_layout(active, group, s2, sz).cell;
  auto m2 = mem;
  m2.cell[10] = m2.cell[12] = -1;
  CHECK(territory_layout(active, group, s2, sz, &m2).cell == fresh);
  auto m3 = mem;
  REQUIRE(L0.territory[300] == 1);
  m3.cell[10] = L0.cell[300];  // a cell of the other territory
  m3.cell[12] = -1;
  CHECK(territory_layout(active, group, s2, sz, &m3).cell == fresh);
  // the memory survives a territory shift: one stock of group 0 leaves, so both ranges move by a few cells
  auto act2 = active;
  act2[299] = false;
  auto L3 = territory_layout(act2, group, s, sz, &mem);
  REQUIRE(L3.territories[0].end != L0.territories[0].end);
  std::size_t kept3 = 0;
  for (std::size_t i = 0; i < n; ++i) kept3 += act2[i] && L3.cell[i] == L0.cell[i] ? 1 : 0;
  CHECK(kept3 >= n - 15);
  // a different lattice size disables the memory
  auto m4 = mem;
  m4.size = LatticeSize{sz.cols + 1, sz.rows};
  CHECK(territory_layout(active, group, s2, sz, &m4).cell == territory_layout(active, group, s2, sz).cell);
  // zero tolerance still keeps a node whose ideal slot equals its old one, and the minimum is 2 slots
  CHECK(territory_layout(active, group, s2, sz, &mem, 0.0).cell == L0.cell);
}
