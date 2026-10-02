#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <set>
#include <vector>

#include "geom/territory.hpp"

using namespace fx;

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
    // the extreme node sits at (one of) the cells nearest the centre
    const double bd = std::hypot(L.cell[best] % static_cast<std::int32_t>(sz.cols) + 0.5 - t.cx,
                                 L.cell[best] / static_cast<std::int32_t>(sz.cols) + 0.5 - t.cy);
    CHECK(bd <= *std::min_element(dv.begin(), dv.end()) + 1e-12);
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
