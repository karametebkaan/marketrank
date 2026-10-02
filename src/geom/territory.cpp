#include "geom/territory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <stdexcept>

namespace fx {

namespace {

long fdiv2(long a) { return a >= 0 ? a / 2 : -((-a + 1) / 2); }  // floor(a / 2)
long sgn(long a) { return (a > 0) - (a < 0); }

void gen(std::vector<std::int32_t>& out, long cols, long x, long y, long ax, long ay, long bx, long by) {
  const long w = std::labs(ax + ay), h = std::labs(bx + by);
  const long dax = sgn(ax), day = sgn(ay), dbx = sgn(bx), dby = sgn(by);
  if (h == 1) {
    for (long i = 0; i < w; ++i, x += dax, y += day) out.push_back(static_cast<std::int32_t>(y * cols + x));
    return;
  }
  if (w == 1) {
    for (long i = 0; i < h; ++i, x += dbx, y += dby) out.push_back(static_cast<std::int32_t>(y * cols + x));
    return;
  }
  long ax2 = fdiv2(ax), ay2 = fdiv2(ay), bx2 = fdiv2(bx), by2 = fdiv2(by);
  const long w2 = std::labs(ax2 + ay2), h2 = std::labs(bx2 + by2);
  if (2 * w > 3 * h) {
    if ((w2 % 2) && w > 2) {
      ax2 += dax;
      ay2 += day;
    }
    gen(out, cols, x, y, ax2, ay2, bx, by);
    gen(out, cols, x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by);
  } else {
    if ((h2 % 2) && h > 2) {
      bx2 += dbx;
      by2 += dby;
    }
    gen(out, cols, x, y, bx2, by2, ax2, ay2);
    gen(out, cols, x + bx2, y + by2, ax, ay, bx - bx2, by - by2);
    gen(out, cols, x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby), -bx2, -by2, -(ax - ax2), -(ay - ay2));
  }
}

double median_of(std::vector<double> v) {
  if (v.empty()) return 0.0;
  const std::size_t m = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m), v.end());
  const double hi = v[m];
  if (v.size() % 2) return hi;
  return 0.5 * (hi + *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m)));
}

}  // namespace

std::vector<std::int32_t> gilbert_order(LatticeSize s) {
  std::vector<std::int32_t> out;
  if (s.cols == 0 || s.rows == 0) return out;
  out.reserve(s.cells());
  const long w = static_cast<long>(s.cols), h = static_cast<long>(s.rows);
  if (w >= h)
    gen(out, w, 0, 0, w, 0, 0, h);
  else
    gen(out, w, 0, 0, 0, h, w, 0);
  return out;
}

TerritoryLayout territory_layout(const std::vector<bool>& active, const std::vector<std::uint32_t>& group,
                                 const std::vector<double>& s, LatticeSize size) {
  const std::size_t n = active.size();
  if (s.size() != n) throw std::invalid_argument("territory_layout: s size mismatch");
  if (!group.empty() && group.size() != n) throw std::invalid_argument("territory_layout: group size mismatch");
  TerritoryLayout out;
  out.cell.assign(n, -1);
  std::map<std::uint32_t, std::vector<std::uint32_t>> members;  // ascending group id
  std::size_t A = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (active[i]) {
      members[group.empty() ? 0u : group[i]].push_back(static_cast<std::uint32_t>(i));
      ++A;
    }
  if (A == 0) return out;
  const std::size_t C = size.cells();
  if (C < A) throw std::invalid_argument("territory_layout: lattice smaller than the active count");
  auto rank = [&](std::uint32_t i) { return std::isfinite(s[i]) ? s[i] : 0.0; };

  std::vector<double> all;
  all.reserve(A);
  for (const auto& kv : members)
    for (auto i : kv.second) all.push_back(rank(i));
  const double med_all = median_of(all);

  // Territory sizes: a_g plus a largest-remainder share of the C - A spare cells.
  const std::uint64_t spare = C - A;
  std::vector<std::size_t> cg;
  std::vector<std::uint64_t> rem;
  std::uint64_t given = 0;
  for (const auto& kv : members) {
    const std::uint64_t a = kv.second.size(), q = a * spare / A;
    cg.push_back(static_cast<std::size_t>(a + q));
    rem.push_back(a * spare % A);
    given += q;
  }
  std::vector<std::size_t> idx(cg.size());
  for (std::size_t k = 0; k < idx.size(); ++k) idx[k] = k;
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return rem[a] > rem[b]; });
  for (std::uint64_t k = 0; k < spare - given; ++k) ++cg[idx[static_cast<std::size_t>(k)]];

  const auto order = gilbert_order(size);
  const auto cols = static_cast<std::int32_t>(size.cols);
  std::size_t pos = 0, g = 0;
  for (const auto& kv : members) {
    Territory t;
    t.group = kv.first;
    t.begin = pos;
    t.end = pos + cg[g];
    t.active = kv.second.size();
    pos = t.end;
    ++g;
    std::vector<std::int32_t> cells(order.begin() + static_cast<std::ptrdiff_t>(t.begin),
                                    order.begin() + static_cast<std::ptrdiff_t>(t.end));
    double sx = 0, sy = 0;
    for (auto c : cells) {
      sx += c % cols + 0.5;
      sy += c / cols + 0.5;
    }
    t.cx = sx / static_cast<double>(cells.size());
    t.cy = sy / static_cast<double>(cells.size());
    auto d2 = [&](std::int32_t c) {
      const double dx = c % cols + 0.5 - t.cx, dy = c / cols + 0.5 - t.cy;
      return dx * dx + dy * dy;
    };
    std::sort(cells.begin(), cells.end(), [&](auto a, auto b) {
      const double da = d2(a), db = d2(b);
      return da < db || (da == db && a < b);
    });
    std::vector<double> sv;
    for (auto i : kv.second) sv.push_back(rank(i));
    t.mountain = median_of(sv) >= med_all;
    std::vector<std::uint32_t> nodes = kv.second;  // ascending index
    std::stable_sort(nodes.begin(), nodes.end(), [&](auto a, auto b) {
      return t.mountain ? rank(a) > rank(b) : rank(a) < rank(b);
    });
    for (std::size_t k = 0; k < nodes.size(); ++k) out.cell[nodes[k]] = cells[k];
    out.territories.push_back(t);
  }
  return out;
}

}  // namespace fx
