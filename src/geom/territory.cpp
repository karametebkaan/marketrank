#include "geom/territory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <map>
#include <numbers>
#include <set>
#include <stdexcept>

namespace mr {

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
                                 const std::vector<double>& s, LatticeSize size, const PlacementMemory* prev,
                                 double rank_tolerance, const std::vector<bool>& median_exclude) {
  const std::size_t n = active.size();
  if (s.size() != n) throw std::invalid_argument("territory_layout: s size mismatch");
  if (!group.empty() && group.size() != n) throw std::invalid_argument("territory_layout: group size mismatch");
  if (!median_exclude.empty() && median_exclude.size() != n)
    throw std::invalid_argument("territory_layout: median_exclude size mismatch");
  auto counted = [&](std::uint32_t i) { return median_exclude.empty() || !median_exclude[i]; };
  if (prev && prev->cell.size() != n)
    throw std::invalid_argument("territory_layout: memory size mismatch");
  TerritoryLayout out;
  out.cell.assign(n, -1);
  out.territory.assign(n, -1);
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
  const bool use_prev = prev && prev->size == size;

  std::vector<double> all;
  all.reserve(A);
  for (const auto& kv : members)
    for (auto i : kv.second)
      if (counted(i)) all.push_back(rank(i));
  if (all.empty())
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
  std::vector<std::int32_t> slot_of(use_prev ? C : 0, -1);  // cell -> slot in its territory's spiral
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
    // Spiral: ring, then angle from -pi (pi itself is mapped to -pi), then cell index.
    struct Key {
      double ring, angle;
      std::int32_t c;
    };
    std::vector<Key> keys;
    keys.reserve(cells.size());
    for (auto c : cells) {
      const double dx = c % cols + 0.5 - t.cx, dy = c / cols + 0.5 - t.cy;
      double a = std::atan2(dy, dx);
      if (a == std::numbers::pi) a = -std::numbers::pi;
      keys.push_back({std::floor(std::sqrt(dx * dx + dy * dy)), a, c});
    }
    std::sort(keys.begin(), keys.end(), [](const Key& a, const Key& b) {
      if (a.ring != b.ring) return a.ring < b.ring;
      if (a.angle != b.angle) return a.angle < b.angle;
      return a.c < b.c;
    });
    for (std::size_t k = 0; k < keys.size(); ++k) cells[k] = keys[k].c;
    std::vector<double> sv;
    for (auto i : kv.second)
      if (counted(i)) sv.push_back(rank(i));
    if (sv.empty())
      for (auto i : kv.second) sv.push_back(rank(i));
    t.mountain = median_of(sv) >= med_all;
    std::vector<std::uint32_t> nodes = kv.second;  // ascending index
    std::stable_sort(nodes.begin(), nodes.end(), [&](auto a, auto b) {
      return t.mountain ? rank(a) > rank(b) : rank(a) < rank(b);
    });
    const auto tid = static_cast<std::int32_t>(out.territories.size());
    for (auto i : nodes) out.territory[i] = tid;
    if (!use_prev) {
      for (std::size_t k = 0; k < nodes.size(); ++k) out.cell[nodes[k]] = cells[k];
      out.territories.push_back(t);
      continue;
    }
    // Hysteresis, in slots of this territory's spiral.
    for (std::size_t k = 0; k < cells.size(); ++k) slot_of[static_cast<std::size_t>(cells[k])] = static_cast<std::int32_t>(k);
    const double tol = std::max(2.0, rank_tolerance * static_cast<double>(cells.size()));
    std::vector<char> taken(cells.size(), 0);
    std::vector<std::int64_t> slot(nodes.size(), -1);  // by rank position
    for (std::size_t k = 0; k < nodes.size(); ++k) {  // pass 1: keep the previous cell
      const auto i = nodes[k];
      const std::int32_t pc = prev->cell[i];
      if (pc < 0 || static_cast<std::size_t>(pc) >= C) continue;
      const std::int32_t ps = slot_of[static_cast<std::size_t>(pc)];
      if (ps < 0 || taken[static_cast<std::size_t>(ps)]) continue;
      if (std::fabs(static_cast<double>(ps) - static_cast<double>(k)) > tol) continue;
      slot[k] = ps;
      taken[static_cast<std::size_t>(ps)] = 1;
    }
    for (std::size_t k = 0; k < nodes.size(); ++k)  // pass 2: the ideal slot if free
      if (slot[k] < 0 && !taken[k]) {
        slot[k] = static_cast<std::int64_t>(k);
        taken[k] = 1;
      }
    std::set<std::size_t> free;
    for (std::size_t k = 0; k < cells.size(); ++k)
      if (!taken[k]) free.insert(k);
    for (std::size_t k = 0; k < nodes.size(); ++k) {  // pass 3: the nearest free slot, ties to the lower one
      if (slot[k] >= 0) continue;
      auto hi = free.lower_bound(k);
      auto pick = hi;
      if (hi == free.end() || (hi != free.begin() && k - *std::prev(hi) <= *hi - k)) pick = std::prev(hi);
      slot[k] = static_cast<std::int64_t>(*pick);
      free.erase(pick);
    }
    for (std::size_t k = 0; k < nodes.size(); ++k) out.cell[nodes[k]] = cells[static_cast<std::size_t>(slot[k])];
    for (auto c : cells) slot_of[static_cast<std::size_t>(c)] = -1;
    out.territories.push_back(t);
  }
  return out;
}

}  // namespace mr
