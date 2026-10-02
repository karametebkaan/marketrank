#include "geom/lattice.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace fx {

LatticeSize lattice_size(std::size_t n) {
  if (n == 0) return {0, 0};
  const auto cols = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
  return {cols, (n + cols - 1) / cols};
}

std::vector<std::int32_t> rcb_assign(const std::vector<double>& xy, const std::vector<bool>& active,
                                     LatticeSize size) {
  const std::size_t n = active.size();
  std::vector<std::int32_t> cell(n, -1);
  std::vector<std::uint32_t> ids;
  for (std::size_t i = 0; i < n; ++i)
    if (active[i]) ids.push_back(static_cast<std::uint32_t>(i));
  if (ids.empty() || size.rows == 0) return cell;
  std::sort(ids.begin(), ids.end(), [&](auto a, auto b) {
    return xy[2 * a] < xy[2 * b] || (xy[2 * a] == xy[2 * b] && a < b);
  });
  for (std::size_t band = 0; band * size.rows < ids.size(); ++band) {
    const std::size_t lo = band * size.rows, hi = std::min(ids.size(), lo + size.rows);
    std::sort(ids.begin() + static_cast<std::ptrdiff_t>(lo), ids.begin() + static_cast<std::ptrdiff_t>(hi),
              [&](auto a, auto b) { return xy[2 * a + 1] < xy[2 * b + 1] || (xy[2 * a + 1] == xy[2 * b + 1] && a < b); });
    for (std::size_t r = lo; r < hi; ++r)
      cell[ids[r]] = static_cast<std::int32_t>((r - lo) * size.cols + band);
  }
  return cell;
}

std::vector<std::int32_t> apply_hysteresis(const std::vector<std::int32_t>& prev,
                                           const std::vector<std::int32_t>& next, LatticeSize size,
                                           int max_shift) {
  const std::size_t n = next.size();
  const auto cols = static_cast<long>(size.cols), rows = static_cast<long>(size.rows);
  for (std::size_t i = 0; i < n; ++i)
    if (next[i] >= 0 && (size.cols == 0 || static_cast<std::size_t>(next[i]) >= size.cells()))
      throw std::invalid_argument("apply_hysteresis: next cell out of range");
  std::vector<std::int32_t> out(n, -1);
  std::vector<char> taken(size.cells(), 0);
  auto cheb = [&](long a, long b) { return std::max(std::labs(a % cols - b % cols), std::labs(a / cols - b / cols)); };
  for (std::size_t i = 0; i < n; ++i) {  // pass 1: keep nearby previous cells
    if (next[i] < 0 || i >= prev.size() || prev[i] < 0 || static_cast<std::size_t>(prev[i]) >= size.cells()) continue;
    if (cheb(prev[i], next[i]) <= max_shift && !taken[static_cast<std::size_t>(prev[i])]) {
      out[i] = prev[i];
      taken[static_cast<std::size_t>(prev[i])] = 1;
    }
  }
  for (std::size_t i = 0; i < n; ++i) {  // pass 2: new cell if free
    if (next[i] < 0 || out[i] >= 0 || taken[static_cast<std::size_t>(next[i])]) continue;
    out[i] = next[i];
    taken[static_cast<std::size_t>(next[i])] = 1;
  }
  // Too many displaced nodes: no stability worth preserving; next is a bijection by construction.
  std::size_t leftovers = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (next[i] >= 0 && out[i] < 0) ++leftovers;
  if (leftovers > 64) return next;
  // pass 3: nearest free cell, ring by ring. Each ring is walked on its perimeter only (O(r)), in row-major
  // order: top row, then for each middle row the left cell then the right cell, then the bottom row.
  for (std::size_t i = 0; i < n; ++i) {
    if (next[i] < 0 || out[i] >= 0) continue;
    const long c0 = next[i] % cols, r0 = next[i] / cols;
    auto try_cell = [&](long dc, long dr) {
      const long c = c0 + dc, rr = r0 + dr;
      if (c < 0 || c >= cols || rr < 0 || rr >= rows) return;
      const auto k = static_cast<std::size_t>(rr * cols + c);
      if (!taken[k]) {
        out[i] = static_cast<std::int32_t>(k);
        taken[k] = 1;
      }
    };
    for (long r = 1; r <= std::max(cols, rows) && out[i] < 0; ++r) {
      for (long dc = -r; dc <= r && out[i] < 0; ++dc) try_cell(dc, -r);
      for (long dr = -r + 1; dr <= r - 1 && out[i] < 0; ++dr) {
        try_cell(-r, dr);
        if (out[i] < 0) try_cell(r, dr);
      }
      for (long dc = -r; dc <= r && out[i] < 0; ++dc) try_cell(dc, r);
    }
  }
  return out;
}

}  // namespace fx
