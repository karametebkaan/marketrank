#include "geom/lattice.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

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
  for (std::size_t i = 0; i < n; ++i) {  // pass 3: nearest free cell, ring by ring
    if (next[i] < 0 || out[i] >= 0) continue;
    const long c0 = next[i] % cols, r0 = next[i] / cols;
    for (long r = 1; r <= std::max(cols, rows) && out[i] < 0; ++r) {
      for (long dr = -r; dr <= r && out[i] < 0; ++dr)
        for (long dc = -r; dc <= r && out[i] < 0; ++dc) {
          if (std::max(std::labs(dc), std::labs(dr)) != r) continue;
          const long c = c0 + dc, rr = r0 + dr;
          if (c < 0 || c >= cols || rr < 0 || rr >= rows) continue;
          const auto k = static_cast<std::size_t>(rr * cols + c);
          if (!taken[k]) {
            out[i] = static_cast<std::int32_t>(k);
            taken[k] = 1;
          }
        }
    }
  }
  return out;
}

}  // namespace fx
