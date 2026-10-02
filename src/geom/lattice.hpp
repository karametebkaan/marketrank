#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fx {

struct LatticeSize {
  std::size_t cols = 0, rows = 0;
  std::size_t cells() const { return cols * rows; }
  bool operator==(const LatticeSize&) const = default;
};

LatticeSize lattice_size(std::size_t n_active);
// Spec 6.2: sort by x into column bands of `rows` nodes, then by y within a band. cell = row*cols+col.
std::vector<std::int32_t> rcb_assign(const std::vector<double>& xy, const std::vector<bool>& active,
                                     LatticeSize size);
// Keep prev cell when within max_shift (Chebyshev) and free; else new cell if free; else nearest free.
std::vector<std::int32_t> apply_hysteresis(const std::vector<std::int32_t>& prev,
                                           const std::vector<std::int32_t>& next, LatticeSize size,
                                           int max_shift);

}  // namespace fx
