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

// Spec 6.2: cols = ceil(sqrt(n_active)), rows = ceil(n_active / cols).
LatticeSize lattice_size(std::size_t n_active);

}  // namespace fx
