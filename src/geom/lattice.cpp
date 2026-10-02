#include "geom/lattice.hpp"

#include <cmath>

namespace mr {

LatticeSize lattice_size(std::size_t n) {
  if (n == 0) return {0, 0};
  const auto cols = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
  return {cols, (n + cols - 1) / cols};
}

}  // namespace mr
