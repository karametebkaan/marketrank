#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "geom/lattice.hpp"

namespace fx {

// Generalized Hilbert ("gilbert2d", Cerveny) curve over a cols x rows rectangle. Visits every cell exactly once,
// consecutive cells at Chebyshev distance 1, starting at cell (0,0). Cell index = row*cols + col.
std::vector<std::int32_t> gilbert_order(LatticeSize s);

struct Territory {
  std::uint32_t group = 0;
  std::size_t begin = 0, end = 0;  // range in gilbert_order
  std::size_t active = 0;          // active nodes in the group
  double cx = 0, cy = 0;           // territory centre in cell units (mean of col+0.5, row+0.5)
  bool mountain = true;            // highest nodes at the centre; otherwise a crater
};

struct TerritoryLayout {
  std::vector<std::int32_t> cell;  // per node, -1 when inactive
  std::vector<Territory> territories;  // ascending group id; groups without active nodes are absent
};

// Sector-territory placement. Each group owns a consecutive gilbert range sized by its active count (spare cells by
// largest remainder, ties to the lower group id). Inside a territory the cells are ordered by squared distance to the
// territory centre (ties by cell index) and the group's nodes take them in order of rank value s: descending for a
// mountain (median s of the group >= median s of all active nodes), ascending for a crater; ties by node index.
// Non-finite s ranks as 0. The farthest cells of each territory stay empty.
// `group` must be empty (one group) or have size active.size(); `s` must have size active.size().
// Requires size.cells() >= active count. O(C log C), deterministic.
TerritoryLayout territory_layout(const std::vector<bool>& active, const std::vector<std::uint32_t>& group,
                                 const std::vector<double>& s, LatticeSize size);

}  // namespace fx
