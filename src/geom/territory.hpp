#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "geom/lattice.hpp"

namespace mr {

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
  std::vector<std::int32_t> cell;       // per node, -1 when inactive
  std::vector<std::int32_t> territory;  // per node, index into territories; -1 when inactive
  std::vector<Territory> territories;   // ascending group id; groups without active nodes are absent
};

// The previous frame's placement, for cell hysteresis. The caller withholds (cell -1) every node that was inactive
// last frame or has changed community or group.
struct PlacementMemory {
  LatticeSize size;                // lattice of the previous frame; a different size disables the memory
  std::vector<std::int32_t> cell;  // per node: previous cell, or -1
};

// Sector-territory placement. Each group owns a consecutive gilbert range sized by its active count (spare cells by
// largest remainder, ties to the lower group id). Inside a territory the cells form a spiral: ordered by ring
// floor(sqrt(d2)) around the territory centre, then by angle atan2(dy, dx) ascending from -pi, then by cell index.
// The group's nodes take the spiral slots in order of rank value s: descending for a mountain (median s of the group
// >= median s of all active nodes), ascending for a crater; ties by node index. Non-finite s ranks as 0. The outer
// slots of each territory stay empty.
// Hysteresis (when `prev` is given): a node keeps its previous cell when the memory has one for it, the lattice size
// is unchanged, that cell still lies inside the node's territory, and the cell's slot in the territory's current
// spiral is within max(2, rank_tolerance * territory cells) of the node's new slot. (Requiring an identical territory
// range would disable the memory on real data, where ~50 stocks join or leave per bar and every range shifts.)
// Conflicts resolve as in a 3-pass assignment, each pass in rank order: kept cells first, then the ideal slot if
// free, then the nearest free slot of the territory (ties to the lower slot).
// `median_exclude` (empty, or size active.size()) leaves nodes out of both medians (a group with only excluded
// nodes uses all of them); they are still placed.
// `group` must be empty (one group) or have size active.size(); `s` must have size active.size().
// Requires size.cells() >= active count. O(C log C), deterministic.
TerritoryLayout territory_layout(const std::vector<bool>& active, const std::vector<std::uint32_t>& group,
                                 const std::vector<double>& s, LatticeSize size, const PlacementMemory* prev = nullptr,
                                 double rank_tolerance = 0.15, const std::vector<bool>& median_exclude = {});

}  // namespace mr
