#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "geom/lattice.hpp"

namespace fx {

struct IdwParams {
  int subdivision = 4;
  double power = 2.0;
  int radius_cells = 3;
};

struct Raster {
  std::size_t w = 0, h = 0;
  std::vector<float> z;  // row-major [py * w + px]
  float zmin = 0, zmax = 0;
};

// Spec 6.3.
Raster idw_raster(const std::vector<std::int32_t>& cell, const std::vector<double>& value, LatticeSize size,
                  const IdwParams& p);

}  // namespace fx
