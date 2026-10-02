#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "geom/lattice.hpp"

namespace mr {

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

// Display smoothing: separable Gaussian with sigma = sigma_cells·subdivision pixels and radius
// ceil(3·sigma_cells·subdivision); edge windows are renormalized by their in-window weight sum. sigma 0 is the
// identity. zmin/zmax are recomputed. Deterministic for any thread count. Throws on negative/non-finite sigma.
Raster smooth_raster(const Raster& r, double sigma_cells, int subdivision);

}  // namespace mr
