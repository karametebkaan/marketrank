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
  bool operator==(const IdwParams&) const = default;
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

// CVT-weighted display smoother (a Lloyd-style relaxation on the raster grid; docs spec 6.3). Display only: stocks
// do not move and the table / tooltips stay exact.
struct CvtParams {
  int iterations = 12;     // 0 = identity
  double lambda = 0.6;     // relaxation step, (0, 1]
  double eps_frac = 0.1;   // density floor as a fraction of the P90 of |z|, (0, 10]
  bool operator==(const CvtParams&) const = default;
};

// In place. Vertices are the raster pixels; the cell neighbourhood is the 3x3 block with area weights 1 (centre),
// 1 (edge neighbours), 0.5 (diagonals), existing neighbours only. Density rho = eps + |z| with eps = eps_frac * P90(|z|)
// (eps = 1 when that P90 is 0). One Jacobi iteration: z' = (1-lambda) z + lambda * sum(a rho z) / sum(a rho).
// zmin/zmax are recomputed. Fixed-order sums, index-owned writes: bit-identical for any thread count.
// Throws on iterations < 0 or lambda / eps_frac outside range.
// Pins (Dirichlet constraint): after every iteration each pinned pixel is reset to its value, so the surface passes
// exactly through it and its neighbours relax toward it. No pins = the unpinned smoother, bit for bit.
struct CvtPin {
  std::size_t px;  // pixel index (py * w + px)
  double z;        // the value the surface must pass through
};
void cvt_smooth(Raster& r, const CvtParams& p, const std::vector<CvtPin>& pins = {});

}  // namespace mr
