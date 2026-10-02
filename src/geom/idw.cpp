#include "geom/idw.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fx {

Raster idw_raster(const std::vector<std::int32_t>& cell, const std::vector<double>& value, LatticeSize size,
                  const IdwParams& p) {
  Raster r;
  const long s = std::max(1, p.subdivision);
  const long cols = static_cast<long>(size.cols), rows = static_cast<long>(size.rows);
  r.w = static_cast<std::size_t>(cols * s);
  r.h = static_cast<std::size_t>(rows * s);
  r.z.assign(r.w * r.h, 0.0f);
  if (r.z.empty()) return r;
  std::vector<double> occ(size.cells(), std::numeric_limits<double>::quiet_NaN());
  bool any = false;
  for (std::size_t i = 0; i < cell.size(); ++i)
    if (cell[i] >= 0 && std::isfinite(value[i])) {
      occ[static_cast<std::size_t>(cell[i])] = value[i];
      any = true;
    }
  if (!any) return r;
  const long R = std::max(0, p.radius_cells);
  // Nearest occupied cell for every lattice cell: multi-source 8-connected BFS, seeded in ascending cell order.
  std::vector<std::int32_t> nearest(occ.size(), -1);
  {
    std::vector<std::int32_t> queue;
    queue.reserve(occ.size());
    for (std::size_t i = 0; i < occ.size(); ++i)
      if (!std::isnan(occ[i])) {
        nearest[i] = static_cast<std::int32_t>(i);
        queue.push_back(static_cast<std::int32_t>(i));
      }
    for (std::size_t head = 0; head < queue.size(); ++head) {
      const long c = queue[head], cx = c % cols, cy = c / cols;
      for (long dy = -1; dy <= 1; ++dy)
        for (long dx = -1; dx <= 1; ++dx) {
          const long nx = cx + dx, ny = cy + dy;
          if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;
          const std::size_t ni = static_cast<std::size_t>(ny * cols + nx);
          if (nearest[ni] >= 0) continue;
          nearest[ni] = nearest[static_cast<std::size_t>(c)];
          queue.push_back(static_cast<std::int32_t>(ni));
        }
    }
  }
#pragma omp parallel for schedule(static)
  for (long py = 0; py < static_cast<long>(r.h); ++py) {
    for (long px = 0; px < static_cast<long>(r.w); ++px) {
      const double u = (static_cast<double>(px) + 0.5) / static_cast<double>(s);
      const double v = (static_cast<double>(py) + 0.5) / static_cast<double>(s);
      const long c0 = static_cast<long>(u), r0 = static_cast<long>(v);
      double num = 0, den = 0, exact = std::numeric_limits<double>::quiet_NaN();
      double near_d = std::numeric_limits<double>::infinity(), near_val = 0;
      bool have_win = false;
      for (long rr = std::max(0L, r0 - R); rr <= std::min(rows - 1, r0 + R) && std::isnan(exact); ++rr)
        for (long cc = std::max(0L, c0 - R); cc <= std::min(cols - 1, c0 + R); ++cc) {
          const double val = occ[static_cast<std::size_t>(rr * cols + cc)];
          if (std::isnan(val)) continue;
          const double d = std::hypot(u - (static_cast<double>(cc) + 0.5), v - (static_cast<double>(rr) + 0.5));
          if (d < 1e-9) {
            exact = val;
            break;
          }
          have_win = true;
          if (d < near_d) {
            near_d = d;
            near_val = val;
          }
          const double w = 1.0 / std::pow(d, p.power);
          num += w * val;
          den += w;
        }
      double z;
      if (!std::isnan(exact)) {
        z = exact;
      } else if (have_win && std::isfinite(num) && std::isfinite(den) && den > 0) {
        z = num / den;
      } else if (have_win) {  // weights underflowed/overflowed: nearest in-window node
        z = near_val;
      } else {
        const long cc = std::min(cols - 1, c0), rr = std::min(rows - 1, r0);
        z = occ[static_cast<std::size_t>(nearest[static_cast<std::size_t>(rr * cols + cc)])];
      }
      r.z[static_cast<std::size_t>(py) * r.w + static_cast<std::size_t>(px)] = static_cast<float>(z);
    }
  }
  r.zmin = *std::min_element(r.z.begin(), r.z.end());
  r.zmax = *std::max_element(r.z.begin(), r.z.end());
  return r;
}

}  // namespace fx
