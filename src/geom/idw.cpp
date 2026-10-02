#include "geom/idw.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mr {

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

Raster smooth_raster(const Raster& r, double sigma_cells, int subdivision) {
  if (!std::isfinite(sigma_cells) || sigma_cells < 0) throw std::invalid_argument("smooth: sigma must be finite and >= 0");
  if (sigma_cells == 0 || r.z.empty()) return r;
  const double sigma = sigma_cells * std::max(1, subdivision);
  const long R = static_cast<long>(std::ceil(3.0 * sigma));
  std::vector<double> wk(static_cast<std::size_t>(2 * R + 1));
  for (long k = -R; k <= R; ++k) wk[static_cast<std::size_t>(k + R)] = std::exp(-0.5 * double(k * k) / (sigma * sigma));
  const long W = static_cast<long>(r.w), H = static_cast<long>(r.h);
  std::vector<double> tmp(r.z.size());
  // Rows pass: each output pixel is written by exactly one iteration; sums run in a fixed order.
#pragma omp parallel for schedule(static)
  for (long y = 0; y < H; ++y)
    for (long x = 0; x < W; ++x) {
      double s = 0, ws = 0;
      for (long k = std::max(-R, -x); k <= std::min(R, W - 1 - x); ++k) {
        const double w = wk[static_cast<std::size_t>(k + R)];
        s += w * r.z[static_cast<std::size_t>(y * W + x + k)];
        ws += w;
      }
      tmp[static_cast<std::size_t>(y * W + x)] = s / ws;
    }
  Raster out;
  out.w = r.w;
  out.h = r.h;
  out.z.assign(r.z.size(), 0.0f);
  // Columns pass into a separate buffer.
#pragma omp parallel for schedule(static)
  for (long x = 0; x < W; ++x)
    for (long y = 0; y < H; ++y) {
      double s = 0, ws = 0;
      for (long k = std::max(-R, -y); k <= std::min(R, H - 1 - y); ++k) {
        const double w = wk[static_cast<std::size_t>(k + R)];
        s += w * tmp[static_cast<std::size_t>((y + k) * W + x)];
        ws += w;
      }
      out.z[static_cast<std::size_t>(y * W + x)] = static_cast<float>(s / ws);
    }
  const auto [mn, mx] = std::minmax_element(out.z.begin(), out.z.end());
  out.zmin = *mn;
  out.zmax = *mx;
  return out;
}

void cvt_smooth(Raster& r, const CvtParams& p) {
  if (p.iterations < 0 || !(std::isfinite(p.lambda) && p.lambda > 0 && p.lambda <= 1) ||
      !(std::isfinite(p.eps_frac) && p.eps_frac > 0 && p.eps_frac <= 10))
    throw std::invalid_argument("cvt: iterations >= 0, lambda in (0, 1], eps_frac in (0, 10] required");
  if (p.iterations == 0 || r.z.empty()) return;
  const long W = static_cast<long>(r.w), H = static_cast<long>(r.h);
  const std::size_t n = r.z.size();
  std::vector<double> a(n), b(n), rho(n), rz(n);
  std::vector<double> mag(n);
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = r.z[i];
    mag[i] = std::fabs(a[i]);
  }
  const std::size_t k = static_cast<std::size_t>(0.9 * static_cast<double>(n - 1));
  std::nth_element(mag.begin(), mag.begin() + static_cast<std::ptrdiff_t>(k), mag.end());
  const double v = mag[k];
  const double eps = v > 0 ? p.eps_frac * v : 1.0;
  const double lam = p.lambda;
  for (int it = 0; it < p.iterations; ++it) {
#pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(n); ++i) {
      rho[static_cast<std::size_t>(i)] = eps + std::fabs(a[static_cast<std::size_t>(i)]);
      rz[static_cast<std::size_t>(i)] = rho[static_cast<std::size_t>(i)] * a[static_cast<std::size_t>(i)];
    }
#pragma omp parallel for schedule(static)
    for (long y = 0; y < H; ++y)
      for (long x = 0; x < W; ++x) {
        double num = 0, den = 0;
        for (long dy = -1; dy <= 1; ++dy) {
          const long yy = y + dy;
          if (yy < 0 || yy >= H) continue;
          for (long dx = -1; dx <= 1; ++dx) {
            const long xx = x + dx;
            if (xx < 0 || xx >= W) continue;
            const double w = (dx != 0 && dy != 0) ? 0.5 : 1.0;
            const std::size_t j = static_cast<std::size_t>(yy * W + xx);
            num += w * rz[j];
            den += w * rho[j];
          }
        }
        const std::size_t i = static_cast<std::size_t>(y * W + x);
        b[i] = (1.0 - lam) * a[i] + lam * (num / den);
      }
    a.swap(b);
  }
  for (std::size_t i = 0; i < n; ++i) r.z[i] = static_cast<float>(a[i]);
  const auto [mn, mx] = std::minmax_element(r.z.begin(), r.z.end());
  r.zmin = *mn;
  r.zmax = *mx;
}

}  // namespace mr
