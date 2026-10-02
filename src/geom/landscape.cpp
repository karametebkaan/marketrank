#include "geom/landscape.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace fx {

HeightMode parse_height_mode(std::string_view s) {
  if (s == "signed-log") return HeightMode::SignedLog;
  if (s == "linear") return HeightMode::Linear;
  throw std::invalid_argument("unknown height mode: " + std::string(s));
}

std::string_view to_string(HeightMode m) { return m == HeightMode::SignedLog ? "signed-log" : "linear"; }

double display_height(double h, HeightMode m) {
  if (!std::isfinite(h)) return std::numeric_limits<double>::quiet_NaN();
  if (m == HeightMode::Linear) return h;
  return h >= 0 ? std::log1p(h) : -std::log1p(-h);
}

std::vector<LandscapeArc> top_arcs(const Csr& P, const std::vector<bool>& active, std::size_t max_arcs) {
  if (active.size() != P.n) throw std::invalid_argument("top_arcs: active size != P.n");
  std::vector<LandscapeArc> arcs;
  for (std::size_t i = 0; i < P.n; ++i) {
    if (!active[i]) continue;
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) {
      const auto j = P.col[e];
      if (j != i && active[j] && P.raw[e] > 0) arcs.push_back({static_cast<std::uint32_t>(i), j, P.raw[e]});
    }
  }
  auto better = [](const LandscapeArc& x, const LandscapeArc& y) {
    if (x.w != y.w) return x.w > y.w;
    if (x.a != y.a) return x.a < y.a;
    return x.b < y.b;
  };
  if (arcs.size() > max_arcs) {
    std::nth_element(arcs.begin(), arcs.begin() + static_cast<std::ptrdiff_t>(max_arcs), arcs.end(), better);
    arcs.resize(max_arcs);
  }
  std::sort(arcs.begin(), arcs.end(), better);
  return arcs;
}

Raster delta_raster(const LandscapeFrame& base, const std::vector<double>& delta, const LandscapeParams& p) {
  if (delta.size() != base.n) throw std::invalid_argument("delta_raster: delta size != base n");
  std::vector<std::int32_t> cell(delta.size(), -1);
  std::vector<double> v(delta.size(), std::numeric_limits<double>::quiet_NaN());
  for (const auto& nd : base.nodes) {
    cell[nd.i] = nd.cell;
    v[nd.i] = display_height(delta[nd.i], p.height);
  }
  return idw_raster(cell, v, base.size, p.idw);
}

LandscapeBuilder::LandscapeBuilder(std::size_t n, LandscapeParams params)
    : n_(n), p_(params), embed_(n, params.embed) {}

LandscapeFrame LandscapeBuilder::build(const Frame& f) {
  const auto t0 = std::chrono::steady_clock::now();
  if (f.active.size() != n_) throw std::invalid_argument("LandscapeBuilder: frame size mismatch");
  if (f.h.size() != n_ || f.pi.size() != n_ || f.P.n != n_ ||
      (!f.forecasts.empty() && f.forecasts.front().score.size() != n_))
    throw std::invalid_argument("LandscapeBuilder: frame vector sizes mismatch");
  xy_ = embed_.positions(f);
  std::size_t n_active = 0;
  double minx = 1e300, maxx = -1e300, miny = 1e300, maxy = -1e300;
  for (std::size_t i = 0; i < n_; ++i) {
    if (!f.active[i]) continue;
    ++n_active;
    minx = std::min(minx, xy_[2 * i]);
    maxx = std::max(maxx, xy_[2 * i]);
    miny = std::min(miny, xy_[2 * i + 1]);
    maxy = std::max(maxy, xy_[2 * i + 1]);
  }
  const LatticeSize size = lattice_size(n_active);
  auto next = rcb_assign(xy_, f.active, size);
  cells_ = (!first_ && size == size_) ? apply_hysteresis(cells_, next, size, p_.max_shift) : next;
  size_ = size;
  first_ = false;

  LandscapeFrame lf;
  lf.t = f.t;
  lf.n = n_;
  lf.size = size;
  std::vector<double> values(n_, std::numeric_limits<double>::quiet_NaN());
  const double rx = maxx > minx ? maxx - minx : 1.0, ry = maxy > miny ? maxy - miny : 1.0;
  const auto& score = f.forecasts.empty() ? f.h : f.forecasts.front().score;
  for (std::size_t i = 0; i < n_; ++i) {
    if (!f.active[i]) continue;
    const double hd = display_height(f.h[i], p_.height);
    values[i] = hd;
    lf.nodes.push_back({static_cast<std::uint32_t>(i), cells_[i], static_cast<float>((xy_[2 * i] - minx) / rx),
                        static_cast<float>((xy_[2 * i + 1] - miny) / ry), f.h[i], hd, f.pi[i], score[i]});
  }
  lf.raster = idw_raster(cells_, values, size, p_.idw);
  lf.arcs = top_arcs(f.P, f.active, p_.max_arcs);
  lf.compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return lf;
}

}  // namespace fx
