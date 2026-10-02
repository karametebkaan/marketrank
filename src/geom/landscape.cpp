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

LandscapeBuilder::LandscapeBuilder(std::size_t n, LandscapeParams params, std::vector<std::uint32_t> group)
    : n_(n), p_(params), group_(std::move(group)), s_prev_(n, 0.0), has_prev_(n, 0) {
  if (!group_.empty() && group_.size() != n) throw std::invalid_argument("LandscapeBuilder: group size mismatch");
  if (!(p_.order_smoothing >= 0.0 && p_.order_smoothing <= 1.0))
    throw std::invalid_argument("LandscapeBuilder: order_smoothing must be in [0, 1]");
}

LandscapeFrame LandscapeBuilder::build(const Frame& f) {
  const auto t0 = std::chrono::steady_clock::now();
  if (f.active.size() != n_) throw std::invalid_argument("LandscapeBuilder: frame size mismatch");
  if (f.h.size() != n_ || f.pi.size() != n_ || f.P.n != n_ ||
      (!f.forecasts.empty() && f.forecasts.front().score.size() != n_))
    throw std::invalid_argument("LandscapeBuilder: frame vector sizes mismatch");
  std::size_t n_active = 0;
  for (std::size_t i = 0; i < n_; ++i) n_active += f.active[i] ? 1 : 0;
  const LatticeSize size = lattice_size(n_active);
  // Ranking hotness: signed-log h blended with the previous frame's value; non-finite h ranks as 0.
  std::vector<double> rank(n_, 0.0);
  for (std::size_t i = 0; i < n_; ++i) {
    if (!f.active[i]) {
      has_prev_[i] = 0;
      continue;
    }
    double cur = display_height(f.h[i], p_.height);
    if (!std::isfinite(cur)) cur = 0.0;
    rank[i] = has_prev_[i] ? (1.0 - p_.order_smoothing) * cur + p_.order_smoothing * s_prev_[i] : cur;
    s_prev_[i] = rank[i];
    has_prev_[i] = 1;
  }
  const std::vector<std::int32_t> cells = territory_layout(f.active, group_, rank, size).cell;

  LandscapeFrame lf;
  lf.t = f.t;
  lf.n = n_;
  lf.size = size;
  const auto cols = static_cast<std::int32_t>(size.cols);
  std::vector<double> values(n_, std::numeric_limits<double>::quiet_NaN());
  const auto& score = f.forecasts.empty() ? f.h : f.forecasts.front().score;
  for (std::size_t i = 0; i < n_; ++i) {
    if (!f.active[i]) continue;
    const double hd = display_height(f.h[i], p_.height);
    values[i] = hd;
    lf.nodes.push_back({static_cast<std::uint32_t>(i), cells[static_cast<std::size_t>(i)],
                        static_cast<float>((cells[i] % cols + 0.5) / static_cast<double>(size.cols)),
                        static_cast<float>((cells[i] / cols + 0.5) / static_cast<double>(size.rows)), f.h[i], hd, f.pi[i], score[i]});
  }
  lf.raster = idw_raster(cells, values, size, p_.idw);
  lf.arcs = top_arcs(f.P, f.active, p_.max_arcs);
  lf.compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return lf;
}

}  // namespace fx
