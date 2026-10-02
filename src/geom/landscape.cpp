#include "geom/landscape.hpp"

#include "graph/hotness.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace mr {

HeightMode parse_height_mode(std::string_view s) {
  if (s == "signed-log") return HeightMode::SignedLog;
  if (s == "linear") return HeightMode::Linear;
  throw std::invalid_argument("unknown height mode: " + std::string(s));
}

TerritoryMode parse_territory_mode(std::string_view s) {
  if (s == "flux") return TerritoryMode::Flux;
  if (s == "sector") return TerritoryMode::Sector;
  throw std::invalid_argument("unknown territory mode: " + std::string(s));
}

std::string_view to_string(TerritoryMode m) { return m == TerritoryMode::Flux ? "flux" : "sector"; }

std::string_view to_string(HeightMode m) { return m == HeightMode::SignedLog ? "signed-log" : "linear"; }

double display_height(double h, HeightMode m) {
  if (!std::isfinite(h)) return std::numeric_limits<double>::quiet_NaN();
  if (m == HeightMode::Linear) return h;
  return h >= 0 ? std::log1p(h) : -std::log1p(-h);
}

Smoother parse_smoother(std::string_view s) {
  if (s == "cvt") return Smoother::Cvt;
  if (s == "gaussian") return Smoother::Gaussian;
  if (s == "none") return Smoother::None;
  throw std::invalid_argument("unknown smoother: " + std::string(s));
}

std::string_view to_string(Smoother s) { return s == Smoother::Cvt ? "cvt" : s == Smoother::Gaussian ? "gaussian" : "none"; }

Raster apply_smoother(Raster r, const LandscapeParams& p) {
  switch (p.smoother) {
    case Smoother::Gaussian: return smooth_raster(r, p.smooth, p.idw.subdivision);
    case Smoother::Cvt: cvt_smooth(r, p.cvt); return r;
    case Smoother::None: break;
  }
  return r;
}

LandscapeValue parse_landscape_value(std::string_view s) {
  if (s == "pi") return LandscapeValue::Pi;
  if (s == "hotness") return LandscapeValue::Hotness;
  if (s == "pi_rel_size") return LandscapeValue::PiRelSize;
  throw std::invalid_argument("unknown landscape value: " + std::string(s));
}

std::string_view to_string(LandscapeValue v) {
  switch (v) {
    case LandscapeValue::Pi: return "pi";
    case LandscapeValue::Hotness: return "hotness";
    case LandscapeValue::PiRelSize: return "pi_rel_size";
  }
  return "?";
}

LandscapeValue default_landscape_value(std::string_view preset) {
  return preset == "marketrank" ? LandscapeValue::PiRelSize : LandscapeValue::Hotness;
}

double landscape_value(double h, double pi, std::size_t n_active, double size_share, LandscapeValue v,
                       HeightMode height) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (v == LandscapeValue::Hotness) return display_height(h, height);
  const double s = v == LandscapeValue::Pi ? pi * static_cast<double>(n_active)
                   : std::isfinite(size_share) && size_share > 0 ? pi / size_share
                                                                 : nan;
  return std::isfinite(s) && s > 0 ? std::log(s) : nan;
}

bool same_placement(const LandscapeParams& a, const LandscapeParams& b) {
  return a.value == b.value && a.order_smoothing == b.order_smoothing && a.rank_tolerance == b.rank_tolerance && a.max_arcs == b.max_arcs &&
         a.territory == b.territory && a.recluster_bars == b.recluster_bars && a.resolution == b.resolution &&
         a.warmup_bars == b.warmup_bars;
}

namespace {
// IDW + display smoothing of the frame's node values (hdisp) at their cells.
Raster node_raster(const LandscapeFrame& f, const LandscapeParams& p) {
  std::vector<std::int32_t> cell(f.n, -1);
  std::vector<double> v(f.n, std::numeric_limits<double>::quiet_NaN());
  for (const auto& nd : f.nodes) {
    cell[nd.i] = nd.cell;
    v[nd.i] = nd.hdisp;
  }
  return apply_smoother(idw_raster(cell, v, f.size, p.idw), p);
}
}  // namespace

LandscapeFrame restyle(const LandscapeFrame& f, const LandscapeParams& p) {
  LandscapeFrame out = f;
  for (auto& nd : out.nodes) nd.hdisp = landscape_value(nd.h, nd.pi, out.nodes.size(), nd.size_share, p.value, p.height);
  out.raster = node_raster(out, p);
  return out;
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
  return apply_smoother(idw_raster(cell, v, base.size, p.idw), p);
}

LandscapeBuilder::LandscapeBuilder(std::size_t n, LandscapeParams params, std::vector<std::uint32_t> group)
    : n_(n), p_(params), group_(std::move(group)), s_prev_(n, 0.0), has_prev_(n, 0),
      tracker_(n, params.recluster_bars, 8, params.resolution) {
  if (!group_.empty() && group_.size() != n) throw std::invalid_argument("LandscapeBuilder: group size mismatch");
  if (!(p_.order_smoothing >= 0.0 && p_.order_smoothing <= 1.0))
    throw std::invalid_argument("LandscapeBuilder: order_smoothing must be in [0, 1]");
  if (!(std::isfinite(p_.rank_tolerance) && p_.rank_tolerance >= 0.0))
    throw std::invalid_argument("LandscapeBuilder: rank_tolerance must be finite and >= 0");
  if (!(std::isfinite(p_.smooth) && p_.smooth >= 0.0))
    throw std::invalid_argument("LandscapeBuilder: smooth must be finite and >= 0");
  if (p_.cvt.iterations < 0 || !(p_.cvt.lambda > 0 && p_.cvt.lambda <= 1) || !(p_.cvt.eps_frac > 0 && p_.cvt.eps_frac <= 10))
    throw std::invalid_argument("LandscapeBuilder: cvt parameters out of range");
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
  // Size shares over the active stocks (PiRelSize) and the teleport-floor flags.
  std::vector<double> share(n_, std::numeric_limits<double>::quiet_NaN());
  if (f.size_ref.size() == n_) {
    std::vector<double> ref(n_, 0.0);
    for (std::size_t i = 0; i < n_; ++i)
      if (f.active[i]) ref[i] = f.size_ref[i];
    share = size_shares(ref);
  }
  std::vector<bool> floor(n_, false);
  for (std::size_t i = 0; i < n_; ++i) floor[i] = f.active[i] && at_teleport_floor(f.pi[i], f.pi_floor);
  // Ranking value: the landscape value (log(pi N) or signed-log h) blended with the previous frame's value;
  // non-finite values rank as 0.
  std::vector<double> rank(n_, 0.0);
  const std::vector<char> was_active = has_prev_;
  for (std::size_t i = 0; i < n_; ++i) {
    if (!f.active[i]) {
      has_prev_[i] = 0;
      continue;
    }
    double cur = landscape_value(f.h[i], f.pi[i], n_active, share[i], p_.value, p_.height);
    if (!std::isfinite(cur)) cur = 0.0;
    rank[i] = has_prev_[i] ? (1.0 - p_.order_smoothing) * cur + p_.order_smoothing * s_prev_[i] : cur;
    s_prev_[i] = rank[i];
    has_prev_[i] = 1;
  }
  const bool flux = p_.territory == TerritoryMode::Flux;
  const std::vector<std::uint32_t>& grp = flux ? tracker_.update(f.P, f.active) : group_;
  // Group identity: the persistent community label in flux mode (-1 = loose pool), the sector id otherwise.
  std::vector<std::int64_t> key(n_, -1);
  for (std::size_t i = 0; i < n_; ++i)
    key[i] = flux ? tracker_.node_group()[i] : (group_.empty() ? 0 : static_cast<std::int64_t>(group_[i]));
  const bool have_mem = !mem_.cell.empty();
  if (have_mem)
    for (std::size_t i = 0; i < n_; ++i)
      if (!f.active[i] || !was_active[i] || key[i] != prev_key_[i]) mem_.cell[i] = -1;
  // Under Pi the floor stocks all tie at one value; they would drag the territory medians (mountain or crater) to
  // the floor, so the medians leave them out.
  const TerritoryLayout layout = territory_layout(f.active, grp, rank, size, have_mem ? &mem_ : nullptr, p_.rank_tolerance,
                                                  p_.value == LandscapeValue::Pi ? floor : std::vector<bool>{});
  const std::vector<std::int32_t>& cells = layout.cell;
  mem_.size = size;
  mem_.cell = cells;
  prev_key_ = std::move(key);

  LandscapeFrame lf;
  lf.t = f.t;
  lf.n = n_;
  lf.size = size;
  const auto cols = static_cast<std::int32_t>(size.cols);
  const auto& score = f.forecasts.empty() ? f.h : f.forecasts.front().score;
  for (std::size_t i = 0; i < n_; ++i) {
    if (!f.active[i]) continue;
    const double hd = landscape_value(f.h[i], f.pi[i], n_active, share[i], p_.value, p_.height);
    lf.nodes.push_back({static_cast<std::uint32_t>(i), cells[static_cast<std::size_t>(i)],
                        static_cast<float>((cells[i] % cols + 0.5) / static_cast<double>(size.cols)),
                        static_cast<float>((cells[i] / cols + 0.5) / static_cast<double>(size.rows)), f.h[i], hd, f.pi[i], score[i],
                        flux ? tracker_.node_group()[i] : (group_.empty() ? 0 : static_cast<std::int32_t>(group_[i])),
                        f.pulse.size() == n_ ? f.pulse[i] : std::numeric_limits<double>::quiet_NaN(), share[i],
                        static_cast<bool>(floor[i])});
  }
  lf.raster = node_raster(lf, p_);
  lf.arcs = top_arcs(f.P, f.active, p_.max_arcs);
  if (flux) {
    lf.communities = tracker_.communities();
    lf.modularity = tracker_.modularity();
    lf.loose = tracker_.loose_nodes();
    lf.cluster_ms = tracker_.cluster_ms();
    lf.reclustered = tracker_.reclustered();
  } else {
    std::set<std::int32_t> ids;
    for (const auto& nd : lf.nodes) ids.insert(nd.group);
    lf.communities = static_cast<int>(ids.size());
  }
  lf.compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return lf;
}

}  // namespace mr
