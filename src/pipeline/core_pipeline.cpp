#include "pipeline/core_pipeline.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fx {

namespace {

// Restrict a full-size transition matrix to the active sub-index (columns remapped).
Csr compact(const Csr& P, const std::vector<std::size_t>& map, std::size_t n_active) {
  Csr C;
  C.n = n_active;
  C.row_ptr.push_back(0);
  for (std::size_t i = 0; i < P.n; ++i) {
    if (map[i] == static_cast<std::size_t>(-1)) continue;
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) {
      C.col.push_back(static_cast<std::uint32_t>(map[P.col[e]]));
      C.val.push_back(P.val[e]);
      C.raw.push_back(P.raw[e]);
    }
    C.row_ptr.push_back(C.col.size());
  }
  return C;
}

}  // namespace

CorePipeline::CorePipeline(std::size_t n, CoreParams params)
    : n_(n), params_(std::move(params)), flux_(n, params_.flux) {}

Frame CorePipeline::step(const Panel& panel, std::size_t t) {
  if (t == 0 || t >= panel.T()) throw std::invalid_argument("CorePipeline::step: t out of range");
  if (panel.N() != n_) throw std::invalid_argument("CorePipeline::step: panel size mismatch");
  const auto t0 = std::chrono::steady_clock::now();
  const double nan = std::numeric_limits<double>::quiet_NaN();

  if (active_.empty()) {
    active_.assign(n_, false);
    for (std::size_t s = 0; s < panel.T(); ++s)
      for (std::size_t i = 0; i < n_; ++i)
        if (std::isfinite(panel.close[panel.idx(s, i)])) active_[i] = true;
  }
  std::vector<std::size_t> map(n_, static_cast<std::size_t>(-1));
  std::size_t n_active = 0;
  for (std::size_t i = 0; i < n_; ++i)
    if (active_[i]) map[i] = n_active++;
  if (n_active == 0) throw std::runtime_error("no nodes with data");

  std::vector<double> returns(n_, nan), dollar_volume(n_, nan);
  for (std::size_t i = 0; i < n_; ++i) {
    const double c = panel.close[panel.idx(t, i)];
    const double c_prev = panel.close[panel.idx(t - 1, i)];
    if (std::isfinite(c) && std::isfinite(c_prev) && c_prev > 0) returns[i] = c / c_prev - 1.0;
    const double v = panel.volume[panel.idx(t, i)];
    const double vw = panel.vwap[panel.idx(t, i)];
    if (std::isfinite(v) && std::isfinite(vw)) dollar_volume[i] = v * vw;
  }
  flux_.step(returns, dollar_volume);

  Frame f;
  f.t = panel.times[t];
  f.active = active_;
  f.P = build_transition(flux_.flux_slow(), n_, params_.top_k, active_);
  f.P_fast = build_transition(flux_.flux_fast(), n_, params_.top_k, active_);

  const Csr Pa = compact(f.P, map, n_active);
  const Csr Pa_fast = compact(f.P_fast, map, n_active);
  std::vector<double> prev_a;
  if (prev_pi_.size() == n_) {
    for (std::size_t i = 0; i < n_; ++i)
      if (active_[i]) prev_a.push_back(prev_pi_[i]);
  }
  f.solve = stationary(Pa, params_.alpha, prev_a);
  const std::vector<double> pi_a = f.solve.pi;
  const std::vector<double> h_a = hotness(pi_a);

  f.pi.assign(n_, 0.0);
  f.h.assign(n_, nan);
  for (std::size_t i = 0; i < n_; ++i) {
    if (!active_[i]) continue;
    f.pi[i] = pi_a[map[i]];
    f.h[i] = h_a[map[i]];
  }
  f.solve.pi = f.pi;
  for (int k : params_.horizons) {
    Forecast fa = forecast(Pa_fast, params_.alpha, pi_a, prev_a, k, params_.beta);
    Forecast full;
    full.k = fa.k;
    full.pi_k.assign(n_, 0.0);
    full.score.assign(n_, nan);
    for (std::size_t i = 0; i < n_; ++i) {
      if (!active_[i]) continue;
      full.pi_k[i] = fa.pi_k[map[i]];
      full.score[i] = fa.score[map[i]];
    }
    f.forecasts.push_back(std::move(full));
  }
  prev_pi_ = f.pi;
  f.compute_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return f;
}

Frame run_panel_last(const Panel& panel, const CoreParams& params) {
  if (panel.T() < 2) throw std::runtime_error("need at least two bars to build flux");
  CorePipeline pipeline(panel.N(), params);
  Frame last;
  for (std::size_t t = 1; t < panel.T(); ++t) last = pipeline.step(panel, t);
  return last;
}

}  // namespace fx
