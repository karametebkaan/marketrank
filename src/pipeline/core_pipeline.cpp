#include "pipeline/core_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

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

constexpr std::size_t kNone = static_cast<std::size_t>(-1);

// Warm start on the active sub-index: carry the full-size previous vector over (newly active
// nodes have 0), renormalized to sum 1. Empty (uniform start) if nothing carries over.
std::vector<double> remap_warm(const std::vector<double>& full, const std::vector<bool>& active,
                               std::size_t n_active) {
  if (full.size() != active.size()) return {};
  std::vector<double> a;
  a.reserve(n_active);
  double sum = 0;
  for (std::size_t i = 0; i < full.size(); ++i) {
    if (!active[i]) continue;
    a.push_back(full[i]);
    sum += full[i];
  }
  if (!(sum > 0)) return {};
  for (double& x : a) x /= sum;
  return a;
}

const CoreParams& validated(const CoreParams& p) {
  p.validate();
  return p;
}

}  // namespace

CoreParams CoreParams::legacy() {
  CoreParams p;
  p.pressure = PressureMode::Dollar;
  p.transition.lift = LiftMode::Off;
  p.transition.k_in = 0;
  p.transition.retention = 0.0;
  p.h_ref = HotRef::Uniform;
  return p;
}

void CoreParams::validate() const {
  auto fail = [](const char* what) {
    throw std::invalid_argument(std::string("CoreParams: ") + what);
  };
  // Half-lives: > 0 and finite, or +infinity (no decay). h > 0 rejects NaN and -infinity.
  auto good_halflife = [](double h) { return h > 0; };
  if (!(std::isfinite(alpha) && alpha > 0 && alpha <= 1)) fail("alpha must be in (0, 1]");
  if (!(std::isfinite(flux.lambda) && flux.lambda >= 0 && flux.lambda <= 1))
    fail("lambda must be in [0, 1]");
  if (!(good_halflife(halflife_slow) && good_halflife(halflife_fast) &&
        good_halflife(halflife_long)))
    fail("half-lives must be > 0 (finite or +infinity)");
  if (transition.k_out == 0) fail("k_out must be >= 1");
  if (!(std::isfinite(transition.retention) && transition.retention >= 0))
    fail("retention must be finite and >= 0");
  if (horizons.empty()) fail("horizons must not be empty");
  for (int k : horizons)
    if (k < 1) fail("horizons must be >= 1");
  if (corr_window < 2) fail("corr_window must be >= 2");
  if (adv_window < 1) fail("adv_window must be >= 1");
  if (stale_bars < 1) fail("stale_bars must be >= 1");
  if (flux.sinks_per_source < 1 || flux.sink_candidates < flux.sinks_per_source)
    fail("need 1 <= sinks_per_source <= sink_candidates");
  if (row_cap < transition.k_out) fail("row_cap must be >= k_out");
}

CorePipeline::CorePipeline(std::size_t n, CoreParams params)
    : n_(n),
      params_(validated(params)),
      pressure_(n, params_.pressure, params_.adv_window),
      window_(n, params_.corr_window),
      slow_(n, params_.halflife_slow, params_.row_cap),
      fast_(n, params_.halflife_fast, params_.row_cap),
      last_close_(n, kNone) {
  if (params_.h_ref == HotRef::LongRun) long_.emplace(n, params_.halflife_long, params_.row_cap);
}

Frame CorePipeline::step(const Panel& panel, std::size_t t) {
  if (t == 0 || t >= panel.T()) throw std::invalid_argument("CorePipeline::step: t out of range");
  if (panel.N() != n_) throw std::invalid_argument("CorePipeline::step: panel size mismatch");
  const auto t0 = std::chrono::steady_clock::now();
  const double nan = std::numeric_limits<double>::quiet_NaN();

  if (t < next_bar_) throw std::invalid_argument("CorePipeline::step: t must increase");

  // Causal active mask: scan only bars not seen yet (and, on a first step, only the bars that can
  // still count as fresh).
  const std::size_t stale = params_.stale_bars;
  for (std::size_t s = std::max(next_bar_, t - std::min(t, stale)); s <= t; ++s)
    for (std::size_t i = 0; i < n_; ++i)
      if (std::isfinite(panel.close[panel.idx(s, i)])) last_close_[i] = s;
  next_bar_ = t + 1;
  std::vector<bool> active(n_, false);
  std::vector<std::size_t> map(n_, kNone);
  std::size_t n_active = 0;
  for (std::size_t i = 0; i < n_; ++i) {
    active[i] = last_close_[i] != kNone && t - last_close_[i] <= stale;
    if (active[i]) map[i] = n_active++;
  }
  if (n_active == 0) throw std::runtime_error("no nodes with data");

  std::vector<double> returns(n_, nan), volume(n_, nan), vwap(n_, nan);
  for (std::size_t i = 0; i < n_; ++i) {
    const double c = panel.close[panel.idx(t, i)];
    const double c_prev = panel.close[panel.idx(t - 1, i)];
    if (std::isfinite(c) && std::isfinite(c_prev) && c_prev > 0) returns[i] = c / c_prev - 1.0;
    volume[i] = panel.volume[panel.idx(t, i)];
    vwap[i] = panel.vwap[panel.idx(t, i)];
  }
  const std::vector<double> pressure = pressure_.step(returns, volume, vwap);
  window_.push(returns);
  std::span<const double> unit;
  if (params_.flux.lambda > 0) unit = window_.unit_vectors();
  const BarFlux bar = bar_flux_sparse(pressure, unit, window_.window(), params_.flux);
  slow_.add(bar);
  fast_.add(bar);
  if (long_) long_->add(bar);

  Frame f;
  f.t = panel.times[t];
  f.active = active;
  f.P = build_transition(slow_, params_.transition, active);
  f.P_fast = build_transition(fast_, params_.transition, active);

  const Csr Pa = compact(f.P, map, n_active);
  const Csr Pa_fast = compact(f.P_fast, map, n_active);
  const std::vector<double> prev_a = remap_warm(prev_pi_, active, n_active);
  f.solve = stationary(Pa, params_.alpha, prev_a);
  const std::vector<double> pi_a = f.solve.pi;

  std::vector<double> h_a;
  switch (params_.h_ref) {
    case HotRef::Uniform:
      h_a = hotness(pi_a);
      break;
    case HotRef::Size: {
      const std::vector<double> mdv = pressure_.median_dollar_volume();
      std::vector<double> ref;
      ref.reserve(n_active);
      for (std::size_t i = 0; i < n_; ++i)
        if (active[i]) ref.push_back(mdv[i]);
      // No volume history yet means a neutral size: use the median of the known references.
      std::vector<double> pos;
      for (double x : ref)
        if (std::isfinite(x) && x > 0) pos.push_back(x);
      if (!pos.empty()) {
        std::nth_element(pos.begin(), pos.begin() + pos.size() / 2, pos.end());
        const double med = pos[pos.size() / 2];
        for (double& x : ref)
          if (!(std::isfinite(x) && x > 0)) x = med;
      }
      h_a = relative_hotness(pi_a, ref);
      break;
    }
    case HotRef::LongRun: {
      const Csr Pl = compact(build_transition(*long_, params_.transition, active), map, n_active);
      const SolveResult lr =
          stationary(Pl, params_.alpha, remap_warm(prev_long_pi_, active, n_active));
      prev_long_pi_.assign(n_, 0.0);
      for (std::size_t i = 0; i < n_; ++i)
        if (active[i]) prev_long_pi_[i] = lr.pi[map[i]];
      h_a = relative_hotness(pi_a, lr.pi);
      break;
    }
  }

  f.pi.assign(n_, 0.0);
  f.h.assign(n_, nan);
  for (std::size_t i = 0; i < n_; ++i) {
    if (!active[i]) continue;
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
      if (!active[i]) continue;
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
