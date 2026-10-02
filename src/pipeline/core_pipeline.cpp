#include "pipeline/core_pipeline.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fx {

CorePipeline::CorePipeline(std::size_t n, CoreParams params)
    : n_(n), params_(std::move(params)), flux_(n, params_.flux) {}

Frame CorePipeline::step(const Panel& panel, std::size_t t) {
  const auto t0 = std::chrono::steady_clock::now();
  const double nan = std::numeric_limits<double>::quiet_NaN();
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
  f.P = build_transition(flux_.flux_slow(), n_, params_.top_k);
  const Csr P_fast = build_transition(flux_.flux_fast(), n_, params_.top_k);
  f.solve = stationary(f.P, params_.alpha, prev_pi_);
  f.pi = f.solve.pi;
  f.h = hotness(f.pi);
  for (int k : params_.horizons)
    f.forecasts.push_back(forecast(P_fast, params_.alpha, f.pi, prev_pi_, k, params_.beta));
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
