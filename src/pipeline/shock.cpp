#include "pipeline/shock.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace fx {

ShockDelta shock_response(const Frame& base, const Frame& shocked) {
  const std::size_t n = base.pi.size();
  if (shocked.pi.size() != n) throw std::invalid_argument("shock_response: frame size mismatch");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  ShockDelta d;
  d.dh.assign(n, nan);
  d.dpi.assign(n, 0.0);
  d.dscore.assign(n, nan);
  for (std::size_t i = 0; i < n; ++i) {
    if (!base.active[i] || !shocked.active[i]) continue;
    d.dh[i] = shocked.h[i] - base.h[i];
    d.dpi[i] = shocked.pi[i] - base.pi[i];
    if (!base.forecasts.empty() && !shocked.forecasts.empty())
      d.dscore[i] = shocked.forecasts.front().score[i] - base.forecasts.front().score[i];
    d.l1_dpi += std::fabs(d.dpi[i]);
  }
  return d;
}

std::pair<Frame, Frame> run_with_shock(const Panel& panel, const CoreParams& params,
                                       const std::vector<Shock>& shocks) {
  if (panel.T() < 3) throw std::runtime_error("shock mode needs at least three bars");
  CorePipeline pipeline(panel.N(), params);
  const std::size_t last = panel.T() - 1;
  for (std::size_t t = 1; t < last; ++t) pipeline.step(panel, t);
  CorePipeline copy = pipeline;
  Frame base = pipeline.step(panel, last);
  Frame shocked = copy.step(panel, last, shocks);
  return {std::move(base), std::move(shocked)};
}

}  // namespace fx
