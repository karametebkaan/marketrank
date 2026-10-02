#pragma once
#include <utility>
#include <vector>

#include "market/panel.hpp"
#include "pipeline/core_pipeline.hpp"

namespace fx {

struct ShockDelta {
  std::vector<double> dh, dpi, dscore;  // shocked - base; dscore uses the first horizon
  double l1_dpi = 0;                    // sum |dpi|
};

// Inactive nodes (in either frame) get NaN dh and dscore and 0 dpi.
ShockDelta shock_response(const Frame& base, const Frame& shocked);

// Runs bars 1..T-2, copies the pipeline, then steps the original (baseline) and the copy (with the
// shocks) at bar T-1. Returns {baseline, shocked}. Throws if T < 3.
std::pair<Frame, Frame> run_with_shock(const Panel& panel, const CoreParams& params,
                                       const std::vector<Shock>& shocks);

}  // namespace fx
