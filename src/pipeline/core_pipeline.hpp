#pragma once
#include <vector>

#include "core/types.hpp"
#include "graph/csr.hpp"
#include "graph/flux_builder.hpp"
#include "graph/forecaster.hpp"
#include "graph/markov_solver.hpp"
#include "market/panel.hpp"

namespace fx {

struct CoreParams {
  FluxParams flux;
  std::size_t top_k = 20;
  double alpha = 0.85;
  double beta = 0.5;
  std::vector<int> horizons{1, 4, 8};
};

struct Frame {
  TimePoint t = 0;
  std::vector<bool> active;         // size n; false = no data in the whole panel
  std::vector<double> pi, h;        // inactive: pi = 0, h = NaN
  SolveResult solve;
  std::vector<Forecast> forecasts;  // parallel to CoreParams::horizons
  Csr P;                            // slow (equilibrium) transition matrix
  Csr P_fast;                       // fast transition matrix
  double compute_ms = 0;
};

class CorePipeline {
 public:
  CorePipeline(std::size_t n, CoreParams params);
  Frame step(const Panel& panel, std::size_t t);

 private:
  std::size_t n_;
  CoreParams params_;
  FluxBuilder flux_;
  std::vector<bool> active_;  // computed from the panel on the first step
  std::vector<double> prev_pi_;  // full size n, 0 for inactive
};

Frame run_panel_last(const Panel& panel, const CoreParams& params);

}  // namespace fx
