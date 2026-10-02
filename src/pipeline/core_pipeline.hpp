#pragma once
#include <optional>
#include <vector>

#include "core/types.hpp"
#include "graph/csr.hpp"
#include "graph/forecaster.hpp"
#include "graph/hotness.hpp"
#include "graph/markov_solver.hpp"
#include "graph/pressure.hpp"
#include "graph/return_window.hpp"
#include "graph/sparse_flux.hpp"
#include "graph/transition.hpp"
#include "market/panel.hpp"

namespace fx {

struct CoreParams {
  PressureMode pressure = PressureMode::Relative;  // (A)
  std::size_t adv_window = 20;
  std::size_t corr_window = 60;
  std::size_t stale_bars = 5;  // active iff the last finite close is at most this many bars old
  SparseFluxParams flux;  // lambda, sink_candidates, sinks_per_source
  double halflife_slow = 20;
  double halflife_fast = 3;
  double halflife_long = 120;  // used only when h_ref == LongRun
  std::size_t row_cap = 256;
  TransitionParams transition;     // (B) lift, (C) k_out / k_in, (E) retention
  HotRef h_ref = HotRef::Uniform;  // (D)
  double alpha = 0.85;
  double beta = 0.5;
  std::vector<int> horizons{1, 4, 8};

  static CoreParams legacy();  // milestone-1 behaviour (spec 5)
  void validate() const;       // throws std::invalid_argument
};

struct Frame {
  TimePoint t = 0;
  std::vector<bool> active;         // size n; causal: a finite close within stale_bars of t
  std::vector<double> pi, h;        // inactive: pi = 0, h = NaN
  SolveResult solve;
  SolveResult solve_long;           // h_ref == LongRun only (pi full size n); else default
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
  PressureModel pressure_;
  ReturnWindow window_;
  FluxAccumulator slow_, fast_;
  std::optional<FluxAccumulator> long_;
  std::vector<std::size_t> last_close_;  // per node: last bar with a finite close (npos = none)
  std::size_t next_bar_ = 0;             // first bar not yet scanned into last_close_
  std::vector<double> prev_pi_;          // full size n, 0 for inactive
  std::vector<double> prev_long_pi_;     // full size n, warm start for the long-run solve
};

Frame run_panel_last(const Panel& panel, const CoreParams& params);

}  // namespace fx
