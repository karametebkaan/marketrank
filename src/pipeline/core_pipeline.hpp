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

namespace mr {

struct CoreParams {
  PressureMode pressure = PressureMode::Sqrt;  // (A)
  std::size_t adv_window = 20;
  std::size_t corr_window = 60;
  double min_dollar_volume = 1e6;  // active only if trailing median dollar volume >= this; 0 = off
  double max_volume_ratio = 5.0;   // relative pressure caps V/ADV here; 0 = uncapped
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
  bool vol_scale = false;       // divide returns by trailing volatility before forming pressure
  std::size_t vol_window = 20;  // trailing returns used for that volatility (>= 5)
  std::vector<int> horizons{1, 4, 8};

  // The MarketRank concept model (spec 5, "MarketRank preset"): the damped chain on the out-shares of
  // the cumulative paired dollar flow, r_i = (1-p) sum_j r_j T_ji / sum_k T_jk + p/N with p = 1 - alpha:
  // dollar pressure, no lift, no retention (no self-loops; a row with nothing to give teleports),
  // uniform reference, no volume cap, no vol scaling, slow half-life 1e9 bars (effectively cumulative).
  static CoreParams market_rank();
  static CoreParams money_flow();  // dollar flux, no lift, two-sided pruning, retention, size ref
  static CoreParams legacy();  // milestone-1 behaviour (spec 5)
  void validate() const;       // throws std::invalid_argument
  bool operator==(const CoreParams&) const = default;
};

struct Frame {
  TimePoint t = 0;
  std::vector<bool> active;         // size n; causal: a recent close and the liquidity floor
  std::vector<double> pi, h;        // inactive: pi = 0, h = NaN
  std::vector<double> inflow;       // size n; slow accumulator in() (structure-gain metric)
  // Heartbeat: log pi_i(t) - log pi_i(t-1) against the previous step's frame; NaN when the node is
  // inactive now or was inactive (or there was no previous step) at t-1.
  std::vector<double> pulse;
  SolveResult solve;
  SolveResult solve_long;           // h_ref == LongRun only (pi full size n); else default
  std::vector<Forecast> forecasts;  // parallel to CoreParams::horizons
  Csr P;                            // slow (equilibrium) transition matrix
  Csr P_fast;                       // fast transition matrix
  double compute_ms = 0;
};

// Counterfactual jolt: an extra size% return at the node's normal volume, added to the bar's actual
// pressure (units of the active pressure mode; duplicate shocks on a node add up).
struct Shock {
  std::size_t node;
  double size;
};

class CorePipeline {
 public:
  CorePipeline(std::size_t n, CoreParams params);
  Frame step(const Panel& panel, std::size_t t, const std::vector<Shock>& shocks = {});
  // Pressure the last step() fed to the flux (after masking and shocks). For tests and diagnostics.
  const std::vector<double>& last_pressure() const { return last_pressure_; }

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
  std::vector<double> last_pressure_;    // full size n, see last_pressure()
};

Frame run_panel_last(const Panel& panel, const CoreParams& params);

// MarketRank score of a node: pi_i * N_active (1 = average).
inline double market_rank_score(double pi, std::size_t n_active) { return pi * static_cast<double>(n_active); }

}  // namespace mr
