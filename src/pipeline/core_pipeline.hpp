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
  // Memory of the flux the landscape clusters (Frame::P_cluster): recent flows, not the cumulative slow flux, so the
  // flux communities follow the market (measured: --cluster-persistence --cp-fast-hl 20). Display only: pi and
  // the forecasts do not use it.
  double halflife_cluster = 20;
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
  // M5: the MarketRank preset on regular-session 15-minute bars (26 per session), bar-count windows rescaled to the
  // same wall-clock meaning: ADV 20 sessions = 520 bars; correlation window 20 sessions = 520 bars (the spec's
  // documented choice; 60 sessions would triple the per-bar affinity cost); liquidity floor $1M per session =
  // 1e6/26 per bar (median per-bar dollar volume); stale after one session (26 bars) without a close; the cumulative
  // chain keeps half-life 1e9 bars; the fast chain is the heartbeat, half-life 26 bars (one session); the landscape
  // cluster flux 20 sessions (520 bars); forecast horizons 1, 6, 26 bars.
  static CoreParams market_rank_intraday();
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
  // Heartbeat, the pulse of the score: log(pi_i(t) N(t)) - log(pi_i(t-1) N(t-1)) against the previous step's
  // frame (N = active count), so a change in N alone is no pulse; NaN when the node is inactive now or was
  // inactive (or there was no previous step) at t-1.
  std::vector<double> pulse;
  // Size reference: trailing median dollar volume (HotRef::Size's reference), NaN for inactive nodes.
  std::vector<double> size_ref;
  // Teleport floor of pi: ((1 - alpha) + alpha * d) / N_active, d = pi mass on dangling (empty) active rows.
  double pi_floor = 0;
  SolveResult solve;
  SolveResult solve_long;           // h_ref == LongRun only (pi full size n); else default
  std::vector<Forecast> forecasts;  // parallel to CoreParams::horizons
  Csr P;                            // slow (equilibrium) transition matrix
  Csr P_fast;                       // fast transition matrix
  Csr P_cluster;                    // transition matrix of the halflife_cluster flux (landscape communities)
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
  // The slow (equilibrium) flux accumulator after the last step: raw kept edges, the source of Frame::P.
  const FluxAccumulator& slow_flux() const { return slow_; }
  // The last step's own (single-bar, un-accumulated) sparse flux. For offline diagnostics.
  const BarFlux& last_bar() const { return last_bar_; }

 private:
  std::size_t n_;
  CoreParams params_;
  PressureModel pressure_;
  ReturnWindow window_;
  FluxAccumulator slow_, fast_, cluster_;
  std::optional<FluxAccumulator> long_;
  std::vector<std::size_t> last_close_;  // per node: last bar with a finite close (npos = none)
  std::size_t next_bar_ = 0;             // first bar not yet scanned into last_close_
  std::vector<double> prev_pi_;          // full size n, 0 for inactive
  std::size_t prev_n_active_ = 0;        // active count of the previous step (heartbeat)
  std::vector<double> prev_long_pi_;     // full size n, warm start for the long-run solve
  std::vector<double> last_pressure_;    // full size n, see last_pressure()
  BarFlux last_bar_;                     // see last_bar()
};

Frame run_panel_last(const Panel& panel, const CoreParams& params);

// Whether pi sits at the teleport floor (within 1e-6 relative); false when no floor is known (floor <= 0).
inline bool at_teleport_floor(double pi, double floor) { return floor > 0 && pi <= floor * (1.0 + 1e-6); }

// MarketRank score of a node: pi_i * N_active (1 = average).
inline double market_rank_score(double pi, std::size_t n_active) { return pi * static_cast<double>(n_active); }

}  // namespace mr
