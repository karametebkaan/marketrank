#pragma once
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "market/panel.hpp"
#include "pipeline/core_pipeline.hpp"
#include "walkforward/backtest.hpp"
#include "walkforward/blend.hpp"
#include "walkforward/external.hpp"
#include "walkforward/signals.hpp"
#include "walkforward/stats.hpp"
#include "walkforward/universe.hpp"

namespace mr {

// Blend windows in rebalance periods: weekly 156/1/104/52 (3 years train, 2 years gate, 1 year minimum), monthly
// 36/1/24/12 (spec section 7).
BlendParams blend_defaults(Rebalance r);

struct WalkForwardParams {
  CoreParams core = CoreParams::market_rank();
  Rebalance rebalance = Rebalance::Weekly;  // user decision: weekly is primary
  std::size_t warmup_bars = 252, elig_window = 20, top_n = 0;
  double min_dollar_volume = 50e6;
  std::vector<int> ic_horizons{1, 2, 5, 20};  // 20 = decay diagnostic only
  BacktestParams bt;
  BlendParams blend;  // = blend_defaults(Weekly); set blend_defaults(Monthly) with Rebalance::Monthly
  // Report only: run id of a sibling run (same --wf-out) on the large-cap sub-universe (--wf-top-n 500) whose
  // gate criteria c1..c4 decide c5 here. Empty = c5 "pending". Not part of the parameter hash.
  std::string largecap_run;
  // M4: "<name>:<digest>" of each external signal (--wf-external), in order. Part of describe() and the params hash
  // only when non-empty, so runs without externals keep their M3a hash. run_walkforward checks it against the
  // externals it is given.
  std::vector<std::string> externals;
};

// The rebalance calendar of run_walkforward: rebalance_dates(panel, p.rebalance, max(p.warmup_bars, 1)) (the
// pipeline steps from bar 1, so a warm-up of 0 acts as 1). Shared with --export-panel.
std::vector<std::size_t> walkforward_dates(const Panel& panel, const WalkForwardParams& p);

// Tag of an external signal for WalkForwardParams::externals.
std::string external_tag(const ExternalSignal& e);

// The spec's base mix (global constraints): the fallback when data/portfolio.json has no holdings.
std::vector<BaseWeight> default_base_mix();

struct IcRow {
  std::string signal;  // built-in name (to_string(Signal)) or external name
  int h;
  MeanT all;
  std::vector<std::pair<int, double>> by_year;  // UTC year -> mean IC, ascending years
};

// Per external signal: its IC at every rebalance against that period's label (spearman(zscore(score, eligible[m]),
// months[m].label)); NaN where the signal has no scores at d_m, the label is undefined or the IC is not finite.
struct ExternalResult {
  std::string name;
  std::vector<double> rebalance_ic;  // size = dates.size()
  std::vector<bool> scored;          // size = dates.size(): the signal has scores at d_m
  std::size_t scored_bars = 0;       // bars with scores (any bar of the panel)
  std::size_t scored_rebalances = 0; // of those, walk-forward rebalance dates (= count of `scored`)
};

// Empty when every scored bar is a rebalance date; otherwise a warning naming how many are not (an off-by-one t,
// or scores on non-rebalance bars, which only the IC rows use).
std::string external_coverage_warning(const ExternalResult& e);

struct WalkForwardResult {
  // Signal-major (built-ins in Signal order, then externals in the order given), then horizon in ic_horizons order.
  std::vector<IcRow> ic_table;
  std::vector<std::size_t> dates;          // rebalance bar indices
  std::vector<MonthRecord> months;         // stored z-scores and labels per rebalance
  std::vector<std::vector<bool>> eligible; // per rebalance: eligible_at(d_m) && frame.active
  std::vector<BlendStep> blend;
  // Equity curves: "blend", one per signal ("sig:<name>"), "bench:buyhold", "bench:rebalanced", "bench:VOO"
  // (the last only when VOO is in the panel), then the base-free sleeves "sleeve:blend" and "sleeve:<name>"
  // (built-ins, then externals; the "sig:" curves likewise)
  // (tilt 1, the top k at 1/k each, a flat blend holds the benchmark) and their benchmark "bench:ew_eligible"
  // (equal weight over each rebalance's eligible names, rebalanced on the same calendar). Sleeves are secondary
  // results: not registry trials and not gated.
  std::vector<std::pair<std::string, EquityCurve>> curves;
  std::vector<ExternalResult> externals;  // empty without external signals
};

// One causal pass over the panel (CorePipeline::step + SignalTracker::update per bar t = 1..T-1; step needs t >= 1,
// so a warm-up of 0 acts as 1), then the blend and the
// simulations. At rebalance date d_m everything stored depends only on bars <= d_m, except label(m) =
// forward_oo_return(d_m, d_{m+1} - d_m) (NaN for the last period), which the blend consumes only after its embargo.
// IC table: per horizon h, every bar t >= warmup with (t - warmup) % h == 0 and t + 1 + h < T contributes
// spearman(zscore(raw_s, eligible_at(t)), forward_oo_return(t, h)) (non-finite ICs skipped).
//
// External signals (M4) are evaluated standalone, never blended (the blend is the pre-registered M3a protocol over
// the built-ins). Their curves "sig:<name>" and "sleeve:<name>" cover the signal's SCORED SPAN only: decisions from
// its first scored rebalance d_a to its last scored rebalance d_b, the curve cut at the close of d_{b+1} (the end
// of the last scored period; the panel's end if d_b is the last rebalance). Each decision is
// zscore(score(d_m), eligible[m]) (the built-ins' mask); an unscored rebalance inside the span gets an empty
// score vector (hold the base / the equal-weight universe, no tilt). Their IC rows sample only the bars
// the signal has scores at (normally its rebalance dates): per horizon h, scored bars t >= warm-up with
// t + 1 + h < T, taken greedily in order with t >= (previous sample) + h (non-overlapping, like the built-ins:
// scores at every bar from the warm-up on give exactly the built-ins' grid), each contributing
// spearman(zscore(score(t), eligible_at(t)), forward_oo_return(t, h)) (non-finite ICs skipped).
// Throws std::invalid_argument on a non-positive IC horizon, an unknown base ticker, an external whose score
// vectors are not of size N, p.externals not matching the externals' tags, or an external none of whose scored
// bars is a rebalance date (checked before the pass).
WalkForwardResult run_walkforward(const Panel& panel, const WalkForwardParams& p,
                                  const std::vector<ExternalSignal>& externals = {});

}  // namespace mr
