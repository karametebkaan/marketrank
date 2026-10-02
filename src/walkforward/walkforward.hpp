#pragma once
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "market/panel.hpp"
#include "pipeline/core_pipeline.hpp"
#include "walkforward/backtest.hpp"
#include "walkforward/blend.hpp"
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
};

// The spec's base mix (global constraints): the fallback when data/portfolio.json has no holdings.
std::vector<BaseWeight> default_base_mix();

struct IcRow {
  Signal s;
  int h;
  MeanT all;
  std::vector<std::pair<int, double>> by_year;  // UTC year -> mean IC, ascending years
};

struct WalkForwardResult {
  std::vector<IcRow> ic_table;             // signal-major, then horizon in ic_horizons order
  std::vector<std::size_t> dates;          // rebalance bar indices
  std::vector<MonthRecord> months;         // stored z-scores and labels per rebalance
  std::vector<std::vector<bool>> eligible; // per rebalance: eligible_at(d_m) && frame.active
  std::vector<BlendStep> blend;
  // Equity curves: "blend", one per signal ("sig:<name>"), "bench:buyhold", "bench:rebalanced", "bench:VOO"
  // (the last only when VOO is in the panel).
  std::vector<std::pair<std::string, EquityCurve>> curves;
};

// One causal pass over the panel (CorePipeline::step + SignalTracker::update per bar t = 1..T-1; step needs t >= 1,
// so a warm-up of 0 acts as 1), then the blend and the
// simulations. At rebalance date d_m everything stored depends only on bars <= d_m, except label(m) =
// forward_oo_return(d_m, d_{m+1} - d_m) (NaN for the last period), which the blend consumes only after its embargo.
// IC table: per horizon h, every bar t >= warmup with (t - warmup) % h == 0 and t + 1 + h < T contributes
// spearman(zscore(raw_s, eligible_at(t)), forward_oo_return(t, h)) (non-finite ICs skipped).
// Throws std::invalid_argument on a non-positive IC horizon or an unknown base ticker.
WalkForwardResult run_walkforward(const Panel& panel, const WalkForwardParams& p);

}  // namespace mr
