#pragma once
#include <cstddef>
#include <string>

#include "walkforward/backtest.hpp"
#include "walkforward/stats.hpp"

namespace mr {

// Conventions (252 trading days per year, rf = 0, daily metrics regardless of rebalance frequency):
//  - EquityCurve.value[0] is the close of the first execution day and already contains day 1's open-to-close move
//    and the funding cost; starting capital is 1.0. So r_0 = value[0]/1.0 - 1 and r_d = value[d]/value[d-1] - 1.
//  - Strategy and benchmark are aligned on common dates (t); the first common date uses base 1.0 for both.
//  - ann_return = (V_end/1.0)^(252/T) - 1; max_drawdown's running peak starts at 1.0.
//  - sd is the sample sd (n-1); skew/kurt are population moments of daily strategy returns, kurt is RAW (3 = normal).
//  - year_hit_rate: share of UTC calendar years (of t) with sum(e) > 0, counting only years with >= 120 days;
//    `years` is the number counted (hit rate 0 if none).
struct Perf {
  double cum_return = 0, ann_return = 0, ann_vol = 0, sharpe = 0, max_drawdown = 0;
  double ann_excess = 0, ir = 0;
  CI excess_ci95;
  double year_hit_rate = 0;
  std::size_t years = 0;
  double skew = 0, kurt = 0;
};
Perf performance(const EquityCurve& s, const EquityCurve& bench);

// Bailey & Lopez de Prado (2014) deflated Sharpe probability. sr = per-day Sharpe, T = days, trial_sr_var = variance of
// the per-day Sharpe across all registered trials, n_trials >= 1 (n_trials == 1 -> SR* = 0); kurt is raw kurtosis.
double deflated_sharpe(double sr, std::size_t T, double skew, double kurt, double trial_sr_var, std::size_t n_trials);

struct GateResult { bool c1, c2, c3, c4, c5, pass; std::string reason; };
GateResult decision_gate(const Perf& strat_vs_buyhold, double base_max_dd, double dsr, bool largecap_ok);

}  // namespace mr
