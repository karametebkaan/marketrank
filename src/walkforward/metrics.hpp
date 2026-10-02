#pragma once
#include <cmath>
#include <cstddef>
#include <string>

#include "walkforward/backtest.hpp"
#include "walkforward/stats.hpp"

namespace mr {

// Conventions (252 trading days per year, rf = 0, daily metrics regardless of rebalance frequency):
//  - EquityCurve.value[0] is the close of the first execution day and already contains day 1's open-to-close move
//    and the funding cost; starting capital is 1.0. So r_0 = value[0]/1.0 - 1 and r_d = value[d]/value[d-1] - 1.
//  - Strategy and benchmark are aligned on common dates (t). On the first common date each return is measured against
//    that curve's previous value (1.0 if the curve starts there); cum/ann return and the drawdown peak start from the
//    strategy's value at that base (1.0 when both curves start together).
//  - ann_return = (V_end/1.0)^(252/T) - 1; max_drawdown's running peak starts at 1.0.
//  - e_d = r_d - rb_d is the daily excess over the benchmark; ann_excess = mean(e) * 252 (an arithmetic mean, so it
//    differs from the gap between the two annualized (geometric) returns).
//  - sd is the sample sd (n-1); skew/kurt are population moments of daily strategy returns and skew_e/kurt_e of the
//    daily excess e; kurtosis is RAW (3 = normal). Moments are NaN when the sd is ~0.
//  - year_hit_rate: share of UTC calendar years (of t) with sum(e) > 0, counting only years with >= 120 days;
//    `years` is the number counted (hit rate 0 if none).
// Ratios are NaN when the sd is ~0 (sd <= 1e-12*max(1,|mean|)) and when there are no common days (T = 0).
struct Perf {
  std::size_t T = 0;  // common days
  double cum_return = 0, ann_return = 0, ann_vol = 0, max_drawdown = 0;
  double sharpe_daily = NAN;  // per-day mean(r)/sd(r); sharpe = sharpe_daily*sqrt(252). Informational.
  double sharpe = NAN;
  double ir_daily = NAN;      // per-day mean(e)/sd(e): the `sr` of the excess that gate c3 deflates
  double ann_excess = 0, ir = NAN;  // ir = ir_daily*sqrt(252)
  CI excess_ci95;
  double year_hit_rate = 0;
  std::size_t years = 0;
  double skew = NAN, kurt = NAN;      // of r
  double skew_e = NAN, kurt_e = NAN;  // of e
};
Perf performance(const EquityCurve& s, const EquityCurve& bench);

// Bailey & Lopez de Prado (2014) deflated Sharpe probability. sr = per-day Sharpe, T = days, trial_sr_var = variance of
// the per-day Sharpe across all registered trials, n_trials >= 1 (n_trials == 1 -> SR* = 0); kurt is raw kurtosis.
double deflated_sharpe(double sr, std::size_t T, double skew, double kurt, double trial_sr_var, std::size_t n_trials);

// Deflated probability of the excess (IR) series: deflated_sharpe(ir_daily, T, skew_e, kurt_e, ...); NaN when
// ir_daily is NaN. This is what gate c3 tests (spec amendment 2026-10-02).
double dsr_excess(const Perf& p, double trial_ir_var, std::size_t n_ir_trials);
// Deflated probability of the total-return Sharpe (informational only: a high-Sharpe base passes it without any
// excess). NaN when sharpe_daily is NaN.
double dsr_total(const Perf& p, double trial_sr_var, std::size_t n_trials);

// c3 = ir_daily is finite and dsr_excess > 0.95 (an undefined IR fails c3 with an "undefined" reason).
struct GateResult { bool c1, c2, c3, c4, c5, pass; std::string reason; };
GateResult decision_gate(const Perf& strat_vs_buyhold, double base_max_dd, double dsr_excess, bool largecap_ok);

}  // namespace mr
