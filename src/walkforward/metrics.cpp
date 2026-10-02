#include "walkforward/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <span>
#include <vector>

#include "core/time.hpp"

namespace mr {
namespace {
constexpr double kDays = 252.0;

double sample_sd(const std::vector<double>& x, double m) {
  if (x.size() < 2) return 0;
  double ss = 0;
  for (double v : x) ss += (v - m) * (v - m);
  return std::sqrt(ss / static_cast<double>(x.size() - 1));
}
// sd <= 1e-12*max(1,|mean|) is treated as zero (constant series), giving NaN ratios instead of noise.
bool sd_zero(double sd, double m) { return !(sd > 1e-12 * std::max(1.0, std::fabs(m))); }
double mean_of(const std::vector<double>& x) {
  if (x.empty()) return 0;
  double s = 0;
  for (double v : x) s += v;
  return s / static_cast<double>(x.size());
}
}  // namespace

Perf performance(const EquityCurve& s, const EquityCurve& bench) {
  Perf p;
  std::vector<TimePoint> ts;
  std::vector<double> r, rb;
  double ps = 1.0, pb = 1.0, peak = 1.0, vend = 1.0, base = 1.0;
  std::size_t i = 0, j = 0;
  const std::size_t ns = std::min(s.t.size(), s.value.size()), nb = std::min(bench.t.size(), bench.value.size());
  while (i < ns && j < nb) {
    if (s.t[i] < bench.t[j]) { ++i; continue; }
    if (s.t[i] > bench.t[j]) { ++j; continue; }
    if (ts.empty()) {  // first common date: measure against the previous own value (1.0 if the curve starts here)
      ps = i > 0 ? s.value[i - 1] : 1.0;
      pb = j > 0 ? bench.value[j - 1] : 1.0;
      base = peak = ps;
    }
    ts.push_back(s.t[i]);
    r.push_back(s.value[i] / ps - 1.0);
    rb.push_back(bench.value[j] / pb - 1.0);
    ps = s.value[i];
    pb = bench.value[j];
    peak = std::max(peak, ps);
    p.max_drawdown = std::max(p.max_drawdown, 1.0 - ps / peak);
    vend = ps;
    ++i; ++j;
  }
  const std::size_t T = r.size();
  if (T == 0) return p;
  std::vector<double> e(T);
  for (std::size_t d = 0; d < T; ++d) e[d] = r[d] - rb[d];

  p.cum_return = vend / base - 1.0;
  p.ann_return = std::pow(vend / base, kDays / static_cast<double>(T)) - 1.0;
  const double m = mean_of(r), sd = sample_sd(r, m);
  p.ann_vol = sd * std::sqrt(kDays);
  p.sharpe_daily = sd_zero(sd, m) ? std::nan("") : m / sd;
  p.sharpe = p.sharpe_daily * std::sqrt(kDays);

  const double me = mean_of(e), sde = sample_sd(e, me);
  p.ann_excess = me * kDays;
  p.ir = sd_zero(sde, me) ? std::nan("") : me / sde * std::sqrt(kDays);
  CI ci = block_bootstrap_mean_ci(std::span<const double>(e), 21, 2000, 11, 0.95);
  p.excess_ci95 = {ci.lo * kDays, ci.hi * kDays};

  // Calendar-year hit rate (UTC year of t).
  struct Y { std::size_t n = 0; double sum = 0; };
  std::map<int, Y> by_year;
  for (std::size_t d = 0; d < T; ++d) {
    Y& y = by_year[civil_from_days(floor_div(ts[d], 86400)).y];
    ++y.n;
    y.sum += e[d];
  }
  std::size_t hits = 0;
  for (const auto& [yr, y] : by_year) {
    if (y.n < 120) continue;
    ++p.years;
    if (y.sum > 0) ++hits;
  }
  p.year_hit_rate = p.years ? static_cast<double>(hits) / static_cast<double>(p.years) : 0.0;

  // Population skew and raw kurtosis of daily strategy returns.
  double m2 = 0, m3 = 0, m4 = 0;
  for (double v : r) {
    const double dv = v - m;
    m2 += dv * dv; m3 += dv * dv * dv; m4 += dv * dv * dv * dv;
  }
  const double n = static_cast<double>(T);
  m2 /= n; m3 /= n; m4 /= n;
  if (!sd_zero(std::sqrt(m2), m)) { p.skew = m3 / std::pow(m2, 1.5); p.kurt = m4 / (m2 * m2); }
  return p;
}

double deflated_sharpe(double sr, std::size_t T, double skew, double kurt, double trial_sr_var, std::size_t n_trials) {
  constexpr double kGamma = 0.5772156649;
  double srstar = 0;
  if (n_trials > 1) {
    const double N = static_cast<double>(n_trials);
    srstar = std::sqrt(std::max(0.0, trial_sr_var)) *
             ((1 - kGamma) * norm_inv(1 - 1 / N) + kGamma * norm_inv(1 - 1 / (N * std::exp(1.0))));
  }
  if (T < 2) return 0.5;
  const double den2 = 1 - skew * sr + (kurt - 1) / 4 * sr * sr;
  if (!(den2 > 0)) return std::nan("");
  return norm_cdf((sr - srstar) * std::sqrt(static_cast<double>(T - 1)) / std::sqrt(den2));
}

GateResult decision_gate(const Perf& p, double base_max_dd, double dsr, bool largecap_ok) {
  GateResult g{};
  g.c1 = p.ann_excess > 0 && p.excess_ci95.lo > 0;
  g.c2 = p.year_hit_rate >= 0.6;
  g.c3 = dsr > 0.95;
  g.c4 = p.max_drawdown <= base_max_dd + 0.05;
  g.c5 = largecap_ok;
  g.pass = g.c1 && g.c2 && g.c3 && g.c4 && g.c5;
  auto add = [&](bool ok, const std::string& msg) {
    if (ok) return;
    if (!g.reason.empty()) g.reason += "; ";
    g.reason += msg;
  };
  add(g.c1, "c1: excess return not significantly > 0 (ann_excess=" + std::to_string(p.ann_excess) +
                ", ci95.lo=" + std::to_string(p.excess_ci95.lo) + ")");
  add(g.c2, "c2: year hit rate " + std::to_string(p.year_hit_rate) + " < 0.6");
  add(g.c3, std::isnan(dsr) ? std::string("c3: deflated Sharpe undefined (non-positive variance term)")
                            : "c3: deflated Sharpe " + std::to_string(dsr) + " <= 0.95");
  add(g.c4, "c4: max drawdown " + std::to_string(p.max_drawdown) + " > base " + std::to_string(base_max_dd) + " + 0.05");
  add(g.c5, "c5: large-cap sub-universe check failed");
  return g;
}

}  // namespace mr
