#include "walkforward/walkforward.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

#include "core/time.hpp"

namespace mr {

std::vector<BaseWeight> default_base_mix() {
  return {{"AAPL", 0.60}, {"VOO", 0.15}, {"NVDA", 0.07}, {"LLY", 0.06},
          {"NVO", 0.05},  {"NKE", 0.035}, {"F", 0.035}};
}

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

int year_of(TimePoint t) { return civil_from_days(floor_div(t, 86400)).y; }

struct YearIc {
  int year;
  double ic;
};
}  // namespace

WalkForwardResult run_walkforward(const Panel& panel, const WalkForwardParams& p) {
  for (int h : p.ic_horizons)
    if (h <= 0) throw std::invalid_argument("run_walkforward: IC horizons must be positive");
  const std::size_t T = panel.T(), N = panel.N(), H = p.ic_horizons.size();
  WalkForwardResult r;
  const std::size_t warmup = std::max<std::size_t>(p.warmup_bars, 1);  // CorePipeline::step starts at bar 1
  r.dates = rebalance_dates(panel, p.rebalance, warmup);
  r.months.reserve(r.dates.size());
  r.eligible.reserve(r.dates.size());

  CorePipeline pipe(N, p.core);
  SignalTracker tracker(N);
  std::vector<std::vector<YearIc>> ic(kSignals * H);  // [s * H + k]
  std::vector<std::size_t> due;                       // horizon slots sampled at this bar
  std::size_t m = 0;
  for (std::size_t t = 1; t < T; ++t) {
    const Frame frame = pipe.step(panel, t);
    const auto raw = tracker.update(frame);
    due.clear();
    if (t >= warmup)
      for (std::size_t k = 0; k < H; ++k) {
        const auto h = static_cast<std::size_t>(p.ic_horizons[k]);
        if ((t - warmup) % h == 0 && t + 1 + h < T) due.push_back(k);
      }
    const bool rebalance = m < r.dates.size() && r.dates[m] == t;
    if (due.empty() && !rebalance) continue;
    const std::vector<bool> elig = eligible_at(panel, t, p.elig_window, p.min_dollar_volume, p.top_n);

    if (!due.empty()) {
      std::vector<std::vector<double>> fwd(due.size());
      for (std::size_t j = 0; j < due.size(); ++j)
        fwd[j] = forward_oo_return(panel, t, static_cast<std::size_t>(p.ic_horizons[due[j]]));
      std::array<std::vector<double>, kSignals> out;  // out[s][j], index-owned per signal
#pragma omp parallel for schedule(static)
      for (std::size_t s = 0; s < kSignals; ++s) {
        const std::vector<double> z = zscore(raw[s], elig);
        out[s].resize(due.size());
        for (std::size_t j = 0; j < due.size(); ++j) out[s][j] = spearman(z, fwd[j]);
      }
      const int year = year_of(panel.times[t]);
      for (std::size_t s = 0; s < kSignals; ++s)
        for (std::size_t j = 0; j < due.size(); ++j)
          if (std::isfinite(out[s][j])) ic[s * H + due[j]].push_back({year, out[s][j]});
    }

    if (rebalance) {
      std::vector<bool> mask(N);
      for (std::size_t i = 0; i < N; ++i) mask[i] = elig[i] && frame.active[i];
      MonthRecord rec;
#pragma omp parallel for schedule(static)
      for (std::size_t s = 0; s < kSignals; ++s) rec.z[s] = zscore(raw[s], mask);
      r.months.push_back(std::move(rec));
      r.eligible.push_back(std::move(mask));
      ++m;
    }
  }

  // Labels: the return from the execution at d_m + 1 to the next execution at d_{m+1} + 1.
  for (std::size_t j = 0; j < r.months.size(); ++j)
    r.months[j].label = j + 1 < r.dates.size() ? forward_oo_return(panel, r.dates[j], r.dates[j + 1] - r.dates[j])
                                               : std::vector<double>(N, kNaN);

  r.ic_table.reserve(kSignals * H);
  for (std::size_t s = 0; s < kSignals; ++s)
    for (std::size_t k = 0; k < H; ++k) {
      const auto& v = ic[s * H + k];
      IcRow row{static_cast<Signal>(s), p.ic_horizons[k], {}, {}};
      std::vector<double> all(v.size());
      for (std::size_t j = 0; j < v.size(); ++j) all[j] = v[j].ic;
      row.all = mean_t(all);
      for (std::size_t j = 0; j < v.size();) {  // ascending years (t increases)
        std::size_t e = j;
        double sum = 0;
        while (e < v.size() && v[e].year == v[j].year) sum += v[e++].ic;
        row.by_year.emplace_back(v[j].year, sum / static_cast<double>(e - j));
        j = e;
      }
      r.ic_table.push_back(std::move(row));
    }

  r.blend = run_blend(r.months, p.blend);

  // Simulations, one strategy at a time so only one copy of the score history exists beyond the months.
  const std::size_t M = r.dates.size();
  auto decisions = [&](auto score_of) {
    std::vector<Decision> d(M);
    for (std::size_t j = 0; j < M; ++j) d[j] = Decision{r.dates[j], score_of(j), r.eligible[j]};
    return d;
  };
  r.curves.emplace_back("blend", simulate(panel, p.bt, decisions([&](std::size_t j) { return r.blend[j].score; })));
  for (std::size_t s = 0; s < kSignals; ++s)
    r.curves.emplace_back("sig:" + std::string(to_string(static_cast<Signal>(s))),
                          simulate(panel, p.bt, decisions([&](std::size_t j) { return r.months[j].z[s]; })));
  std::vector<Decision> cal(M);
  for (std::size_t j = 0; j < M; ++j) cal[j] = Decision{r.dates[j], {}, {}};
  r.curves.emplace_back("bench:buyhold", simulate(panel, p.bt, cal, BenchKind::BuyHoldBase));
  r.curves.emplace_back("bench:rebalanced", simulate(panel, p.bt, cal, BenchKind::RebalancedBase));
  if (std::find(panel.tickers.begin(), panel.tickers.end(), "VOO") != panel.tickers.end())
    r.curves.emplace_back("bench:VOO", simulate(panel, p.bt, cal, BenchKind::Single, "VOO"));
  return r;
}

}  // namespace mr
