#include "walkforward/walkforward.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>

#include "core/time.hpp"

namespace mr {

std::vector<BaseWeight> default_base_mix() {
  return {{"AAPL", 0.60}, {"VOO", 0.15}, {"NVDA", 0.07}, {"LLY", 0.06},
          {"NVO", 0.05},  {"NKE", 0.035}, {"F", 0.035}};
}

BlendParams blend_defaults(Rebalance r) {
  BlendParams b;  // weekly defaults
  if (r == Rebalance::Monthly) b.train_months = 36, b.embargo = 1, b.gate_months = 24, b.gate_min = 12;
  return b;
}

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

int year_of(TimePoint t) { return civil_from_days(floor_div(t, 86400)).y; }

struct YearIc {
  int year;
  double ic;
};

// IC row from samples in increasing bar order: overall mean/t and the mean per UTC year (ascending).
IcRow make_ic_row(std::string signal, int h, const std::vector<YearIc>& v) {
  IcRow row{std::move(signal), h, {}, {}};
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
  return row;
}
}  // namespace

std::vector<std::size_t> walkforward_dates(const Panel& panel, const WalkForwardParams& p) {
  // CorePipeline::step starts at bar 1, so a warm-up of 0 acts as 1.
  return rebalance_dates(panel, p.rebalance, std::max<std::size_t>(p.warmup_bars, 1));
}

std::string external_tag(const ExternalSignal& e) { return e.name + ":" + e.digest; }

std::string external_coverage_warning(const ExternalResult& e) {
  if (e.scored_rebalances >= e.scored_bars) return "";
  return "external " + e.name + ": " + std::to_string(e.scored_bars - e.scored_rebalances) + " of its " +
         std::to_string(e.scored_bars) +
         " scored bars are not walk-forward rebalance dates (off-by-one t?); they enter the IC rows only";
}

WalkForwardResult run_walkforward(const Panel& panel, const WalkForwardParams& p,
                                  const std::vector<ExternalSignal>& externals) {
  for (int h : p.ic_horizons)
    if (h <= 0) throw std::invalid_argument("run_walkforward: IC horizons must be positive");
  const std::size_t T = panel.T(), N = panel.N(), H = p.ic_horizons.size();
  {
    std::vector<std::string> tags;
    for (const auto& e : externals) {
      tags.push_back(external_tag(e));
      for (const auto& [bar, v] : e.scores)
        if (v.size() != N || bar >= T)
          throw std::invalid_argument("run_walkforward: external signal " + e.name + " does not fit the panel");
    }
    if (tags != p.externals)
      throw std::invalid_argument("run_walkforward: params.externals does not match the external signals given");
  }
  WalkForwardResult r;
  const std::size_t warmup = std::max<std::size_t>(p.warmup_bars, 1);  // CorePipeline::step starts at bar 1
  r.dates = walkforward_dates(panel, p);
  const std::size_t M = r.dates.size();
  for (const auto& e : externals) {  // coverage, before the long pass
    ExternalResult er{e.name, std::vector<double>(M, kNaN), std::vector<bool>(M, false), e.scores.size(), 0};
    for (std::size_t j = 0; j < M; ++j)
      if (e.scores.count(r.dates[j])) er.scored[j] = true, ++er.scored_rebalances;
    if (er.scored_rebalances == 0)
      throw std::invalid_argument("external signal " + e.name + ": none of its " + std::to_string(er.scored_bars) +
                                  " scored bars is a walk-forward rebalance date (t must be the unix time of a "
                                  "rebalance bar, as in the --export-panel meta.json)");
    r.externals.push_back(std::move(er));
  }
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

  r.ic_table.reserve((kSignals + externals.size()) * H);
  for (std::size_t s = 0; s < kSignals; ++s)
    for (std::size_t k = 0; k < H; ++k)
      r.ic_table.push_back(make_ic_row(std::string(to_string(static_cast<Signal>(s))), p.ic_horizons[k], ic[s * H + k]));

  // External signals' IC rows: their scored bars only, non-overlapping per horizon (see the header).
  if (!externals.empty()) {
    std::map<std::size_t, std::vector<bool>> elig_cache;  // bar -> eligible_at(bar)
    std::map<std::pair<std::size_t, std::size_t>, std::vector<double>> fwd_cache;  // (bar, h) -> forward return
    for (const auto& e : externals) {
      std::vector<std::vector<YearIc>> samples(H);
      for (std::size_t k = 0; k < H; ++k) {
        const auto h = static_cast<std::size_t>(p.ic_horizons[k]);
        bool have_last = false;
        std::size_t last = 0;
        for (const auto& [t, score] : e.scores) {
          if (t < warmup || t + 1 + h >= T) continue;
          if (have_last && t < last + h) continue;
          have_last = true;
          last = t;
          auto el = elig_cache.find(t);
          if (el == elig_cache.end())
            el = elig_cache.emplace(t, eligible_at(panel, t, p.elig_window, p.min_dollar_volume, p.top_n)).first;
          auto fw = fwd_cache.find({t, h});
          if (fw == fwd_cache.end()) fw = fwd_cache.emplace(std::make_pair(t, h), forward_oo_return(panel, t, h)).first;
          const double v = spearman(zscore(score, el->second), fw->second);
          if (std::isfinite(v)) samples[k].push_back({year_of(panel.times[t]), v});
        }
      }
      for (std::size_t k = 0; k < H; ++k) r.ic_table.push_back(make_ic_row(e.name, p.ic_horizons[k], samples[k]));
    }
  }

  r.blend = run_blend(r.months, p.blend);

  // Simulations, one strategy at a time so only one copy of the score history exists beyond the months.
  // External z-scores at rebalance j under the built-ins' mask; empty (no tilt) where the signal has no scores.
  auto ext_z = [&](const ExternalSignal& e, std::size_t j) {
    const auto it = e.scores.find(r.dates[j]);
    return it == e.scores.end() ? std::vector<double>{} : zscore(it->second, r.eligible[j]);
  };
  for (std::size_t x = 0; x < externals.size(); ++x)
    for (std::size_t j = 0; j < M; ++j) {
      if (!r.externals[x].scored[j]) continue;
      const double v = spearman(ext_z(externals[x], j), r.months[j].label);
      if (std::isfinite(v)) r.externals[x].rebalance_ic[j] = v;
    }
  // An external's curve over its scored span: decisions at rebalances a..b (first..last scored), cut at the close of
  // d_{b+1} (the end of the last scored period).
  auto ext_curve = [&](std::size_t x, const BacktestParams& bt) {
    const auto& sc = r.externals[x].scored;
    std::size_t a = 0, b = M - 1;
    while (!sc[a]) ++a;
    while (!sc[b]) --b;
    std::vector<Decision> d;
    for (std::size_t j = a; j <= b; ++j) d.push_back(Decision{r.dates[j], ext_z(externals[x], j), r.eligible[j]});
    EquityCurve c = simulate(panel, bt, d);
    if (b + 1 < M) {
      const TimePoint end = panel.times[r.dates[b + 1]];
      std::size_t keep = 0;
      while (keep < c.t.size() && c.t[keep] <= end) ++keep;
      c.t.resize(keep);
      c.value.resize(keep);
    }
    return c;
  };
  auto decisions = [&](auto score_of) {
    std::vector<Decision> d(M);
    for (std::size_t j = 0; j < M; ++j) d[j] = Decision{r.dates[j], score_of(j), r.eligible[j]};
    return d;
  };
  r.curves.emplace_back("blend", simulate(panel, p.bt, decisions([&](std::size_t j) { return r.blend[j].score; })));
  for (std::size_t s = 0; s < kSignals; ++s)
    r.curves.emplace_back("sig:" + std::string(to_string(static_cast<Signal>(s))),
                          simulate(panel, p.bt, decisions([&](std::size_t j) { return r.months[j].z[s]; })));
  for (std::size_t x = 0; x < externals.size(); ++x) r.curves.emplace_back("sig:" + externals[x].name, ext_curve(x, p.bt));
  std::vector<Decision> cal(M);
  for (std::size_t j = 0; j < M; ++j) cal[j] = Decision{r.dates[j], {}, {}};
  r.curves.emplace_back("bench:buyhold", simulate(panel, p.bt, cal, BenchKind::BuyHoldBase));
  r.curves.emplace_back("bench:rebalanced", simulate(panel, p.bt, cal, BenchKind::RebalancedBase));
  if (std::find(panel.tickers.begin(), panel.tickers.end(), "VOO") != panel.tickers.end())
    r.curves.emplace_back("bench:VOO", simulate(panel, p.bt, cal, BenchKind::Single, "VOO"));

  // Base-free sleeves (secondary, pre-registered for M3c): 100% in the top k at 1/k each, against the equal-weight
  // eligible universe on the same calendar. A flat blend (gate closed) holds that universe.
  BacktestParams sleeve = p.bt;
  sleeve.base.clear();
  sleeve.tilt = 1.0;
  sleeve.max_name_tilt = 1.0;
  sleeve.flat_holds_equal_weight = true;
  r.curves.emplace_back("sleeve:blend",
                        simulate(panel, sleeve, decisions([&](std::size_t j) { return r.blend[j].score; })));
  for (std::size_t s = 0; s < kSignals; ++s)
    r.curves.emplace_back("sleeve:" + std::string(to_string(static_cast<Signal>(s))),
                          simulate(panel, sleeve, decisions([&](std::size_t j) { return r.months[j].z[s]; })));
  for (std::size_t x = 0; x < externals.size(); ++x)
    r.curves.emplace_back("sleeve:" + externals[x].name, ext_curve(x, sleeve));
  r.curves.emplace_back("bench:ew_eligible", simulate(panel, sleeve, decisions([](std::size_t) {
                                                        return std::vector<double>{};
                                                      }),
                                                      BenchKind::EqualWeightEligible));
  return r;
}

}  // namespace mr
