#include "pipeline/evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace mr {

double gini(std::span<const double> x) {
  const std::size_t n = x.size();
  if (n == 0) return 0.0;
  std::vector<double> v(x.begin(), x.end());
  std::sort(v.begin(), v.end());
  double sum = 0, weighted = 0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += v[i];
    weighted += static_cast<double>(i + 1) * v[i];
  }
  if (!(sum > 0)) return 0.0;
  const double nd = static_cast<double>(n);
  return 2.0 * weighted / (nd * sum) - (nd + 1.0) / nd;
}

namespace {

std::vector<double> ranks(std::span<const double> x) {
  const std::size_t n = x.size();
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](auto a, auto b) { return x[a] < x[b]; });
  std::vector<double> r(n);
  for (std::size_t i = 0; i < n;) {
    std::size_t j = i;
    while (j + 1 < n && x[order[j + 1]] == x[order[i]]) ++j;
    const double avg = 0.5 * static_cast<double>(i + j) + 1.0;
    for (std::size_t k = i; k <= j; ++k) r[order[k]] = avg;
    i = j + 1;
  }
  return r;
}

void summarize(const std::vector<double>& v, double& mean, double& tstat) {
  mean = 0;
  tstat = 0;
  if (v.empty()) return;
  for (double x : v) mean += x;
  mean /= static_cast<double>(v.size());
  if (v.size() < 2) return;
  double ss = 0;
  for (double x : v) ss += (x - mean) * (x - mean);
  const double sd = std::sqrt(ss / static_cast<double>(v.size() - 1));
  if (sd > 0) tstat = mean / (sd / std::sqrt(static_cast<double>(v.size())));
}

bool known_sector(const std::string& s) { return !s.empty() && s != "Unclassified" && s != "Extra"; }

}  // namespace

double spearman(std::span<const double> a_in, std::span<const double> b_in) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (a_in.size() != b_in.size()) return nan;
  // Only pairs where both values are finite take part (NaN would break the rank sort).
  std::vector<double> a, b;
  a.reserve(a_in.size());
  b.reserve(b_in.size());
  for (std::size_t i = 0; i < a_in.size(); ++i) {
    if (!std::isfinite(a_in[i]) || !std::isfinite(b_in[i])) continue;
    a.push_back(a_in[i]);
    b.push_back(b_in[i]);
  }
  if (a.size() < 3) return nan;
  const auto ra = ranks(a), rb = ranks(b);
  const double n = static_cast<double>(a.size());
  double ma = 0, mb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    ma += ra[i];
    mb += rb[i];
  }
  ma /= n;
  mb /= n;
  double sab = 0, saa = 0, sbb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double da = ra[i] - ma, db = rb[i] - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  if (!(saa > 0 && sbb > 0)) return nan;
  return sab / std::sqrt(saa * sbb);
}

double floor_share(const Frame& f, double alpha) {
  std::size_t n_active = 0;
  for (bool a : f.active) n_active += a ? 1 : 0;
  if (n_active == 0) return 0.0;
  const double floor = (1.0 - alpha) / static_cast<double>(n_active);
  std::size_t at_floor = 0;
  for (std::size_t i = 0; i < f.active.size(); ++i)
    if (f.active[i] && f.pi[i] <= floor * (1.0 + 1e-6)) ++at_floor;
  return static_cast<double>(at_floor) / static_cast<double>(n_active);
}

double structure_gain(const Frame& f) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  double total = 0;
  for (std::size_t i = 0; i < f.active.size() && i < f.inflow.size(); ++i)
    if (f.active[i]) total += f.inflow[i];
  if (!(total > 0)) return nan;
  std::vector<double> pis, shares;
  for (std::size_t i = 0; i < f.active.size() && i < f.inflow.size(); ++i) {
    if (!f.active[i]) continue;
    pis.push_back(f.pi[i]);
    shares.push_back(f.inflow[i] / total);
  }
  return 1.0 - spearman(pis, shares);
}

double sector_coherence(const Frame& f, const std::vector<Security>& nodes) {
  double same = 0, total = 0;
  for (std::size_t i = 0; i < f.P.n; ++i) {
    if (!f.active[i] || !known_sector(nodes[i].sector)) continue;
    for (auto e = f.P.row_ptr[i]; e < f.P.row_ptr[i + 1]; ++e) {
      const std::size_t j = f.P.col[e];
      if (j == i || !f.active[j] || !known_sector(nodes[j].sector)) continue;
      total += f.P.raw[e];
      if (nodes[j].sector == nodes[i].sector) same += f.P.raw[e];
    }
  }
  return total > 0 ? same / total : 0.0;
}

double oo_ic_for_frame(const Panel& panel, const Frame& f, std::size_t t, bool use_h,
                       std::size_t score_horizon) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (t + 2 >= panel.T() || f.active.size() != panel.N()) return nan;
  std::vector<double> x, r;
  for (std::size_t i = 0; i < panel.N(); ++i) {
    const double o1 = panel.open[panel.idx(t + 1, i)], o2 = panel.open[panel.idx(t + 2, i)];
    const double v = use_h ? f.h[i] : f.forecasts.at(score_horizon).score[i];
    if (!f.active[i] || !std::isfinite(o1) || !std::isfinite(o2) || !(o1 > 0) ||
        !std::isfinite(v))
      continue;
    x.push_back(v);
    r.push_back(o2 / o1 - 1.0);
  }
  return spearman(x, r);
}

EvalMetrics evaluate(const Panel& panel, const std::vector<Security>& nodes,
                     const CoreParams& params, std::size_t eval_bars) {
  const std::size_t T = panel.T(), N = panel.N();
  if (T < 3) throw std::runtime_error("evaluate: need at least three bars");
  if (nodes.size() != N) throw std::invalid_argument("evaluate: nodes/panel size mismatch");
  const std::size_t e0 = eval_bars >= T ? 1 : T - eval_bars;  // no eval_bars + 1 overflow
  // Clamp before casting: 3 * halflife_slow may be +inf, and the cast would be undefined.
  const double warm_d = std::min(std::ceil(3.0 * params.halflife_slow), static_cast<double>(T));
  const std::size_t warmup =
      std::max<std::size_t>(params.corr_window, static_cast<std::size_t>(warm_d));
  const std::size_t s0 = e0 > warmup + 1 ? e0 - warmup : 1;

  CorePipeline pipe(N, params);
  EvalMetrics m;
  std::vector<double> ics, ics_h, ics_oo, ics_h_oo;
  std::size_t frames = 0, sg_frames = 0;
  double ms = 0;
  // IC uses the one-bar-ahead forecast when the horizons include k = 1, else the first one.
  std::size_t fc = 0;
  for (std::size_t k = 0; k < params.horizons.size(); ++k)
    if (params.horizons[k] == 1) {
      fc = k;
      break;
    }
  for (std::size_t t = s0; t < T; ++t) {
    const Frame f = pipe.step(panel, t);
    if (t < e0) continue;
    ++frames;
    ms += f.compute_ms;
    m.floor_share += floor_share(f, params.alpha);
    std::vector<double> pis;
    for (std::size_t i = 0; i < N; ++i)
      if (f.active[i]) pis.push_back(f.pi[i]);
    m.gini += gini(pis);
    m.sector_coherence += sector_coherence(f, nodes);
    if (const double sg = structure_gain(f); std::isfinite(sg)) {
      m.structure_gain += sg;
      ++sg_frames;
    }
    if (t + 1 < T) {
      std::vector<double> s, h, r;
      for (std::size_t i = 0; i < N; ++i) {
        const double c0 = panel.close[panel.idx(t, i)], c1 = panel.close[panel.idx(t + 1, i)];
        const double sc = f.forecasts[fc].score[i];
        if (!f.active[i] || !std::isfinite(c0) || !std::isfinite(c1) || !(c0 > 0) ||
            !std::isfinite(sc) || !std::isfinite(f.h[i]))
          continue;
        s.push_back(sc);
        h.push_back(f.h[i]);
        r.push_back(c1 / c0 - 1.0);
      }
      const double ic = spearman(s, r), ich = spearman(h, r);
      if (std::isfinite(ic)) ics.push_back(ic);
      if (std::isfinite(ich)) ics_h.push_back(ich);
    }
    if (const double v = oo_ic_for_frame(panel, f, t, false, fc); std::isfinite(v))
      ics_oo.push_back(v);
    if (const double v = oo_ic_for_frame(panel, f, t, true, fc); std::isfinite(v))
      ics_h_oo.push_back(v);
  }
  if (frames > 0) {
    const double fr = static_cast<double>(frames);
    m.floor_share /= fr;
    m.gini /= fr;
    m.sector_coherence /= fr;
    m.mean_frame_ms = ms / fr;
  }
  if (sg_frames > 0) m.structure_gain /= static_cast<double>(sg_frames);
  summarize(ics, m.ic_mean, m.ic_t);
  summarize(ics_h, m.ic_h_mean, m.ic_h_t);
  summarize(ics_oo, m.ic_oo_mean, m.ic_oo_t);
  summarize(ics_h_oo, m.ic_h_oo_mean, m.ic_h_oo_t);
  m.ic_samples = ics.size();
  return m;
}

std::vector<EvalConfig> evaluation_grid() {
  const CoreParams L = CoreParams::legacy();
  std::vector<EvalConfig> g;
  g.push_back({"legacy", L});
  {
    CoreParams p = L;
    p.pressure = PressureMode::Relative;
    g.push_back({"+A relative", p});
  }
  {
    CoreParams p = L;
    p.transition.lift = LiftMode::Excess;
    g.push_back({"+B excess", p});
  }
  {
    CoreParams p = L;
    p.transition.k_in = 10;
    g.push_back({"+C k_in=10", p});
  }
  {
    CoreParams p = L;
    p.h_ref = HotRef::Size;
    g.push_back({"+D size", p});
  }
  {
    CoreParams p = L;
    p.h_ref = HotRef::LongRun;
    g.push_back({"+D longrun", p});
  }
  {
    CoreParams p = L;
    p.transition.retention = 1.0;
    g.push_back({"+E retention", p});
  }
  g.push_back({"defaults", CoreParams{}});
  {
    CoreParams p;
    p.pressure = PressureMode::Relative;
    g.push_back({"defaults relative", p});
  }
  {
    CoreParams p;
    p.h_ref = HotRef::LongRun;
    g.push_back({"defaults+longrun", p});
  }
  {
    CoreParams p;
    p.h_ref = HotRef::NetFlow;
    g.push_back({"defaults+netflow", p});
  }
  g.push_back({"money-flow", CoreParams::money_flow()});
  {
    CoreParams p = CoreParams::money_flow();
    p.h_ref = HotRef::NetFlow;
    g.push_back({"money-flow+netflow", p});
  }
  {
    CoreParams p;
    p.vol_scale = true;
    g.push_back({"defaults+volscale", p});
  }
  {
    CoreParams p = CoreParams::money_flow();
    p.vol_scale = true;
    g.push_back({"money-flow+volscale", p});
  }
  return g;
}

}  // namespace mr
