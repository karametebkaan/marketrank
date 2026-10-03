#include "cli/export_panel.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <nlohmann/json.hpp>

#include "cli/args.hpp"
#include "core/time.hpp"
#include "market/session_calendar.hpp"
#include "pipeline/core_pipeline.hpp"
#include "walkforward/report.hpp"
#include "walkforward/universe.hpp"

namespace mr {
namespace fs = std::filesystem;

static_assert(std::endian::native == std::endian::little, "--export-panel writes the native float layout");

namespace {
constexpr float kNaNf = std::numeric_limits<float>::quiet_NaN();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

double dollar_volume(const Panel& p, std::size_t t, std::size_t i) {
  return p.close[p.idx(t, i)] * p.volume[p.idx(t, i)];
}

double median_of(std::vector<double>& v) {
  const std::size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
  double m = v[mid];
  if (v.size() % 2 == 0) m = 0.5 * (m + *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid)));
  return m;
}

void write_f32(const fs::path& file, const std::vector<float>& v) {
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write " + file.string());
  out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
  out.flush();
  if (!out) throw std::runtime_error("cannot write " + file.string());
}
}  // namespace

std::vector<float> export_ret1(const Panel& p) {
  const std::size_t T = p.T(), N = p.N();
  std::vector<float> out(T * N, kNaNf);
  for (std::size_t t = 1; t < T; ++t)
    for (std::size_t i = 0; i < N; ++i) {
      const double a = p.close[p.idx(t - 1, i)], b = p.close[p.idx(t, i)];
      if (std::isfinite(a) && std::isfinite(b) && a > 0 && b > 0) out[p.idx(t, i)] = static_cast<float>(std::log(b / a));
    }
  return out;
}

std::vector<float> export_ldv(const Panel& p) {
  std::vector<float> out(p.T() * p.N(), kNaNf);
  for (std::size_t t = 0; t < p.T(); ++t)
    for (std::size_t i = 0; i < p.N(); ++i) {
      const double dv = dollar_volume(p, t, i);
      if (std::isfinite(dv) && dv > 0) out[p.idx(t, i)] = static_cast<float>(std::log(dv));
    }
  return out;
}

std::vector<float> export_dvshock(const Panel& p) {
  constexpr std::size_t kWin = 20, kNeed = 10;
  const std::size_t T = p.T(), N = p.N();
  std::vector<float> out(T * N, kNaNf);
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < N; ++i) {
    std::vector<double> buf;
    buf.reserve(kWin);
    for (std::size_t t = 1; t < T; ++t) {
      const double dv = dollar_volume(p, t, i);
      if (!std::isfinite(dv)) continue;
      buf.clear();
      for (std::size_t u = t >= kWin ? t - kWin : 0; u < t; ++u) {
        const double x = dollar_volume(p, u, i);
        if (std::isfinite(x)) buf.push_back(x);
      }
      if (buf.size() < kNeed) continue;
      const double med = median_of(buf);
      if (med > 0) out[p.idx(t, i)] = static_cast<float>(dv / med);
    }
  }
  return out;
}

std::vector<float> export_vol(const Panel& p, std::size_t kWin, std::size_t kNeed) {
  if (kWin == 0 || kNeed < 2) throw std::invalid_argument("export_vol: need window >= 1 and need >= 2");
  const std::size_t T = p.T(), N = p.N();
  std::vector<float> out(T * N, kNaNf);
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < N; ++i) {
    std::vector<double> r(T, kNaN);
    for (std::size_t t = 1; t < T; ++t) {
      const double a = p.close[p.idx(t - 1, i)], b = p.close[p.idx(t, i)];
      if (std::isfinite(a) && std::isfinite(b) && a > 0 && b > 0) r[t] = std::log(b / a);
    }
    for (std::size_t t = 1; t < T; ++t) {
      double sum = 0;
      std::size_t n = 0;
      const std::size_t lo = t + 1 >= kWin ? t + 1 - kWin : 0;
      for (std::size_t u = lo; u <= t; ++u)
        if (std::isfinite(r[u])) sum += r[u], ++n;
      if (n < kNeed) continue;
      const double mean = sum / static_cast<double>(n);
      double ss = 0;
      for (std::size_t u = lo; u <= t; ++u)
        if (std::isfinite(r[u])) ss += (r[u] - mean) * (r[u] - mean);
      out[p.idx(t, i)] = static_cast<float>(std::sqrt(ss / static_cast<double>(n - 1)));
    }
  }
  return out;
}

std::vector<float> export_vol20(const Panel& p) { return export_vol(p, 20, 10); }

PressureActive export_pressure_active(const Panel& p, const CoreParams& core) {
  const std::size_t T = p.T(), N = p.N();
  PressureActive out{std::vector<float>(T * N, kNaNf), std::vector<float>(T * N, 0.0f)};
  CorePipeline pipe(N, core);
  for (std::size_t t = 1; t < T; ++t) {
    const Frame f = pipe.step(p, t);
    const auto& phi = pipe.last_pressure();
    for (std::size_t i = 0; i < N; ++i)
      if (f.active[i]) {
        if (!std::isfinite(phi[i])) throw std::logic_error("export_pressure_active: non-finite pressure of an active node");
        out.pressure[p.idx(t, i)] = static_cast<float>(phi[i]);
        out.active[p.idx(t, i)] = 1.0f;
      }
  }
  return out;
}

std::vector<float> export_elig(const Panel& p, const WalkForwardParams& wf) {
  const std::size_t T = p.T(), N = p.N();
  std::vector<float> out(T * N, 0.0f);
#pragma omp parallel for schedule(dynamic, 16)
  for (std::size_t t = 0; t < T; ++t) {
    const std::vector<bool> e = eligible_at(p, t, wf.elig_window, wf.min_dollar_volume, wf.top_n);
    for (std::size_t i = 0; i < N; ++i) out[p.idx(t, i)] = e[i] ? 1.0f : 0.0f;
  }
  return out;
}

std::vector<float> export_label(const Panel& p, std::size_t h) {
  const std::size_t T = p.T(), N = p.N();
  std::vector<float> out(T * N, kNaNf);
#pragma omp parallel for schedule(static)
  for (std::size_t t = 0; t < T; ++t) {
    const std::vector<double> r = forward_oo_return(p, t, h);
    for (std::size_t i = 0; i < N; ++i) out[p.idx(t, i)] = static_cast<float>(r[i]);
  }
  return out;
}

const std::vector<std::string>& export_array_names() {
  static const std::vector<std::string> names{"ret1",     "ldv",    "dvshock", "vol20",
                                              "pressure", "active", "elig",    "label_w"};
  return names;
}

void export_panel(const Panel& p, const std::vector<std::string>& sectors, const PanelExportParams& ep,
                  const fs::path& dir, bool log) {
  if (sectors.size() != p.N()) throw std::invalid_argument("export_panel: sectors size != N");
  if (ep.label_h == 0) throw std::invalid_argument("export_panel: label horizon must be >= 1");
  fs::create_directories(dir);
  const std::size_t T = p.T(), N = p.N();
  nlohmann::json files = nlohmann::json::object();
  std::vector<float> active;  // computed with the pressure (one pipeline pass), written next
  for (const auto& name : export_array_names()) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> v;
    if (name == "ret1") v = export_ret1(p);
    else if (name == "ldv") v = export_ldv(p);
    else if (name == "dvshock") v = export_dvshock(p);
    else if (name == "vol20") v = export_vol20(p);
    else if (name == "pressure") {
      PressureActive pa = export_pressure_active(p, ep.wf.core);
      v = std::move(pa.pressure);
      active = std::move(pa.active);
    } else if (name == "active") v = std::move(active);
    else if (name == "elig") v = export_elig(p, ep.wf);
    else v = export_label(p, ep.label_h);
    const fs::path file = dir / (name + ".f32");
    write_f32(file, v);
    files[name] = name + ".f32";
    if (log)
      std::cerr << "  " << name << ": " << v.size() * sizeof(float) / (1024.0 * 1024.0) << " MiB in "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
  }

  nlohmann::json j;
  j["format"] = "marketrank-panel-export/1";
  j["N"] = N;
  j["T"] = T;
  j["tickers"] = p.tickers;
  j["sectors"] = sectors;
  j["times"] = p.times;
  if (T) j["first"] = format_rfc3339(p.times.front()), j["last"] = format_rfc3339(p.times.back());
  j["layout"] = "float32 little-endian, row-major [T][N] (element t*N + i), NaN = missing";
  j["files"] = files;
  j["features"] = {
      {"ret1", "log(close[t] / close[t-1]); NaN at t = 0 or with a missing / non-positive close"},
      {"ldv", "log(close[t] * volume[t]), log dollar volume"},
      {"dvshock", "close*volume at t / median of close*volume over bars t-20..t-1 (>= 10 finite values; bar t excluded)"},
      {"vol20", "sample sd (n-1) of ret1 over bars t-19..t (>= 10 finite values)"},
      {"pressure", "CorePipeline::last_pressure() after step(t) with the walk-forward core model (dollar pressure "
                   "phi of bar t); NaN at t = 0 and where the node is inactive at t"},
      {"active", "1 if the pipeline's frame.active at t (a recent close and the liquidity floor of the core model), "
                 "else 0 (0 at t = 0); equal to isfinite(pressure)"},
      {"elig", "1 if eligible_at(panel, t, window, min_dollar_volume, top_n) (the walk-forward universe), else 0"},
      {"label_w", "forward_oo_return(panel, t, h) = open[t+1+h] / open[t+1] - 1 (NaN if t+1+h >= T): the label"}};
  j["feature_names"] = {"ret1", "ldv", "dvshock", "vol20", "pressure"};
  j["label_names"] = {"label_w"};
  j["label_horizons"] = {{"label_w", ep.label_h}};
  j["decision_mask"] =
      "The walk-forward z-scores and trades at rebalance d over elig[d] AND active[d] (eligible_at AND "
      "frame.active; active == isfinite(pressure)). A trainer scoring rebalances should use the same mask.";
  j["label_note"] =
      "label_w is a FIXED horizon: forward_oo_return(t, " + std::to_string(ep.label_h) +
      ") = open[t+1+h]/open[t+1] - 1 at every bar. The walk-forward's period label at rebalance d_m is "
      "forward_oo_return(d_m, d_{m+1} - d_m) (open after d_m to open after d_{m+1}), which differs from label_w "
      "when the period is not 5 bars (holiday weeks: 4). The IC table's h = 5 row uses the fixed horizon.";
  j["causality"] =
      "Every feature and elig at bar t reads only bars <= t. label_w is forward by construction: it reads opens at "
      "t+1 and t+1+h. A model scoring rebalance date d may use features at bars <= d and labels only of bars t with "
      "t+1+h <= d (realized by the open of d), or stricter with an embargo.";
  j["eligibility"] = {{"window", ep.wf.elig_window}, {"min_dollar_volume", ep.wf.min_dollar_volume},
                      {"top_n", ep.wf.top_n}};
  j["rebalance"] = {{"mode", ep.wf.rebalance == Rebalance::Weekly ? "weekly" : "monthly"},
                    {"warmup_bars", ep.wf.warmup_bars},
                    {"bars", walkforward_dates(p, ep.wf)},
                    {"note", "rebalance bar indices exactly as run_walkforward computes them (walkforward_dates); "
                             "score CSVs for --wf-external use times[bar] as t"}};
  j["wf_params"] = describe(ep.wf);
  j["wf_params_hash"] = params_hash(ep.wf);
  std::ofstream(dir / "meta.json") << j.dump(1) << "\n";
}

// ---------------------------------------------------------------------------------------------------------------
// M5: 15-minute export
// ---------------------------------------------------------------------------------------------------------------

std::vector<float> export_dvshock_slot(const Panel& p, const SessionIndex& si, std::size_t sessions, std::size_t need) {
  const std::size_t T = p.T(), N = p.N();
  if (si.session.size() != T) throw std::invalid_argument("export_dvshock_slot: session index does not fit the panel");
  std::vector<float> out(T * N, kNaNf);
  if (sessions == 0 || need == 0) return out;
  // bar_of[s][slot] = bar index, or -1.
  std::size_t max_slot = 0;
  for (std::size_t t = 0; t < T; ++t) max_slot = std::max<std::size_t>(max_slot, si.slot[t]);
  std::vector<std::vector<long>> bar_of(si.sessions, std::vector<long>(max_slot + 1, -1));
  for (std::size_t t = 0; t < T; ++t) bar_of[si.session[t]][si.slot[t]] = static_cast<long>(t);
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < N; ++i) {
    std::vector<double> buf;
    buf.reserve(sessions);
    for (std::size_t t = 0; t < T; ++t) {
      const double dv = dollar_volume(p, t, i);
      if (!std::isfinite(dv)) continue;
      const std::size_t s = si.session[t], k = si.slot[t];
      buf.clear();
      for (std::size_t q = s >= sessions ? s - sessions : 0; q < s; ++q) {
        const long u = bar_of[q][k];
        if (u < 0) continue;
        const double x = dollar_volume(p, static_cast<std::size_t>(u), i);
        if (std::isfinite(x)) buf.push_back(x);
      }
      if (buf.size() < need) continue;
      const double med = median_of(buf);
      if (med > 0) out[p.idx(t, i)] = static_cast<float>(dv / med);
    }
  }
  return out;
}

std::vector<float> export_session_label(const Panel& p, const SessionIndex& si, std::size_t h) {
  const std::size_t T = p.T(), N = p.N();
  if (si.session.size() != T) throw std::invalid_argument("export_session_label: session index does not fit the panel");
  if (h == 0) throw std::invalid_argument("export_session_label: horizon must be >= 1");
  std::vector<float> out(T * N, kNaNf);
  for (std::size_t t = 0; t + 1 + h < T; ++t) {
    const std::size_t a = t + 1, b = t + 1 + h;
    if (si.session[a] != si.session[b]) continue;
    if (p.times[b] - p.times[a] != static_cast<TimePoint>(h) * 900) continue;  // a missing bar inside the window
    for (std::size_t i = 0; i < N; ++i) {
      const double oa = p.open[p.idx(a, i)], ob = p.open[p.idx(b, i)];
      if (std::isfinite(oa) && std::isfinite(ob)) out[p.idx(t, i)] = static_cast<float>(ob / oa - 1.0);
    }
  }
  return out;
}

const std::vector<std::string>& intraday_array_names() {
  static const std::vector<std::string> names{"ret1",     "ldv",    "dvshock", "vol20",
                                              "pressure", "active", "elig",    "label_6"};
  return names;
}

PriorEdges read_prior(const fs::path& prior_dir) {
  PriorEdges out;
  auto read_all = [](const fs::path& f, auto& vec) {
    std::ifstream in(f, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot read " + f.string());
    const auto size = static_cast<std::size_t>(in.tellg());
    using V = typename std::decay_t<decltype(vec)>::value_type;
    if (size % sizeof(V) != 0) throw std::runtime_error(f.string() + ": size is not a whole number of records");
    vec.resize(size / sizeof(V));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(vec.data()), static_cast<std::streamsize>(size));
    if (!in) throw std::runtime_error("cannot read " + f.string());
  };
  read_all(prior_dir / "offsets.u64", out.offsets);
  read_all(prior_dir / "edges.bin", out.edges);
  if (out.offsets.empty() || out.offsets.back() != out.edges.size())
    throw std::runtime_error(prior_dir.string() + ": offsets do not match edges.bin");
  return out;
}

namespace {

// One pipeline pass: pressure and active (as in M4) and, when prior_dir is set, the kept edges of Frame::P per bar.
PressureActive pressure_active_prior(const Panel& p, const CoreParams& core, const fs::path& prior_dir,
                                     std::size_t& n_edges, bool log) {
  const std::size_t T = p.T(), N = p.N();
  PressureActive out{std::vector<float>(T * N, kNaNf), std::vector<float>(T * N, 0.0f)};
  std::ofstream edges_out;
  std::vector<std::uint64_t> offsets{0};
  if (!prior_dir.empty()) {
    fs::create_directories(prior_dir);
    edges_out.open(prior_dir / "edges.bin", std::ios::binary | std::ios::trunc);
    if (!edges_out) throw std::runtime_error("cannot write " + (prior_dir / "edges.bin").string());
    offsets.push_back(0);  // bar 0
  }
  n_edges = 0;
  CorePipeline pipe(N, core);
  std::vector<PriorEdge> bar_edges;
  const auto t0 = std::chrono::steady_clock::now();
  for (std::size_t t = 1; t < T; ++t) {
    const Frame f = pipe.step(p, t);
    const auto& phi = pipe.last_pressure();
    for (std::size_t i = 0; i < N; ++i)
      if (f.active[i]) {
        if (!std::isfinite(phi[i])) throw std::logic_error("export: non-finite pressure of an active node");
        out.pressure[p.idx(t, i)] = static_cast<float>(phi[i]);
        out.active[p.idx(t, i)] = 1.0f;
      }
    if (!prior_dir.empty()) {
      bar_edges.clear();
      for (std::size_t i = 0; i < N; ++i) {
        if (!f.active[i]) continue;  // inactive rows are a placeholder self-loop, not an estimated edge
        for (auto e = f.P.row_ptr[i]; e < f.P.row_ptr[i + 1]; ++e)
          bar_edges.push_back({static_cast<std::uint32_t>(t), static_cast<std::uint32_t>(i), f.P.col[e],
                               static_cast<float>(f.P.val[e]), static_cast<float>(f.P.raw[e])});
      }
      edges_out.write(reinterpret_cast<const char*>(bar_edges.data()),
                      static_cast<std::streamsize>(bar_edges.size() * sizeof(PriorEdge)));
      n_edges += bar_edges.size();
      offsets.push_back(n_edges);
    }
    if (log && (t % 1000 == 0 || t + 1 == T))
      std::cerr << "  pipeline: bar " << t << " / " << T - 1 << ", " << n_edges << " prior edges, "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
  }
  if (!prior_dir.empty()) {
    edges_out.flush();
    if (!edges_out) throw std::runtime_error("cannot write " + (prior_dir / "edges.bin").string());
    edges_out.close();
    if (T == 0) offsets.assign(1, 0);
    std::ofstream off(prior_dir / "offsets.u64", std::ios::binary | std::ios::trunc);
    off.write(reinterpret_cast<const char*>(offsets.data()),
              static_cast<std::streamsize>(offsets.size() * sizeof(std::uint64_t)));
    off.flush();
    if (!off) throw std::runtime_error("cannot write " + (prior_dir / "offsets.u64").string());
  }
  return out;
}

std::vector<float> intraday_elig(const Panel& p, const IntradayExportParams& ep) {
  const std::size_t T = p.T(), N = p.N();
  std::vector<float> out(T * N, 0.0f);
#pragma omp parallel for schedule(dynamic, 16)
  for (std::size_t t = 0; t < T; ++t) {
    const std::vector<bool> e = eligible_at(p, t, ep.elig_window, ep.elig_min_dollar_volume, ep.elig_top_n);
    for (std::size_t i = 0; i < N; ++i) out[p.idx(t, i)] = e[i] ? 1.0f : 0.0f;
  }
  return out;
}

}  // namespace

void export_panel_intraday(const Panel& p, const std::vector<std::string>& sectors, const IntradayExportParams& ep,
                           const fs::path& dir, bool log) {
  if (sectors.size() != p.N()) throw std::invalid_argument("export_panel_intraday: sectors size != N");
  if (ep.label_h == 0) throw std::invalid_argument("export_panel_intraday: label horizon must be >= 1");
  for (TimePoint t : p.times)
    if (!is_regular_session_bar(t))
      throw std::invalid_argument("export_panel_intraday: bar " + format_rfc3339(t) + " is not a regular-session 15m bar");
  fs::create_directories(dir);
  const std::size_t T = p.T(), N = p.N();
  const SessionIndex si = session_index(p.times);
  nlohmann::json files = nlohmann::json::object();
  std::vector<float> active;
  std::size_t n_edges = 0;
  double pipeline_s = 0;
  for (const auto& name : intraday_array_names()) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> v;
    if (name == "ret1") v = export_ret1(p);
    else if (name == "ldv") v = export_ldv(p);
    else if (name == "dvshock") v = export_dvshock_slot(p, si, ep.dv_sessions, ep.dv_need);
    else if (name == "vol20") v = export_vol(p, ep.vol_window, ep.vol_need);
    else if (name == "pressure") {
      PressureActive pa = pressure_active_prior(p, ep.core, ep.write_prior ? dir / "prior" : fs::path{}, n_edges, log);
      v = std::move(pa.pressure);
      active = std::move(pa.active);
    } else if (name == "active") v = std::move(active);
    else if (name == "elig") v = intraday_elig(p, ep);
    else v = export_session_label(p, si, ep.label_h);
    write_f32(dir / (name + ".f32"), v);
    files[name] = name + ".f32";
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (name == "pressure") pipeline_s = secs;
    if (log)
      std::cerr << "  " << name << ": " << v.size() * sizeof(float) / (1024.0 * 1024.0) << " MiB in " << secs << " s\n";
  }
  files["vol"] = "vol20.f32";       // M5 name; vol20 = 20 sessions of bars
  files["label_w"] = "label_6.f32";  // the M4 loader's label key

  std::vector<std::size_t> decision;
  for (std::size_t t = 1; t + 2 <= T; ++t) decision.push_back(t);
  const std::string lab = "label_" + std::to_string(ep.label_h);

  nlohmann::json j;
  j["format"] = "marketrank-panel-export/1";
  j["timeframe"] = "15m";
  j["bar_seconds"] = 900;
  j["N"] = N;
  j["T"] = T;
  j["tickers"] = p.tickers;
  j["sectors"] = sectors;
  j["times"] = p.times;
  if (T) j["first"] = format_rfc3339(p.times.front()), j["last"] = format_rfc3339(p.times.back());
  j["layout"] = "float32 little-endian, row-major [T][N] (element t*N + i), NaN = missing";
  j["files"] = files;
  j["session"] = {
      {"index", si.session},
      {"slot", si.slot},
      {"first_of_session", si.first},
      {"last_of_session", si.last},
      {"dates", si.dates},
      {"count", si.sessions},
      {"rules",
       "Regular session only: a bar is kept iff it starts on the 15-minute grid on a weekday with 09:30 ET <= start and "
       "start + 15 min <= the close, 16:00 ET, or 13:00 ET on NYSE early-close days (the day after Thanksgiving; July 3 "
       "and December 24 when Monday-Thursday). ET is UTC-5 / UTC-4 by the US DST rules (second Sunday of March to first "
       "Sunday of November). Full holidays have no regular-session bars in the feed, so they form no session. A session "
       "is one ET date; index counts sessions from 0, slot is the 15-minute slot from 09:30 (0..25; 0..13 on early "
       "closes), first/last_of_session flag the session's first / last bar in the panel (the final bar counts as "
       "last). Times are the union of the names' bar times; a name without a bar at a time is NaN there."}};
  j["features"] = {
      {"ret1", "log(close[t] / close[t-1]); NaN at t = 0 or with a missing / non-positive close. At the first bar of a "
               "session it spans the overnight gap (previous session's last close to this bar's close)."},
      {"ldv", "log(close[t] * volume[t]), log dollar volume of the 15-minute bar"},
      {"dvshock", "close*volume at t / median of close*volume at the SAME slot in the previous " +
                      std::to_string(ep.dv_sessions) + " sessions (>= " + std::to_string(ep.dv_need) +
                      " finite values; bar t excluded): the volume shock net of the time-of-day profile"},
      {"vol20", "sample sd (n-1) of ret1 over bars t-" + std::to_string(ep.vol_window - 1) + "..t (" +
                    std::to_string(ep.vol_window) + " bars = 20 sessions; >= " + std::to_string(ep.vol_need) +
                    " finite values); also listed as files.vol"},
      {"pressure", "CorePipeline::last_pressure() after step(t) with core_params (dollar pressure phi of bar t); NaN at "
                   "t = 0 and where the node is inactive at t"},
      {"active", "1 if the pipeline's frame.active at t (a close within stale_bars and the per-bar liquidity floor), "
                 "else 0; equal to isfinite(pressure)"},
      {"elig", "1 if eligible_at(panel, t, " + std::to_string(ep.elig_window) + ", " +
                   std::to_string(ep.elig_min_dollar_volume) + ", " + std::to_string(ep.elig_top_n) +
                   "): trailing median per-bar vwap*volume over 20 sessions >= M4's $50M/day / 26"},
      {lab, "open[t+1+" + std::to_string(ep.label_h) + "] / open[t+1] - 1, NaN unless bars t+1 .. t+1+" +
                std::to_string(ep.label_h) + " are contiguous bars of ONE session (same session index, times 15 min "
                "apart): masked across the close, no overnight returns. Also listed as files.label_w."}};
  j["feature_names"] = {"ret1", "ldv", "dvshock", "vol20", "pressure"};
  j["label_names"] = {lab};
  j["label_horizons"] = {{lab, ep.label_h}, {"label_w", ep.label_h}};
  j["windows_bars"] = {{"bars_per_session", 26},
                       {"adv_window", ep.core.adv_window},
                       {"corr_window", ep.core.corr_window},
                       {"stale_bars", ep.core.stale_bars},
                       {"halflife_slow", ep.core.halflife_slow},
                       {"halflife_fast_heartbeat", ep.core.halflife_fast},
                       {"halflife_cluster", ep.core.halflife_cluster},
                       {"min_dollar_volume_per_bar", ep.core.min_dollar_volume},
                       {"vol_window", ep.vol_window},
                       {"dvshock_sessions", ep.dv_sessions},
                       {"elig_window", ep.elig_window}};
  j["core_params"] = describe(ep.core);
  j["decision_mask"] = "elig[t] AND active[t] (active == isfinite(pressure)), as in M4.";
  j["rebalance"] = {{"mode", "every_bar"},
                    {"warmup_bars", 0},
                    {"bars", decision},
                    {"note", "M5 decides at every bar t in [1, T-2] (a next open exists); bars whose label is NaN "
                             "(the last 7 bars of each session) carry no target."}};
  if (ep.write_prior)
    j["prior"] = {{"dir", "prior"},
                  {"edges", "edges.bin"},
                  {"offsets", "offsets.u64"},
                  {"count", n_edges},
                  {"record", "20 bytes little-endian: t uint32 (bar index), src uint32, dst uint32 (node indices into "
                             "tickers), P float32, raw float32"},
                  {"order", "sorted by (t, src, dst); offsets.u64 holds T + 1 uint64: bar t's edges are records "
                            "[offsets[t], offsets[t+1]); bar 0 has none"},
                  {"definition", "after CorePipeline::step(t) with core_params: every kept edge of an ACTIVE source "
                                 "row of Frame::P, the transition matrix of the cumulative chain (halflife_slow; top "
                                 "k_out per row, k_in per column; no lift, no retention, dangling rows teleport). P = "
                                 "P[src][dst] (row-stochastic over the kept edges of a non-dangling row), raw = the "
                                 "accumulated dollar flux of the edge (un-normalized). Inactive rows are omitted."}};
  j["causality"] =
      "Every feature, elig, active and the prior edges at bar t read only bars <= t (the pipeline is stepped in time "
      "order; the session index is a calendar fact). label_6 is forward by construction: it reads opens at t+1 and "
      "t+7. A model deciding at bar d may use features and prior at bars <= d and labels only of bars t with "
      "t+7 <= d (realized by the open of d), or stricter with an embargo (M5: one session).";
  j["universe"] = "static list (data/universe/intraday_top1000.csv): top names by median daily dollar volume over the "
                  "window; survivorship bias acknowledged";
  j["timings_s"] = {{"pipeline_and_prior", pipeline_s}};
  std::ofstream(dir / "meta.json") << j.dump(1) << "\n";
}

std::vector<float> read_f32(const fs::path& file) {
  std::ifstream in(file, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot read " + file.string());
  const auto size = static_cast<std::size_t>(in.tellg());
  std::vector<float> v(size / sizeof(float));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
  if (!in) throw std::runtime_error("cannot read " + file.string());
  return v;
}

}  // namespace mr
