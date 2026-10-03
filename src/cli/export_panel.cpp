#include "cli/export_panel.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "core/time.hpp"
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

std::vector<float> export_vol20(const Panel& p) {
  constexpr std::size_t kWin = 20, kNeed = 10;
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
