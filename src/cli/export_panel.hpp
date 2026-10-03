#pragma once
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "market/panel.hpp"
#include "walkforward/walkforward.hpp"

namespace mr {

// --export-panel (M4): the replay panel as float32 little-endian arrays, row-major [T][N], NaN = missing, plus
// meta.json, for the graph-learning model (research/graphlearn). Eligibility, labels and rebalance dates come from
// the walk-forward's own functions, so the model trains and is scored on exactly the M3a calendar and universe.
//
// Features at bar t read only bars <= t (label_w is forward by construction):
//  ret1     log(close[t] / close[t-1]); NaN at t = 0 or if either close is non-finite or <= 0.
//  ldv      log(close[t] * volume[t]) (log dollar volume); NaN if not finite and > 0.
//  dvshock  dv[t] / median(dv[t-20..t-1]), dv = close * volume: the trailing median over the 20 previous bars
//           (bar t excluded; finite dv only, >= 10 values, median > 0), NaN otherwise or if dv[t] is not finite.
//  vol20    sample sd (n - 1) of ret1 over bars t-19..t (finite values only, >= 10 of them), NaN otherwise.
//  pressure the CorePipeline's pressure phi fed to the flux at bar t (last_pressure() after step(t), p.core =
//           CoreParams::market_rank(): dollar pressure), NaN at t = 0 and where the node is inactive at t.
//  elig     1 if eligible_at(panel, t, p.elig_window, p.min_dollar_volume, p.top_n), else 0.
//  label_w  forward_oo_return(panel, t, label_h): open[t+1+h] / open[t+1] - 1, the weekly (h = 5) label.
struct PanelExportParams {
  WalkForwardParams wf;    // calendar (weekly, warm-up 252), eligibility (20, 50e6, 0) and core model (market_rank)
  std::size_t label_h = 5;  // label_w horizon, bars
};

// Row-major [T][N] float32 arrays (the double values rounded to float).
std::vector<float> export_ret1(const Panel& p);
std::vector<float> export_ldv(const Panel& p);
std::vector<float> export_dvshock(const Panel& p);
std::vector<float> export_vol20(const Panel& p);
std::vector<float> export_pressure(const Panel& p, const CoreParams& core);  // one pipeline pass over every bar
std::vector<float> export_elig(const Panel& p, const WalkForwardParams& wf);
std::vector<float> export_label(const Panel& p, std::size_t h);

// The feature names in file order (each written as <name>.f32).
const std::vector<std::string>& export_array_names();

// Writes every array (<dir>/<name>.f32) and meta.json; creates dir. sectors[i] belongs to panel.tickers[i]
// (throws std::invalid_argument on a size mismatch). Progress goes to stderr when `log` is true.
void export_panel(const Panel& p, const std::vector<std::string>& sectors, const PanelExportParams& ep,
                  const std::filesystem::path& dir, bool log = false);

// Reads a float32 little-endian file (for tests and checks). Throws std::runtime_error if unreadable.
std::vector<float> read_f32(const std::filesystem::path& file);

}  // namespace mr
