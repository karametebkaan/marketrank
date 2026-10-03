#pragma once
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "market/intraday.hpp"
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
//  active   1 if the pipeline's frame.active at bar t (a recent close and the liquidity floor), else 0 (0 at t = 0);
//           pressure is finite exactly where active is 1. The walk-forward's decision mask at rebalance d is
//           elig AND active.
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
struct PressureActive {
  std::vector<float> pressure, active;
};
PressureActive export_pressure_active(const Panel& p, const CoreParams& core);  // one pipeline pass over every bar
std::vector<float> export_elig(const Panel& p, const WalkForwardParams& wf);
std::vector<float> export_label(const Panel& p, std::size_t h);

// The feature names in file order (each written as <name>.f32).
const std::vector<std::string>& export_array_names();

// Writes every array (<dir>/<name>.f32) and meta.json; creates dir. sectors[i] belongs to panel.tickers[i]
// (throws std::invalid_argument on a size mismatch). Progress goes to stderr when `log` is true.
void export_panel(const Panel& p, const std::vector<std::string>& sectors, const PanelExportParams& ep,
                  const std::filesystem::path& dir, bool log = false);

// --- M5: --export-panel DIR --timeframe 15m --------------------------------------------------------------------
// The regular-session 15-minute panel (26 bars per session, 14 on early-close days) with the same feature arrays as
// M4 plus a session-masked label and MarketRank's prior edges. Windows are in bars with the same wall-clock meaning
// as the daily export (20 sessions = 520 bars). Decision bars: every bar.
struct IntradayExportParams {
  CoreParams core = CoreParams::market_rank_intraday();  // pressure, active and the prior edges
  std::size_t label_h = 6;                              // label_6 = open[t+1+6] / open[t+1] - 1, session-masked
  std::size_t vol_window = 520, vol_need = 260;         // vol20: sd of ret1 over 20 sessions of bars
  std::size_t dv_sessions = 20, dv_need = 10;           // dvshock: same slot over the previous 20 sessions
  std::size_t elig_window = 520;                        // eligible_at window (bars)
  double elig_min_dollar_volume = 50e6 / 26;            // M4's $50M per day, per bar
  std::size_t elig_top_n = 0;
  bool write_prior = true;
};

// Generic trailing volatility: sample sd of ret1 over bars t-window+1..t (>= need finite values). The M4 vol20 is
// export_vol(p, 20, 10).
std::vector<float> export_vol(const Panel& p, std::size_t window, std::size_t need);
// dv[t] / median(dv at the same slot in sessions s-sessions .. s-1) (finite values only, >= need of them, median > 0),
// dv = close * volume: the intraday volume shock net of the time-of-day profile. Reads bars <= t only.
std::vector<float> export_dvshock_slot(const Panel& p, const SessionIndex& si, std::size_t sessions, std::size_t need);
// open[t+1+h] / open[t+1] - 1 when bars t+1 .. t+1+h are contiguous bars of one session (same session index and
// times[t+1+h] - times[t+1] = h * 900 s), else NaN: never an overnight return.
std::vector<float> export_session_label(const Panel& p, const SessionIndex& si, std::size_t h);
// The 15m array names in file order (<name>.f32): ret1 ldv dvshock vol20 pressure active elig label_6.
const std::vector<std::string>& intraday_array_names();

// prior/edges.bin: one 20-byte little-endian record per kept transition edge of an ACTIVE source row of the
// CorePipeline's Frame::P after step(t) (the cumulative MarketRank chain): t (uint32 bar index), src (uint32),
// dst (uint32), P (float32, the transition probability P[src][dst]), raw (float32, the accumulated dollar flux of the
// edge). Sorted by (t, src, dst). prior/offsets.u64: T + 1 uint64 record offsets; bar t's edges are records
// [offsets[t], offsets[t+1]). Bar 0 has none (the pipeline starts at bar 1).
struct PriorEdge {
  std::uint32_t t, src, dst;
  float p, raw;
};
static_assert(sizeof(PriorEdge) == 20, "PriorEdge is the 20-byte on-disk record");
struct PriorEdges {
  std::vector<std::uint64_t> offsets;
  std::vector<PriorEdge> edges;
};
PriorEdges read_prior(const std::filesystem::path& prior_dir);

// Writes the 15m arrays, prior/ and meta.json (session index, windows, label mask, causality); creates dir.
void export_panel_intraday(const Panel& p, const std::vector<std::string>& sectors, const IntradayExportParams& ep,
                           const std::filesystem::path& dir, bool log = false);

// Reads a float32 little-endian file (for tests and checks). Throws std::runtime_error if unreadable.
std::vector<float> read_f32(const std::filesystem::path& file);

}  // namespace mr
