#pragma once
#include <filesystem>
#include <string>

#include "market/panel.hpp"
#include "walkforward/walkforward.hpp"

namespace mr {

// Canonical one-line description of every parameter that affects results (core model, calendar, eligibility,
// IC horizons, backtest, blend). largecap_run is excluded.
std::string describe(const WalkForwardParams& p);
// 8 lowercase hex digits: FNV-1a 64 of describe(p), folded to 32 bits.
std::string params_hash(const WalkForwardParams& p);

// Run ids name a directory and a registry key: 1-128 characters from [A-Za-z0-9._:+-], not "." or "..".
// Throws std::invalid_argument naming `what` otherwise.
void check_run_id(const std::string& id, const std::string& what);

// Trials in a registry.csv: rows that parse (>= 8 fields) with a finite sharpe_daily (n_trials, trial_sr_var), and,
// separately, rows with a finite 9th field ir_daily (n_ir_trials, trial_ir_var; 8-column rows written before
// 2026-10-02 are not IR trials). Variances are sample variances (0 with fewer than 2). A missing file has no trials.
struct RegistryStats {
  std::size_t n_trials = 0;
  double trial_sr_var = 0;
  std::size_t n_ir_trials = 0;
  double trial_ir_var = 0;
};
RegistryStats registry_stats(const std::filesystem::path& registry_csv);

// Read-only pre-pass check (no lock, nothing written): throws std::invalid_argument if run_id is already in
// out_dir/registry.csv with a params hash other than `hash`. write_report repeats the check under the lock.
void check_registry_conflict(const std::filesystem::path& out_dir, const std::string& run_id, const std::string& hash);

// Writes results.json, equity.csv, trades.csv, report.md under out_dir/<run_id>/ and records the run in
// out_dir/registry.csv (run_id,strategy,params_hash,cost_bps,top_n,sharpe_daily,T,ann_excess,ir_daily): one row per
// strategy curve (every curve not named "bench:*") with at least one simulated day. A run id already in the
// registry with the same params hash has its rows replaced; with a different hash it is rejected
// (std::invalid_argument) before anything is written. The registry is rewritten atomically (temp file + rename) under an advisory flock on
// out_dir/registry.csv.lock. Returns the run directory. Perf is against "bench:buyhold". Gate c3 deflates the blend's
// daily IR of the excess (dsr_excess) over the registry's IR trials; DSR(total) over the Sharpe trials is reported
// as informational. Both use registry_stats() of the updated registry (this run's rows included). c5 comes
// from the sibling run p.largecap_run (its results.json gate.core_pass) or is reported "pending". Throws
// std::runtime_error if the sibling run is named but unreadable (before anything is written).
std::filesystem::path write_report(const WalkForwardResult&, const WalkForwardParams&, const Panel&,
                                   const std::filesystem::path& out_dir, const std::string& run_id);

// Developer path (--wf-rereport): regenerates results.json and report.md of out_dir/<run_id> from its stored
// results.json (params, panel, IC table, blend, costs, turnover) and equity.csv, re-registering the run's rows under
// its stored params hash exactly as write_report does. equity.csv and trades.csv are not touched. Throws if the
// files are missing or malformed.
std::filesystem::path rereport(const std::filesystem::path& out_dir, const std::string& run_id);

}  // namespace mr
