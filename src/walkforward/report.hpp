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

// Trials in a registry.csv: rows that parse (>= 8 fields) with a finite sharpe_daily; trial_sr_var is the sample
// variance of those values (0 with fewer than 2). A missing file has no trials.
struct RegistryStats {
  std::size_t n_trials = 0;
  double trial_sr_var = 0;
};
RegistryStats registry_stats(const std::filesystem::path& registry_csv);

// Writes results.json, equity.csv, trades.csv, report.md under out_dir/<run_id>/ and records the run in
// out_dir/registry.csv (run_id,strategy,params_hash,cost_bps,top_n,sharpe_daily,T,ann_excess): one row per strategy
// curve (every curve not named "bench:*") with at least one simulated day. A run id already in the registry with the
// same params hash has its rows replaced; with a different hash it is rejected (std::invalid_argument) before
// anything is written. The registry is rewritten atomically (temp file + rename) under an advisory flock on
// out_dir/registry.csv.lock. Returns the run directory. Perf is against "bench:buyhold". The deflated Sharpe uses
// registry_stats() of the updated registry (this run's rows included). The decision gate is the blend's; c5 comes
// from the sibling run p.largecap_run (its results.json gate.core_pass) or is reported "pending". Throws
// std::runtime_error if the sibling run is named but unreadable (before anything is written).
std::filesystem::path write_report(const WalkForwardResult&, const WalkForwardParams&, const Panel&,
                                   const std::filesystem::path& out_dir, const std::string& run_id);

}  // namespace mr
