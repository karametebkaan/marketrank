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

// Writes results.json, equity.csv, trades.csv, report.md under out_dir/<run_id>/ and appends registry.csv rows
// (run_id,strategy,params_hash,cost_bps,top_n,sharpe_daily,T,ann_excess; one per strategy curve, i.e. every curve
// not named "bench:*", with at least one simulated day) under out_dir; returns the run directory. Perf is against
// "bench:buyhold". The deflated Sharpe uses every registry row (this run's included): n_trials = row count, trial_sr_var = sample variance of the
// finite sharpe_daily values. The decision gate is the blend's; c5 comes from the sibling run p.largecap_run
// (its results.json gate.core_pass) or is reported "pending". Throws std::runtime_error if the sibling run is
// named but unreadable (before anything is written).
std::filesystem::path write_report(const WalkForwardResult&, const WalkForwardParams&, const Panel&,
                                   const std::filesystem::path& out_dir, const std::string& run_id);

}  // namespace mr
