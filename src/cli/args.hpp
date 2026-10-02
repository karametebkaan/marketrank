#pragma once
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/types.hpp"
#include "pipeline/core_pipeline.hpp"
#include "walkforward/walkforward.hpp"

namespace mr {

enum class UniverseSource { Auto, Sp500, Snapshot };
enum class RankBy { Pi, Hotness };  // primary rank table: MarketRank pi (default) or hotness h

struct CliArgs {
  std::string mode = "synthetic";
  Timeframe tf = Timeframe::Day;
  int lookback_days = -1;  // resolved per timeframe: 1h 60, 1d 365, 1w 1825
  std::size_t top = 15;
  std::filesystem::path data = "data";
  UniverseSource universe = UniverseSource::Auto;
  std::size_t universe_size = 10000;
  bool refresh_universe = false;
  bool eval = false;
  std::size_t eval_bars = 120;
  int threads = 0;  // 0 = OpenMP default
  bool migrate_cache = false;
  std::filesystem::path migrate_from = "data/cache";
  bool maintain = false;
  bool sync_sectors = false;  // fetch SEC SIC sectors for the universe; needs no Alpaca keys
  bool refetch_full = false;  // alpaca only: refetch every ticker's full stored history once
  std::size_t export_slice = 0;  // replay only: export a graph slice of this many stocks (--export-slice [N], 6)
  std::filesystem::path slice_out = "docs/img/marketrank-slice.json";
  std::vector<std::pair<std::string, double>> shocks;  // --shock TICKER:SIZE, repeatable
  bool serve = false;
  int port = 8765;
  std::string host = "127.0.0.1";
  std::filesystem::path web = "web";
  bool help = false;
  RankBy rank_by = RankBy::Pi;
  std::string preset = "marketrank";  // the preset the model parameters started from: marketrank | money-flow | legacy | defaults
  CoreParams params = CoreParams::market_rank();
  // Walk-forward evaluation (--walkforward, replay only); the rest of WalkForwardParams keeps its defaults.
  bool walkforward = false;
  Rebalance wf_rebalance = Rebalance::Weekly;
  std::size_t wf_warmup = 252;   // warm-up bars before the first rebalance
  std::size_t wf_top_n = 0;      // 0 = every eligible name; 500 = the large-cap sub-universe
  double wf_cost_bps = 10;
  double wf_tilt = 0.20;
  std::size_t wf_k = 10;
  std::filesystem::path wf_out;  // empty = <data>/walkforward
  std::string wf_run_id;         // empty = <UTC timestamp>-<params hash>
  std::string wf_largecap_run;   // sibling large-cap run id for gate c5
  std::string wf_rereport;       // dev: regenerate results.json/report.md of this stored run (no lake, no pass)
  std::optional<BlendParams> wf_blend;  // --wf-blend TRAIN/EMBARGO/GATE/MIN; default blend_defaults(wf_rebalance)
  std::vector<std::string> warnings;    // non-fatal problems found while parsing (main prints them)
};

// Time range of bars to load and analyse: everything for synthetic (fixed historical dates),
// [now - lookback, now - 16 min] for alpaca (free-tier delay), [now - lookback, now] for replay.
std::pair<TimePoint, TimePoint> data_window(const CliArgs& args, TimePoint now);

CliArgs parse_cli(const std::vector<std::string>& args);
// Walk-forward parameters from the CLI (model preset, --wf-* flags; blend windows by calendar unless --wf-blend).
// The base mix is left empty for the caller (main loads it from data/portfolio.json).
WalkForwardParams walkforward_params(const CliArgs& a);
std::string cli_usage();
std::string describe(const CoreParams& p);

}  // namespace mr
