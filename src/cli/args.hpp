#pragma once
#include <cstddef>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "core/types.hpp"
#include "pipeline/core_pipeline.hpp"

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
  // --- M3c 13F comparison (replay only) ---
  bool compare_13f = false;               // --compare-13f: observed 13F flows vs estimated flows, calibration grid
  std::vector<std::string> quarters_13f;  // --13f-quarters Q1,Q2,... (YYYYQn); empty = every quarter found
};

// Time range of bars to load and analyse: everything for synthetic (fixed historical dates),
// [now - lookback, now - 16 min] for alpaca (free-tier delay), [now - lookback, now] for replay.
std::pair<TimePoint, TimePoint> data_window(const CliArgs& args, TimePoint now);

CliArgs parse_cli(const std::vector<std::string>& args);
std::string cli_usage();
std::string describe(const CoreParams& p);

}  // namespace mr
