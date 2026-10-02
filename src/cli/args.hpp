#pragma once
#include <cstddef>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "core/types.hpp"
#include "pipeline/core_pipeline.hpp"

namespace fx {

enum class UniverseSource { Auto, Sp500, Snapshot };

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
  bool help = false;
  CoreParams params;
};

// Time range of bars to load and analyse: everything for synthetic (fixed historical dates),
// [now - lookback, now - 16 min] for alpaca (free-tier delay), [now - lookback, now] for replay.
std::pair<TimePoint, TimePoint> data_window(const CliArgs& args, TimePoint now);

CliArgs parse_cli(const std::vector<std::string>& args);
std::string cli_usage();
std::string describe(const CoreParams& p);

}  // namespace fx
