#pragma once
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "market/bar_store.hpp"
#include "market/universe.hpp"

namespace fx {

struct AssetInfo {
  std::string symbol, name, exchange;
  bool tradable = false;
};

std::vector<AssetInfo> parse_assets(const std::string& json);  // throws std::runtime_error

struct UniverseRules {
  std::set<std::string> always_include;  // S&P 500, portfolio holdings, include.csv
  std::set<std::string> exclude;         // exclude.csv
};

// Spec 3: tradable; exclude list; always_include; exchange; warrant/unit/right names only.
bool passes_universe_rules(const AssetInfo& a, const UniverseRules& rules);

struct RankedAsset {
  AssetInfo asset;
  double median_dollar_volume = 0;
};

// Median v*vwap over the last `window` Day bars; no bars = dropped; descending (ties by symbol).
std::vector<RankedAsset> rank_by_liquidity(const std::vector<AssetInfo>& assets,
                                           const BarStore& store, std::size_t window,
                                           std::size_t top_n);

// ticker,name,sector,exchange,median_dollar_volume (sector from sp500, else Unclassified).
void write_universe_snapshot(const std::filesystem::path& path,
                             const std::vector<RankedAsset>& ranked, const Universe& sp500);
// Snapshot names: universe_<YYYY-MM-DD>.csv (old) or universe_<YYYY-MM-DD>_n<size>.csv, where
// size is the requested --universe-size.
std::optional<std::string> snapshot_date(const std::filesystem::path& path);  // "YYYY-MM-DD"
std::optional<std::size_t> snapshot_size(const std::filesystem::path& path);  // new form only
std::optional<std::filesystem::path> latest_snapshot(const std::filesystem::path& dir);
// Newest snapshot named for exactly `size` whose date is less than max_age_days before `now`.
std::optional<std::filesystem::path> find_snapshot(const std::filesystem::path& dir,
                                                   std::size_t size, TimePoint now,
                                                   int max_age_days);
Universe load_snapshot(const std::filesystem::path& path, const std::filesystem::path& funds_csv);
std::set<std::string> read_ticker_list(const std::filesystem::path& path);

}  // namespace fx
