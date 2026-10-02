#pragma once
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/types.hpp"

namespace fx {

struct LakeRow {
  std::string ticker;
  Bar bar;
};

// POSIX open + fsync + close of a file or directory; throws std::runtime_error on failure.
void fsync_path(const std::filesystem::path& p);
// create_directories that fsyncs the parent of every directory it creates (so the new entries are
// durable). Returns the created directories, outermost first.
std::vector<std::filesystem::path> create_dirs_synced(const std::filesystem::path& p);

struct RetentionPolicy {
  std::map<Timeframe, std::optional<int>> keep_days;  // nullopt = keep forever
  static RetentionPolicy defaults();                  // 1h: 730, 1d/1w: forever
  static RetentionPolicy load(const std::filesystem::path& json);  // missing file = defaults
};

// Spec 4.3: Hive-partitioned Parquet bars + a small DuckDB catalog. Single process.
class Lake {
 public:
  explicit Lake(std::filesystem::path root);
  ~Lake();
  Lake(const Lake&) = delete;
  Lake& operator=(const Lake&) = delete;

  // One write batch (new seq). Data becomes visible atomically per file; coverage and complete marks
  // are applied after. complete: per ticker, the latest bar time stored after its session or bucket
  // had closed (only moves later).
  void write(Timeframe tf, const std::vector<LakeRow>& rows,
             const std::vector<std::pair<std::string, TimePoint>>& coverage,
             const std::vector<std::pair<std::string, TimePoint>>& complete = {});
  std::map<std::string, std::vector<Bar>> read(Timeframe tf, const std::vector<std::string>& tickers,
                                               TimePoint start, TimePoint end);
  std::map<std::string, TimePoint> coverage(Timeframe tf, const std::vector<std::string>& tickers);
  std::map<std::string, TimePoint> complete(Timeframe tf, const std::vector<std::string>& tickers);
  // Earliest stored bar time per ticker (over the whole lake, not a window).
  std::map<std::string, TimePoint> first_times(Timeframe tf, const std::vector<std::string>& tickers);
  std::size_t compact(Timeframe tf, std::size_t max_files);
  std::size_t apply_retention(const RetentionPolicy& policy, TimePoint now);
  std::size_t file_count(Timeframe tf) const;
  const std::filesystem::path& root() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fx
