#pragma once
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/types.hpp"
#include "storage/lake.hpp"

namespace fx {

// In-memory bar series with upsert semantics, persisted to a Lake (spec 4.3). Merge-only use
// (synthetic markets, tests) never opens the lake.
class BarStore {
 public:
  explicit BarStore(std::filesystem::path lake_root);
  ~BarStore();
  BarStore(const BarStore&) = delete;
  BarStore& operator=(const BarStore&) = delete;

  void merge(const std::string& ticker, Timeframe tf, const std::vector<Bar>& incoming);
  const std::vector<Bar>& bars(const std::string& ticker, Timeframe tf) const;
  std::optional<TimePoint> last_time(const std::string& ticker, Timeframe tf) const;
  std::optional<TimePoint> first_time(const std::string& ticker, Timeframe tf) const;
  // Earliest start ever requested; only moves earlier; persisted after that flush's bars.
  std::optional<TimePoint> covered_from(const std::string& ticker, Timeframe tf) const;
  void set_covered_from(const std::string& ticker, Timeframe tf, TimePoint t);
  // Latest bar time that was stored after its session or bucket had closed; only moves later;
  // persisted with the same flush as the bars.
  std::optional<TimePoint> complete_through(const std::string& ticker, Timeframe tf) const;
  void set_complete_through(const std::string& ticker, Timeframe tf, TimePoint t);
  void save(const std::string& ticker, Timeframe tf);  // mark queued bars for the next flush
  void flush();                                        // persist everything saved so far
  void load_range(const std::vector<std::string>& tickers, Timeframe tf, TimePoint start, TimePoint end);
  void load_all(const std::vector<std::string>& tickers, Timeframe tf);
  Lake& lake();

 private:
  using Key = std::pair<std::string, Timeframe>;
  void upsert(const Key& key, const std::vector<Bar>& incoming);
  std::filesystem::path root_;
  std::unique_ptr<Lake> lake_;
  std::map<Key, std::vector<Bar>> series_;
  std::map<Key, TimePoint> covered_;
  std::map<Key, std::vector<Bar>> queued_;    // merged, not yet saved
  std::map<Key, std::vector<Bar>> to_write_;  // saved, awaiting flush
  std::map<Key, TimePoint> cov_to_write_;
  std::map<Key, TimePoint> complete_;
  std::map<Key, TimePoint> complete_to_write_;
};

}  // namespace fx
