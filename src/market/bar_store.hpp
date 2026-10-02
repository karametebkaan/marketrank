#pragma once
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/types.hpp"

namespace fx {

class BarStore {
 public:
  explicit BarStore(std::filesystem::path cache_dir);

  void merge(const std::string& ticker, Timeframe tf, const std::vector<Bar>& incoming);
  const std::vector<Bar>& bars(const std::string& ticker, Timeframe tf) const;
  std::optional<TimePoint> last_time(const std::string& ticker, Timeframe tf) const;
  void save(const std::string& ticker, Timeframe tf) const;
  void load_all(const std::vector<std::string>& tickers, Timeframe tf);

 private:
  std::filesystem::path file_for(const std::string& ticker, Timeframe tf) const;
  std::filesystem::path cache_dir_;
  std::map<std::pair<std::string, Timeframe>, std::vector<Bar>> series_;
};

}  // namespace fx
