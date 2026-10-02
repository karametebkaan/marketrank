#include "market/bar_store.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "core/csv.hpp"

namespace fx {

BarStore::BarStore(std::filesystem::path cache_dir) : cache_dir_(std::move(cache_dir)) {}

void BarStore::merge(const std::string& ticker, Timeframe tf, const std::vector<Bar>& incoming) {
  auto& series = series_[{ticker, tf}];
  std::map<TimePoint, Bar> by_time;
  for (const Bar& b : series) by_time[b.t] = b;
  for (const Bar& b : incoming) by_time[b.t] = b;
  series.clear();
  series.reserve(by_time.size());
  for (const auto& [t, b] : by_time) series.push_back(b);
}

const std::vector<Bar>& BarStore::bars(const std::string& ticker, Timeframe tf) const {
  static const std::vector<Bar> kEmpty;
  auto it = series_.find({ticker, tf});
  return it == series_.end() ? kEmpty : it->second;
}

std::optional<TimePoint> BarStore::last_time(const std::string& ticker, Timeframe tf) const {
  const auto& b = bars(ticker, tf);
  if (b.empty()) return std::nullopt;
  return b.back().t;
}

std::filesystem::path BarStore::file_for(const std::string& ticker, Timeframe tf) const {
  return cache_dir_ / std::string(to_string(tf)) / (ticker + ".csv");
}

void BarStore::save(const std::string& ticker, Timeframe tf) const {
  const auto path = file_for(ticker, tf);
  std::filesystem::create_directories(path.parent_path());
  auto tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::trunc);
    out << "t,o,h,l,c,v,vw\n" << std::setprecision(15);
    for (const Bar& b : bars(ticker, tf)) {
      out << b.t << ',' << b.o << ',' << b.h << ',' << b.l << ',' << b.c << ',' << b.v << ','
          << b.vw << '\n';
    }
    out.flush();
    if (!out) {
      out.close();
      std::filesystem::remove(tmp);
      throw std::runtime_error("failed to write " + tmp.string());
    }
  }
  std::filesystem::rename(tmp, path);
}

void BarStore::load_all(const std::vector<std::string>& tickers, Timeframe tf) {
  static const std::vector<std::string> kHeader = {"t", "o", "h", "l", "c", "v", "vw"};
  for (const auto& ticker : tickers) {
    try {
      const auto path = file_for(ticker, tf);
      if (!std::filesystem::exists(path)) continue;
      const CsvRows rows = read_csv_file(path);
      if (rows.empty() || rows[0] != kHeader) continue;
      std::vector<Bar> loaded;
      for (std::size_t r = 1; r < rows.size(); ++r) {
        const auto& f = rows[r];
        if (f.size() < 7) continue;
        try {
          loaded.push_back({std::stoll(f[0]), std::stod(f[1]), std::stod(f[2]), std::stod(f[3]),
                            std::stod(f[4]), std::stod(f[5]), std::stod(f[6])});
        } catch (const std::exception&) {
          continue;  // malformed row
        }
      }
      merge(ticker, tf, loaded);
    } catch (const std::exception&) {
      continue;  // unreadable file: treat as missing
    }
  }
}

}  // namespace fx

namespace fx {
std::optional<TimePoint> BarStore::first_time(const std::string& ticker, Timeframe tf) const {
  const auto& b = bars(ticker, tf);
  if (b.empty()) return std::nullopt;
  return b.front().t;
}
}  // namespace fx
