#include "storage/csv_migration.hpp"

#include <fstream>

#include "core/csv.hpp"

namespace mr {

std::size_t migrate_csv_cache(const std::filesystem::path& csv_root, BarStore& store) {
  namespace fs = std::filesystem;
  static const std::vector<std::string> kHeader = {"t", "o", "h", "l", "c", "v", "vw"};
  std::size_t imported = 0, since_flush = 0;
  for (Timeframe tf : {Timeframe::Hour, Timeframe::Day, Timeframe::Week}) {
    const fs::path dir = csv_root / std::string(to_string(tf));
    if (!fs::is_directory(dir)) continue;
    for (const auto& e : fs::directory_iterator(dir)) {
      if (!e.is_regular_file()) continue;
      const std::string ticker = e.path().stem().string();
      if (e.path().extension() == ".from") {
        std::ifstream in(e.path());
        long long v = 0;
        if (in >> v) store.set_covered_from(ticker, tf, v);
        continue;
      }
      if (e.path().extension() != ".csv") continue;
      try {
        const CsvRows rows = read_csv_file(e.path());
        if (rows.empty() || rows[0] != kHeader) continue;
        std::vector<Bar> bars;
        for (std::size_t r = 1; r < rows.size(); ++r) {
          const auto& f = rows[r];
          if (f.size() < 7) continue;
          try {
            bars.push_back({std::stoll(f[0]), std::stod(f[1]), std::stod(f[2]), std::stod(f[3]),
                            std::stod(f[4]), std::stod(f[5]), std::stod(f[6])});
          } catch (const std::exception&) {
          }
        }
        if (bars.empty()) continue;
        store.merge(ticker, tf, bars);
        store.save(ticker, tf);
        ++imported;
        if (++since_flush >= 500) {
          store.flush();
          since_flush = 0;
        }
      } catch (const std::exception&) {
      }
    }
  }
  store.flush();
  return imported;
}

}  // namespace mr
