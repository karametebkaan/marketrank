#pragma once
#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "market/panel.hpp"

namespace mr {

// An externally computed signal (--wf-external NAME=PATH, M4): raw scores per bar, evaluated by the walk-forward
// next to the built-in signals (IC table, tilt and sleeve curves, registry rows) but never part of the blend.
struct ExternalSignal {
  std::string name;
  std::string digest;  // 16 hex digits, FNV-1a 64 of the CSV bytes (part of the walk-forward params hash)
  std::map<std::size_t, std::vector<double>> scores;  // bar index -> raw score per node (size N, NaN = missing)
  std::size_t unknown_tickers = 0;  // CSV rows skipped because the ticker is not in the panel
};

// Names: 1-64 characters from [A-Za-z0-9_.-], not a built-in signal name and not "blend". Throws
// std::invalid_argument naming the problem otherwise.
void check_external_name(const std::string& name);

// Parses a score CSV with header "t,ticker,score" (t = unix seconds of a panel bar, normally a rebalance bar;
// score a finite number, or empty/"nan" for missing). Rows for tickers not in the panel are skipped and counted.
// Throws std::runtime_error on an unreadable file, a bad header or row, a t that is not a panel bar time, or a
// duplicate (t, ticker).
ExternalSignal parse_external_signal(const std::string& name, const std::string& csv_text, const Panel& panel);
ExternalSignal load_external_signal(const std::string& name, const std::filesystem::path& csv, const Panel& panel);

}  // namespace mr
