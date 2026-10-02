#pragma once
#include <string>
#include <vector>

#include "core/types.hpp"
#include "market/bar_store.hpp"

namespace fx {

struct Panel {
  std::vector<TimePoint> times;
  std::vector<std::string> tickers;
  std::vector<double> close, volume, vwap;  // row-major [t * N + i], NaN = missing

  std::size_t T() const { return times.size(); }
  std::size_t N() const { return tickers.size(); }
  std::size_t idx(std::size_t t, std::size_t i) const { return t * tickers.size() + i; }
};

Panel build_panel(const BarStore& store, const std::vector<std::string>& tickers, Timeframe tf);

}  // namespace fx
