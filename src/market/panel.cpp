#include "market/panel.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace fx {

Panel build_panel(const BarStore& store, const std::vector<std::string>& tickers, Timeframe tf) {
  Panel p;
  p.tickers = tickers;
  for (const auto& ticker : tickers)
    for (const Bar& b : store.bars(ticker, tf)) p.times.push_back(b.t);
  std::sort(p.times.begin(), p.times.end());
  p.times.erase(std::unique(p.times.begin(), p.times.end()), p.times.end());

  std::unordered_map<TimePoint, std::size_t> row_of;
  for (std::size_t t = 0; t < p.times.size(); ++t) row_of[p.times[t]] = t;

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::size_t cells = p.T() * p.N();
  p.close.assign(cells, nan);
  p.volume.assign(cells, nan);
  p.vwap.assign(cells, nan);
  for (std::size_t i = 0; i < tickers.size(); ++i) {
    for (const Bar& b : store.bars(tickers[i], tf)) {
      const std::size_t k = p.idx(row_of[b.t], i);
      p.close[k] = b.c;
      p.volume[k] = b.v;
      p.vwap[k] = b.vw > 0 ? b.vw : b.c;
    }
  }
  return p;
}

}  // namespace fx
