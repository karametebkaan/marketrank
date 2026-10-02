#include "market/panel.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace mr {

Panel build_panel(const BarStore& store, const std::vector<std::string>& tickers, Timeframe tf,
                  TimePoint start, TimePoint end) {
  Panel p;
  p.tickers = tickers;
  for (const auto& ticker : tickers)
    for (const Bar& b : store.bars(ticker, tf))
      if (start <= b.t && b.t <= end) p.times.push_back(b.t);
  std::sort(p.times.begin(), p.times.end());
  p.times.erase(std::unique(p.times.begin(), p.times.end()), p.times.end());

  std::unordered_map<TimePoint, std::size_t> row_of;
  for (std::size_t t = 0; t < p.times.size(); ++t) row_of[p.times[t]] = t;

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::size_t cells = p.T() * p.N();
  p.open.assign(cells, nan);
  p.high.assign(cells, nan);
  p.low.assign(cells, nan);
  p.close.assign(cells, nan);
  p.volume.assign(cells, nan);
  p.vwap.assign(cells, nan);
  for (std::size_t i = 0; i < tickers.size(); ++i) {
    for (const Bar& b : store.bars(tickers[i], tf)) {
      if (b.t < start || b.t > end) continue;
      const std::size_t k = p.idx(row_of[b.t], i);
      p.open[k] = b.o;
      p.high[k] = b.h;
      p.low[k] = b.l;
      p.close[k] = b.c;
      p.volume[k] = b.v;
      p.vwap[k] = b.vw > 0 ? b.vw : b.c;
    }
  }
  return p;
}

}  // namespace mr
