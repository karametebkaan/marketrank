#include "market/market_sync.hpp"

#include <map>

#include "market/session_calendar.hpp"

namespace fx {

void sync_bars(AlpacaClient& client, BarStore& store, const std::vector<std::string>& tickers,
               Timeframe tf, TimePoint start, TimePoint end) {
  std::map<TimePoint, std::vector<std::string>> by_start;
  for (const auto& t : tickers) by_start[store.last_time(t, tf).value_or(start)].push_back(t);

  for (const auto& [from, group] : by_start) {
    if (from >= end) continue;
    auto fetched = client.fetch_bars(group, alpaca_timeframe(tf), from, end);
    for (auto& [ticker, bars] : fetched) {
      store.merge(ticker, tf, tf == Timeframe::Hour ? aggregate_session_hours(bars) : bars);
      store.save(ticker, tf);
    }
  }
}

}  // namespace fx
