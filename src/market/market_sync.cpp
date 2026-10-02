#include "market/market_sync.hpp"

#include <map>

#include "market/session_calendar.hpp"

namespace fx {

std::vector<std::string> sync_bars(AlpacaClient& client, BarStore& store, const std::vector<std::string>& tickers,
                                   Timeframe tf, TimePoint start, TimePoint end) {
  std::vector<std::string> stale;
  std::map<TimePoint, std::vector<std::string>> by_start;
  for (const auto& t : tickers) by_start[store.last_time(t, tf).value_or(start)].push_back(t);

  for (const auto& [from, group] : by_start) {
    if (from >= end) continue;
    FetchResult fetched = client.fetch_bars(group, alpaca_timeframe(tf), from, end);
    stale.insert(stale.end(), fetched.stale.begin(), fetched.stale.end());
    for (auto& [ticker, bars] : fetched.bars) {
      store.merge(ticker, tf, tf == Timeframe::Hour ? aggregate_session_hours(bars) : bars);
      store.save(ticker, tf);
    }
  }
  return stale;
}

}  // namespace fx
