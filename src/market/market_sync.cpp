#include "market/market_sync.hpp"

#include <algorithm>
#include <map>
#include <set>

#include "market/session_calendar.hpp"

namespace fx {
namespace {

TimePoint backfill_tolerance(Timeframe tf) {
  switch (tf) {
    case Timeframe::Hour: return 4 * 86400;
    case Timeframe::Day: return 5 * 86400;
    case Timeframe::Week: return 14 * 86400;
  }
  return 5 * 86400;
}

void fetch_into(AlpacaClient& client, BarStore& store, const std::vector<std::string>& group,
                Timeframe tf, TimePoint from, TimePoint to, std::set<std::string>& touched,
                std::set<std::string>& stale) {
  if (group.empty() || from >= to) return;
  FetchResult r = client.fetch_bars(group, alpaca_timeframe(tf), from, to);
  for (auto& [ticker, bars] : r.bars) {
    store.merge(ticker, tf, tf == Timeframe::Hour ? aggregate_session_hours(bars) : bars);
    touched.insert(ticker);
  }
  stale.insert(r.stale.begin(), r.stale.end());
}

}  // namespace

std::vector<std::string> sync_bars(AlpacaClient& client, BarStore& store,
                                   const std::vector<std::string>& tickers, Timeframe tf,
                                   TimePoint start, TimePoint end) {
  std::set<std::string> touched, stale;
  // 1. Back-fill history missing before the first cached bar (one group, up to the latest first bar).
  std::vector<std::string> backfill;
  TimePoint backfill_to = start;
  for (const auto& t : tickers) {
    const auto first = store.first_time(t, tf);
    if (first && *first > start + backfill_tolerance(tf)) {
      backfill.push_back(t);
      backfill_to = std::max(backfill_to, *first);
    }
  }
  fetch_into(client, store, backfill, tf, start, std::min(backfill_to, end), touched, stale);
  // 2. Incremental tail from each ticker's last cached bar (inclusive: refreshes a partial bar).
  std::map<TimePoint, std::vector<std::string>> by_start;
  for (const auto& t : tickers) by_start[store.last_time(t, tf).value_or(start)].push_back(t);
  for (const auto& [from, group] : by_start) fetch_into(client, store, group, tf, from, end, touched, stale);
  for (const auto& t : touched) store.save(t, tf);
  return {stale.begin(), stale.end()};
}

}  // namespace fx
