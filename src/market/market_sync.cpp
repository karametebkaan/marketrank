#include "market/market_sync.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
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
                Timeframe tf, TimePoint from, TimePoint to, std::set<std::string>& stale,
                const std::set<std::string>& cover, std::set<std::string>* readjusted = nullptr) {
  if (group.empty() || from >= to) return;
  // Commit each batch (bars, then coverage) before the next request, so a crash loses at most one batch.
  FetchResult r = client.fetch_bars(
      group, alpaca_timeframe(tf), from, to,
      [&](const std::vector<std::string>& batch_symbols,
          const std::map<std::string, std::vector<Bar>>& batch_bars) {
        for (const auto& [ticker, raw] : batch_bars) {
          const std::vector<Bar> bars = tf == Timeframe::Hour ? aggregate_session_hours(raw) : raw;
          if (readjusted) {
            // The inclusive tail re-fetches the last cached bar. A changed close means Alpaca
            // re-adjusted the history (split or dividend), so the cached bars are on an old basis.
            const auto& old = store.bars(ticker, tf);
            if (!old.empty() && old.back().t == from) {
              const double c_old = old.back().c;
              for (const Bar& b : bars) {
                if (b.t != from) continue;
                if (std::fabs(b.c - c_old) / std::max(std::fabs(c_old), 1e-12) > 1e-6) {
                  std::cerr << "sync: " << ticker << " re-adjusted; refetching history\n";
                  readjusted->insert(ticker);
                }
                break;
              }
            }
          }
          store.merge(ticker, tf, bars);
          store.save(ticker, tf);
        }
        for (const auto& t : batch_symbols)
          if (cover.count(t)) store.set_covered_from(t, tf, from);
        store.flush();
      });
  stale.insert(r.stale.begin(), r.stale.end());
}

}  // namespace

std::vector<std::string> sync_bars(AlpacaClient& client, BarStore& store,
                                   const std::vector<std::string>& tickers, Timeframe tf,
                                   TimePoint start, TimePoint end) {
  std::set<std::string> stale;
  // 1. Back-fill history missing before the first cached bar (one group, up to the latest first bar).
  std::vector<std::string> backfill;
  TimePoint backfill_to = start;
  for (const auto& t : tickers) {
    const auto first = store.first_time(t, tf);
    const auto cov = store.covered_from(t, tf);
    if (first && *first > start + backfill_tolerance(tf) && (!cov || start < *cov)) {
      backfill.push_back(t);
      backfill_to = std::max(backfill_to, *first);
    }
  }
  fetch_into(client, store, backfill, tf, start, std::min(backfill_to, end), stale,
             std::set<std::string>(backfill.begin(), backfill.end()));
  // 2. Incremental tail from each ticker's last cached bar (inclusive: refreshes a partial bar).
  std::map<TimePoint, std::vector<std::string>> by_start;
  for (const auto& t : tickers) by_start[store.last_time(t, tf).value_or(start)].push_back(t);
  std::set<std::string> readjusted;
  for (const auto& [from, group] : by_start) {
    // Tickers with no cache are fetched from `start`, so that history is now covered.
    std::set<std::string> cover;
    for (const auto& t : group)
      if (!store.last_time(t, tf)) cover.insert(t);
    fetch_into(client, store, group, tf, from, end, stale, cover, &readjusted);
  }
  // 3. Re-adjusted tickers: re-fetch the full window so every bar is on the new basis (the newer
  // write wins in the lake).
  const std::vector<std::string> refetch(readjusted.begin(), readjusted.end());
  fetch_into(client, store, refetch, tf, start, end, stale, readjusted);
  return {stale.begin(), stale.end()};
}

}  // namespace fx
