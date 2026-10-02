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

// When a bar's session (1d) or bucket (intraday; a week for 1w) has closed.
TimePoint bar_end(Timeframe tf, TimePoint t) {
  switch (tf) {
    case Timeframe::Hour: return t + 3600;
    case Timeframe::Day: return session_close(t);
    case Timeframe::Week: return t + 7 * 86400;
  }
  return session_close(t);
}

bool close_changed(double c_new, double c_old) {
  return std::fabs(c_new - c_old) / std::max(std::fabs(c_old), 1e-12) > 1e-6;
}

// Stored bar at time t, or nullptr. Series are sorted by t.
const Bar* stored_at(const std::vector<Bar>& series, TimePoint t) {
  auto it = std::lower_bound(series.begin(), series.end(), t, [](const Bar& b, TimePoint x) { return b.t < x; });
  return it != series.end() && it->t == t ? &*it : nullptr;
}

// readjusted != nullptr turns on the adjustment check (tail fetches only). A ticker found re-adjusted
// is neither merged nor saved: its stored data stays as it is until the full refetch replaces it, so
// a failed refetch is detected again on the next sync.
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
            // Compare every re-fetched bar that was complete when stored (t <= complete_through). A
            // changed close means Alpaca re-adjusted the history (split or dividend). Bars stored
            // before their session or bucket closed may change legitimately and are not compared.
            if (const auto done = store.complete_through(ticker, tf)) {
              const auto& old = store.bars(ticker, tf);
              bool changed = false;
              for (const Bar& b : bars) {
                if (b.t > *done) break;
                const Bar* o = stored_at(old, b.t);
                if (o && close_changed(b.c, o->c)) {
                  changed = true;
                  break;
                }
              }
              if (changed) {
                std::cerr << "sync: " << ticker << " re-adjusted; refetching history\n";
                readjusted->insert(ticker);
                continue;
              }
            }
          }
          store.merge(ticker, tf, bars);
          store.save(ticker, tf);
          // Record the latest bar whose session or bucket had closed by this fetch's end.
          for (auto it = bars.rbegin(); it != bars.rend(); ++it)
            if (bar_end(tf, it->t) <= to) {
              store.set_complete_through(ticker, tf, it->t);
              break;
            }
        }
        for (const auto& t : batch_symbols)
          if (cover.count(t)) store.set_covered_from(t, tf, from);
        store.flush();
      });
  stale.insert(r.stale.begin(), r.stale.end());
}

// Full refetch of each ticker from the earliest of start, its first stored bar and its coverage
// through end, grouped by that start. The newer seq wins in the lake.
void refetch_full(AlpacaClient& client, BarStore& store, const std::set<std::string>& tickers, Timeframe tf,
                  TimePoint start, TimePoint end, std::set<std::string>& stale) {
  std::map<TimePoint, std::vector<std::string>> by_from;
  for (const auto& t : tickers) {
    TimePoint from = start;
    if (const auto first = store.first_time(t, tf)) from = std::min(from, *first);
    if (const auto cov = store.covered_from(t, tf)) from = std::min(from, *cov);
    by_from[from].push_back(t);
  }
  for (const auto& [from, group] : by_from)
    fetch_into(client, store, group, tf, from, end, stale, std::set<std::string>(group.begin(), group.end()));
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
  // 2. Incremental tail (inclusive: refreshes a partial bar). It starts at the last bar that was
  // complete when stored, so the adjustment check always has a complete bar to compare.
  std::map<TimePoint, std::vector<std::string>> by_start;
  for (const auto& t : tickers) {
    TimePoint from = start;
    if (const auto last = store.last_time(t, tf)) {
      from = *last;
      if (const auto done = store.complete_through(t, tf)) from = std::min(from, *done);
    }
    by_start[from].push_back(t);
  }
  std::set<std::string> readjusted;
  for (const auto& [from, group] : by_start) {
    // Tickers with no cache are fetched from `start`, so that history is now covered.
    std::set<std::string> cover;
    for (const auto& t : group)
      if (!store.last_time(t, tf)) cover.insert(t);
    fetch_into(client, store, group, tf, from, end, stale, cover, &readjusted);
  }
  // 3. Re-adjusted tickers: re-fetch all stored history so every bar is on the new basis.
  refetch_full(client, store, readjusted, tf, start, end, stale);
  return {stale.begin(), stale.end()};
}

}  // namespace fx
