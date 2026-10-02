#pragma once
#include <string>
#include <vector>

#include "market/alpaca_client.hpp"
#include "market/bar_store.hpp"

namespace fx {

// Back-fills missing history, then fetches each ticker's tail; saves what succeeded; returns stale tickers.
// A ticker whose history was re-adjusted (split or dividend) is refetched in full. refetch_all treats
// every ticker as re-adjusted (one-time repair): each is refetched from the earliest of start, its first
// stored bar and its coverage through end, committed per batch; a failed ticker stays untouched.
std::vector<std::string> sync_bars(AlpacaClient& client, BarStore& store, const std::vector<std::string>& tickers,
                                   Timeframe tf, TimePoint start, TimePoint end, bool refetch_all = false);

}  // namespace fx
