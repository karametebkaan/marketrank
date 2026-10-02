#pragma once
#include <string>
#include <vector>

#include "market/alpaca_client.hpp"
#include "market/bar_store.hpp"

namespace fx {

// Back-fills missing history, then fetches each ticker's tail; saves what succeeded; returns stale tickers.
std::vector<std::string> sync_bars(AlpacaClient& client, BarStore& store, const std::vector<std::string>& tickers,
                                   Timeframe tf, TimePoint start, TimePoint end);

}  // namespace fx
