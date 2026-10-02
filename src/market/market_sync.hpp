#pragma once
#include <string>
#include <vector>

#include "market/alpaca_client.hpp"
#include "market/bar_store.hpp"

namespace fx {

void sync_bars(AlpacaClient& client, BarStore& store, const std::vector<std::string>& tickers,
               Timeframe tf, TimePoint start, TimePoint end);

}  // namespace fx
