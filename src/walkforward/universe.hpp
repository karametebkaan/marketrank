#pragma once
#include <cstddef>
#include <string_view>
#include <vector>

#include "market/panel.hpp"

namespace mr {

enum class Rebalance { Monthly, Weekly };
Rebalance parse_rebalance(std::string_view name);  // "monthly" | "weekly"; throws std::invalid_argument otherwise

// Last bar of each calendar month (monthly) or Monday-based ISO week (weekly), by UTC date.
// Only indices in [warmup_bars, T-2] are returned so a next open exists.
std::vector<std::size_t> rebalance_dates(const Panel& p, Rebalance mode, std::size_t warmup_bars);

// Point-in-time eligibility at bar t (reads only bars <= t): trailing median of vwap*volume over
// [t-window+1, t] (non-finite skipped, >= window/2 values required) >= min_dollar_volume, and within
// the top_n by that median (0 = no limit; ties broken by lower index).
std::vector<bool> eligible_at(const Panel& p, std::size_t t, std::size_t window, double min_dollar_volume,
                              std::size_t top_n);

// open[t+1+h] / open[t+1] - 1 per stock; NaN if either open is non-finite or t+1+h >= T.
std::vector<double> forward_oo_return(const Panel& p, std::size_t t, std::size_t h);

}  // namespace mr
