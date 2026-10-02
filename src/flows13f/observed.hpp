#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace mr {
struct Holding { std::uint64_t cik; std::string ticker; double shares; double value_usd; };
// Reads holdings_<q>.csv joined with cusip_map.csv; rows whose CUSIP has no ticker are dropped (counted).
struct QuarterHoldings { std::string quarter; std::vector<Holding> rows; double dropped_value = 0, total_value = 0; };
QuarterHoldings load_quarter(const std::filesystem::path& dir, const std::string& quarter);
struct FlowEdge { std::uint32_t from, to; double dollars; };
struct ObservedFlows { std::string quarter; std::vector<FlowEdge> edges; double paired = 0, unpaired_in = 0, unpaired_out = 0; std::size_t managers = 0; };
// Node indices are positions in `tickers`; prices = mean close of quarter q per ticker (NaN = unknown -> value_usd/shares
// fallback when finite, else the position is skipped). split_ratio(ticker) adjusts q-1 shares to q's basis (1.0 = none).
// Pairing per manager: F_ij = out_i*in_j/sum(in)*min(1, sum(in)/sum(out)). If sources*sinks > kMaxPairs (1e6) only the
// top kMaxSinks (1000) sinks by in-dollars are kept and the in-dollars are rescaled to preserve sum(in).
ObservedFlows observed_flows(const QuarterHoldings& prev, const QuarterHoldings& cur, const std::vector<std::string>& tickers,
                             const std::vector<double>& prices, const std::function<double(const std::string&)>& split_ratio);
}  // namespace mr
