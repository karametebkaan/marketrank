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
struct QuarterHoldings { std::string quarter; std::vector<Holding> rows; double dropped_value = 0, total_value = 0; std::size_t bad_rows = 0; };
QuarterHoldings load_quarter(const std::filesystem::path& dir, const std::string& quarter);
struct FlowEdge { std::uint32_t from, to; double dollars; };
struct ObservedParams { std::size_t top_n = 2000; double prune_rel = 1e-9; };
// nodes = indices into `tickers` of the top_n tickers by total 13F value (ascending index); edge from/to are such indices.
// outside_* = |d| of positions outside the node set; skipped_value = value of positions with unknown price / unknown ticker.
struct ObservedFlows { std::string quarter; std::vector<FlowEdge> edges; std::vector<std::uint32_t> nodes;
                       double paired = 0, unpaired_in = 0, unpaired_out = 0, outside_in = 0, outside_out = 0, skipped_value = 0;
                       std::size_t managers = 0; };
// Node indices are positions in `tickers`; prices = mean close of quarter q per ticker (NaN = unknown -> value_usd/shares
// fallback when finite, else the position is skipped). split_ratio(ticker) adjusts q-1 shares to q's basis (1.0 = none).
// Pairing per manager: F_ij = out_i*in_j/sum(in)*min(1, sum(in)/sum(out)), accumulated densely over the node set
// (no sink cap); edges with dollars <= prune_rel * sum(paired) are dropped.
ObservedFlows observed_flows(const QuarterHoldings& prev, const QuarterHoldings& cur, const std::vector<std::string>& tickers,
                             const std::vector<double>& prices, const std::function<double(const std::string&)>& split_ratio,
                             const ObservedParams& p = {});
}  // namespace mr
