#pragma once
// M3c: estimated (bar-flux) vs observed (13F) quarter flows, MarketRank agreement and the calibration grid.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/types.hpp"
#include "flows13f/observed.hpp"
#include "market/panel.hpp"
#include "pipeline/core_pipeline.hpp"
#include "pipeline/evaluation.hpp"  // spearman

namespace mr {

// ---- quarters (UTC calendar quarters, "YYYYQn") ----
// [start, end) in UTC seconds; throws std::invalid_argument on a malformed quarter.
std::pair<TimePoint, TimePoint> quarter_bounds(const std::string& q);
bool is_quarter(const std::string& q);
std::string previous_quarter(const std::string& q);
// Inclusive [first, last] panel bars whose time falls in the quarter; nullopt if none.
std::optional<std::pair<std::size_t, std::size_t>> quarter_bar_range(const Panel& panel, const std::string& q);

// ---- estimated flows ----
// Sum of the pipeline's slow-flux bar contributions over the bars of one quarter (fresh accumulator, half-life 1e9).
// The pipeline runs from bar 1 (warm-up for ADV, the return window and the active mask) and the exact BarFlux of
// each bar t in [first_bar, last_bar] (inclusive; bar 0 has no return and is never stepped) is added to a separate
// FluxAccumulator with half-life 1e9 and no row cap. Edges are indices into panel.tickers.
std::vector<FlowEdge> estimated_quarter_flows(const Panel&, const CoreParams&, std::size_t first_bar, std::size_t last_bar);
// The same for several inclusive ranges in one pipeline pass (one fresh accumulator per range).
std::vector<std::vector<FlowEdge>> estimated_range_flows(const Panel&, const CoreParams&,
                                                         const std::vector<std::pair<std::size_t, std::size_t>>& ranges);

// ---- prices and split ratios for observed_flows ----
// Lake bars are adjustment=all (split- and dividend-adjusted to today's basis); 13F shares are raw at each quarter
// end. Per ticker and quarter end, factor f = raw/adjusted = median(value_usd/shares over that quarter's 13F rows)
// / last adjusted close at or before the quarter end. Then
//   prices[i] = mean adjusted close over q's bars * f_q    (q's raw basis, the basis of q's 13F shares)
//   ratio[i]  = f_{q-1} / f_q                              (q-1 raw shares -> q raw shares)
// so d = (shares_q - ratio * shares_{q-1}) * prices = (adjusted shares_q - adjusted shares_{q-1}) * mean adjusted close.
// A ratio within 25% of 1 (dividend drift, price noise) is snapped to 1, and a split also needs the median holder
// share ratio shares_q/shares_{q-1} (managers holding the ticker in both quarters) within 10% of it; unknown factors
// give ratio 1, and an
// unknown f_q gives price NaN (observed_flows then falls back to value/shares, also q's raw basis).
struct QuarterPricing {
  std::vector<double> mean_close, factor_prev, factor_cur, prices, ratio;  // per panel ticker
  std::size_t splits = 0;  // tickers with ratio != 1
};
QuarterPricing quarter_pricing(const Panel& panel, const QuarterHoldings& prev, const QuarterHoldings& cur);

// Edges with both ends in `nodes` (ascending indices into a space of n_total), remapped to positions in `nodes`.
std::vector<FlowEdge> restrict_to(const std::vector<FlowEdge>& edges, const std::vector<std::uint32_t>& nodes,
                                  std::size_t n_total);

// ---- agreement ----
struct Agreement {
  double edge_spearman, top500_overlap, top2000_overlap, in_spearman, out_spearman, pi_spearman, top50_pi_overlap;
  double shuf_edge_spearman, shuf_pi_spearman;  // label-permutation baselines (seed 13)
};
// Duplicate (from, to) pairs are summed first. Edge Spearman over the union of each list's top-5000 edges (a pair
// missing from a list counts 0); top-k overlaps |A n B| / k; node in/out Spearman over nodes with any flow in either
// list; pi Spearman over all n nodes and top-50 overlap / 50. Baselines: estimated node labels permuted (mt19937,
// seed 13), edge and pi Spearman recomputed.
Agreement compare_flows(const std::vector<FlowEdge>& observed, const std::vector<FlowEdge>& estimated, std::size_t n);
// pi on an edge list: row-normalised out-shares, p = 0.15, dangling -> teleport (same solver as the engine).
std::vector<double> pi_of(const std::vector<FlowEdge>&, std::size_t n);

// ---- the --compare-13f run ----
struct Compare13fOptions {
  std::filesystem::path data;          // reads <data>/13f/, writes report.md and report.json there
  std::vector<std::string> quarters;   // empty = every quarter found
  CoreParams base = CoreParams::market_rank();
  std::string preset = "marketrank";
  ObservedParams observed;
};
// Lists the holdings_<q>.csv quarters found, compares each with a previous quarter and complete panel coverage,
// runs the calibration grid (lambda in {0, 0.5, 1} x pressure in {dollar, sqrt}, other params from `base`), writes
// the reports and returns the JSON report. Progress goes to `log`.
nlohmann::json run_compare_13f(const Panel& panel, const Compare13fOptions& opt, std::ostream& log);
// The markdown report rendered from run_compare_13f's JSON.
std::string compare_report_md(const nlohmann::json& report);

}  // namespace mr
