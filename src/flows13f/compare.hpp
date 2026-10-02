#pragma once
// M3c: estimated (bar-flux) vs observed (13F) quarter flows, MarketRank agreement and the calibration grid.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
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
std::string next_quarter(const std::string& q);
// Inclusive [first, last] panel bars whose time falls in the quarter; nullopt if none.
std::optional<std::pair<std::size_t, std::size_t>> quarter_bar_range(const Panel& panel, const std::string& q);
// Temporal placebo for quarter q: q-4 if available, else q+1, else q-1; "" if none is available.
std::string placebo_quarter(const std::string& q, const std::function<bool(const std::string&)>& available);

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
// / last adjusted close within that quarter. Then
//   prices[i] = mean adjusted close over q's bars * f_q    (q's raw basis, the basis of q's 13F shares)
//   ratio[i]  = f_{q-1} / f_q                              (q-1 raw shares -> q raw shares)
// so d = (shares_q - ratio * shares_{q-1}) * prices = (adjusted shares_q - adjusted shares_{q-1}) * mean adjusted close.
// A price-factor ratio r with |log r| < log(1.08) (dividend drift ~1-2%/quarter, price noise) is snapped to 1. A
// larger r is a split only if the median holder share ratio shares_q/shares_{q-1} (managers holding the ticker in
// both quarters) is within 10% of r and closer to r than to 1 (log scale); otherwise ratio stays 1 and the ticker is listed in `unconfirmed`. Unknown
// factors give ratio 1; an unknown f_q falls back to f_{q-1} (full exits), else price NaN (observed_flows then uses
// value/shares, also q's raw basis).
struct SplitCandidate {
  std::size_t node;    // panel ticker index
  double price_ratio;  // f_{q-1} / f_q
  double share_ratio;  // median holder shares_q / shares_{q-1} (NaN: no holder in both quarters)
  double value_usd;    // q's 13F value of the ticker
};
struct QuarterPricing {
  std::vector<double> mean_close, factor_prev, factor_cur, prices, ratio;  // per panel ticker
  std::size_t splits = 0;                  // tickers with ratio != 1
  std::vector<SplitCandidate> unconfirmed;  // by value, descending
};
QuarterPricing quarter_pricing(const Panel& panel, const QuarterHoldings& prev, const QuarterHoldings& cur);

// Edges with both ends in `nodes` (ascending indices into a space of n_total), remapped to positions in `nodes`.
std::vector<FlowEdge> restrict_to(const std::vector<FlowEdge>& edges, const std::vector<std::uint32_t>& nodes,
                                  std::size_t n_total);

// ---- agreement ----
// Metrics of an estimated matrix E against the observed matrix O on n nodes (duplicate pairs summed, self-pairs and
// non-positive weights dropped):
//   obs_topk_spearman  Spearman over O's top-5000 edges of (O, E looked up; 0 if missing)          [primary]
//   full_spearman      Spearman over all n(n-1) ordered pairs (missing = 0)                         [primary]
//   row_cosine         cosine of the row-normalised share matrices (each row of O and E sums to 1)  [primary]
//   top500/2000_overlap |A n B| / min(k, |A|, |B|) of the top-k edge sets (descriptive)
//   union_spearman     the earlier union-of-top-5000 Spearman (biased; continuity only)
//   in/out_spearman    node in/out dollars over nodes with any flow in O or E
//   pi_spearman        pi (pi_of) over all n nodes; pi_spearman_flow over nodes with any flow; top50_pi_overlap
struct FlowMetrics {
  double obs_topk_spearman, full_spearman, row_cosine;
  double top500_overlap, top2000_overlap, union_spearman;
  double in_spearman, out_spearman, pi_spearman, pi_spearman_flow, top50_pi_overlap;
};
// Field names and members, in report order.
inline constexpr std::pair<const char*, double FlowMetrics::*> kFlowMetricFields[] = {
    {"obs_topk_spearman", &FlowMetrics::obs_topk_spearman}, {"full_spearman", &FlowMetrics::full_spearman},
    {"row_cosine", &FlowMetrics::row_cosine},               {"top500_overlap", &FlowMetrics::top500_overlap},
    {"top2000_overlap", &FlowMetrics::top2000_overlap},     {"union_spearman", &FlowMetrics::union_spearman},
    {"in_spearman", &FlowMetrics::in_spearman},             {"out_spearman", &FlowMetrics::out_spearman},
    {"pi_spearman", &FlowMetrics::pi_spearman},             {"pi_spearman_flow", &FlowMetrics::pi_spearman_flow},
    {"top50_pi_overlap", &FlowMetrics::top50_pi_overlap}};

struct NullStats {
  FlowMetrics mean{}, sd{};
  std::size_t draws = 0;
};
struct Agreement {
  FlowMetrics est{};      // O vs E
  FlowMetrics gravity{};  // O vs G(E) = out_E in_E^T / total_E (E's own margins): structure beyond the marginals
  NullStats perm;         // O vs E with node labels permuted, seeds 1..R (mt19937 Fisher-Yates)
  bool has_placebo = false;
  FlowMetrics placebo{};  // O vs another quarter's estimate (temporal placebo), when given
  double top500_expected = 0, top2000_expected = 0;  // hypergeometric max(|A|,|B|) / (n(n-1)) for the overlaps
};
Agreement compare_flows(const std::vector<FlowEdge>& observed, const std::vector<FlowEdge>& estimated, std::size_t n,
                        std::size_t perms = 100, const std::vector<FlowEdge>* placebo = nullptr);
// Gravity model of an edge list: out_i * in_j / total for i != j (positive margins only).
std::vector<FlowEdge> gravity_null(const std::vector<FlowEdge>& edges, std::size_t n);
// pi on an edge list: row-normalised out-shares, p = 0.15, dangling -> teleport (same solver as the engine).
std::vector<double> pi_of(const std::vector<FlowEdge>&, std::size_t n);

// ---- the --compare-13f run ----
struct Compare13fOptions {
  std::filesystem::path data;          // reads <data>/13f/, writes report.md and report.json there
  std::vector<std::string> quarters;   // empty = every quarter found
  CoreParams base = CoreParams::market_rank();
  std::string preset = "marketrank";
  ObservedParams observed;
  std::size_t perms = 100;   // permutation-null draws per quarter and config
  // recorded in report.json
  int lookback_days = 0;
  std::string timeframe;
  std::string git_sha;
};
// Lists the holdings_<q>.csv quarters found and compares each that has a previous quarter, complete panel coverage
// and a warm-up of at least max(corr_window, adv_window, vol_window) bars before its first bar. Runs the calibration
// grid (lambda in {0, 0.5, 1} x pressure in {dollar, sqrt}, other params from `base`) with gravity, temporal-placebo
// and permutation nulls; the best config is the one with the highest mean lift over the placebo. Writes the reports
// and returns the JSON report. Progress goes to `log`.
nlohmann::json run_compare_13f(const Panel& panel, const Compare13fOptions& opt, std::ostream& log);
// The markdown report rendered from run_compare_13f's JSON.
std::string compare_report_md(const nlohmann::json& report);

}  // namespace mr
