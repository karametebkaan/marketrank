#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "market/panel.hpp"

namespace mr {

struct BaseWeight {
  std::string ticker;
  double weight;
};

struct BacktestParams {
  std::vector<BaseWeight> base;  // the core mix (sums to <= 1; remainder is cash)
  double tilt = 0.20;            // share of value moved toward the signal
  std::size_t k = 10;            // names in the tilt
  double max_name_tilt = 0.10;   // cap on one name's tilt weight
  double cost_bps = 10;          // per side, on traded notional
  double max_turnover = 0.5;     // per rebalance, sum |delta w| / 2; larger moves are scaled down
};

// One decision per rebalance date: the score vector (NaN = not a candidate) and the eligibility mask;
// an empty score vector means "hold the base, no tilt" (gate closed / no signal). An empty eligibility
// mask means every stock is eligible.
struct Decision {
  std::size_t date;
  std::vector<double> score;
  std::vector<bool> eligible;
};

enum class BenchKind { None, BuyHoldBase, RebalancedBase, Single };  // Single = 100% in one ticker

struct EquityCurve {
  std::vector<TimePoint> t;             // every bar from the first execution day on
  std::vector<double> value;            // marked at the close; starting capital is 1.0
  double costs = 0;                     // total cost paid (as fraction of starting value)
  double turnover = 0;                  // sum over rebalances (initial funding excluded)
  std::vector<std::string> trades_csv;  // "t,ticker,delta_weight,price"
};

// Target weight per stock (size N): (1 - tilt) * base + tilt * tiltpart, where the tilt part gives
// min(1/k, max_name_tilt/tilt) to each of the top-k finite-score eligible names (ties -> lower index).
// Unallocated tilt and any base remainder are cash. Empty score -> base alone.
// Throws std::invalid_argument on an unknown base ticker or a score/eligible size mismatch.
std::vector<double> target_weights(const Panel&, const BacktestParams&, const Decision&);

// Simulates the portfolio from starting capital 1.0 (all cash). Bars are processed in order; at each
// bar the open happens before the close. Decisions must have strictly increasing dates with date+1 < T.
//
// Execution of a decision at the open of date+1:
//  1. V = cash + sum shares*px, px = open if finite else the last finite price seen (open or close).
//  2. delta_i = w_i*V - shares_i*px_i. Stocks with no finite open that day get delta 0 (keep shares).
//  3. Turnover cap: turnover = sum|delta_i| / (2V); every delta is scaled by min(1, max_turnover/turnover).
//     The first execution (initial funding from all cash) is exempt from the cap.
//  4. Funding: if buys exceed cash + sells (possible when an untradable stock could not be sold), the
//     buys are scaled down so cash never goes below zero from trading.
//  5. Cost = cost_bps*1e-4 * sum|delta_i| on the final deltas (pre-trade value convention, so moving
//     100% of V from A to B costs exactly 2V*cost_bps*1e-4 with turnover 1.0). The cost is funded by
//     scaling every tradable post-trade position and the cash by f = (W - cost)/W, where W is the
//     post-trade value of the tradable positions plus cash. Post-trade value is therefore exactly
//     V - cost, and cash stays 0 when the target is fully invested. (The extra notional implied by
//     this scaling is not itself charged: an O(cost_bps^2) approximation.) The simulator checks
//     sum shares*px + cash == V - cost to 1e-12 relative and throws std::logic_error otherwise.
//  6. Shares change at the open price. curve.turnover adds the final sum|delta_i|/(2V) of every
//     execution but the first; curve.costs adds every cost, including the initial funding's.
//
// Marking: every close from the first execution bar on, value = cash + sum shares*close (forward-filled).
//
// Benchmarks (use the decisions' dates only; scores are ignored; same execution rules and costs):
//  BuyHoldBase    - trade to the base weights at the first decision only, then hold.
//  RebalancedBase - trade to the base weights at every decision date.
//  Single         - 100% in single_ticker at the first decision, then hold (throws if unknown).
// No decisions -> empty curve.
EquityCurve simulate(const Panel&, const BacktestParams&, const std::vector<Decision>&, BenchKind bench = BenchKind::None,
                     const std::string& single_ticker = "");

}  // namespace mr
