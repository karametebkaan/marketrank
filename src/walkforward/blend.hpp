#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "walkforward/signals.hpp"

namespace mr {
// All counts are REBALANCE PERIODS (weekly by default: 156 = 3 years, 104 = 2 years, 52 = 1 year), not months.
// The field names keep the "_months" suffix for linkage with later tasks.
struct BlendParams { std::size_t train_months = 156, embargo = 1, gate_months = 104, gate_min = 52; double gate_t = 2.0; };
// Per rebalance period m: z[s] = cross-sectional z-scores (size N, NaN = not eligible), label = forward return to the
// next rebalance execution (size N), both from the walk-forward pass.
struct MonthRecord { std::array<std::vector<double>, kSignals> z; std::vector<double> label; };
struct BlendStep { std::array<double, kSignals> w{}; bool gate_open = false; double oos_ic = NAN; std::vector<double> score; };
// A NaN/non-finite z for a weighted signal contributes 0 (imputes the cross-sectional mean) without renormalising
// the weights, so partially covered names are shrunk toward 0.
// Causal: step m reads labels of periods <= m - embargo - 1 only (label(m) is used only for its own oos_ic,
// which later steps consume after the embargo). Serial, deterministic. Throws std::invalid_argument if embargo < 1.
std::vector<BlendStep> run_blend(const std::vector<MonthRecord>& months, const BlendParams& p);
}  // namespace mr
