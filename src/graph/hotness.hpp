#pragma once
#include <span>
#include <string_view>
#include <vector>

namespace mr {

enum class HotRef { Uniform, Size, LongRun, NetFlow };  // (D)

HotRef parse_hot_ref(std::string_view s);
std::string_view to_string(HotRef r);

// h_i = pi_i / ref_i - 1 with ref normalized to sum 1 (spec 5 D).
std::vector<double> relative_hotness(std::span<const double> pi, std::span<const double> ref);

// Size share of each node: ref_i / sum of the valid refs (finite and > 0); NaN for an invalid ref. This is the
// normalization relative_hotness applies to its reference (HotRef::Size: the trailing median dollar volume),
// without its 1e-12 floor for invalid refs.
std::vector<double> size_shares(std::span<const double> ref);

// h_i = (in_i - out_i) / (in_i + out_i + kappa), kappa = median of (in + out) over the given
// (active) nodes; bounded in (-1, 1). Non-finite inputs count as 0; a 0/0 node gets h = 0.
std::vector<double> net_flow_hotness(std::span<const double> in, std::span<const double> out);

}  // namespace mr
