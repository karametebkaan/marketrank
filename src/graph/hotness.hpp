#pragma once
#include <span>
#include <string_view>
#include <vector>

namespace fx {

enum class HotRef { Uniform, Size, LongRun };  // (D)

HotRef parse_hot_ref(std::string_view s);
std::string_view to_string(HotRef r);

// h_i = pi_i / ref_i - 1 with ref normalized to sum 1 (spec 5 D).
std::vector<double> relative_hotness(std::span<const double> pi, std::span<const double> ref);

}  // namespace fx
