#pragma once
#include <array>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include "pipeline/core_pipeline.hpp"

namespace mr {

enum class Signal { Score, Pulse1, Pulse5, Pulse20, PiRelSize, NegHotness, InflowMom5, Forecast };
constexpr std::size_t kSignals = 8;

std::string_view to_string(Signal s);
Signal parse_signal(std::string_view name);  // throws std::invalid_argument on an unknown name

// Raw per-node signals from a stream of frames (call update() once per step, in order). NaN where undefined.
class SignalTracker {
 public:
  explicit SignalTracker(std::size_t n);
  std::array<std::vector<double>, kSignals> update(const Frame& f);

 private:
  static constexpr std::size_t kRing = 21;
  std::size_t n_;
  std::size_t head_ = kRing - 1;  // slot of the latest frame; the first update advances it to 0
  std::array<std::vector<double>, kRing> logs_;    // log(pi N), NaN where inactive
  std::array<std::vector<double>, kRing> inflow_;  // raw inflow
};

// Cross-sectional z-score over the finite entries inside the mask: (x - mean) / sd (population sd), winsorized
// at +-3. Outside the mask or non-finite: NaN. sd == 0: all (finite masked) values 0.
// Throws std::invalid_argument if mask.size() != x.size().
std::vector<double> zscore(std::span<const double> x, const std::vector<bool>& mask);

}  // namespace mr
