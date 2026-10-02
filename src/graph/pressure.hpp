#pragma once
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace fx {

enum class PressureMode { Dollar, Sqrt, Relative };

PressureMode parse_pressure_mode(std::string_view s);
std::string_view to_string(PressureMode m);

// Spec 5 (A). Tracks per-node trailing volumes so relative pressure uses ADV from previous bars.
class PressureModel {
 public:
  // max_volume_ratio > 0 caps V/ADV in relative mode; 0 = uncapped.
  PressureModel(std::size_t n, PressureMode mode, std::size_t adv_window = 20,
                double max_volume_ratio = 0.0);

  std::vector<double> step(std::span<const double> returns, std::span<const double> volume,
                           std::span<const double> vwap);
  std::vector<double> median_dollar_volume() const;  // 0 where no history

 private:
  static double median_of(const double* first, std::size_t count);
  std::size_t n_, w_;
  PressureMode mode_;
  double max_ratio_;
  std::vector<double> vol_, dollar_;  // [i * w + slot]
  std::vector<std::size_t> count_, head_;
};

}  // namespace fx
