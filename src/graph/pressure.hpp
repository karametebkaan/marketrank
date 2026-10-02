#pragma once
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace mr {

enum class PressureMode { Dollar, Sqrt, Relative };

PressureMode parse_pressure_mode(std::string_view s);
std::string_view to_string(PressureMode m);

// Spec 5 (A). Tracks per-node trailing volumes so relative pressure uses ADV from previous bars.
class PressureModel {
 public:
  // max_volume_ratio > 0 caps V/ADV in relative mode; 0 = uncapped.
  // vol_window >= 1 turns on volatility scaling: the return is divided by max(sigma, 1e-4), where
  // sigma is the sample std of the node's previous (up to vol_window) returns, excluding the
  // current bar. Fewer than 5 previous returns gives pressure 0. 0 = off.
  PressureModel(std::size_t n, PressureMode mode, std::size_t adv_window = 20,
                double max_volume_ratio = 0.0, std::size_t vol_window = 0);

  std::vector<double> step(std::span<const double> returns, std::span<const double> volume,
                           std::span<const double> vwap);
  std::vector<double> median_dollar_volume() const;  // 0 where no history
  // Per-node multiplier applied to returns in the last step(): 1 when scaling is off, else
  // 1 / max(sigma, floor) (0 with fewer than 5 previous returns).
  const std::vector<double>& return_scale() const { return scale_; }

 private:
  static double median_of(const double* first, std::size_t count);
  std::size_t n_, w_, vw_;
  PressureMode mode_;
  double max_ratio_;
  std::vector<double> vol_, dollar_;  // [i * w + slot]
  std::vector<std::size_t> count_, head_;
  std::vector<double> ret_;  // [i * vw + slot] trailing finite returns
  std::vector<std::size_t> ret_count_, ret_head_;
  std::vector<double> scale_;
};

}  // namespace mr
