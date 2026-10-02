#include "graph/pressure.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mr {

PressureMode parse_pressure_mode(std::string_view s) {
  if (s == "dollar") return PressureMode::Dollar;
  if (s == "sqrt") return PressureMode::Sqrt;
  if (s == "relative") return PressureMode::Relative;
  throw std::invalid_argument("unknown pressure mode: " + std::string(s));
}

std::string_view to_string(PressureMode m) {
  switch (m) {
    case PressureMode::Dollar: return "dollar";
    case PressureMode::Sqrt: return "sqrt";
    case PressureMode::Relative: return "relative";
  }
  return "?";
}

PressureModel::PressureModel(std::size_t n, PressureMode mode, std::size_t adv_window,
                             double max_volume_ratio, std::size_t vol_window)
    : n_(n),
      w_(adv_window),
      vw_(vol_window),
      mode_(mode),
      max_ratio_(max_volume_ratio),
      vol_(n * adv_window, 0.0),
      dollar_(n * adv_window, 0.0),
      count_(n, 0),
      head_(n, 0),
      ret_(n * vol_window, 0.0),
      ret_count_(n, 0),
      ret_head_(n, 0),
      scale_(n, 1.0) {
  if (adv_window == 0) throw std::invalid_argument("PressureModel: adv_window must be >= 1");
}

double PressureModel::median_of(const double* first, std::size_t count) {
  std::vector<double> v(first, first + count);
  const std::size_t mid = count / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
  if (count % 2 == 1) return v[mid];
  const double upper = v[mid];
  const double lower = *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid));
  return 0.5 * (lower + upper);
}

std::vector<double> PressureModel::step(std::span<const double> returns,
                                        std::span<const double> volume,
                                        std::span<const double> vwap) {
  std::vector<double> p(n_, 0.0);
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < n_; ++i) {
    double r = returns[i];
    const double r_raw = r;
    if (vw_ > 0) {
      const std::size_t cnt = ret_count_[i];
      if (cnt < 5) {
        scale_[i] = 0.0;
      } else {
        const double* x = &ret_[i * vw_];
        double mean = 0;
        for (std::size_t k = 0; k < cnt; ++k) mean += x[k];
        mean /= static_cast<double>(cnt);
        double ss = 0;
        for (std::size_t k = 0; k < cnt; ++k) ss += (x[k] - mean) * (x[k] - mean);
        const double sigma = std::sqrt(ss / static_cast<double>(cnt - 1));
        scale_[i] = 1.0 / std::max(sigma, 1e-4);
      }
      r *= scale_[i];  // NaN stays NaN; 0 * finite = 0 gives pressure 0
    }
    const double v = volume[i], vw = vwap[i];
    const bool have_bar = std::isfinite(v) && std::isfinite(vw) && v >= 0 && vw > 0;
    if (std::isfinite(r) && have_bar) {
      switch (mode_) {
        case PressureMode::Dollar: p[i] = r * v * vw; break;
        case PressureMode::Sqrt: p[i] = r * std::sqrt(v * vw); break;
        case PressureMode::Relative:
          if (count_[i] > 0) {
            const double adv = median_of(&vol_[i * w_], count_[i]);
            if (adv > 0) {
              double ratio = v / adv;
              if (max_ratio_ > 0) ratio = std::min(ratio, max_ratio_);
              p[i] = r * ratio;
            }
          }
          break;
      }
    }
    if (have_bar) {
      vol_[i * w_ + head_[i]] = v;
      dollar_[i * w_ + head_[i]] = v * vw;
      head_[i] = (head_[i] + 1) % w_;
      count_[i] = std::min(count_[i] + 1, w_);
    }
    if (vw_ > 0 && std::isfinite(r_raw)) {
      ret_[i * vw_ + ret_head_[i]] = r_raw;
      ret_head_[i] = (ret_head_[i] + 1) % vw_;
      ret_count_[i] = std::min(ret_count_[i] + 1, vw_);
    }
  }
  return p;
}

std::vector<double> PressureModel::median_dollar_volume() const {
  std::vector<double> m(n_, 0.0);
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < n_; ++i)
    if (count_[i] > 0) m[i] = median_of(&dollar_[i * w_], count_[i]);
  return m;
}

}  // namespace mr
