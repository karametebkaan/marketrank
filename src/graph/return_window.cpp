#include "graph/return_window.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fx {

ReturnWindow::ReturnWindow(std::size_t n, std::size_t window)
    : n_(n), w_(window), ring_(n * window, 0.0), unit_(n * window, 0.0) {
  if (window < 2) throw std::invalid_argument("ReturnWindow: window must be >= 2");
}

void ReturnWindow::push(std::span<const double> returns) {
  double* slot = &ring_[head_ * n_];
  for (std::size_t i = 0; i < n_; ++i) slot[i] = std::isfinite(returns[i]) ? returns[i] : 0.0;
  head_ = (head_ + 1) % w_;
  count_ = std::min(count_ + 1, w_);
  dirty_ = true;
}

const std::vector<double>& ReturnWindow::unit_vectors() {
  if (!dirty_) return unit_;
  const std::size_t c = count_;
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < n_; ++i) {
    double* u = &unit_[i * w_];
    std::fill(u, u + w_, 0.0);
    if (c < 2) continue;
    double mean = 0;
    for (std::size_t s = 0; s < c; ++s) mean += ring_[s * n_ + i];
    mean /= static_cast<double>(c);
    double norm2 = 0;
    for (std::size_t s = 0; s < c; ++s) {
      u[s] = ring_[s * n_ + i] - mean;
      norm2 += u[s] * u[s];
    }
    if (norm2 < 1e-24) {
      std::fill(u, u + w_, 0.0);
      continue;
    }
    const double inv = 1.0 / std::sqrt(norm2);
    for (std::size_t s = 0; s < c; ++s) u[s] *= inv;
  }
  dirty_ = false;
  return unit_;
}

double ReturnWindow::correlation(std::size_t i, std::size_t j) {
  const auto& u = unit_vectors();
  double d = 0;
  for (std::size_t s = 0; s < w_; ++s) d += u[i * w_ + s] * u[j * w_ + s];
  return d;
}

}  // namespace fx
