#include "graph/flux_builder.hpp"

#include <algorithm>
#include <cmath>

namespace fx {

void bar_flux(std::span<const double> pressure, std::span<const double> corr, double lambda,
              std::vector<double>& out) {
  const std::size_t n = pressure.size();
  out.assign(n * n, 0.0);
  const bool affinity = !corr.empty() && lambda != 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!(pressure[i] < 0)) continue;
    double denom = 0;
    for (std::size_t k = 0; k < n; ++k) {
      if (!(pressure[k] > 0)) continue;
      const double a = affinity ? 1.0 + lambda * std::max(0.0, corr[i * n + k]) : 1.0;
      denom += pressure[k] * a;
    }
    if (denom <= 0) continue;
    const double outflow = -pressure[i];
    for (std::size_t j = 0; j < n; ++j) {
      if (!(pressure[j] > 0)) continue;
      const double a = affinity ? 1.0 + lambda * std::max(0.0, corr[i * n + j]) : 1.0;
      out[i * n + j] = outflow * pressure[j] * a / denom;
    }
  }
}

FluxBuilder::FluxBuilder(std::size_t n, FluxParams params)
    : n_(n),
      params_(params),
      slow_(n * n, 0.0),
      fast_(n * n, 0.0),
      corr_(n * n, 0.0),
      ring_(static_cast<std::size_t>(std::max(params.corr_window, 2)) * n, 0.0),
      sx_(n, 0.0),
      sxx_(n, 0.0),
      sxy_(n * n, 0.0) {
  for (std::size_t i = 0; i < n; ++i) corr_[i * n + i] = 1.0;
}

void FluxBuilder::push_correlation_sample(const std::vector<double>& x) {
  const std::size_t window = ring_.size() / n_;
  double* slot = &ring_[head_ * n_];
  if (count_ == window) {  // evict oldest
    for (std::size_t i = 0; i < n_; ++i) {
      sx_[i] -= slot[i];
      sxx_[i] -= slot[i] * slot[i];
      for (std::size_t j = 0; j < n_; ++j) sxy_[i * n_ + j] -= slot[i] * slot[j];
    }
  } else {
    ++count_;
  }
  for (std::size_t i = 0; i < n_; ++i) {
    slot[i] = x[i];
    sx_[i] += x[i];
    sxx_[i] += x[i] * x[i];
    for (std::size_t j = 0; j < n_; ++j) sxy_[i * n_ + j] += x[i] * x[j];
  }
  head_ = (head_ + 1) % window;
  if (head_ == 0) recompute_sums();  // bound floating-point drift once per window
}

void FluxBuilder::recompute_sums() {
  std::fill(sx_.begin(), sx_.end(), 0.0);
  std::fill(sxx_.begin(), sxx_.end(), 0.0);
  std::fill(sxy_.begin(), sxy_.end(), 0.0);
  for (std::size_t w = 0; w < count_; ++w) {
    const double* row = &ring_[w * n_];
    for (std::size_t i = 0; i < n_; ++i) {
      sx_[i] += row[i];
      sxx_[i] += row[i] * row[i];
      for (std::size_t j = 0; j < n_; ++j) sxy_[i * n_ + j] += row[i] * row[j];
    }
  }
}

void FluxBuilder::update_correlation() {
  const double c = static_cast<double>(count_);
  for (std::size_t i = 0; i < n_; ++i) {
    const double vi = c * sxx_[i] - sx_[i] * sx_[i];
    for (std::size_t j = 0; j < n_; ++j) {
      if (i == j) {
        corr_[i * n_ + j] = 1.0;
        continue;
      }
      const double vj = c * sxx_[j] - sx_[j] * sx_[j];
      const double denom = std::sqrt(std::max(0.0, vi) * std::max(0.0, vj));
      corr_[i * n_ + j] = denom > 1e-18 ? (c * sxy_[i * n_ + j] - sx_[i] * sx_[j]) / denom : 0.0;
    }
  }
}

void FluxBuilder::step(std::span<const double> returns, std::span<const double> dollar_volume) {
  std::vector<double> pressure(n_, 0.0), sample(n_, 0.0);
  for (std::size_t i = 0; i < n_; ++i) {
    const double r = returns[i], dv = dollar_volume[i];
    if (std::isfinite(r)) sample[i] = r;
    if (std::isfinite(r) && std::isfinite(dv)) pressure[i] = r * dv;
  }
  const bool use_corr = params_.lambda != 0.0;
  if (use_corr) {
    push_correlation_sample(sample);
    update_correlation();
  }
  bar_flux(pressure, use_corr ? std::span<const double>(corr_) : std::span<const double>(),
           params_.lambda, bar_);
  const double ds = std::exp2(-1.0 / params_.halflife_slow);
  const double df = std::exp2(-1.0 / params_.halflife_fast);
  for (std::size_t k = 0; k < bar_.size(); ++k) {
    slow_[k] = ds * slow_[k] + bar_[k];
    fast_[k] = df * fast_[k] + bar_[k];
  }
}

}  // namespace fx
