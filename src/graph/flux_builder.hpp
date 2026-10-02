#pragma once
#include <cstddef>
#include <span>
#include <vector>

namespace fx {

struct FluxParams {
  double lambda = 1.0;
  double halflife_slow = 20;
  double halflife_fast = 3;
  int corr_window = 60;
};

void bar_flux(std::span<const double> pressure, std::span<const double> corr, double lambda,
              std::vector<double>& out);

class FluxBuilder {
 public:
  FluxBuilder(std::size_t n, FluxParams params);

  void step(std::span<const double> returns, std::span<const double> dollar_volume);
  const std::vector<double>& flux_slow() const { return slow_; }
  const std::vector<double>& flux_fast() const { return fast_; }
  double correlation(std::size_t i, std::size_t j) const { return corr_[i * n_ + j]; }
  std::size_t size() const { return n_; }

 private:
  void push_correlation_sample(const std::vector<double>& x);
  void recompute_sums();
  void update_correlation();

  std::size_t n_;
  FluxParams params_;
  std::vector<double> slow_, fast_, bar_, corr_;
  // rolling window of returns: ring buffer [window][n]
  std::vector<double> ring_;
  std::size_t head_ = 0, count_ = 0;
  std::vector<double> sx_, sxx_, sxy_;
};

}  // namespace fx
