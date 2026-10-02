#pragma once
#include <cstddef>
#include <span>
#include <vector>

namespace fx {

// Rolling window of the last `window` returns per node. Pearson correlation is a dot product of
// the centered, unit-length vectors: corr(i, j) = u_i . u_j. Memory O(n * window).
class ReturnWindow {
 public:
  ReturnWindow(std::size_t n, std::size_t window);

  void push(std::span<const double> returns);  // non-finite -> 0
  const std::vector<double>& unit_vectors();   // n x window, row-major
  double correlation(std::size_t i, std::size_t j);
  std::size_t size() const { return n_; }
  std::size_t window() const { return w_; }
  std::size_t count() const { return count_; }

 private:
  std::size_t n_, w_;
  std::vector<double> ring_;  // [slot * n + i]
  std::size_t head_ = 0, count_ = 0;
  std::vector<double> unit_;  // [i * w + s]
  bool dirty_ = true;
};

}  // namespace fx
