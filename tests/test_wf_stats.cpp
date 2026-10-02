#include <doctest/doctest.h>
#include <cmath>
#include <vector>
#include "walkforward/stats.hpp"
using namespace mr;

TEST_CASE("spearman handles ties, NaN and degenerate input") {
  std::vector<double> a{1, 2, 3, 4, 5}, b{10, 20, 30, 40, 50};
  CHECK(spearman(a, b) == doctest::Approx(1.0));
  std::vector<double> c{5, 4, 3, 2, 1};
  CHECK(spearman(a, c) == doctest::Approx(-1.0));
  std::vector<double> t{1, 1, 2, 2, 3}, u{1, 2, 3, 4, 5};
  CHECK(spearman(t, u) == doctest::Approx(0.9486833).epsilon(1e-6));  // average ranks
  std::vector<double> n{1, NAN, 3, 4, 5};
  CHECK(spearman(n, b) == doctest::Approx(1.0));
  std::vector<double> k{2, 2, 2, 2, 2};
  CHECK(std::isnan(spearman(k, b)));
}

TEST_CASE("mean_t and normal functions") {
  std::vector<double> x{1, 2, 3, NAN};
  const MeanT m = mean_t(x);
  CHECK(m.n == 3);
  CHECK(m.mean == doctest::Approx(2.0));
  CHECK(m.t == doctest::Approx(2.0 / (1.0 / std::sqrt(3.0))));
  CHECK(norm_cdf(0) == doctest::Approx(0.5));
  CHECK(norm_cdf(1.96) == doctest::Approx(0.9750021).epsilon(1e-6));
  CHECK(norm_inv(0.975) == doctest::Approx(1.959964).epsilon(1e-6));
  CHECK(norm_inv(0.99) == doctest::Approx(2.326348).epsilon(1e-6));  // upper tail (p > 0.97575 branch)
  CHECK(norm_inv(0.999) == doctest::Approx(3.090232).epsilon(1e-6));
  CHECK(norm_inv(norm_cdf(-2.3)) == doctest::Approx(-2.3).epsilon(1e-8));
}

TEST_CASE("block bootstrap CI covers the mean and is deterministic") {
  std::vector<double> x(500);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = 0.001 + 0.01 * std::sin(0.7 * static_cast<double>(i));
  const CI a = block_bootstrap_mean_ci(x, 21, 2000, 7, 0.95), b = block_bootstrap_mean_ci(x, 21, 2000, 7, 0.95);
  CHECK(a.lo == b.lo);
  CHECK(a.hi == b.hi);
  double mean = 0;
  for (double v : x) mean += v;
  mean /= static_cast<double>(x.size());
  CHECK(a.lo < mean);
  CHECK(a.hi > mean);
}
