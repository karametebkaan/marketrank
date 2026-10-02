#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "graph/return_window.hpp"

using namespace mr;

namespace {
double pearson(const std::vector<std::vector<double>>& hist, std::size_t i, std::size_t j,
               std::size_t w) {
  const std::size_t m = std::min(hist.size(), w), start = hist.size() - m;
  double mi = 0, mj = 0;
  for (std::size_t t = start; t < hist.size(); ++t) {
    mi += hist[t][i];
    mj += hist[t][j];
  }
  mi /= static_cast<double>(m);
  mj /= static_cast<double>(m);
  double sij = 0, sii = 0, sjj = 0;
  for (std::size_t t = start; t < hist.size(); ++t) {
    const double a = hist[t][i] - mi, b = hist[t][j] - mj;
    sij += a * b;
    sii += a * a;
    sjj += b * b;
  }
  return sij / std::sqrt(sii * sjj);
}
}  // namespace

TEST_CASE("ReturnWindow correlation matches brute-force Pearson across wraps") {
  const std::size_t n = 4, W = 10;
  ReturnWindow rw(n, W);
  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0.0, 0.01);
  std::vector<std::vector<double>> hist;
  for (int step = 1; step <= 35; ++step) {
    std::vector<double> r(n);
    for (auto& x : r) x = noise(rng);
    r[3] = 0.7 * r[0] + noise(rng);
    rw.push(r);
    hist.push_back(r);
    if (step == 7 || step == 10 || step == 11 || step == 23 || step == 35) {
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          INFO("step " << step << " i " << i << " j " << j);
          const double expected = i == j ? 1.0 : pearson(hist, i, j, W);
          CHECK(rw.correlation(i, j) == doctest::Approx(expected).epsilon(1e-9));
        }
      }
    }
  }
  CHECK(rw.count() == W);
}

TEST_CASE("ReturnWindow treats NaN as 0 and constant series as uncorrelated") {
  ReturnWindow rw(3, 5);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  rw.push(std::vector<double>{0.01, 0.02, nan});
  rw.push(std::vector<double>{-0.01, 0.02, nan});
  rw.push(std::vector<double>{0.02, 0.02, nan});
  CHECK(rw.count() == 3);
  CHECK(rw.correlation(0, 1) == 0.0);
  CHECK(rw.correlation(0, 2) == 0.0);
  CHECK(rw.correlation(1, 1) == 0.0);
  CHECK(rw.correlation(0, 0) == doctest::Approx(1.0));
}

TEST_CASE("ReturnWindow needs window >= 2 and two samples") {
  CHECK_THROWS_AS(ReturnWindow(2, 1), std::invalid_argument);
  ReturnWindow rw(2, 4);
  rw.push(std::vector<double>{0.01, -0.01});
  CHECK(rw.correlation(0, 1) == 0.0);
}
