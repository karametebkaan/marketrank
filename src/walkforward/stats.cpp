#include "walkforward/stats.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
namespace mr {
namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

MeanT mean_t(std::span<const double> x) {
  MeanT r;
  double sum = 0;
  for (double v : x)
    if (!std::isnan(v)) { sum += v; ++r.n; }
  if (r.n == 0) { r.mean = kNaN; r.t = kNaN; return r; }
  const double n = static_cast<double>(r.n);
  r.mean = sum / n;
  if (r.n < 2) { r.t = kNaN; return r; }
  double ss = 0;
  for (double v : x)
    if (!std::isnan(v)) ss += (v - r.mean) * (v - r.mean);
  const double sd = std::sqrt(ss / (n - 1));
  r.t = sd > 0 ? r.mean / (sd / std::sqrt(n)) : kNaN;
  return r;
}

double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

double norm_inv(double p) {
  static constexpr double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                                 1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
  static constexpr double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                                 6.680131188771972e+01,  -1.328068155288572e+01};
  static constexpr double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                                 -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
  static constexpr double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                                 3.754408661907416e+00};
  constexpr double plow = 0.02425, phigh = 1 - plow;
  if (!(p > 0 && p < 1)) {
    if (p == 0) return -std::numeric_limits<double>::infinity();
    if (p == 1) return std::numeric_limits<double>::infinity();
    return kNaN;
  }
  if (p < plow) {
    const double q = std::sqrt(-2 * std::log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    const double q = std::sqrt(-2 * std::log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const double q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

CI block_bootstrap_mean_ci(std::span<const double> x, std::size_t block, std::size_t reps, std::uint64_t seed,
                           double level) {
  const std::size_t n = x.size();
  if (n == 0 || reps == 0) return {kNaN, kNaN};
  if (block == 0) block = 1;
  std::mt19937_64 rng(seed);
  const std::size_t nblocks = (n + block - 1) / block;
  std::vector<double> means(reps);
  for (std::size_t r = 0; r < reps; ++r) {
    double sum = 0;
    std::size_t cnt = 0;
    for (std::size_t bi = 0; bi < nblocks; ++bi) {
      const std::size_t s = static_cast<std::size_t>(rng() % n);
      for (std::size_t k = 0; k < block && cnt < n; ++k, ++cnt) sum += x[(s + k) % n];
    }
    means[r] = sum / static_cast<double>(n);
  }
  std::sort(means.begin(), means.end());
  const double alpha = (1 - level) / 2;
  const auto at = [&](double q) { return means[static_cast<std::size_t>(std::floor(q * static_cast<double>(reps - 1)))]; };
  return {at(alpha), at(1 - alpha)};
}
}  // namespace mr
