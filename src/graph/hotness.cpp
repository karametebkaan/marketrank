#include "graph/hotness.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace mr {

HotRef parse_hot_ref(std::string_view s) {
  if (s == "uniform") return HotRef::Uniform;
  if (s == "size") return HotRef::Size;
  if (s == "longrun") return HotRef::LongRun;
  if (s == "netflow") return HotRef::NetFlow;
  throw std::invalid_argument("unknown hotness reference: " + std::string(s));
}

std::string_view to_string(HotRef r) {
  switch (r) {
    case HotRef::Uniform: return "uniform";
    case HotRef::Size: return "size";
    case HotRef::LongRun: return "longrun";
    case HotRef::NetFlow: return "netflow";
  }
  return "?";
}

std::vector<double> relative_hotness(std::span<const double> pi, std::span<const double> ref) {
  const std::size_t n = pi.size();
  std::vector<double> r(n, 0.0);
  double mx = 0;
  for (std::size_t i = 0; i < n; ++i) {
    r[i] = std::isfinite(ref[i]) && ref[i] > 0 ? ref[i] : 0.0;
    mx = std::max(mx, r[i]);
  }
  if (!(mx > 0)) std::fill(r.begin(), r.end(), 1.0), mx = 1.0;
  double sum = 0;
  for (auto& x : r) {
    x = std::max(x, 1e-12 * mx);
    sum += x;
  }
  std::vector<double> h(n);
  for (std::size_t i = 0; i < n; ++i) h[i] = pi[i] / (r[i] / sum) - 1.0;
  return h;
}

std::vector<double> size_shares(std::span<const double> ref) {
  auto valid = [](double x) { return std::isfinite(x) && x > 0; };
  double sum = 0;
  for (double x : ref)
    if (valid(x)) sum += x;
  std::vector<double> s(ref.size(), std::numeric_limits<double>::quiet_NaN());
  if (!(sum > 0)) return s;
  for (std::size_t i = 0; i < ref.size(); ++i)
    if (valid(ref[i])) s[i] = ref[i] / sum;
  return s;
}

std::vector<double> net_flow_hotness(std::span<const double> in, std::span<const double> out) {
  const std::size_t n = std::min(in.size(), out.size());
  auto clean = [](double x) { return std::isfinite(x) ? x : 0.0; };
  std::vector<double> total(n);
  for (std::size_t i = 0; i < n; ++i) total[i] = clean(in[i]) + clean(out[i]);
  double kappa = 0;
  if (n > 0) {
    std::vector<double> s = total;
    std::sort(s.begin(), s.end());
    kappa = n % 2 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
  }
  std::vector<double> h(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double den = total[i] + kappa;
    if (den > 0) h[i] = (clean(in[i]) - clean(out[i])) / den;
  }
  return h;
}

}  // namespace mr
