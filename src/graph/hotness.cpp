#include "graph/hotness.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace fx {

HotRef parse_hot_ref(std::string_view s) {
  if (s == "uniform") return HotRef::Uniform;
  if (s == "size") return HotRef::Size;
  if (s == "longrun") return HotRef::LongRun;
  throw std::invalid_argument("unknown hotness reference: " + std::string(s));
}

std::string_view to_string(HotRef r) {
  switch (r) {
    case HotRef::Uniform: return "uniform";
    case HotRef::Size: return "size";
    case HotRef::LongRun: return "longrun";
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

}  // namespace fx
