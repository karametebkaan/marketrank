#include "graph/csr.hpp"

#include <algorithm>
#include <utility>

namespace fx {

Csr build_transition(std::span<const double> flux, std::size_t n, std::size_t k,
                     const std::vector<bool>& active) {
  Csr P;
  P.n = n;
  P.row_ptr.reserve(n + 1);
  P.row_ptr.push_back(0);
  std::vector<std::pair<double, std::uint32_t>> row;
  for (std::size_t i = 0; i < n; ++i) {
    row.clear();
    const bool row_active = active.empty() || active[i];
    for (std::size_t j = 0; j < n && row_active; ++j) {
      if (!active.empty() && !active[j]) continue;
      const double w = flux[i * n + j];
      if (j != i && w > 0) row.emplace_back(w, static_cast<std::uint32_t>(j));
    }
    if (row.size() > k) {
      std::partial_sort(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(k), row.end(),
                        [](const auto& a, const auto& b) {
                          return a.first > b.first || (a.first == b.first && a.second < b.second);
                        });
      row.resize(k);
    }
    if (row.empty()) {
      P.col.push_back(static_cast<std::uint32_t>(i));
      P.val.push_back(1.0);
      P.raw.push_back(0.0);
    } else {
      std::sort(row.begin(), row.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
      double sum = 0;
      for (const auto& [w, j] : row) sum += w;
      for (const auto& [w, j] : row) {
        P.col.push_back(j);
        P.val.push_back(w / sum);
        P.raw.push_back(w);
      }
    }
    P.row_ptr.push_back(P.col.size());
  }
  return P;
}

std::vector<double> left_multiply(const Csr& P, std::span<const double> x) {
  std::vector<double> y(P.n, 0.0);
  for (std::size_t i = 0; i < P.n; ++i) {
    const double xi = x[i];
    if (xi == 0) continue;
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) y[P.col[e]] += xi * P.val[e];
  }
  return y;
}

}  // namespace fx
