#include "graph/csr.hpp"

namespace mr {

std::vector<double> left_multiply(const Csr& P, std::span<const double> x) {
  std::vector<double> y(P.n, 0.0);
  for (std::size_t i = 0; i < P.n; ++i) {
    const double xi = x[i];
    if (xi == 0) continue;
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) y[P.col[e]] += xi * P.val[e];
  }
  return y;
}

Csr transpose(const Csr& P) {
  Csr T;
  T.n = P.n;
  T.row_ptr.assign(P.n + 1, 0);
  for (auto c : P.col) ++T.row_ptr[c + 1];
  for (std::size_t i = 0; i < P.n; ++i) T.row_ptr[i + 1] += T.row_ptr[i];
  T.col.resize(P.col.size());
  T.val.resize(P.val.size());
  T.raw.resize(P.raw.size());
  std::vector<std::size_t> next(T.row_ptr.begin(), T.row_ptr.end() - 1);
  for (std::size_t i = 0; i < P.n; ++i) {
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) {
      const std::size_t slot = next[P.col[e]]++;
      T.col[slot] = static_cast<std::uint32_t>(i);
      T.val[slot] = P.val[e];
      if (!P.raw.empty()) T.raw[slot] = P.raw[e];
    }
  }
  return T;
}

std::vector<double> left_multiply_transposed(const Csr& PT, std::span<const double> x) {
  std::vector<double> y(PT.n, 0.0);
#pragma omp parallel for schedule(dynamic, 256)
  for (std::size_t j = 0; j < PT.n; ++j) {
    double s = 0;
    for (auto e = PT.row_ptr[j]; e < PT.row_ptr[j + 1]; ++e) s += x[PT.col[e]] * PT.val[e];
    y[j] = s;
  }
  return y;
}

}  // namespace mr
