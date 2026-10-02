#pragma once
#include <cstdint>
#include <limits>
#include <vector>

#include "graph/sparse_flux.hpp"
#include "graph/transition.hpp"

namespace mr::test {

// Accumulator holding exactly the dense n x n matrix F: out = row sums, in = column sums.
inline FluxAccumulator acc_from_dense(const std::vector<double>& F, std::size_t n) {
  BarFlux bar;
  bar.rows.resize(n);
  bar.out.assign(n, 0.0);
  bar.in.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double w = F[i * n + j];
      if (w == 0) continue;
      bar.rows[i].push_back({static_cast<std::uint32_t>(j), w});
      bar.out[i] += w;
      bar.in[j] += w;
    }
  }
  FluxAccumulator acc(n, std::numeric_limits<double>::infinity(), n == 0 ? 1 : n);
  acc.add(bar);
  return acc;
}

inline TransitionParams legacy_tp(std::size_t k) {
  TransitionParams tp;
  tp.lift = LiftMode::Off;
  tp.k_out = k;
  tp.k_in = 0;
  tp.retention = 0.0;
  return tp;
}

}  // namespace mr::test
