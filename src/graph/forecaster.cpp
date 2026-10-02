#include "graph/forecaster.hpp"

#include "graph/markov_solver.hpp"

namespace fx {

Forecast forecast(const Csr& P_fast, double alpha, std::span<const double> pi_now,
                  std::span<const double> pi_prev, int k, double beta) {
  Forecast f;
  f.k = k;
  f.pi_k = propagate(P_fast, alpha, pi_now, k);
  const std::size_t n = pi_now.size();
  const double N = static_cast<double>(n);
  const bool drift = pi_prev.size() == n;
  f.score.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    f.score[i] = N * (f.pi_k[i] - pi_now[i]);
    if (drift) f.score[i] += beta * N * (pi_now[i] - pi_prev[i]);
  }
  return f;
}

}  // namespace fx
