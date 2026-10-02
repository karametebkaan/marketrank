#include "graph/markov_solver.hpp"

#include <cmath>

namespace fx {

std::vector<double> damped_step(const Csr& P, double alpha, std::span<const double> pi) {
  std::vector<double> next = left_multiply(P, pi);
  const double teleport = (1.0 - alpha) / static_cast<double>(P.n);
  double sum = 0;
  for (double& x : next) {
    x = alpha * x + teleport;
    sum += x;
  }
  for (double& x : next) x /= sum;
  return next;
}

SolveResult stationary(const Csr& P, double alpha, std::span<const double> warm_start,
                       double tol, int max_iter) {
  SolveResult r;
  if (warm_start.size() == P.n) {
    r.pi.assign(warm_start.begin(), warm_start.end());
  } else {
    r.pi.assign(P.n, 1.0 / static_cast<double>(P.n));
  }
  for (r.iterations = 1; r.iterations <= max_iter; ++r.iterations) {
    std::vector<double> next = damped_step(P, alpha, r.pi);
    r.residual = 0;
    for (std::size_t i = 0; i < P.n; ++i) r.residual += std::abs(next[i] - r.pi[i]);
    r.pi = std::move(next);
    if (r.residual < tol) {
      r.converged = true;
      return r;
    }
  }
  r.iterations = max_iter;
  return r;
}

std::vector<double> propagate(const Csr& P, double alpha, std::span<const double> pi, int k) {
  std::vector<double> x(pi.begin(), pi.end());
  for (int s = 0; s < k; ++s) x = damped_step(P, alpha, x);
  return x;
}

std::vector<double> hotness(std::span<const double> pi) {
  std::vector<double> h(pi.size());
  const double n = static_cast<double>(pi.size());
  for (std::size_t i = 0; i < pi.size(); ++i) h[i] = n * pi[i] - 1.0;
  return h;
}

}  // namespace fx
