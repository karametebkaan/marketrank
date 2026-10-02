#include "graph/markov_solver.hpp"

#include <cmath>
#include <utility>

namespace mr {

namespace {
std::vector<double> step_t(const Csr& PT, double alpha, std::span<const double> pi) {
  std::vector<double> next = left_multiply_transposed(PT, pi);
  const double teleport = (1.0 - alpha) / static_cast<double>(PT.n);
  double sum = 0;
  for (double& x : next) {
    x = alpha * x + teleport;
    sum += x;
  }
  for (double& x : next) x /= sum;
  return next;
}
}  // namespace

std::vector<double> damped_step(const Csr& P, double alpha, std::span<const double> pi) {
  return step_t(transpose(P), alpha, pi);
}

SolveResult stationary(const Csr& P, double alpha, std::span<const double> warm_start, double tol,
                       int max_iter) {
  const Csr PT = transpose(P);
  SolveResult r;
  if (warm_start.size() == P.n) {
    r.pi.assign(warm_start.begin(), warm_start.end());
  } else {
    r.pi.assign(P.n, 1.0 / static_cast<double>(P.n));
  }
  for (r.iterations = 1; r.iterations <= max_iter; ++r.iterations) {
    std::vector<double> next = step_t(PT, alpha, r.pi);
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
  const Csr PT = transpose(P);
  std::vector<double> x(pi.begin(), pi.end());
  for (int s = 0; s < k; ++s) x = step_t(PT, alpha, x);
  return x;
}

std::vector<double> hotness(std::span<const double> pi) {
  std::vector<double> h(pi.size());
  const double n = static_cast<double>(pi.size());
  for (std::size_t i = 0; i < pi.size(); ++i) h[i] = n * pi[i] - 1.0;
  return h;
}

}  // namespace mr
