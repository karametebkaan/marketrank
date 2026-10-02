#include "graph/markov_solver.hpp"

#include <cmath>
#include <utility>

namespace mr {

namespace {
// Rows of P without any entry (dangling nodes, DanglingMode::Teleport).
std::vector<std::uint32_t> dangling_rows(const Csr& P) {
  std::vector<std::uint32_t> d;
  for (std::size_t i = 0; i < P.n; ++i)
    if (P.row_ptr[i] == P.row_ptr[i + 1]) d.push_back(static_cast<std::uint32_t>(i));
  return d;
}

// One damped step pi' = alpha * (pi P + d/N) + (1 - alpha)/N, with d the mass on dangling rows (standard
// PageRank: a dangling node teleports all its mass uniformly), then renormalized to sum 1. Without
// dangling rows this is exactly the earlier step.
std::vector<double> step_t(const Csr& PT, double alpha, std::span<const double> pi,
                           const std::vector<std::uint32_t>& dangling) {
  std::vector<double> next = left_multiply_transposed(PT, pi);
  double teleport = (1.0 - alpha) / static_cast<double>(PT.n);
  if (!dangling.empty()) {
    double d = 0;
    for (auto i : dangling) d += pi[i];
    teleport += alpha * d / static_cast<double>(PT.n);
  }
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
  return step_t(transpose(P), alpha, pi, dangling_rows(P));
}

SolveResult stationary(const Csr& P, double alpha, std::span<const double> warm_start, double tol,
                       int max_iter) {
  const Csr PT = transpose(P);
  const auto dangling = dangling_rows(P);
  SolveResult r;
  if (warm_start.size() == P.n) {
    r.pi.assign(warm_start.begin(), warm_start.end());
  } else {
    r.pi.assign(P.n, 1.0 / static_cast<double>(P.n));
  }
  for (r.iterations = 1; r.iterations <= max_iter; ++r.iterations) {
    std::vector<double> next = step_t(PT, alpha, r.pi, dangling);
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
  const auto dangling = dangling_rows(P);
  std::vector<double> x(pi.begin(), pi.end());
  for (int s = 0; s < k; ++s) x = step_t(PT, alpha, x, dangling);
  return x;
}

std::vector<double> hotness(std::span<const double> pi) {
  std::vector<double> h(pi.size());
  const double n = static_cast<double>(pi.size());
  for (std::size_t i = 0; i < pi.size(); ++i) h[i] = n * pi[i] - 1.0;
  return h;
}

}  // namespace mr
