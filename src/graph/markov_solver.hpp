#pragma once
#include <span>
#include <vector>

#include "graph/csr.hpp"

namespace mr {

struct SolveResult {
  std::vector<double> pi;
  int iterations = 0;
  double residual = 0;
  bool converged = false;
};

std::vector<double> damped_step(const Csr& P, double alpha, std::span<const double> pi);
SolveResult stationary(const Csr& P, double alpha, std::span<const double> warm_start,
                       double tol = 1e-10, int max_iter = 1000);
std::vector<double> propagate(const Csr& P, double alpha, std::span<const double> pi, int k);
std::vector<double> hotness(std::span<const double> pi);

}  // namespace mr
