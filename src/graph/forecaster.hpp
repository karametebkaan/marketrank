#pragma once
#include <span>
#include <vector>

#include "graph/csr.hpp"

namespace mr {

struct Forecast {
  int k = 0;
  std::vector<double> pi_k;
  std::vector<double> score;
};

Forecast forecast(const Csr& P_fast, double alpha, std::span<const double> pi_now,
                  std::span<const double> pi_prev, int k, double beta);

}  // namespace mr
