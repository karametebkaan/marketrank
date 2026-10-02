#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fx {

struct Csr {
  std::size_t n = 0;
  std::vector<std::size_t> row_ptr;
  std::vector<std::uint32_t> col;
  std::vector<double> val;
  std::vector<double> raw;  // accumulated un-lifted flux for kept edges; retention*in for self-loops
};

std::vector<double> left_multiply(const Csr& P, std::span<const double> x);  // y = x P
Csr transpose(const Csr& P);
// y = x P computed from PT = transpose(P) as a parallel pull (deterministic).
std::vector<double> left_multiply_transposed(const Csr& PT, std::span<const double> x);

}  // namespace fx
