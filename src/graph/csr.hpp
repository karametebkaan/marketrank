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
  std::vector<double> raw;  // un-normalized pruned flux weight, parallel to val (0 for self-loop)
};

// `active` (empty = all active): inactive nodes get no edges in or out, only a self-loop.
Csr build_transition(std::span<const double> flux, std::size_t n, std::size_t k,
                     const std::vector<bool>& active = {});
std::vector<double> left_multiply(const Csr& P, std::span<const double> x);

}  // namespace fx
