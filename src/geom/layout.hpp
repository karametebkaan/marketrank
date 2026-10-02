#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "graph/csr.hpp"

namespace fx {

struct LayoutParams {
  int iterations_init = 150;
  int iterations_step = 20;
  double theta = 0.8;
  double repulsion = 0.05;  // multiplied by 1 / active count
  double attraction = 1.0;
  double gravity = 0.1;
  double max_step = 0.05;
  std::uint64_t seed = 7;
};

struct LayoutEdge {
  std::uint32_t a, b;
  double w;
};

std::vector<double> initial_positions(std::size_t n, std::uint64_t seed);
std::vector<LayoutEdge> layout_edges(const Csr& P, const std::vector<bool>& active, std::size_t per_node);
std::vector<double> exact_repulsion(const std::vector<double>& xy, const std::vector<bool>& active);
std::vector<double> barnes_hut_repulsion(const std::vector<double>& xy, const std::vector<bool>& active,
                                         double theta);
// Spec 6.1: springs on edges (weight / max weight), Barnes-Hut repulsion, gravity, cooling step cap.
void run_layout(std::vector<double>& xy, const std::vector<bool>& active,
                const std::vector<LayoutEdge>& edges, const LayoutParams& p, int iterations);

}  // namespace fx
