#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "graph/csr.hpp"

namespace mr {

// One bar's flow neighbourhoods, kept per cached frame for the graph view: for every active node its k strongest
// kept out-edges and k strongest kept in-edges of the chain P (by raw dollars), deduplicated. Compact: an edge is
// 12 bytes; out_total[i] is the raw sum of i's kept off-diagonal out-edges (share = raw / out_total[a]).
struct FlowNeighbours {
  struct Edge {
    std::uint32_t a, b;  // a -> b
    float raw;
  };
  std::vector<Edge> edges;      // sorted by (a, b)
  std::vector<float> out_total;  // size n
};

FlowNeighbours flow_neighbours(const Csr& P, const std::vector<bool>& active, std::size_t k);

// The graph around `focus`: focus nodes plus each focus node's up to k strongest neighbours (in or out, by raw),
// and every stored edge among that node set. Nodes in ascending index; edges sorted by (a, b).
struct FlowSubgraph {
  std::vector<std::uint32_t> nodes;
  std::vector<FlowNeighbours::Edge> edges;
};
FlowSubgraph flow_subgraph(const FlowNeighbours& g, std::span<const std::uint32_t> focus, std::size_t k);

}  // namespace mr
