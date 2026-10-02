#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/types.hpp"
#include "graph/sparse_flux.hpp"
#include "market/panel.hpp"
#include "market/universe.hpp"
#include "pipeline/core_pipeline.hpp"

namespace mr {

// A small slice of the money-flow graph for the README figure (--export-slice): the top-pi stock and its strongest
// partners, with the raw flux among them and MarketRank re-solved on the slice alone.
struct SliceEdge {
  std::size_t from = 0, to = 0;  // positions in GraphSlice::nodes
  double dollars = 0;            // raw accumulated flux from -> to (slow accumulator)
};

struct GraphSlice {
  TimePoint t = 0;
  std::size_t n_active = 0;
  double alpha = 0.85;              // the slice is solved with p = 1 - alpha
  std::vector<std::size_t> nodes;   // universe indices; nodes[0] is the top-pi stock, then partners by strength
  std::vector<double> pi, mr;       // global pi and pi * N_active, parallel to nodes
  std::vector<SliceEdge> edges;     // every kept accumulator edge among the slice nodes (from != to, dollars > 0)
  std::vector<double> slice_pi;     // stationary distribution of the slice's own damped chain, parallel to nodes
};

// Picks the slice from the slow accumulator's kept edges among active nodes: the top-pi active node (ties to the
// lower index) and its n - 1 partners with the largest combined raw flux in both directions (ties to the lower
// index). Fewer partners than n - 1 give a smaller slice. n >= 1.
GraphSlice pick_slice(const FluxAccumulator& slow, const Frame& f, std::size_t n, double alpha);

// MarketRank on k nodes from their edges alone: P = row-normalized raw flux, damped with alpha, dangling rows
// teleport (the solver of the full model).
std::vector<double> slice_market_rank(std::size_t k, const std::vector<SliceEdge>& edges, double alpha);

// Runs the pipeline over the whole panel and picks the slice of the latest frame.
GraphSlice export_slice(const Panel& panel, const CoreParams& params, std::size_t n);

nlohmann::json slice_json(const GraphSlice& s, const std::vector<Security>& universe, const std::string& bar,
                          const std::string& preset);

}  // namespace mr
