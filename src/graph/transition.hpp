#pragma once
#include <cstddef>
#include <string_view>
#include <vector>

#include "graph/csr.hpp"
#include "graph/sparse_flux.hpp"

namespace mr {

enum class LiftMode { Off, Excess, Ratio };

LiftMode parse_lift_mode(std::string_view s);
std::string_view to_string(LiftMode m);

// What an active row with no kept off-diagonal edge (and no retained mass) becomes: a self-loop of 1
// (milestone-1 behaviour), or an empty row, i.e. a dangling node that the solver teleports uniformly.
enum class DanglingMode { SelfLoop, Teleport };

struct TransitionParams {
  LiftMode lift = LiftMode::Excess;  // (B)
  std::size_t k_out = 20;            // (C) per-row top edges
  std::size_t k_in = 10;             // (C) per-column top edges
  double retention = 1.0;            // (E)
  DanglingMode dangling = DanglingMode::SelfLoop;
  bool operator==(const TransitionParams&) const = default;
};

// Spec 5 (B, C, E). Row-stochastic except dangling rows (empty under DanglingMode::Teleport); rows
// ascending by column; raw = un-lifted F (self: retention*in). With retention 0 no active row has a
// diagonal entry unless it is a SelfLoop dangling row. Inactive rows are always a self-loop of 1.
Csr build_transition(const FluxAccumulator& acc, const TransitionParams& params,
                     const std::vector<bool>& active = {});

}  // namespace mr
