#pragma once
#include <cstddef>
#include <string_view>
#include <vector>

#include "graph/csr.hpp"
#include "graph/sparse_flux.hpp"

namespace fx {

enum class LiftMode { Off, Excess, Ratio };

LiftMode parse_lift_mode(std::string_view s);
std::string_view to_string(LiftMode m);

struct TransitionParams {
  LiftMode lift = LiftMode::Excess;  // (B)
  std::size_t k_out = 20;            // (C) per-row top edges
  std::size_t k_in = 10;             // (C) per-column top edges
  double retention = 1.0;            // (E)
};

// Spec 5 (B, C, E). Row-stochastic; rows ascending by column; raw = un-lifted F (self: retention*in).
Csr build_transition(const FluxAccumulator& acc, const TransitionParams& params,
                     const std::vector<bool>& active = {});

}  // namespace fx
