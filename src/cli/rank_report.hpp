#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include "cli/args.hpp"
#include "pipeline/core_pipeline.hpp"

namespace mr {

// Active nodes of f in rank order: pi descending (RankBy::Pi) or h descending, ties by lower index; the same
// order /api/top uses.
std::vector<std::size_t> rank_order(const Frame& f, RankBy by);

// The bottom section of the rank table. By pi, the stocks at the teleport floor (all tied) are folded into one
// line and `rows` lists only non-floor names; by hotness nothing is folded.
struct BottomSection {
  std::size_t floor_count = 0;
  double floor_score = 0;         // pi*N_active of the floor
  std::vector<std::size_t> rows;  // at most `top` nodes, lowest first
};
BottomSection bottom_section(const Frame& f, const std::vector<std::size_t>& order, RankBy by, std::size_t top);
// "N stocks tied at the teleport floor (π·N = x)".
std::string floor_line(const BottomSection& b);

}  // namespace mr
