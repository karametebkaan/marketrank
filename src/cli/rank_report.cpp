#include "cli/rank_report.hpp"

#include <algorithm>
#include <cstdio>

namespace mr {

std::vector<std::size_t> rank_order(const Frame& f, RankBy by) {
  std::vector<std::size_t> order;
  for (std::size_t i = 0; i < f.active.size(); ++i)
    if (f.active[i]) order.push_back(i);
  auto key = [&](std::size_t i) { return by == RankBy::Pi ? f.pi[i] : f.h[i]; };
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return key(a) > key(b); });
  return order;
}

BottomSection bottom_section(const Frame& f, const std::vector<std::size_t>& order, RankBy by, std::size_t top) {
  BottomSection b;
  const std::size_t n_active = order.size();
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    const std::size_t i = *it;
    if (by == RankBy::Pi && at_teleport_floor(f.pi[i], f.pi_floor)) {
      ++b.floor_count;
      continue;
    }
    if (b.rows.size() < top) b.rows.push_back(i);
  }
  b.floor_score = market_rank_score(f.pi_floor, n_active);
  return b;
}

std::string floor_line(const BottomSection& b) {
  char buf[96];
  std::snprintf(buf, sizeof buf, "%zu stocks tied at the teleport floor (π·N = %.4f)", b.floor_count, b.floor_score);
  return buf;
}

}  // namespace mr
