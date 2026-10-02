#include "server/top_list.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fx {

namespace {
// Positions of f's nodes with finite h, sorted hottest first (ties by lower i).
std::vector<std::size_t> ranking(const LandscapeFrame& f) {
  std::vector<std::size_t> pos;
  pos.reserve(f.nodes.size());
  for (std::size_t k = 0; k < f.nodes.size(); ++k)
    if (std::isfinite(f.nodes[k].h)) pos.push_back(k);
  std::sort(pos.begin(), pos.end(), [&](std::size_t a, std::size_t b) {
    const auto &x = f.nodes[a], &y = f.nodes[b];
    return x.h > y.h || (x.h == y.h && x.i < y.i);
  });
  return pos;
}

// Index of node i in f.nodes (ascending i), or npos.
std::size_t find_node(const LandscapeFrame& f, std::uint32_t i) {
  auto it = std::lower_bound(f.nodes.begin(), f.nodes.end(), i,
                             [](const LandscapeNode& nd, std::uint32_t v) { return nd.i < v; });
  return it != f.nodes.end() && it->i == i ? static_cast<std::size_t>(it - f.nodes.begin()) : std::size_t(-1);
}
}  // namespace

std::vector<TopRow> top_hot(const std::vector<std::shared_ptr<const LandscapeFrame>>& frames, std::size_t n,
                            std::size_t bars) {
  std::vector<TopRow> rows;
  if (frames.empty() || !frames.back()) return rows;
  const LandscapeFrame& cur = *frames.back();
  const auto order = ranking(cur);
  // Rank of every node position in the previous frame (0 = unranked).
  const LandscapeFrame* prev = frames.size() >= 2 ? frames[frames.size() - 2].get() : nullptr;
  std::vector<std::size_t> prev_rank_at;
  if (prev) {
    prev_rank_at.assign(prev->nodes.size(), 0);
    const auto po = ranking(*prev);
    for (std::size_t r = 0; r < po.size(); ++r) prev_rank_at[po[r]] = r + 1;
  }
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t r = 0; r < std::min(n, order.size()); ++r) {
    const LandscapeNode& nd = cur.nodes[order[r]];
    TopRow row;
    row.rank = r + 1;
    row.i = nd.i;
    row.h = nd.h;
    row.hdisp = nd.hdisp;
    row.pi = nd.pi;
    row.score = nd.score;
    if (prev) {
      const std::size_t k = find_node(*prev, nd.i);
      if (k != std::size_t(-1) && prev_rank_at[k] > 0) row.prev_rank = prev_rank_at[k];
    }
    row.series.assign(bars, nan);
    // series[bars-1] is frames.back(); walk back through the cached frames.
    for (std::size_t q = 0; q < bars && q < frames.size(); ++q) {
      const LandscapeFrame* f = frames[frames.size() - 1 - q].get();
      if (!f) continue;
      const std::size_t k = find_node(*f, nd.i);
      if (k != std::size_t(-1) && std::isfinite(f->nodes[k].h)) row.series[bars - 1 - q] = f->nodes[k].h;
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

}  // namespace fx
