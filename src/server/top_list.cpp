#include "server/top_list.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mr {

namespace {
// The metric of node k of f: pi·N_active (Pi; same order as pi) or h.
double metric(const LandscapeFrame& f, std::size_t k, TopBy by) {
  const LandscapeNode& nd = f.nodes[k];
  return by == TopBy::Pi ? nd.pi * static_cast<double>(f.nodes.size()) : nd.h;
}

// Positions of f's nodes with a finite metric, sorted highest first (ties by lower i). By pi the sort key is
// nd.pi itself (the same order as pi*N, without the product's rounding), matching the CLI.
std::vector<std::size_t> ranking(const LandscapeFrame& f, TopBy by) {
  std::vector<std::size_t> pos;
  pos.reserve(f.nodes.size());
  std::vector<double> m(f.nodes.size());
  for (std::size_t k = 0; k < f.nodes.size(); ++k) {
    m[k] = by == TopBy::Pi ? f.nodes[k].pi : f.nodes[k].h;
    if (std::isfinite(m[k])) pos.push_back(k);
  }
  std::sort(pos.begin(), pos.end(), [&](std::size_t a, std::size_t b) {
    return m[a] > m[b] || (m[a] == m[b] && f.nodes[a].i < f.nodes[b].i);
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
                            std::size_t bars, TopBy by) {
  std::vector<TopRow> rows;
  if (frames.empty() || !frames.back()) return rows;
  const LandscapeFrame& cur = *frames.back();
  const auto order = ranking(cur, by);
  // Rank of every node position in the previous frame (0 = unranked).
  const LandscapeFrame* prev = frames.size() >= 2 ? frames[frames.size() - 2].get() : nullptr;
  std::vector<std::size_t> prev_rank_at;
  if (prev) {
    prev_rank_at.assign(prev->nodes.size(), 0);
    const auto po = ranking(*prev, by);
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
    row.mr = nd.pi * static_cast<double>(cur.nodes.size());
    row.pulse = nd.pulse;
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
      if (k == std::size_t(-1)) continue;
      const double v = metric(*f, k, by);
      if (std::isfinite(v)) row.series[bars - 1 - q] = v;
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

}  // namespace mr
