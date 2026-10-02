#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "geom/landscape.hpp"

namespace mr {

// Ranking metric: the MarketRank score pi (series in pi·N_active) or exact hotness h (series in h).
enum class TopBy { Pi, Hotness };

struct TopRow {
  std::size_t rank = 0;  // 1-based
  std::uint32_t i = 0;
  double h = 0, hdisp = 0, pi = 0, score = 0;
  double mr = 0;                                            // MarketRank score pi·N_active (1 = average)
  double pulse = std::numeric_limits<double>::quiet_NaN();  // heartbeat delta log pi
  std::optional<std::size_t> prev_rank;  // rank at the previous cached bar by the same metric; empty if absent there
  std::vector<double> series;            // the metric over the last `bars` bars ending at t, oldest first (NaN = missing)
};

// Top nodes of frames.back() by the metric (descending, ties by lower i; non-finite values excluded), at most n.
// frames: cached frames ending at the target bar, oldest first; frames[size-2] (if any) is the previous bar.
// O(N log N + n·bars·log N).
std::vector<TopRow> top_hot(const std::vector<std::shared_ptr<const LandscapeFrame>>& frames, std::size_t n,
                            std::size_t bars, TopBy by = TopBy::Hotness);

}  // namespace mr
