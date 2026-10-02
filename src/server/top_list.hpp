#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "geom/landscape.hpp"

namespace fx {

struct TopRow {
  std::size_t rank = 0;  // 1-based
  std::uint32_t i = 0;
  double h = 0, hdisp = 0, pi = 0, score = 0;
  std::optional<std::size_t> prev_rank;  // rank at the previous cached bar; empty if absent there
  std::vector<double> series;            // exact h over the last `bars` bars ending at t, oldest first (NaN = missing)
};

// Hottest nodes of frames.back() by exact h (descending, ties by lower i; non-finite h excluded), at most n.
// frames: cached frames ending at the target bar, oldest first; frames[size-2] (if any) is the previous bar.
// O(N log N + n·bars·log N).
std::vector<TopRow> top_hot(const std::vector<std::shared_ptr<const LandscapeFrame>>& frames, std::size_t n,
                            std::size_t bars);

}  // namespace fx
