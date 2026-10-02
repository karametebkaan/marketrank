#pragma once
#include <cstdint>
#include <vector>

#include "core/time.hpp"
#include "market/bar_store.hpp"
#include "market/universe.hpp"

namespace fx {

struct SyntheticConfig {
  int sectors = 5;
  int per_sector = 10;
  int bars = 300;
  std::uint64_t seed = 42;
  int rotation_from = 0;
  int rotation_to = 1;
  int rotation_start = 150;
  double rotation_strength = 0.004;
  Timeframe tf = Timeframe::Day;
  TimePoint start = utc_seconds(2025, 1, 2, 21, 0);
};

std::vector<Security> generate_synthetic(const SyntheticConfig& cfg, BarStore& store);

}  // namespace fx
