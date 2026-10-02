#pragma once
#include <cstdint>
#include <string_view>

namespace mr {

using TimePoint = std::int64_t;  // seconds since Unix epoch, UTC

struct Bar {
  TimePoint t = 0;  // bar start
  double o = 0, h = 0, l = 0, c = 0, v = 0, vw = 0;
};

enum class Timeframe { Hour, Day, Week };

std::string_view to_string(Timeframe tf);
Timeframe parse_timeframe(std::string_view s);
TimePoint timeframe_seconds(Timeframe tf);

}  // namespace mr
