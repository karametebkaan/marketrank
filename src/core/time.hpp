#pragma once
#include <cstdint>
#include <string>
#include <string_view>

#include "core/types.hpp"

namespace fx {

struct Civil {
  int y;
  unsigned m;
  unsigned d;
};

std::int64_t floor_div(std::int64_t a, std::int64_t b);
std::int64_t days_from_civil(int y, unsigned m, unsigned d);
Civil civil_from_days(std::int64_t z);
unsigned weekday_from_days(std::int64_t z);  // 0 = Sunday
TimePoint utc_seconds(int y, unsigned m, unsigned d, int hh = 0, int mm = 0, int ss = 0);
TimePoint parse_rfc3339(std::string_view s);
std::string format_rfc3339(TimePoint t);

}  // namespace fx
