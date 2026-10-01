#pragma once
#include <vector>

#include "core/time.hpp"
#include "core/types.hpp"

namespace fx {

bool is_us_dst(TimePoint utc);

struct EtTime {
  Civil date;
  int hour;
  int minute;
  unsigned weekday;  // 0 = Sunday
};

EtTime to_eastern(TimePoint utc);

// Session hours 09:30-10:30 ... 14:30-15:30 and 15:30-16:00 ET (7 buckets).
std::vector<Bar> aggregate_session_hours(const std::vector<Bar>& bars30m);

}  // namespace fx
