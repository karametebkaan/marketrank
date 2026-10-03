#pragma once
#include <vector>

#include "core/time.hpp"
#include "core/types.hpp"

namespace mr {

bool is_us_dst(TimePoint utc);

struct EtTime {
  Civil date;
  int hour;
  int minute;
  unsigned weekday;  // 0 = Sunday
};

EtTime to_eastern(TimePoint utc);

// 16:00 ET (as UTC) on the ET date of utc: the regular session close. Early closes are not
// modelled, so this is never earlier than the real close.
TimePoint session_close(TimePoint utc);

// --- M5 intraday (15-minute regular-session bars) ---
// NYSE early closes (13:00 ET), by rule: the day after Thanksgiving (4th Thursday of November + 1), and July 3 and
// December 24 when they fall on Monday-Thursday (on a Friday they are the observed holiday). Exact for 2015-2027.
bool is_early_close(const Civil& et_date);
// Regular-session close of an ET date in minutes after ET midnight: 13:00 on early-close days, else 16:00.
// Full holidays are not modelled: the feed has no regular-session bars on them, so they never form a session.
int session_close_minute(const Civil& et_date);
// Whether a bar starting at t (UTC) and lasting bar_seconds is a regular-session bar: a weekday, the start on the
// bar grid, 09:30 ET <= start and start + bar_seconds <= the session close (DST handled by to_eastern).
bool is_regular_session_bar(TimePoint t, TimePoint bar_seconds = 900);
std::vector<Bar> filter_regular_session(const std::vector<Bar>& bars, TimePoint bar_seconds = 900);

// Session hours 09:30-10:30 ... 14:30-15:30 and 15:30-16:00 ET (7 buckets).
std::vector<Bar> aggregate_session_hours(const std::vector<Bar>& bars30m);

}  // namespace mr
