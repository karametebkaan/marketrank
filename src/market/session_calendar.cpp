#include "market/session_calendar.hpp"

#include <algorithm>

namespace fx {
namespace {

// Day of month of the n-th Sunday (n >= 1) of the given month.
unsigned nth_sunday(int y, unsigned m, unsigned n) {
  const unsigned w = weekday_from_days(days_from_civil(y, m, 1));
  return 1 + (7 - w) % 7 + 7 * (n - 1);
}

constexpr int kSessionOpenMin = 9 * 60 + 30;
constexpr int kSessionCloseMin = 16 * 60;

}  // namespace

bool is_us_dst(TimePoint utc) {
  const int y = civil_from_days(floor_div(utc, 86400)).y;
  const TimePoint start = utc_seconds(y, 3, nth_sunday(y, 3, 2), 7, 0);
  const TimePoint end = utc_seconds(y, 11, nth_sunday(y, 11, 1), 6, 0);
  return utc >= start && utc < end;
}

EtTime to_eastern(TimePoint utc) {
  const TimePoint local = utc + (is_us_dst(utc) ? -4 : -5) * 3600;
  const std::int64_t days = floor_div(local, 86400);
  const std::int64_t secs = local - days * 86400;
  return {civil_from_days(days), static_cast<int>(secs / 3600),
          static_cast<int>(secs % 3600 / 60), weekday_from_days(days)};
}

std::vector<Bar> aggregate_session_hours(const std::vector<Bar>& bars30m) {
  std::vector<Bar> out;
  std::int64_t cur_day = 0;
  int cur_bucket = -1;
  double vw_num = 0;
  for (const Bar& b : bars30m) {
    const EtTime et = to_eastern(b.t);
    const int minute_of_day = et.hour * 60 + et.minute;
    if (minute_of_day < kSessionOpenMin || minute_of_day >= kSessionCloseMin) continue;
    const int bucket = (minute_of_day - kSessionOpenMin) / 60;
    const std::int64_t day = days_from_civil(et.date.y, et.date.m, et.date.d);
    const double bar_vw = b.vw > 0 ? b.vw : b.c;
    if (out.empty() || day != cur_day || bucket != cur_bucket) {
      if (!out.empty()) out.back().vw = out.back().v > 0 ? vw_num / out.back().v : out.back().c;
      const int offset_min = minute_of_day - (kSessionOpenMin + bucket * 60);
      out.push_back({b.t - offset_min * 60, b.o, b.h, b.l, b.c, b.v, 0});
      vw_num = bar_vw * b.v;
      cur_day = day;
      cur_bucket = bucket;
    } else {
      Bar& a = out.back();
      a.h = std::max(a.h, b.h);
      a.l = std::min(a.l, b.l);
      a.c = b.c;
      a.v += b.v;
      vw_num += bar_vw * b.v;
    }
  }
  if (!out.empty()) out.back().vw = out.back().v > 0 ? vw_num / out.back().v : out.back().c;
  return out;
}

}  // namespace fx
