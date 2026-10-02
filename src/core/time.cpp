#include "core/time.hpp"

#include <cstdio>
#include <stdexcept>

namespace mr {

std::int64_t floor_div(std::int64_t a, std::int64_t b) {
  std::int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

// Howard Hinnant's civil-date algorithms (proleptic Gregorian).
std::int64_t days_from_civil(int y, unsigned m, unsigned d) {
  y -= m <= 2;
  const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

Civil civil_from_days(std::int64_t z) {
  z += 719468;
  const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const std::int64_t y = static_cast<std::int64_t>(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  const unsigned d = doy - (153 * mp + 2) / 5 + 1;
  const unsigned m = mp < 10 ? mp + 3 : mp - 9;
  return {static_cast<int>(y + (m <= 2)), m, d};
}

unsigned weekday_from_days(std::int64_t z) {
  return static_cast<unsigned>(((z % 7) + 11) % 7);  // 1970-01-01 was a Thursday (4)
}

TimePoint utc_seconds(int y, unsigned m, unsigned d, int hh, int mm, int ss) {
  return days_from_civil(y, m, d) * 86400 + hh * 3600 + mm * 60 + ss;
}

TimePoint parse_rfc3339(std::string_view s) {
  int y = 0, mo = 0, d = 0, hh = 0, mi = 0, ss = 0;
  std::string str(s);
  if (str.empty() || str.back() != 'Z' ||
      std::sscanf(str.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d", &y, &mo, &d, &hh, &mi, &ss) != 6) {
    throw std::invalid_argument("not an RFC 3339 UTC timestamp: " + str);
  }
  return utc_seconds(y, static_cast<unsigned>(mo), static_cast<unsigned>(d), hh, mi, ss);
}

std::string format_rfc3339(TimePoint t) {
  const std::int64_t days = floor_div(t, 86400);
  const std::int64_t secs = t - days * 86400;
  const Civil c = civil_from_days(days);
  char buf[32];
  std::snprintf(buf, sizeof buf, "%04d-%02u-%02uT%02d:%02d:%02dZ", c.y, c.m, c.d,
                static_cast<int>(secs / 3600), static_cast<int>(secs % 3600 / 60),
                static_cast<int>(secs % 60));
  return buf;
}

}  // namespace mr
