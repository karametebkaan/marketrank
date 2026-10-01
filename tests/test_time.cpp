#include <doctest/doctest.h>

#include <stdexcept>

#include "core/time.hpp"
#include "core/types.hpp"

using namespace fx;

TEST_CASE("civil date round trip and known values") {
  CHECK(days_from_civil(1970, 1, 1) == 0);
  CHECK(utc_seconds(2000, 1, 1) == 946684800);
  for (int64_t z = -800000; z <= 800000; z += 997) {
    Civil c = civil_from_days(z);
    CHECK(days_from_civil(c.y, c.m, c.d) == z);
  }
  CHECK(weekday_from_days(days_from_civil(2026, 10, 1)) == 4);  // Thursday
  CHECK(weekday_from_days(days_from_civil(2026, 3, 8)) == 0);   // Sunday
}

TEST_CASE("floor_div rounds toward negative infinity") {
  CHECK(floor_div(7, 2) == 3);
  CHECK(floor_div(-7, 2) == -4);
  CHECK(floor_div(-8, 2) == -4);
}

TEST_CASE("rfc3339 parse and format") {
  CHECK(parse_rfc3339("2026-09-30T13:30:00Z") == utc_seconds(2026, 9, 30, 13, 30));
  CHECK(parse_rfc3339("2026-09-30T13:30:00.123Z") == utc_seconds(2026, 9, 30, 13, 30));
  CHECK(format_rfc3339(utc_seconds(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05Z");
  CHECK_THROWS_AS(parse_rfc3339("2026-09-30T13:30:00+02:00"), std::invalid_argument);
  CHECK_THROWS_AS(parse_rfc3339("garbage"), std::invalid_argument);
}

TEST_CASE("timeframe strings") {
  CHECK(parse_timeframe("1h") == Timeframe::Hour);
  CHECK(parse_timeframe("1d") == Timeframe::Day);
  CHECK(parse_timeframe("1w") == Timeframe::Week);
  CHECK(to_string(Timeframe::Week) == "1w");
  CHECK(timeframe_seconds(Timeframe::Day) == 86400);
  CHECK_THROWS_AS(parse_timeframe("5m"), std::invalid_argument);
}
