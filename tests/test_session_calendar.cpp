#include <doctest/doctest.h>

#include "core/time.hpp"
#include "market/session_calendar.hpp"

using namespace fx;

TEST_CASE("US DST boundaries 2026") {
  // DST starts Sunday 2026-03-08 02:00 EST = 07:00 UTC
  CHECK_FALSE(is_us_dst(utc_seconds(2026, 3, 8, 6, 59)));
  CHECK(is_us_dst(utc_seconds(2026, 3, 8, 7, 0)));
  // DST ends Sunday 2026-11-01 02:00 EDT = 06:00 UTC
  CHECK(is_us_dst(utc_seconds(2026, 11, 1, 5, 59)));
  CHECK_FALSE(is_us_dst(utc_seconds(2026, 11, 1, 6, 0)));
  CHECK_FALSE(is_us_dst(utc_seconds(2026, 1, 15, 12, 0)));
  CHECK(is_us_dst(utc_seconds(2026, 7, 15, 12, 0)));
}

TEST_CASE("to_eastern converts across DST") {
  EtTime a = to_eastern(utc_seconds(2026, 3, 9, 13, 30));  // EDT
  CHECK(a.hour == 9);
  CHECK(a.minute == 30);
  CHECK(a.date.d == 9);
  CHECK(a.weekday == 1);  // Monday
  EtTime b = to_eastern(utc_seconds(2026, 3, 6, 14, 30));  // EST
  CHECK(b.hour == 9);
  CHECK(b.minute == 30);
  EtTime c = to_eastern(utc_seconds(2026, 3, 7, 2, 0));  // previous ET evening
  CHECK(c.date.d == 6);
  CHECK(c.hour == 21);
}

TEST_CASE("30Min bars aggregate into session-aligned hours") {
  // 2026-09-30 is EDT (UTC-4). ET 09:00 (pre-market), 09:30, 10:00, 10:30, 15:30.
  std::vector<Bar> in = {
      {utc_seconds(2026, 9, 30, 13, 0), 9, 9, 9, 9, 100, 9},
      {utc_seconds(2026, 9, 30, 13, 30), 10, 12, 9.5, 11, 1000, 10.5},
      {utc_seconds(2026, 9, 30, 14, 0), 11, 13, 10, 12, 3000, 12.5},
      {utc_seconds(2026, 9, 30, 14, 30), 12, 12.5, 11.5, 12, 500, 12},
      {utc_seconds(2026, 9, 30, 19, 30), 20, 21, 19, 20.5, 800, 20},
  };
  auto out = aggregate_session_hours(in);
  REQUIRE(out.size() == 3);
  CHECK(out[0].t == utc_seconds(2026, 9, 30, 13, 30));
  CHECK(out[0].o == 10);
  CHECK(out[0].h == 13);
  CHECK(out[0].l == 9.5);
  CHECK(out[0].c == 12);
  CHECK(out[0].v == 4000);
  CHECK(out[0].vw == doctest::Approx((10.5 * 1000 + 12.5 * 3000) / 4000));
  CHECK(out[1].t == utc_seconds(2026, 9, 30, 14, 30));
  CHECK(out[1].v == 500);
  CHECK(out[2].t == utc_seconds(2026, 9, 30, 19, 30));
  CHECK(out[2].c == 20.5);
}
