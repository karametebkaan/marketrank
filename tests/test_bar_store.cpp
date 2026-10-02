#include <doctest/doctest.h>

#include <cmath>

#include "market/bar_store.hpp"
#include "market/panel.hpp"
#include "test_util.hpp"

using namespace fx;

TEST_CASE("merge keeps bars sorted, unique, newest wins") {
  BarStore s(test::temp_dir("bars_merge"));
  s.merge("AAPL", Timeframe::Day, {{200, 1, 1, 1, 2, 10, 2}, {100, 1, 1, 1, 1, 10, 1}});
  s.merge("AAPL", Timeframe::Day, {{200, 1, 1, 1, 5, 10, 5}, {300, 1, 1, 1, 3, 10, 3}});
  const auto& b = s.bars("AAPL", Timeframe::Day);
  REQUIRE(b.size() == 3);
  CHECK(b[0].t == 100);
  CHECK(b[1].c == 5);
  CHECK(b[2].t == 300);
  CHECK(s.last_time("AAPL", Timeframe::Day).value() == 300);
  CHECK_FALSE(s.last_time("MSFT", Timeframe::Day).has_value());
  CHECK(s.bars("MSFT", Timeframe::Day).empty());
}

TEST_CASE("save and load round trip") {
  auto dir = test::temp_dir("bars_io");
  {
    BarStore s(dir);
    s.merge("BRK.B", Timeframe::Hour, {{1759239000, 470.25, 471.5, 469.0, 470.75, 123456, 470.4}});
    s.save("BRK.B", Timeframe::Hour);
  }
  BarStore s2(dir);
  s2.load_all({"BRK.B", "MISSING"}, Timeframe::Hour);
  const auto& b = s2.bars("BRK.B", Timeframe::Hour);
  REQUIRE(b.size() == 1);
  CHECK(b[0].t == 1759239000);
  CHECK(b[0].c == doctest::Approx(470.75));
  CHECK(b[0].v == doctest::Approx(123456));
  CHECK(b[0].vw == doctest::Approx(470.4));
}

TEST_CASE("panel aligns tickers on the union of times with NaN gaps") {
  BarStore s(test::temp_dir("panel"));
  s.merge("A", Timeframe::Day, {{100, 0, 0, 0, 10, 5, 10}, {200, 0, 0, 0, 11, 6, 11}});
  s.merge("B", Timeframe::Day, {{200, 0, 0, 0, 20, 7, 20}, {300, 0, 0, 0, 21, 8, 21}});
  Panel p = build_panel(s, {"A", "B"}, Timeframe::Day);
  REQUIRE(p.T() == 3);
  REQUIRE(p.N() == 2);
  CHECK(p.times == std::vector<TimePoint>{100, 200, 300});
  CHECK(p.close[p.idx(0, 0)] == 10);
  CHECK(std::isnan(p.close[p.idx(0, 1)]));
  CHECK(p.close[p.idx(1, 1)] == 20);
  CHECK(std::isnan(p.volume[p.idx(2, 0)]));
  CHECK(p.vwap[p.idx(2, 1)] == 21);
}
