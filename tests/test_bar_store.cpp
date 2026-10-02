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

TEST_CASE("load_all skips malformed rows and keeps the good ones") {
  auto dir = test::temp_dir("bars_corrupt_row");
  test::write_file(dir / "1d" / "AAA.csv",
                   "t,o,h,l,c,v,vw\n100,1,1,1,1,10,1\nnot,a,row,at,all,x,y\n200,2,2,2,2,20,2\n"
                   "300,3,3\n");
  BarStore s(dir);
  CHECK_NOTHROW(s.load_all({"AAA"}, Timeframe::Day));
  const auto& b = s.bars("AAA", Timeframe::Day);
  REQUIRE(b.size() == 2);
  CHECK(b[0].t == 100);
  CHECK(b[1].t == 200);
}

TEST_CASE("load_all skips files with a wrong header") {
  auto dir = test::temp_dir("bars_bad_header");
  test::write_file(dir / "1d" / "BAD.csv", "time,open\n100,1,1,1,1,10,1\n");
  test::write_file(dir / "1d" / "EMPTY.csv", "");
  BarStore s(dir);
  CHECK_NOTHROW(s.load_all({"BAD", "EMPTY"}, Timeframe::Day));
  CHECK(s.bars("BAD", Timeframe::Day).empty());
  CHECK(s.bars("EMPTY", Timeframe::Day).empty());
}

TEST_CASE("save is atomic and leaves no temp file") {
  auto dir = test::temp_dir("bars_atomic");
  BarStore s(dir);
  s.merge("AAPL", Timeframe::Day, {{100, 1, 1, 1, 1, 10, 1}});
  s.save("AAPL", Timeframe::Day);
  CHECK(std::filesystem::exists(dir / "1d" / "AAPL.csv"));
  CHECK_FALSE(std::filesystem::exists(dir / "1d" / "AAPL.csv.tmp"));
}
