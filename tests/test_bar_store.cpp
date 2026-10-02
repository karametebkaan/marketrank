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

TEST_CASE("merge-only stores never touch disk") {
  auto dir = test::temp_dir("bars_mem");
  {
    BarStore s(dir / "lake");
    s.merge("AAPL", Timeframe::Day, {{100, 1, 1, 1, 1, 10, 1}});
  }
  CHECK_FALSE(std::filesystem::exists(dir / "lake"));
}

TEST_CASE("saved bars and coverage persist losslessly through the lake") {
  auto dir = test::temp_dir("bars_lake");
  {
    BarStore s(dir);
    s.merge("BRK.B", Timeframe::Hour, {{1759239000, 470.25, 471.5, 469.0, 0.1 + 0.2, 123456, 1.0 / 3.0}});
    s.save("BRK.B", Timeframe::Hour);
    s.set_covered_from("BRK.B", Timeframe::Hour, 1759000000);
    s.flush();
    s.merge("BRK.B", Timeframe::Hour, {{1759239000, 1, 1, 1, 9.5, 1, 1}});  // upsert, saved by destructor
    s.save("BRK.B", Timeframe::Hour);
  }
  BarStore s2(dir);
  s2.load_all({"BRK.B", "MISSING"}, Timeframe::Hour);
  const auto& b = s2.bars("BRK.B", Timeframe::Hour);
  REQUIRE(b.size() == 1);
  CHECK(b[0].c == 9.5);
  CHECK(s2.covered_from("BRK.B", Timeframe::Hour).value() == 1759000000);
  CHECK(s2.bars("MISSING", Timeframe::Hour).empty());
}

TEST_CASE("load_range only loads the window and panel windows and carries OHLC") {
  auto dir = test::temp_dir("bars_window");
  {
    BarStore s(dir);
    s.merge("A", Timeframe::Day, {{100, 9, 12, 8, 10, 5, 10}, {200, 10, 13, 9, 11, 6, 11}, {300, 11, 14, 10, 12, 7, 12}});
    s.save("A", Timeframe::Day);
  }
  BarStore s2(dir);
  s2.load_range({"A"}, Timeframe::Day, 150, 400);
  CHECK(s2.bars("A", Timeframe::Day).size() == 2);
  Panel p = build_panel(s2, {"A"}, Timeframe::Day, 250, 400);
  REQUIRE(p.T() == 1);
  CHECK(p.open[p.idx(0, 0)] == 11);
  CHECK(p.high[p.idx(0, 0)] == 14);
  CHECK(p.low[p.idx(0, 0)] == 10);
  CHECK(p.close[p.idx(0, 0)] == 12);
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

TEST_CASE("unsaved series are not persisted") {
  auto dir = test::temp_dir("bars_unsaved");
  {
    BarStore s(dir);
    s.merge("A", Timeframe::Day, {{100, 1, 1, 1, 1, 1, 1}});
    s.merge("B", Timeframe::Day, {{100, 2, 2, 2, 2, 2, 2}});
    s.save("A", Timeframe::Day);
    s.flush();
  }
  BarStore s2(dir);
  s2.load_all({"A", "B"}, Timeframe::Day);
  CHECK(s2.bars("A", Timeframe::Day).size() == 1);
  CHECK(s2.bars("B", Timeframe::Day).empty());
}

TEST_CASE("repeated merges of one bar persist the last value") {
  auto dir = test::temp_dir("bars_lastwins");
  {
    BarStore s(dir);
    s.merge("A", Timeframe::Day, {{100, 1, 1, 1, 10, 1, 1}});
    s.merge("A", Timeframe::Day, {{100, 1, 1, 1, 12, 1, 1}});
    s.save("A", Timeframe::Day);
    s.flush();
  }
  BarStore s2(dir);
  s2.load_all({"A"}, Timeframe::Day);
  REQUIRE(s2.bars("A", Timeframe::Day).size() == 1);
  CHECK(s2.bars("A", Timeframe::Day)[0].c == 12);
}

TEST_CASE("destructor flushes saved bars") {
  auto dir = test::temp_dir("bars_dtor");
  {
    BarStore s(dir);
    s.merge("A", Timeframe::Day, {{100, 1, 1, 1, 1, 1, 1}});
    s.save("A", Timeframe::Day);
  }
  BarStore s2(dir);
  s2.load_all({"A"}, Timeframe::Day);
  CHECK(s2.bars("A", Timeframe::Day).size() == 1);
}

TEST_CASE("coverage persists without any bars") {
  auto dir = test::temp_dir("bars_cov_only");
  {
    BarStore s(dir);
    s.set_covered_from("A", Timeframe::Day, 77);
    s.flush();
  }
  BarStore s2(dir);
  s2.load_all({"A"}, Timeframe::Day);
  CHECK(s2.covered_from("A", Timeframe::Day).value() == 77);
  CHECK(s2.bars("A", Timeframe::Day).empty());
}

TEST_CASE("load_range queues nothing") {
  auto dir = test::temp_dir("bars_noqueue");
  {
    BarStore s(dir);
    s.merge("A", Timeframe::Day, {{100, 1, 1, 1, 1, 1, 1}});
    s.save("A", Timeframe::Day);
  }
  BarStore s2(dir);
  s2.load_range({"A"}, Timeframe::Day, 0, 1000);
  const auto before = s2.lake().file_count(Timeframe::Day);
  s2.flush();
  CHECK(s2.lake().file_count(Timeframe::Day) == before);
  CHECK(s2.bars("A", Timeframe::Day).size() == 1);
}

TEST_CASE("a corrupt lake file does not abort a load") {
  auto dir = test::temp_dir("bars_corrupt");
  {
    BarStore s(dir);
    s.merge("A", Timeframe::Day, {{1759239000, 1, 1, 1, 1, 1, 1}});
    s.save("A", Timeframe::Day);
  }
  std::filesystem::path part;
  for (const auto& e : std::filesystem::recursive_directory_iterator(dir))
    if (e.path().extension() == ".parquet") part = e.path().parent_path();
  REQUIRE_FALSE(part.empty());
  test::write_file(part / "part-x.parquet", "this is not parquet");
  BarStore s2(dir);
  CHECK_NOTHROW(s2.load_range({"A"}, Timeframe::Day, 0, 2000000000));
}
