#include <doctest/doctest.h>

#include "market/bar_store.hpp"
#include "storage/csv_migration.hpp"
#include "test_util.hpp"

using namespace mr;

TEST_CASE("migrate_csv_cache imports bars and .from coverage, skipping bad rows") {
  auto dir = test::temp_dir("migrate");
  test::write_file(dir / "csv" / "1d" / "AAPL.csv", "t,o,h,l,c,v,vw\n100,1,2,0.5,1.5,10,1.4\nbad,row\n200,1,2,0.5,1.75,11,1.6\n");
  test::write_file(dir / "csv" / "1d" / "AAPL.from", "50\n");
  test::write_file(dir / "csv" / "1d" / "JUNK.csv", "wrong,header\n1,2\n");
  {
    BarStore s(dir / "lake");
    CHECK(migrate_csv_cache(dir / "csv", s) == 1);
  }
  BarStore s(dir / "lake");
  s.load_all({"AAPL", "JUNK"}, Timeframe::Day);
  REQUIRE(s.bars("AAPL", Timeframe::Day).size() == 2);
  CHECK(s.bars("AAPL", Timeframe::Day)[1].c == 1.75);
  CHECK(s.covered_from("AAPL", Timeframe::Day).value() == 50);
  CHECK(s.bars("JUNK", Timeframe::Day).empty());
  CHECK(migrate_csv_cache(dir / "nothing", s) == 0);
}

TEST_CASE("migrate_csv_cache skips malformed rows and keeps the good ones") {
  auto dir = test::temp_dir("migrate_row");
  test::write_file(dir / "csv" / "1d" / "AAA.csv",
                   "t,o,h,l,c,v,vw\n100,1,1,1,1,10,1\nnot,a,row,at,all,x,y\n200,2,2,2,2,20,2\n"
                   "300,3,3\n");
  {
    BarStore s(dir / "lake");
    CHECK(migrate_csv_cache(dir / "csv", s) == 1);
  }
  BarStore s(dir / "lake");
  s.load_all({"AAA"}, Timeframe::Day);
  const auto& b = s.bars("AAA", Timeframe::Day);
  REQUIRE(b.size() == 2);
  CHECK(b[0].t == 100);
  CHECK(b[1].t == 200);
}

TEST_CASE("migrate_csv_cache skips files with a wrong header and leaves no temp files") {
  auto dir = test::temp_dir("migrate_header");
  test::write_file(dir / "csv" / "1d" / "BAD.csv", "time,open\n100,1,1,1,1,10,1\n");
  test::write_file(dir / "csv" / "1d" / "EMPTY.csv", "");
  {
    BarStore s(dir / "lake");
    CHECK(migrate_csv_cache(dir / "csv", s) == 0);
  }
  if (std::filesystem::exists(dir / "lake"))
    for (const auto& e : std::filesystem::recursive_directory_iterator(dir / "lake"))
      CHECK(e.path().extension() != ".tmp");
  BarStore s(dir / "lake");
  s.load_all({"BAD", "EMPTY"}, Timeframe::Day);
  CHECK(s.bars("BAD", Timeframe::Day).empty());
  CHECK(s.bars("EMPTY", Timeframe::Day).empty());
}

TEST_CASE("migrate_csv_cache imports nothing from a file with only malformed rows") {
  auto dir = test::temp_dir("migrate_allbad");
  test::write_file(dir / "csv" / "1d" / "ZZZ.csv", "t,o,h,l,c,v,vw\nnot,a,row,at,all,x,y\n1,2,3\n");
  BarStore s(dir / "lake");
  CHECK(migrate_csv_cache(dir / "csv", s) == 0);
}
