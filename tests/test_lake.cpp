#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>

#include "core/time.hpp"
#include "storage/lake.hpp"
#include "test_util.hpp"

using namespace fx;

namespace {
LakeRow row(const std::string& ticker, TimePoint t, double c) {
  return {ticker, Bar{t, c - 1, c + 1, c - 2, c, 1000 + c, c + 0.5}};
}
}  // namespace

TEST_CASE("lake round-trips rows losslessly and filters by ticker and window") {
  auto dir = test::temp_dir("lake_rt");
  const TimePoint d0 = utc_seconds(2026, 9, 1, 4);
  {
    Lake lake(dir);
    std::vector<LakeRow> rows = {row("AAPL", d0, 0.1 + 0.2), row("AAPL", d0 + 86400, 1.0 / 3.0),
                                 row("BRK.B", d0, 470.25), row("MSFT", d0 + 40 * 86400, 5.0)};
    lake.write(Timeframe::Day, rows, {});
  }
  Lake lake(dir);
  auto got = lake.read(Timeframe::Day, {"AAPL", "BRK.B"}, d0, d0 + 86400);
  REQUIRE(got.size() == 2);
  REQUIRE(got["AAPL"].size() == 2);
  CHECK(got["AAPL"][0].c == 0.1 + 0.2);  // exact
  CHECK(got["AAPL"][1].c == 1.0 / 3.0);
  CHECK(got["AAPL"][1].vw == 1.0 / 3.0 + 0.5);
  CHECK(got["BRK.B"][0].t == d0);
  CHECK(lake.read(Timeframe::Day, {"MSFT"}, d0, d0 + 86400).empty());
  CHECK(lake.read(Timeframe::Day, {"MSFT"}, d0, d0 + 50 * 86400)["MSFT"].size() == 1);
  CHECK(lake.read(Timeframe::Hour, {"AAPL"}, d0, d0 + 86400).empty());
}

TEST_CASE("newest write wins for the same ticker and time") {
  auto dir = test::temp_dir("lake_upsert");
  const TimePoint d0 = utc_seconds(2026, 9, 1, 4);
  Lake lake(dir);
  lake.write(Timeframe::Day, {row("AAPL", d0, 10)}, {});
  lake.write(Timeframe::Day, {row("AAPL", d0, 11)}, {});
  auto got = lake.read(Timeframe::Day, {"AAPL"}, d0, d0);
  REQUIRE(got["AAPL"].size() == 1);
  CHECK(got["AAPL"][0].c == 11);
}

TEST_CASE("coverage only moves earlier and persists across reopen") {
  auto dir = test::temp_dir("lake_cov");
  {
    Lake lake(dir);
    lake.write(Timeframe::Day, {}, {{"AAPL", 500}});
    lake.write(Timeframe::Day, {}, {{"AAPL", 900}, {"NVO", 700}});
  }
  Lake lake(dir);
  auto cov = lake.coverage(Timeframe::Day, {"AAPL", "NVO", "MSFT"});
  CHECK(cov.at("AAPL") == 500);
  CHECK(cov.at("NVO") == 700);
  CHECK_FALSE(cov.count("MSFT"));
  CHECK(lake.coverage(Timeframe::Hour, {"AAPL"}).empty());
}

TEST_CASE("leftover staging is removed on open") {
  auto dir = test::temp_dir("lake_staging");
  { Lake lake(dir); }
  test::write_file(dir / "_staging" / "junk" / "half.parquet", "not parquet");
  Lake lake(dir);
  CHECK_FALSE(std::filesystem::exists(dir / "_staging" / "junk"));
  CHECK(lake.read(Timeframe::Day, {"AAPL"}, 0, 1LL << 40).empty());
}

TEST_CASE("compaction merges partition files and keeps the newest rows") {
  auto dir = test::temp_dir("lake_compact");
  const TimePoint d0 = utc_seconds(2026, 9, 1, 4);
  Lake lake(dir);
  for (int b = 0; b < 5; ++b) lake.write(Timeframe::Day, {row("AAPL", d0, 10 + b), row("F", d0 + b * 86400, 3)}, {});
  CHECK(lake.file_count(Timeframe::Day) == 5);
  CHECK(lake.compact(Timeframe::Day, 2) == 1);
  CHECK(lake.file_count(Timeframe::Day) == 1);
  auto got = lake.read(Timeframe::Day, {"AAPL", "F"}, d0, d0 + 10 * 86400);
  CHECK(got["AAPL"].size() == 1);
  CHECK(got["AAPL"][0].c == 14);
  CHECK(got["F"].size() == 5);
  CHECK(lake.compact(Timeframe::Day, 2) == 0);
}

TEST_CASE("retention drops whole months older than the cutoff only") {
  auto dir = test::temp_dir("lake_ret");
  Lake lake(dir);
  lake.write(Timeframe::Hour, {row("AAPL", utc_seconds(2024, 1, 15, 15), 1), row("AAPL", utc_seconds(2026, 9, 2, 15), 2)}, {});
  lake.write(Timeframe::Day, {row("AAPL", utc_seconds(2020, 1, 15, 4), 1)}, {});
  RetentionPolicy p = RetentionPolicy::defaults();
  p.keep_days[Timeframe::Hour] = 30;
  CHECK(lake.apply_retention(p, utc_seconds(2026, 10, 1)) == 1);
  auto hours = lake.read(Timeframe::Hour, {"AAPL"}, 0, 1LL << 40);
  REQUIRE(hours["AAPL"].size() == 1);
  CHECK(hours["AAPL"][0].c == 2);
  CHECK(lake.read(Timeframe::Day, {"AAPL"}, 0, 1LL << 40)["AAPL"].size() == 1);
}

TEST_CASE("retention policy file") {
  auto dir = test::temp_dir("lake_policy");
  auto d = RetentionPolicy::defaults();
  CHECK(d.keep_days.at(Timeframe::Hour).value() == 730);
  CHECK_FALSE(d.keep_days.at(Timeframe::Day).has_value());
  CHECK(RetentionPolicy::load(dir / "missing.json").keep_days.at(Timeframe::Hour).value() == 730);
  auto p = RetentionPolicy::load(test::write_file(dir / "r.json", R"({"1h": 90, "1w": 3650})"));
  CHECK(p.keep_days.at(Timeframe::Hour).value() == 90);
  CHECK(p.keep_days.at(Timeframe::Week).value() == 3650);
  CHECK_FALSE(p.keep_days.at(Timeframe::Day).has_value());
  CHECK_THROWS_AS(RetentionPolicy::load(test::write_file(dir / "bad.json", "{")), std::runtime_error);
}

TEST_CASE("duplicate rows inside one batch resolve to the last, also after compaction") {
  auto dir = test::temp_dir("lake_dup");
  const TimePoint d0 = utc_seconds(2026, 9, 1, 4);
  Lake lake(dir);
  lake.write(Timeframe::Day, {row("AAPL", d0, 10), row("AAPL", d0, 12)}, {});
  CHECK(lake.read(Timeframe::Day, {"AAPL"}, d0, d0)["AAPL"][0].c == 12);
  for (int b = 0; b < 3; ++b) lake.write(Timeframe::Day, {row("F", d0 + b * 86400, 3)}, {});
  CHECK(lake.compact(Timeframe::Day, 2) == 1);
  auto got = lake.read(Timeframe::Day, {"AAPL"}, d0, d0);
  REQUIRE(got["AAPL"].size() == 1);
  CHECK(got["AAPL"][0].c == 12);
}

TEST_CASE("fsync_path syncs an existing file and directory and throws on a missing path") {
  auto dir = test::temp_dir("lake_fsync");
  const auto f = test::write_file(dir / "a.bin", "data");
  CHECK_NOTHROW(fsync_path(f));
  CHECK_NOTHROW(fsync_path(dir));
  CHECK_THROWS_AS(fsync_path(dir / "missing.bin"), std::runtime_error);
}

TEST_CASE("an unreadable Parquet file is quarantined and the rest of the timeframe still reads") {
  namespace fs = std::filesystem;
  auto dir = test::temp_dir("lake_quarantine");
  const TimePoint d0 = utc_seconds(2026, 9, 1, 4);
  Lake lake(dir);
  lake.write(Timeframe::Day, {row("AAPL", d0, 10), row("F", d0 + 86400, 3)}, {{"AAPL", d0}, {"F", d0}});
  lake.write(Timeframe::Hour, {row("AAPL", d0 + 3600, 7)}, {{"AAPL", d0}});
  fs::path part;
  for (const auto& e : fs::recursive_directory_iterator(dir / "bars" / "tf=1d"))
    if (e.path().extension() == ".parquet") part = e.path().parent_path();
  REQUIRE_FALSE(part.empty());
  const fs::path bad = part / "part-x.parquet";
  test::write_file(bad, "this is not parquet");
  auto got = lake.read(Timeframe::Day, {"AAPL", "F"}, d0, d0 + 10 * 86400);
  REQUIRE(got["AAPL"].size() == 1);
  CHECK(got["AAPL"][0].c == 10);
  CHECK(got["F"].size() == 1);
  CHECK_FALSE(fs::exists(bad));
  CHECK(fs::exists(dir / "_quarantine" / fs::relative(bad, dir)));
  CHECK(lake.coverage(Timeframe::Day, {"AAPL", "F"}).empty());
  CHECK(lake.coverage(Timeframe::Hour, {"AAPL"}).at("AAPL") == d0);  // other timeframes untouched
  CHECK(lake.read(Timeframe::Hour, {"AAPL"}, d0, d0 + 86400)["AAPL"].size() == 1);
}

TEST_CASE("complete marks only move later and persist across reopen") {
  auto dir = test::temp_dir("lake_complete");
  {
    Lake lake(dir);
    lake.write(Timeframe::Day, {}, {}, {{"AAPL", 900}});
    lake.write(Timeframe::Day, {}, {}, {{"AAPL", 500}, {"NVO", 700}});
  }
  Lake lake(dir);
  auto c = lake.complete(Timeframe::Day, {"AAPL", "NVO", "MSFT"});
  CHECK(c.at("AAPL") == 900);
  CHECK(c.at("NVO") == 700);
  CHECK_FALSE(c.count("MSFT"));
  CHECK(lake.complete(Timeframe::Hour, {"AAPL"}).empty());
}
