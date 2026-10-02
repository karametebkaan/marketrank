#include <doctest/doctest.h>

#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "market/asset_universe.hpp"
#include "test_util.hpp"

using namespace fx;

TEST_CASE("parse_assets reads symbols, names, exchanges and tradable flags") {
  auto a = parse_assets(R"([
    {"symbol":"AAPL","name":"Apple Inc. Common Stock","exchange":"NASDAQ","tradable":true,"status":"active"},
    {"symbol":"XYZW","name":null,"exchange":"NYSE","tradable":false},
    {"symbol":"BRK.B","name":"Berkshire Hathaway Inc.","exchange":"NYSE","tradable":true},
    {"name":"no symbol"}])");
  REQUIRE(a.size() == 3);
  CHECK(a[0].symbol == "AAPL");
  CHECK(a[0].exchange == "NASDAQ");
  CHECK(a[0].tradable);
  CHECK(a[1].name.empty());
  CHECK_FALSE(a[1].tradable);
  CHECK_THROWS_AS(parse_assets("{}"), std::runtime_error);
  CHECK_THROWS_AS(parse_assets("not json"), std::runtime_error);
}

TEST_CASE("universe rules filter exchanges, warrants, units, rights and ETF-like names") {
  UniverseRules rules;
  auto ok = [&](std::string sym, std::string name, std::string exch, bool tradable = true) {
    return passes_universe_rules({sym, name, exch, tradable}, rules);
  };
  CHECK(ok("AAPL", "Apple Inc. Common Stock", "NASDAQ"));
  CHECK(ok("FRT", "Federal Realty Investment Trust", "NYSE"));
  CHECK(ok("UNIT", "Uniti Group Inc.", "NASDAQ"));
  CHECK_FALSE(ok("ACMW", "Acme Corp Warrant", "NASDAQ"));
  CHECK_FALSE(ok("ACMU", "Acme Acquisition Corp Units", "NASDAQ"));
  CHECK_FALSE(ok("ACMR", "Acme Acquisition Corp Rights", "NASDAQ"));
  CHECK_FALSE(ok("SPY", "SPDR S&P 500 ETF Trust", "ARCA"));
  CHECK_FALSE(ok("TQQQ", "ProShares UltraPro QQQ", "NASDAQ"));
  CHECK_FALSE(ok("ABCD", "Abcd Holdings", "OTC"));
  CHECK_FALSE(ok("AAPL", "Apple Inc.", "NASDAQ", false));
  rules.always_include = {"SPY"};
  CHECK(ok("SPY", "SPDR S&P 500 ETF Trust", "ARCA"));
  rules.exclude = {"AAPL"};
  CHECK_FALSE(ok("AAPL", "Apple Inc. Common Stock", "NASDAQ"));
}

TEST_CASE("rank_by_liquidity uses the median dollar volume of the last bars") {
  BarStore store(test::temp_dir("rank"));
  auto bar = [](TimePoint t, double v, double vw) { return Bar{t, 1, 1, 1, 1, v, vw}; };
  store.merge("BIG", Timeframe::Day, {bar(1, 1000, 10), bar(2, 10, 10), bar(3, 900, 10)});
  store.merge("MID", Timeframe::Day, {bar(1, 1, 10), bar(2, 300, 10), bar(3, 300, 10)});
  store.merge("SMALL", Timeframe::Day, {bar(3, 10, 10)});
  std::vector<AssetInfo> assets = {{"SMALL", "", "NYSE", true},
                                   {"MID", "", "NYSE", true},
                                   {"BIG", "", "NYSE", true},
                                   {"NONE", "", "NYSE", true}};
  auto ranked = rank_by_liquidity(assets, store, 2, 2);
  REQUIRE(ranked.size() == 2);
  CHECK(ranked[0].asset.symbol == "BIG");
  CHECK(ranked[0].median_dollar_volume == doctest::Approx(4550.0));
  CHECK(ranked[1].asset.symbol == "MID");
  CHECK(rank_by_liquidity(assets, store, 2, 10).size() == 3);
}

TEST_CASE("snapshots round-trip and the latest one is found") {
  auto dir = test::temp_dir("snap");
  Universe sp = Universe::from_securities({{"AAPL", "Apple Inc.", "Information Technology"}});
  std::vector<RankedAsset> ranked = {{{"AAPL", "Apple Inc.", "NASDAQ", true}, 5e9},
                                     {{"ZZZ", "Zeta, Inc.", "NYSE", true}, 1e6}};
  write_universe_snapshot(dir / "universe_2026-09-01.csv", ranked, sp);
  write_universe_snapshot(dir / "universe_2026-10-01.csv", ranked, sp);
  test::write_file(dir / "funds.csv", "ticker,tracks\nVOO,sp500\n");
  auto latest = latest_snapshot(dir);
  REQUIRE(latest.has_value());
  CHECK(latest->filename() == "universe_2026-10-01.csv");
  CHECK(snapshot_date(*latest).value() == "2026-10-01");
  CHECK_FALSE(snapshot_date(dir / "funds.csv").has_value());
  Universe u = load_snapshot(*latest, dir / "funds.csv");
  REQUIRE(u.nodes().size() == 2);
  CHECK(u.nodes()[0].sector == "Information Technology");
  CHECK(u.nodes()[1].name == "Zeta, Inc.");
  CHECK(u.nodes()[1].sector == "Unclassified");
  CHECK(u.is_fund("VOO"));
  CHECK_FALSE(latest_snapshot(dir / "missing").has_value());
  CHECK_FALSE(std::filesystem::exists(dir / "universe_2026-10-01.csv.tmp"));
}

TEST_CASE("read_ticker_list skips comments and blanks; a missing file is empty") {
  auto dir = test::temp_dir("tickers");
  auto p = test::write_file(dir / "x.csv", "# header\nAAPL\n\n  MSFT \r\n");
  CHECK(read_ticker_list(p) == std::set<std::string>{"AAPL", "MSFT"});
  CHECK(read_ticker_list(dir / "none.csv").empty());
}

TEST_CASE("snapshot write failure throws and leaves no tmp file") {
  auto dir = test::temp_dir("snapfail");
  auto blocker = test::write_file(dir / "file", "x");
  Universe sp = Universe::from_securities({});
  std::vector<RankedAsset> ranked = {{{"AAPL", "Apple", "NASDAQ", true}, 1.0}};
  CHECK_THROWS(write_universe_snapshot(blocker / "universe_2026-10-01.csv", ranked, sp));
  CHECK_FALSE(std::filesystem::exists(blocker / "universe_2026-10-01.csv.tmp"));
}

TEST_CASE("snapshot names carry the date and, in the new form, the requested size") {
  CHECK(snapshot_date("universe_2026-10-01.csv").value() == "2026-10-01");
  CHECK(snapshot_date("dir/universe_2026-10-01_n500.csv").value() == "2026-10-01");
  CHECK_FALSE(snapshot_size("universe_2026-10-01.csv").has_value());
  CHECK(snapshot_size("dir/universe_2026-10-01_n500.csv").value() == 500);
  CHECK(snapshot_size("universe_2026-10-01_n10000.csv").value() == 10000);
  CHECK_FALSE(snapshot_date("universe_2026-10-01_n.csv").has_value());
  CHECK_FALSE(snapshot_size("universe_2026-10-01_nx.csv").has_value());
  CHECK_FALSE(snapshot_size("funds.csv").has_value());

  auto dir = test::temp_dir("snapnames");
  Universe sp = Universe::from_securities({});
  std::vector<RankedAsset> ranked = {{{"AAPL", "Apple", "NASDAQ", true}, 1.0}};
  write_universe_snapshot(dir / "universe_2026-09-01.csv", ranked, sp);
  write_universe_snapshot(dir / "universe_2026-09-15_n500.csv", ranked, sp);
  CHECK(latest_snapshot(dir)->filename() == "universe_2026-09-15_n500.csv");
  write_universe_snapshot(dir / "universe_2026-10-01.csv", ranked, sp);
  CHECK(latest_snapshot(dir)->filename() == "universe_2026-10-01.csv");
  write_universe_snapshot(dir / "universe_2026-10-02_n2000.csv", ranked, sp);
  CHECK(latest_snapshot(dir)->filename() == "universe_2026-10-02_n2000.csv");
}

TEST_CASE("find_snapshot picks the newest fresh file of exactly the requested size") {
  auto dir = test::temp_dir("findsnap");
  Universe sp = Universe::from_securities({});
  std::vector<RankedAsset> ranked = {{{"AAPL", "Apple", "NASDAQ", true}, 1.0}};
  const TimePoint now = 1790812800;  // 2026-10-01
  CHECK_FALSE(find_snapshot(dir / "missing", 500, now, 7).has_value());
  write_universe_snapshot(dir / "universe_2026-10-01_n500.csv", ranked, sp);
  write_universe_snapshot(dir / "universe_2026-10-01_n2000.csv", ranked, sp);
  write_universe_snapshot(dir / "universe_2026-09-28_n500.csv", ranked, sp);
  write_universe_snapshot(dir / "universe_2026-09-20_n3000.csv", ranked, sp);  // too old
  write_universe_snapshot(dir / "universe_2026-10-01.csv", ranked, sp);        // no size
  CHECK(find_snapshot(dir, 500, now, 7)->filename() == "universe_2026-10-01_n500.csv");
  CHECK(find_snapshot(dir, 2000, now, 7)->filename() == "universe_2026-10-01_n2000.csv");
  CHECK_FALSE(find_snapshot(dir, 3000, now, 7).has_value());  // 11 days old
  CHECK_FALSE(find_snapshot(dir, 1000, now, 7).has_value());
  CHECK(find_snapshot(dir, 3000, now, 30)->filename() == "universe_2026-09-20_n3000.csv");
  // Exactly max_age_days old is stale; one day younger is fresh.
  write_universe_snapshot(dir / "universe_2026-09-24_n700.csv", ranked, sp);
  write_universe_snapshot(dir / "universe_2026-09-25_n800.csv", ranked, sp);
  CHECK_FALSE(find_snapshot(dir, 700, now, 7).has_value());
  CHECK(find_snapshot(dir, 800, now, 7).has_value());
}
