#include <doctest/doctest.h>

#include "core/csv.hpp"
#include "market/universe.hpp"
#include "test_util.hpp"

using namespace fx;

TEST_CASE("parse_csv handles quotes, escaped quotes, CRLF and blank lines") {
  auto rows = parse_csv("a,b,c\r\n\"x, y\",\"say \"\"hi\"\"\",z\n\nlast,,\n");
  REQUIRE(rows.size() == 3);
  CHECK(rows[1][0] == "x, y");
  CHECK(rows[1][1] == "say \"hi\"");
  CHECK(rows[2].size() == 3);
  CHECK(rows[2][1] == "");
}

TEST_CASE("universe loads, adds non-fund extras, indexes tickers") {
  auto dir = test::temp_dir("universe");
  auto sp = test::write_file(dir / "sp500.csv",
                             "ticker,name,sector\nAAPL,Apple Inc.,Information Technology\n"
                             "NKE,\"Nike, Inc.\",Consumer Discretionary\nF,Ford Motor Company,"
                             "Consumer Discretionary\n");
  auto fu = test::write_file(dir / "funds.csv", "ticker,tracks\nVOO,sp500\n");
  Universe u = Universe::load(sp, fu);
  CHECK(u.nodes().size() == 3);
  CHECK(u.nodes()[1].name == "Nike, Inc.");

  PortfolioSpec p{1e6, "2025-10-01", {{"AAPL", 0.6}, {"VOO", 0.15}, {"NVO", 0.25}}};
  u.add_extras(p);
  REQUIRE(u.nodes().size() == 4);
  CHECK(u.nodes()[3].ticker == "NVO");
  CHECK(u.nodes()[3].sector == "Extra");
  CHECK(u.index_of("NVO").value() == 3);
  CHECK_FALSE(u.index_of("VOO").has_value());
  CHECK(u.is_fund("VOO"));
  CHECK(u.price_tickers().size() == 5);
  u.add_extras(p);  // idempotent
  CHECK(u.nodes().size() == 4);
}

TEST_CASE("load_portfolio validates weights") {
  auto dir = test::temp_dir("portfolio");
  auto ok = test::write_file(dir / "ok.json",
                             R"({"initial_cash":1000000,"inception":"2025-10-01",
                                 "holdings":[{"ticker":"AAPL","weight":0.6},
                                             {"ticker":"VOO","weight":0.4}]})");
  PortfolioSpec p = load_portfolio(ok);
  CHECK(p.initial_cash == 1000000);
  CHECK(p.holdings.size() == 2);
  auto bad = test::write_file(dir / "bad.json",
                              R"({"initial_cash":1,"inception":"x",
                                  "holdings":[{"ticker":"AAPL","weight":0.6}]})");
  CHECK_THROWS_AS(load_portfolio(bad), std::runtime_error);
}

TEST_CASE("bundled data files are consistent") {
  Universe u = Universe::load(FLUX_SOURCE_DIR "/data/universe/sp500.csv",
                              FLUX_SOURCE_DIR "/data/universe/funds.csv");
  CHECK(u.nodes().size() >= 490);
  CHECK(u.nodes().size() <= 510);
  CHECK(u.index_of("AAPL").has_value());
  PortfolioSpec p = load_portfolio(FLUX_SOURCE_DIR "/data/portfolio.json");
  u.add_extras(p);
  CHECK(u.index_of("NVO").has_value());
  CHECK(u.is_fund("VOO"));
}
