#include <doctest/doctest.h>

#include <cstdlib>

#include "core/time.hpp"
#include "market/alpaca_client.hpp"
#include "market/market_sync.hpp"
#include "test_util.hpp"

using namespace fx;

namespace {
const char* kPage1 = R"({"bars":{"AAPL":[
  {"t":"2026-09-29T04:00:00Z","o":250.1,"h":252.0,"l":249.5,"c":251.2,"v":41000000,"n":500000,"vw":250.9}],
  "NVO":[{"t":"2026-09-29T04:00:00Z","o":60.0,"h":61.0,"l":59.5,"c":60.5,"v":5000000,"n":40000,"vw":60.4}]},
  "next_page_token":"QUFQTHxE+/="})";
const char* kPage2 = R"({"bars":{"AAPL":[
  {"t":"2026-09-30T04:00:00Z","o":251.2,"h":253.0,"l":250.0,"c":252.8,"v":39000000,"n":480000,"vw":252.1}]},
  "next_page_token":null})";
AlpacaConfig test_config() {
  AlpacaConfig c;
  c.key_id = "k";
  c.secret = "s";
  c.backoff_initial_ms = 0;
  c.backoff_max_ms = 0;
  return c;
}
}  // namespace

TEST_CASE("parse_bars_page reads bars and token") {
  BarsPage p = parse_bars_page(kPage1);
  REQUIRE(p.bars.at("AAPL").size() == 1);
  CHECK(p.bars.at("AAPL")[0].t == utc_seconds(2026, 9, 29, 4, 0));
  CHECK(p.bars.at("AAPL")[0].vw == doctest::Approx(250.9));
  CHECK(p.bars.at("NVO")[0].c == doctest::Approx(60.5));
  CHECK(p.next_page_token.value() == "QUFQTHxE+/=");
  CHECK_FALSE(parse_bars_page(kPage2).next_page_token.has_value());
  CHECK(parse_bars_page(R"({"bars":{}})").bars.empty());
}

TEST_CASE("fetch_bars follows pages, URL-encodes the token and retries 429") {
  std::vector<std::string> paths;
  int calls = 0;
  HttpGet fake = [&](const std::string& path) -> HttpResponse {
    paths.push_back(path);
    ++calls;
    if (calls == 1) return {429, "rate limited"};
    if (path.find("page_token=") == std::string::npos) return {200, kPage1};
    return {200, kPage2};
  };
  AlpacaClient client(test_config(), fake);
  auto res = client.fetch_bars({"AAPL", "NVO"}, "1Day", utc_seconds(2026, 9, 29),
                               utc_seconds(2026, 10, 1));
  const auto& bars = res.bars;
  CHECK(res.stale.empty());
  CHECK(bars.at("AAPL").size() == 2);
  CHECK(bars.at("NVO").size() == 1);
  REQUIRE(paths.size() == 3);
  CHECK(paths[0].find("/v2/stocks/bars?symbols=AAPL,NVO&timeframe=1Day") == 0);
  CHECK(paths[0].find("&adjustment=all&feed=sip") != std::string::npos);
  CHECK(paths[0].find("start=2026-09-29T00:00:00Z") != std::string::npos);
  CHECK(paths[2].find("page_token=QUFQTHxE%2B%2F%3D") != std::string::npos);
}

TEST_CASE("fetch_bars marks failing symbols stale on 403 and exhausted retries") {
  int calls = 0;
  AlpacaClient forbidden(test_config(), [&](const std::string&) {
    ++calls;
    return HttpResponse{403, "no"};
  });
  auto r = forbidden.fetch_bars({"AAPL"}, "1Day", 0, 1);
  CHECK(r.bars.empty());
  CHECK(r.stale == std::vector<std::string>{"AAPL"});
  CHECK(calls == 1);

  calls = 0;
  auto cfg = test_config();
  cfg.max_retries = 6;
  AlpacaClient down(cfg, [&](const std::string&) {
    ++calls;
    return HttpResponse{503, ""};
  });
  auto r2 = down.fetch_bars({"AAPL"}, "1Day", 0, 1);
  CHECK(r2.stale == std::vector<std::string>{"AAPL"});
  CHECK(calls == 7);
}

TEST_CASE("fetch_bars treats a malformed body as a stale batch") {
  AlpacaClient bad(test_config(), [](const std::string&) { return HttpResponse{200, "not json"}; });
  auto r = bad.fetch_bars({"AAPL"}, "1Day", 0, 1);
  CHECK(r.stale == std::vector<std::string>{"AAPL"});
  AlpacaClient typed(test_config(), [](const std::string&) {
    return HttpResponse{200, R"({"bars":{"AAPL":[{"t":"2026-09-29T04:00:00Z","o":"x"}]}})"};
  });
  CHECK(typed.fetch_bars({"AAPL"}, "1Day", 0, 1).stale.size() == 1);
}

TEST_CASE("a failing second batch keeps the first batch's bars") {
  std::vector<std::string> symbols = {"AAPL"};
  for (int i = 0; i < 99; ++i) symbols.push_back("Y" + std::to_string(i));
  for (int i = 0; i < 100; ++i) symbols.push_back("Z" + std::to_string(i));
  AlpacaClient client(test_config(), [](const std::string& path) {
    if (path.find("symbols=AAPL,") != std::string::npos) return HttpResponse{200, kPage2};
    return HttpResponse{403, "denied"};
  });
  auto r = client.fetch_bars(symbols, "1Day", 0, 1);
  REQUIRE(r.bars.count("AAPL") == 1);
  CHECK(r.bars.at("AAPL").size() == 1);
  REQUIRE(r.stale.size() == 100);
  CHECK(r.stale.front() == "Z0");
  CHECK(r.stale.back() == "Z99");
}

TEST_CASE("fetch_bars batches 100 symbols per request") {
  std::vector<std::string> symbols;
  for (int i = 0; i < 250; ++i) symbols.push_back("T" + std::to_string(i));
  int calls = 0;
  AlpacaClient client(test_config(), [&](const std::string&) {
    ++calls;
    return HttpResponse{200, R"({"bars":{}})"};
  });
  client.fetch_bars(symbols, "1Day", 0, 1);
  CHECK(calls == 3);
}

TEST_CASE("sync_bars fetches incrementally and saves") {
  auto dir = test::temp_dir("sync");
  BarStore store(dir);
  std::vector<std::string> paths;
  AlpacaClient client(test_config(), [&](const std::string& path) {
    paths.push_back(path);
    return HttpResponse{200, path.find("page_token") == std::string::npos ? kPage1 : kPage2};
  });
  auto stale = sync_bars(client, store, {"AAPL", "NVO"}, Timeframe::Day, utc_seconds(2026, 9, 1),
                         utc_seconds(2026, 10, 1));
  CHECK(stale.empty());
  CHECK(store.bars("AAPL", Timeframe::Day).size() == 2);
  CHECK(std::filesystem::exists(dir / "1d" / "AAPL.csv"));

  paths.clear();
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 9, 1),
            utc_seconds(2026, 10, 2));
  REQUIRE_FALSE(paths.empty());
  CHECK(paths[0].find("start=2026-09-30T04:00:00Z") != std::string::npos);
}

TEST_CASE("sync_bars reports stale tickers and keeps the rest") {
  BarStore store(test::temp_dir("sync_stale"));
  std::vector<std::string> tickers = {"AAPL"};
  for (int i = 0; i < 99; ++i) tickers.push_back("Y" + std::to_string(i));
  for (int i = 0; i < 100; ++i) tickers.push_back("Z" + std::to_string(i));
  AlpacaClient client(test_config(), [](const std::string& path) {
    if (path.find("symbols=AAPL,") != std::string::npos)
      return HttpResponse{200, R"({"bars":{"AAPL":[{"t":"2026-09-30T04:00:00Z","o":1,"h":1,"l":1,"c":1,"v":1,"vw":1}]}})"};
    return HttpResponse{500, ""};
  });
  auto stale = sync_bars(client, store, tickers, Timeframe::Day, utc_seconds(2026, 9, 1),
                         utc_seconds(2026, 10, 1));
  CHECK(stale.size() == 100);
  CHECK(store.bars("AAPL", Timeframe::Day).size() == 1);
}

TEST_CASE("load_dotenv sets unset variables only") {
  auto dir = test::temp_dir("dotenv");
  ::setenv("FLUX_TEST_KEEP", "original", 1);
  ::unsetenv("FLUX_TEST_NEW");
  auto path = test::write_file(dir / ".env",
                               "# comment\nFLUX_TEST_NEW=\"hello\"\nFLUX_TEST_KEEP=changed\n");
  load_dotenv(path);
  CHECK(std::string(std::getenv("FLUX_TEST_NEW")) == "hello");
  CHECK(std::string(std::getenv("FLUX_TEST_KEEP")) == "original");
}

TEST_CASE("load_dotenv trims trailing whitespace before unquoting") {
  for (const char* k : {"FLUX_TEST_TRIM", "FLUX_TEST_TAB", "FLUX_TEST_QUOTED"}) ::unsetenv(k);
  auto path = test::write_file(test::temp_dir("dotenv_trim") / ".env",
                               "FLUX_TEST_TRIM=sip \nFLUX_TEST_TAB=iex\t\nFLUX_TEST_QUOTED=\"a b\" \n");
  load_dotenv(path);
  CHECK(std::string(std::getenv("FLUX_TEST_TRIM")) == "sip");
  CHECK(std::string(std::getenv("FLUX_TEST_TAB")) == "iex");
  CHECK(std::string(std::getenv("FLUX_TEST_QUOTED")) == "a b");
}
