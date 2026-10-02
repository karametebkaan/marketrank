#include <doctest/doctest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

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
  c.min_request_interval_ms = 0;
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
  {
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

  paths.clear();
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 9, 1),
            utc_seconds(2026, 10, 2));
  REQUIRE_FALSE(paths.empty());
  bool tail = false;
  for (const auto& p : paths) tail = tail || p.find("start=2026-09-30T04:00:00Z") != std::string::npos;
  CHECK(tail);
  }
  BarStore reload(dir);
  reload.load_all({"AAPL"}, Timeframe::Day);
  CHECK(reload.bars("AAPL", Timeframe::Day).size() == 2);
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

TEST_CASE("sync_bars back-fills history before the first cached bar") {
  BarStore store(test::temp_dir("backfill"));
  store.merge("AAPL", Timeframe::Day, {{utc_seconds(2026, 9, 29, 4), 1, 1, 1, 1, 1, 1}});
  CHECK(store.first_time("AAPL", Timeframe::Day).value() == utc_seconds(2026, 9, 29, 4));
  std::vector<std::string> paths;
  AlpacaClient client(test_config(), [&](const std::string& path) {
    paths.push_back(path);
    return HttpResponse{200, R"({"bars":{}})"};
  });
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 9, 1), utc_seconds(2026, 10, 1));
  REQUIRE(paths.size() == 2);
  CHECK(paths[0].find("start=2026-09-01T00:00:00Z") != std::string::npos);
  CHECK(paths[0].find("end=2026-09-29T04:00:00Z") != std::string::npos);
  CHECK(paths[1].find("start=2026-09-29T04:00:00Z") != std::string::npos);

  paths.clear();
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 9, 27), utc_seconds(2026, 10, 1));
  CHECK(paths.size() == 1);  // first bar within tolerance of start: no back-fill
}

TEST_CASE("client spaces requests by min_request_interval_ms and get returns the body") {
  AlpacaConfig c = test_config();
  c.min_request_interval_ms = 40;
  int calls = 0;
  AlpacaClient client(c, [&](const std::string&) {
    ++calls;
    return HttpResponse{200, "[]"};
  });
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 3; ++i) CHECK(client.get("/v2/assets") == "[]");
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  CHECK(calls == 3);
  CHECK(ms >= 80);
  AlpacaClient bad(test_config(), [](const std::string&) { return HttpResponse{403, "no"}; });
  CHECK_THROWS_AS(bad.get("/v2/assets"), std::runtime_error);
}

TEST_CASE("covered history is not re-requested; an earlier start back-fills again") {
  BarStore store(test::temp_dir("covered"));
  store.merge("AAPL", Timeframe::Day, {{utc_seconds(2026, 9, 29, 4), 1, 1, 1, 1, 1, 1}});
  std::vector<std::string> paths;
  AlpacaClient client(test_config(), [&](const std::string& path) {
    paths.push_back(path);
    return HttpResponse{200, R"({"bars":{}})"};
  });
  const auto end = utc_seconds(2026, 10, 1);
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 9, 1), end);
  CHECK(paths.size() == 2);
  CHECK(store.covered_from("AAPL", Timeframe::Day).value() == utc_seconds(2026, 9, 1));
  paths.clear();
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 9, 1), end);
  REQUIRE(paths.size() == 1);
  CHECK(paths[0].find("start=2026-09-29T04:00:00Z") != std::string::npos);
  paths.clear();
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 8, 1), end);
  CHECK(paths.size() == 2);
  CHECK(store.covered_from("AAPL", Timeframe::Day).value() == utc_seconds(2026, 8, 1));
  store.set_covered_from("AAPL", Timeframe::Day, utc_seconds(2026, 9, 1));  // never moves later
  CHECK(store.covered_from("AAPL", Timeframe::Day).value() == utc_seconds(2026, 8, 1));
}

TEST_CASE("covered_from persists through the lake") {
  auto dir = test::temp_dir("covered_persist");
  {
    BarStore store(dir);
    store.set_covered_from("AAPL", Timeframe::Day, utc_seconds(2026, 9, 1));
  }
  BarStore fresh(dir);
  CHECK_FALSE(fresh.covered_from("AAPL", Timeframe::Day).has_value());
  fresh.load_all({"AAPL"}, Timeframe::Day);
  CHECK(fresh.covered_from("AAPL", Timeframe::Day).value() == utc_seconds(2026, 9, 1));
}

TEST_CASE("sync_bars saves each group as it completes") {
  auto dir = test::temp_dir("pergroup");
  {
  BarStore store(dir);
  store.merge("AAPL", Timeframe::Day, {{utc_seconds(2026, 9, 29, 4), 1, 1, 1, 1, 1, 1}});
  AlpacaClient client(test_config(), [&](const std::string& path) {
    if (path.find("end=2026-09-29T04:00:00Z") != std::string::npos)
      return HttpResponse{200, R"({"bars":{"AAPL":[{"t":"2026-09-10T04:00:00Z","o":1,"h":1,"l":1,"c":1,"v":1,"n":1,"vw":1}]}})"};
    return HttpResponse{403, "no"};
  });
  auto stale = sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 9, 1),
                         utc_seconds(2026, 10, 1));
  CHECK(stale.size() == 1);
  }
  BarStore reload(dir);
  reload.load_all({"AAPL"}, Timeframe::Day);
  CHECK(reload.bars("AAPL", Timeframe::Day).size() == 2);
}

TEST_CASE("sync data is durable before sync returns (no destructor flush needed)") {
  auto dir = test::temp_dir("sync_durable");
  std::vector<std::string> tickers;
  for (int i = 0; i < 150; ++i) tickers.push_back("T" + std::to_string(i));
  BarStore store(dir);
  AlpacaClient client(test_config(), [&](const std::string& path) -> HttpResponse {
    if (path.find("symbols=T0,") == std::string::npos) return {403, "no"};  // second batch fails
    return {200, R"({"bars":{"T0":[{"t":"2026-09-29T04:00:00Z","o":1,"h":1,"l":1,"c":1,"v":1,"vw":1}]}})"};
  });
  sync_bars(client, store, tickers, Timeframe::Day, utc_seconds(2026, 9, 1), utc_seconds(2026, 10, 1));
  // Store is still alive: read straight from disk through the lake.
  const auto on_disk = store.lake().read(Timeframe::Day, {"T0"}, 0, 1LL << 40);
  REQUIRE(on_disk.count("T0") == 1);
  CHECK(on_disk.at("T0").size() == 1);
  const auto cov = store.lake().coverage(Timeframe::Day, {"T0", "T120"});
  CHECK(cov.count("T0") == 1);
  CHECK(cov.count("T120") == 0);
}

TEST_CASE("sync commits each 100-symbol batch before fetching the next") {
  auto dir = test::temp_dir("sync_batches");
  std::vector<std::string> tickers;
  for (int i = 0; i < 150; ++i) tickers.push_back("T" + std::to_string(i));
  int calls = 0;
  {
    BarStore store(dir);
    AlpacaClient client(test_config(), [&](const std::string& path) -> HttpResponse {
      ++calls;
      if (path.find("symbols=T0,") == std::string::npos) return {403, "no"};  // second batch fails
      return {200, R"({"bars":{"T0":[{"t":"2026-09-29T04:00:00Z","o":1,"h":1,"l":1,"c":1,"v":1,"vw":1}]}})"};
    });
    auto stale = sync_bars(client, store, tickers, Timeframe::Day, utc_seconds(2026, 9, 1), utc_seconds(2026, 10, 1));
    CHECK(stale.size() == 50);
  }
  BarStore reloaded(dir);
  reloaded.load_all({"T0", "T120"}, Timeframe::Day);
  CHECK(reloaded.bars("T0", Timeframe::Day).size() == 1);
  CHECK(reloaded.covered_from("T0", Timeframe::Day).has_value());   // batch 1 committed
  CHECK_FALSE(reloaded.covered_from("T120", Timeframe::Day).has_value());  // failed batch not covered
  CHECK(calls == 2);
}

TEST_CASE("fetch_bars reports each successful batch to the callback") {
  std::vector<std::string> symbols;
  for (int i = 0; i < 250; ++i) symbols.push_back("S" + std::to_string(i));
  AlpacaClient client(test_config(), [](const std::string&) { return HttpResponse{200, R"({"bars":{}})"}; });
  std::vector<std::size_t> sizes;
  client.fetch_bars(symbols, "1Day", 0, 1,
                    [&](const std::vector<std::string>& batch, const std::map<std::string, std::vector<Bar>>&) {
                      sizes.push_back(batch.size());
                    });
  CHECK(sizes == std::vector<std::size_t>{100, 100, 50});
}

TEST_CASE("a re-adjusted overlap bar triggers a full-window refetch") {
  BarStore store(test::temp_dir("readjust"));
  const TimePoint t0 = utc_seconds(2026, 9, 1, 4), t1 = utc_seconds(2026, 9, 2, 4);
  store.merge("AAPL", Timeframe::Day, {{t0, 100, 100, 100, 100, 1, 100}, {t1, 110, 110, 110, 110, 1, 110}});
  store.set_complete_through("AAPL", Timeframe::Day, t1);  // t1 was stored after its session closed
  std::vector<std::string> paths;
  AlpacaClient client(test_config(), [&](const std::string& path) -> HttpResponse {
    paths.push_back(path);
    if (path.find("start=2026-09-02T04:00:00Z") != std::string::npos)  // tail: split-adjusted overlap bar
      return {200, R"({"bars":{"AAPL":[{"t":"2026-09-02T04:00:00Z","o":55,"h":55,"l":55,"c":55,"v":2,"vw":55}]}})"};
    return {200, R"({"bars":{"AAPL":[{"t":"2026-09-01T04:00:00Z","o":50,"h":50,"l":50,"c":50,"v":2,"vw":50},
                                     {"t":"2026-09-02T04:00:00Z","o":55,"h":55,"l":55,"c":55,"v":2,"vw":55}]}})"};
  });
  auto stale = sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 9, 1), utc_seconds(2026, 10, 1));
  CHECK(stale.empty());
  REQUIRE(paths.size() == 2);
  CHECK(paths[0].find("start=2026-09-02T04:00:00Z") != std::string::npos);
  CHECK(paths[1].find("start=2026-09-01T00:00:00Z") != std::string::npos);
  CHECK(paths[1].find("end=2026-10-01T00:00:00Z") != std::string::npos);
  const auto& b = store.bars("AAPL", Timeframe::Day);
  REQUIRE(b.size() == 2);
  CHECK(b[0].c == 50);
  CHECK(b[1].c == 55);
}

TEST_CASE("an unchanged overlap bar does not refetch history") {
  BarStore store(test::temp_dir("noreadjust"));
  const TimePoint t1 = utc_seconds(2026, 9, 2, 4);
  store.merge("AAPL", Timeframe::Day, {{utc_seconds(2026, 9, 1, 4), 100, 100, 100, 100, 1, 100}, {t1, 110, 110, 110, 110, 1, 110}});
  store.set_complete_through("AAPL", Timeframe::Day, t1);
  int calls = 0;
  AlpacaClient client(test_config(), [&](const std::string&) -> HttpResponse {
    ++calls;
    return {200, R"({"bars":{"AAPL":[{"t":"2026-09-02T04:00:00Z","o":110,"h":110,"l":110,"c":110,"v":2,"vw":110}]}})"};
  });
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 9, 1), utc_seconds(2026, 10, 1));
  CHECK(calls == 1);
}

TEST_CASE("fetch_bars with a callback hands bars to it and does not accumulate them") {
  AlpacaClient client(test_config(), [](const std::string& path) {
    return HttpResponse{200, path.find("page_token") == std::string::npos ? kPage1 : kPage2};
  });
  std::map<std::string, std::size_t> seen;
  auto r = client.fetch_bars({"AAPL", "NVO"}, "1Day", 0, 1,
                             [&](const std::vector<std::string>&, const std::map<std::string, std::vector<Bar>>& bars) {
                               for (const auto& [sym, b] : bars) seen[sym] += b.size();
                             });
  CHECK(r.bars.empty());
  CHECK(r.stale.empty());
  CHECK(seen["AAPL"] == 2);
  CHECK(seen["NVO"] == 1);
}

TEST_CASE("only a bar that was complete when stored is checked for re-adjustment") {
  BarStore store(test::temp_dir("readjust_partial"));
  const TimePoint start = utc_seconds(2026, 9, 1), t0 = utc_seconds(2026, 9, 1, 4), t1 = utc_seconds(2026, 9, 2, 4);
  auto bars = [](std::initializer_list<std::pair<const char*, double>> v) {
    std::string s = R"({"bars":{"AAPL":[)";
    bool first = true;
    for (const auto& [t, c] : v) {
      const std::string cs = std::to_string(c);
      s += std::string(first ? "" : ",") + R"({"t":")" + t + R"(","o":)" + cs + R"(,"h":)" + cs + R"(,"l":)" + cs +
           R"(,"c":)" + cs + R"(,"v":2,"vw":)" + cs + "}";
      first = false;
    }
    return s + "]}}";
  };
  std::string full = bars({{"2026-09-01T04:00:00Z", 100}, {"2026-09-02T04:00:00Z", 110}});
  std::string tail;
  std::vector<std::string> paths;
  AlpacaClient client(test_config(), [&](const std::string& path) -> HttpResponse {
    paths.push_back(path);
    return {200, path.find("start=2026-09-01T00:00:00Z") != std::string::npos ? full : tail};
  });
  auto sync = [&](TimePoint end) {
    paths.clear();
    sync_bars(client, store, {"AAPL"}, Timeframe::Day, start, end);
    return paths.size();
  };
  // 1. First sync at 10:00 ET on 09-02: t0 is complete, t1 is a partial session bar.
  CHECK(sync(utc_seconds(2026, 9, 2, 14)) == 1);
  CHECK(store.complete_through("AAPL", Timeframe::Day).value() == t0);
  // 2. Intraday re-sync: the partial t1 changed, which is not a re-adjustment.
  tail = bars({{"2026-09-02T04:00:00Z", 112}});
  CHECK(sync(utc_seconds(2026, 9, 2, 15)) == 1);
  CHECK(store.bars("AAPL", Timeframe::Day).back().c == 112);
  // 3. After the close: t1 is final now, but the stored copy was partial, so still no refetch.
  tail = bars({{"2026-09-02T04:00:00Z", 115}});
  CHECK(sync(utc_seconds(2026, 9, 3, 1)) == 1);
  CHECK(store.complete_through("AAPL", Timeframe::Day).value() == t1);
  // 4. A changed complete bar (2:1 split) still refetches the full window.
  tail = bars({{"2026-09-02T04:00:00Z", 57.5}});
  full = bars({{"2026-09-01T04:00:00Z", 50}, {"2026-09-02T04:00:00Z", 57.5}});
  CHECK(sync(utc_seconds(2026, 9, 4)) == 2);
  REQUIRE(paths.size() == 2);
  CHECK(paths[1].find("start=2026-09-01T00:00:00Z") != std::string::npos);
  CHECK(store.bars("AAPL", Timeframe::Day).front().c == 50);
}

namespace {
// Reviewer scenario vendor: weekday daily bars at 04:00Z from 2026-06-01; the true close of day k is
// 100 + k. A bar whose session (20:00Z) has not closed by `now` gets a partial close (+0.37). From
// split_at on, the whole history is divided by 2. fail_full fails any request spanning > 10 days.
struct SplitVendor {
  TimePoint now = 0;
  TimePoint split_at = std::numeric_limits<TimePoint>::max();
  bool fail_full = false;
  static TimePoint param(const std::string& path, const std::string& key) {
    const auto p = path.find(key + "=");
    return parse_rfc3339(path.substr(p + key.size() + 1, 20));
  }
  HttpResponse operator()(const std::string& path) const {
    const TimePoint s = param(path, "start"), e = param(path, "end");
    if (fail_full && e - s > 10 * 86400) return {500, "x"};
    std::string out = R"({"bars":{"AAPL":[)";
    bool first = true;
    const TimePoint d0 = utc_seconds(2026, 6, 1, 4);
    for (TimePoint t = d0; t <= e && t <= now; t += 86400) {
      if (t < s) continue;
      const auto wd = weekday_from_days(floor_div(t, 86400));
      if (wd == 0 || wd == 6) continue;
      double c = 100 + static_cast<double>((t - d0) / 86400);
      if (t + 16 * 3600 > now) c += 0.37;
      if (now >= split_at) c /= 2;
      char buf[256];
      std::snprintf(buf, sizeof buf, R"(%s{"t":"%s","o":%.6f,"h":%.6f,"l":%.6f,"c":%.6f,"v":1000,"vw":%.6f})",
                    first ? "" : ",", format_rfc3339(t).c_str(), c, c, c, c, c);
      out += buf;
      first = false;
    }
    return {200, out + "]}}"};
  }
};

// Bars whose basis differs from the latest bar's (a ratio of ~2 after a split).
int mixed_basis(const BarStore& s) {
  const auto& b = s.bars("AAPL", Timeframe::Day);
  const TimePoint d0 = utc_seconds(2026, 6, 1, 4);
  const double basis = (100 + static_cast<double>((b.back().t - d0) / 86400)) / b.back().c;
  int bad = 0;
  for (const Bar& x : b)
    if (std::abs((100 + static_cast<double>((x.t - d0) / 86400)) / x.c / basis - 1) > 0.05) ++bad;
  return bad;
}

AlpacaConfig no_retry_config() {
  AlpacaConfig c = test_config();
  c.max_retries = 0;
  return c;
}

bool weekend(TimePoint t) {
  const auto wd = weekday_from_days(floor_div(t, 86400));
  return wd == 0 || wd == 6;
}
}  // namespace

TEST_CASE("intraday syncs still catch a split (tail starts at the last complete bar)") {
  for (int hour : {14, 22}) {  // 10:30 EDT intraday (minute 30 below), and 18:00 EDT evening
    INFO("sync hour UTC " << hour);
    BarStore store(test::temp_dir("split_intraday"));
    SplitVendor v;
    v.split_at = utc_seconds(2026, 9, 16, 13);
    AlpacaClient client(no_retry_config(), [&](const std::string& p) { return v(p); });
    for (unsigned day = 1; day <= 20; ++day) {
      const TimePoint now = utc_seconds(2026, 9, day, hour, hour == 14 ? 30 : 0);
      if (weekend(now)) continue;
      v.now = now;
      sync_bars(client, store, {"AAPL"}, Timeframe::Day, utc_seconds(2026, 6, 1), now);
    }
    CHECK(store.bars("AAPL", Timeframe::Day).size() == 80);
    CHECK(mixed_basis(store) == 0);
  }
}

TEST_CASE("a split found by a narrow-window sync refetches all stored history") {
  BarStore store(test::temp_dir("split_narrow"));
  SplitVendor v;
  AlpacaClient client(no_retry_config(), [&](const std::string& p) { return v(p); });
  const TimePoint start = utc_seconds(2026, 6, 1);
  v.now = utc_seconds(2026, 9, 15, 22);
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, start, v.now);
  v.now = utc_seconds(2026, 9, 16, 22);
  v.split_at = utc_seconds(2026, 9, 16, 13);
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, v.now - 40 * 86400, v.now);  // ranking-style window
  CHECK(mixed_basis(store) == 0);
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, start, v.now);
  CHECK(mixed_basis(store) == 0);
}

TEST_CASE("a failed re-adjustment refetch leaves the data untouched and is retried next sync") {
  BarStore store(test::temp_dir("split_failed"));
  SplitVendor v;
  AlpacaClient client(no_retry_config(), [&](const std::string& p) { return v(p); });
  const TimePoint start = utc_seconds(2026, 6, 1);
  v.now = utc_seconds(2026, 9, 15, 22);
  sync_bars(client, store, {"AAPL"}, Timeframe::Day, start, v.now);
  const std::vector<Bar> before = store.bars("AAPL", Timeframe::Day);
  v.now = utc_seconds(2026, 9, 16, 22);
  v.split_at = utc_seconds(2026, 9, 16, 13);
  v.fail_full = true;
  const auto st = sync_bars(client, store, {"AAPL"}, Timeframe::Day, start, v.now);
  CHECK(st == std::vector<std::string>{"AAPL"});
  const auto& after = store.bars("AAPL", Timeframe::Day);
  REQUIRE(after.size() == before.size());
  for (std::size_t i = 0; i < after.size(); ++i) CHECK(after[i].c == before[i].c);  // stale: untouched
  v.fail_full = false;
  for (unsigned d = 17; d <= 18; ++d) {
    v.now = utc_seconds(2026, 9, d, 22);
    sync_bars(client, store, {"AAPL"}, Timeframe::Day, start, v.now);
  }
  CHECK(mixed_basis(store) == 0);
}
