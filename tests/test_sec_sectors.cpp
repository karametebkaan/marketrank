#include <doctest/doctest.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "market/sec_sectors.hpp"
#include "test_util.hpp"

using namespace fx;

namespace {
const char* kTickers = R"({"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."},
 "1":{"cik_str":1067983,"ticker":"BRK-B","title":"BERKSHIRE HATHAWAY INC"},
 "2":{"cik_str":789019,"ticker":"msft","title":"MICROSOFT CORP"}})";
const char* kSub = R"({"cik":"320193","sic":"3571","sicDescription":"Electronic Computers","name":"Apple Inc."})";
const char* kSubEmpty = R"({"cik":"1","sic":"","sicDescription":""})";

constexpr std::int64_t kNow = 1790000000;  // arbitrary fixed "now"
constexpr std::int64_t kDay = 86400;

struct FakeSec {
  int calls = 0;
  std::vector<std::string> paths;
  int throw_logic_on_call = -1;  // submission call index that throws std::logic_error
  int fail_ciks = 0;             // submissions that return 404
  SecHttpGet fn() {
    return [this](const std::string& host, const std::string& path) -> HttpResponse {
      ++calls;
      paths.push_back(host + path);
      if (path == "/files/company_tickers.json")
        return {200, R"({"0":{"cik_str":1,"ticker":"AAA","title":"a"},
                         "1":{"cik_str":2,"ticker":"BBB","title":"b"},
                         "2":{"cik_str":3,"ticker":"CCC","title":"c"},
                         "3":{"cik_str":4,"ticker":"DDD","title":"d"}})"};
      if (throw_logic_on_call >= 0 && static_cast<int>(paths.size()) - 1 == throw_logic_on_call)
        throw std::logic_error("interrupted");
      if (fail_ciks > 0) {
        --fail_ciks;
        return {404, "nope"};
      }
      return {200, R"({"sic":"2834","sicDescription":"Pharma"})"};
    };
  }
};
SecConfig fast_config() {
  SecConfig c;
  c.user_agent = "Test test@example.com";
  c.min_request_interval_ms = 0;
  c.backoff_initial_ms = 0;
  c.backoff_max_ms = 0;
  return c;
}
}  // namespace

TEST_CASE("parse_company_tickers uppercases and maps class-share forms") {
  const auto m = parse_company_tickers(kTickers);
  CHECK(m.at("AAPL") == "320193");
  CHECK(m.at("MSFT") == "789019");
  CHECK(m.at("BRK-B") == "1067983");
  CHECK(m.at("BRK.B") == "1067983");  // Alpaca spelling
}

TEST_CASE("parse_submission reads sic and description") {
  const auto s = parse_submission(kSub);
  CHECK(s.sic == "3571");
  CHECK(s.description == "Electronic Computers");
  CHECK(parse_submission(kSubEmpty).sic.empty());
  CHECK(parse_submission(R"({"sic":3571})").sic == "3571");  // tolerate a numeric sic
  CHECK_THROWS_AS(parse_submission("not json"), std::runtime_error);
}

TEST_CASE("sic_to_sector range table") {
  CHECK(sic_to_sector(2834) == "Health Care");
  CHECK(sic_to_sector(3674) == "Information Technology");
  CHECK(sic_to_sector(7372) == "Information Technology");
  CHECK(sic_to_sector(6021) == "Financials");
  CHECK(sic_to_sector(1311) == "Energy");
  CHECK(sic_to_sector(4911) == "Utilities");
  CHECK(sic_to_sector(6798) == "Real Estate");
  CHECK(sic_to_sector(2080) == "Consumer Staples");
  CHECK(sic_to_sector(3711) == "Consumer Discretionary");
  CHECK(sic_to_sector(4813) == "Communication Services");
  CHECK(sic_to_sector(2821) == "Materials");
  CHECK(sic_to_sector(3560) == "Industrials");
  CHECK(sic_to_sector(6770) == "Financials");
  CHECK(sic_to_sector(9999) == "");
  CHECK(sic_to_sector(std::string("")) == "");
  CHECK(sic_to_sector(std::string("2834")) == "Health Care");
  CHECK(sic_to_sector(std::string("abc")) == "");
  CHECK(sic_to_sector(6500) == "Real Estate");
  CHECK(sic_to_sector(4512) == "Industrials");
  CHECK(sic_to_sector(5812) == "Consumer Discretionary");
}

TEST_CASE("cache round-trips and tolerates a missing file or stray temp file") {
  auto dir = test::temp_dir("sec");
  const auto path = dir / "sectors" / "sec_sic.csv";
  CHECK(load_sec_cache(path).empty());
  SecCache c;
  c["AAPL"] = {"AAPL", "320193", "3571", "Electronic Computers, Inc", "Information Technology", kNow};
  c["ZZZ"] = {"ZZZ", "", "", "", "", kNow - 5};
  save_sec_cache(path, c);
  test::write_file(path.string() + ".tmp", "garbage,half");
  const auto r = load_sec_cache(path);
  REQUIRE(r.size() == 2);
  CHECK(r.at("AAPL").sic_description == "Electronic Computers, Inc");
  CHECK(r.at("AAPL").sector == "Information Technology");
  CHECK(r.at("AAPL").fetched_at == kNow);
  CHECK(r.at("ZZZ").sic.empty());
}

TEST_CASE("sync fetches only missing and stale tickers; no-CIK is cached") {
  auto dir = test::temp_dir("sec");
  const auto path = dir / "sec_sic.csv";
  SecCache c;
  c["AAA"] = {"AAA", "1", "2834", "x", "Health Care", kNow - 10 * kDay};    // fresh
  c["BBB"] = {"BBB", "2", "2834", "x", "Health Care", kNow - 91 * kDay};    // stale
  save_sec_cache(path, c);
  FakeSec fake;
  SecSyncOptions opt;
  opt.now = kNow;
  const auto stats = sync_sec_sectors({"AAA", "BBB", "CCC", "NOCIK"}, path,
                                      [&] { return SecClient(fast_config(), fake.fn()); }, opt);
  CHECK(stats.fresh == 1);
  CHECK(stats.fetched == 2);  // BBB, CCC
  CHECK(stats.no_cik == 1);
  CHECK(fake.calls == 3);  // company_tickers + BBB + CCC
  const auto r = load_sec_cache(path);
  CHECK(r.at("AAA").fetched_at == kNow - 10 * kDay);
  CHECK(r.at("BBB").fetched_at == kNow);
  CHECK(r.at("CCC").sector == "Health Care");
  CHECK(r.at("NOCIK").sic.empty());
  CHECK(r.at("NOCIK").fetched_at == kNow);

  FakeSec again;  // everything fresh now: no client is even built
  bool built = false;
  sync_sec_sectors({"AAA", "BBB", "CCC", "NOCIK"}, path,
                   [&] { built = true; return SecClient(fast_config(), again.fn()); }, opt);
  CHECK_FALSE(built);
  CHECK(again.calls == 0);
}

TEST_CASE("sync resumes after an interruption using committed progress") {
  auto dir = test::temp_dir("sec");
  const auto path = dir / "sec_sic.csv";
  FakeSec fake;
  fake.throw_logic_on_call = 4;  // company_tickers + AAA, BBB, CCC ok; DDD interrupts
  SecSyncOptions opt;
  opt.now = kNow;
  opt.commit_every = 2;
  CHECK_THROWS_AS(sync_sec_sectors({"AAA", "BBB", "CCC", "DDD"}, path,
                                   [&] { return SecClient(fast_config(), fake.fn()); }, opt),
                  std::logic_error);
  CHECK(load_sec_cache(path).size() == 2);  // the first commit of 2 survived
  FakeSec resume;
  const auto stats = sync_sec_sectors({"AAA", "BBB", "CCC", "DDD"}, path,
                                      [&] { return SecClient(fast_config(), resume.fn()); }, opt);
  CHECK(stats.fetched == 2);
  CHECK(load_sec_cache(path).size() == 4);
}

TEST_CASE("failed submission fetches are not cached so they retry next run") {
  auto dir = test::temp_dir("sec");
  const auto path = dir / "sec_sic.csv";
  FakeSec fake;
  fake.fail_ciks = 1;  // 404 is not retryable
  SecSyncOptions opt;
  opt.now = kNow;
  const auto stats = sync_sec_sectors({"AAA", "BBB"}, path,
                                      [&] { return SecClient(fast_config(), fake.fn()); }, opt);
  CHECK(stats.failed == 1);
  CHECK(load_sec_cache(path).size() == 1);
}

TEST_CASE("client retries 429 and 5xx up to 3 attempts, then throws") {
  int calls = 0;
  SecClient ok(fast_config(), [&](const std::string&, const std::string&) -> HttpResponse {
    return ++calls < 3 ? HttpResponse{429, ""} : HttpResponse{200, "body"};
  });
  CHECK(ok.get("h", "/p") == "body");
  CHECK(calls == 3);
  calls = 0;
  SecClient bad(fast_config(), [&](const std::string&, const std::string&) -> HttpResponse {
    ++calls;
    return {503, ""};
  });
  CHECK_THROWS_AS(bad.get("h", "/p"), std::runtime_error);
  CHECK(calls == 3);
}

TEST_CASE("client sends the configured User-Agent to the transport via config only") {
  // The transport is injected; the config carries the agent and the error text never includes it.
  SecConfig c = fast_config();
  c.user_agent = "Secret Agent secret@example.com";
  SecClient bad(c, [](const std::string&, const std::string&) -> HttpResponse { return {500, "x"}; });
  try {
    bad.get("h", "/p");
    FAIL("expected throw");
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()).find("secret@example.com") == std::string::npos);
  }
}

TEST_CASE("missing SEC_USER_AGENT gives the setup error") {
  const char* old = std::getenv("SEC_USER_AGENT");
  const std::string saved = old ? old : "";
  ::unsetenv("SEC_USER_AGENT");
  const std::string expected =
      "SEC_USER_AGENT is not set: add SEC_USER_AGENT=\"Your Name your@email\" to .env (SEC requires a contact)";
  try {
    sec_config_from_env();
    FAIL("expected throw");
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()) == expected);
  }
  ::setenv("SEC_USER_AGENT", "", 1);
  CHECK_THROWS_WITH_AS(make_sec_client(), expected.c_str(), std::runtime_error);
  ::setenv("SEC_USER_AGENT", "Test t@example.com", 1);
  CHECK_NOTHROW(sec_config_from_env());
  if (old) ::setenv("SEC_USER_AGENT", saved.c_str(), 1);
  else ::unsetenv("SEC_USER_AGENT");
}

TEST_CASE("fund heuristic") {
  CHECK(looks_like_fund("State Street SPDR S&P 500 ETF Trust"));
  CHECK(looks_like_fund("iShares Core S&P 500"));
  CHECK(looks_like_fund("ProShares UltraPro QQQ"));
  CHECK(looks_like_fund("Invesco QQQ Trust, Series 1"));
  CHECK(looks_like_fund("Barclays iPath Series B ETN"));
  CHECK(looks_like_fund("Nuveen Municipal Value Fund Inc."));
  CHECK(looks_like_fund("Grayscale Bitcoin Trust Shares"));
  CHECK(looks_like_fund("Some Index Trust"));
  CHECK_FALSE(looks_like_fund("Apple Inc. Common Stock"));
  CHECK_FALSE(looks_like_fund("Fundamental Global Inc"));
  CHECK_FALSE(looks_like_fund("Northern Trust Corporation"));
}

TEST_CASE("sector fill order: GICS, then SEC, then ETF/Fund, then Unclassified") {
  SecCache cache;
  cache["AAPL"] = {"AAPL", "1", "3571", "", "Information Technology", kNow};
  cache["NKE"] = {"NKE", "2", "3021", "", "Consumer Discretionary", kNow};
  cache["EMPTY"] = {"EMPTY", "", "", "", "", kNow};
  // GICS wins over SEC
  CHECK(resolve_sector({"AAPL", "Apple", "Health Care"}, false, cache) == "Health Care");
  CHECK(resolve_sector({"AAPL", "Apple", "Unclassified"}, false, cache) == "Information Technology");
  // SEC wins over fund name
  CHECK(resolve_sector({"NKE", "Nike Trust ETF", "Unclassified"}, false, cache) == "Consumer Discretionary");
  CHECK(resolve_sector({"SPY", "State Street SPDR S&P 500 ETF Trust", "Unclassified"}, false, cache) == "ETF/Fund");
  CHECK(resolve_sector({"EMPTY", "Plain Co", "Unclassified"}, false, cache) == "Unclassified");
  CHECK(resolve_sector({"VOO", "Plain Co", "Unclassified"}, true, cache) == "ETF/Fund");
  CHECK(resolve_sector({"X", "Plain Co", ""}, false, cache) == "Unclassified");
  CHECK(resolve_sector({"X", "Plain Co", "Extra"}, false, cache) == "Extra");

  auto u = Universe::from_securities({{"AAPL", "Apple", "Unclassified"},
                                      {"SPY", "SPDR S&P 500 ETF Trust", "Unclassified"},
                                      {"MSFT", "Microsoft", "Information Technology"}},
                                     {});
  CHECK(apply_sector_fill(u, cache) == 2);
  CHECK(u.nodes()[0].sector == "Information Technology");
  CHECK(u.nodes()[1].sector == "ETF/Fund");
}
