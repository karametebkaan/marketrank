#include <doctest/doctest.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "market/asset_universe.hpp"
#include "market/sec_sectors.hpp"
#include "test_util.hpp"

using namespace mr;

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
    return [this](const std::string& host, const std::string& path, const std::string&) -> HttpResponse {
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
  c.sleep = [](std::chrono::milliseconds) {};
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

TEST_CASE("parse_company_tickers rejects an empty map and prefers real tickers over variants") {
  CHECK_THROWS_AS(parse_company_tickers("{}"), std::runtime_error);
  // The synthetic dash variant of AB.C is inserted before the real AB-C appears.
  const auto m = parse_company_tickers(R"({"0":{"cik_str":1,"ticker":"AB.C","title":"x"},
                                           "1":{"cik_str":2,"ticker":"AB-C","title":"y"}})");
  CHECK(m.at("AB.C") == "1");
  CHECK(m.at("AB-C") == "2");
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

TEST_CASE("sic_to_sector carve-outs") {
  const char* cd = "Consumer Discretionary";
  CHECK(sic_to_sector(3020) == cd);
  CHECK(sic_to_sector(3021) == cd);  // footwear, not Materials rubber/plastics
  CHECK(sic_to_sector(3029) == cd);
  CHECK(sic_to_sector(3089) == "Materials");
  CHECK(sic_to_sector(7812) == "Communication Services");
  CHECK(sic_to_sector(7841) == "Communication Services");
  CHECK(sic_to_sector(7900) == cd);
  CHECK(sic_to_sector(7999) == cd);
  CHECK(sic_to_sector(6324) == "Health Care");
  CHECK(sic_to_sector(6411) == "Financials");
  CHECK(sic_to_sector(5331) == "Consumer Staples");
  CHECK(sic_to_sector(5399) == "Consumer Staples");
  CHECK(sic_to_sector(5411) == "Consumer Staples");
  CHECK(sic_to_sector(5600) == cd);
  CHECK(sic_to_sector(3826) == "Health Care");
  CHECK(sic_to_sector(3829) == "Health Care");
  CHECK(sic_to_sector(3812) == "Industrials");
  CHECK(sic_to_sector(4922) == "Energy");
  CHECK(sic_to_sector(4924) == "Energy");
  CHECK(sic_to_sector(4911) == "Utilities");
  CHECK(sic_to_sector(4941) == "Utilities");
  CHECK(sic_to_sector(1220) == "Energy");
  CHECK(sic_to_sector(1241) == "Energy");
  CHECK(sic_to_sector(1040) == "Materials");
  CHECK(sic_to_sector(7389) == "Industrials");
  CHECK(sic_to_sector(7311) == "Communication Services");
  CHECK(sic_to_sector(3852) == "Industrials");
  CHECK(sic_to_sector(3899) == "Industrials");
  CHECK(sic_to_sector(8300) == "Industrials");
  CHECK(sic_to_sector(8699) == "Industrials");
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
  save_sec_cache(path, c);  // a stale temp file from a crashed run does not get in the way
  CHECK(std::filesystem::exists(path.string() + ".tmp"));  // unique temp names: the stray one is left alone
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
  SecClient ok(fast_config(), [&](const std::string&, const std::string&, const std::string&) -> HttpResponse {
    return ++calls < 3 ? HttpResponse{429, ""} : HttpResponse{200, "body"};
  });
  CHECK(ok.get("h", "/p") == "body");
  CHECK(calls == 3);
  calls = 0;
  SecClient bad(fast_config(), [&](const std::string&, const std::string&, const std::string&) -> HttpResponse {
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
  SecClient bad(c, [](const std::string&, const std::string&, const std::string&) -> HttpResponse { return {500, "x"}; });
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
  CHECK_FALSE(looks_like_fund("Tetfield Holdings"));   // " etf" must be a whole word
  CHECK_FALSE(looks_like_fund("Acme Etfa Corp"));
  CHECK(looks_like_fund("Global X ETF"));
}

TEST_CASE("sector fill order: GICS, then SEC, then ETF/Fund, then Unclassified") {
  SecCache cache;
  cache["AAPL"] = {"AAPL", "1", "3571", "", "Information Technology", kNow};
  cache["NKE"] = {"NKE", "2", "3021", "", sic_to_sector(3021), kNow};
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

TEST_CASE("403 aborts immediately with a clear error") {
  auto dir = test::temp_dir("sec");
  int calls = 0;
  SecSyncOptions opt;
  opt.now = kNow;
  auto factory = [&] {
    return SecClient(fast_config(), [&](const std::string&, const std::string&, const std::string&) -> HttpResponse {
      ++calls;
      return {403, "denied"};
    });
  };
  CHECK_THROWS_AS(sync_sec_sectors({"AAA", "BBB"}, dir / "c.csv", factory, opt), SecAbort);
  CHECK(calls == 1);
}

TEST_CASE("403 mid-sync saves progress before aborting") {
  auto dir = test::temp_dir("sec");
  const auto path = dir / "c.csv";
  int sub = 0;
  SecSyncOptions opt;
  opt.now = kNow;
  opt.commit_every = 1000;  // only the abort path can have saved the two rows
  auto factory = [&] {
    return SecClient(fast_config(), [&](const std::string&, const std::string& p, const std::string&) -> HttpResponse {
      if (p == "/files/company_tickers.json")
        return {200, R"({"0":{"cik_str":1,"ticker":"AAA","title":"a"},"1":{"cik_str":2,"ticker":"BBB","title":"b"},
                         "2":{"cik_str":3,"ticker":"CCC","title":"c"}})"};
      return ++sub <= 2 ? HttpResponse{200, R"({"sic":"2834"})"} : HttpResponse{403, ""};
    });
  };
  CHECK_THROWS_AS(sync_sec_sectors({"AAA", "BBB", "CCC"}, path, factory, opt), SecAbort);
  CHECK(load_sec_cache(path).size() == 2);
}

TEST_CASE("20 consecutive failed tickers abort and save progress") {
  auto dir = test::temp_dir("sec");
  const auto path = dir / "c.csv";
  std::string json = "{";
  std::vector<std::string> tickers;
  for (int i = 0; i < 40; ++i) {
    const std::string t = "T" + std::to_string(i);
    tickers.push_back(t);
    json += (i ? "," : "") + std::string("\"") + std::to_string(i) + "\":{\"cik_str\":" + std::to_string(i + 1) +
            ",\"ticker\":\"" + t + "\",\"title\":\"x\"}";
  }
  json += "}";
  int sub_calls = 0;
  SecSyncOptions opt;
  opt.now = kNow;
  auto factory = [&] {
    return SecClient(fast_config(), [&](const std::string&, const std::string& p, const std::string&) -> HttpResponse {
      if (p == "/files/company_tickers.json") return {200, json};
      ++sub_calls;
      return {429, ""};
    });
  };
  CHECK_THROWS_AS(sync_sec_sectors(tickers, path, factory, opt), SecAbort);
  CHECK(sub_calls == 20 * 3);  // 20 tickers, 3 attempts each
}

TEST_CASE("Retry-After is honoured on 429 and capped at 60 seconds") {
  std::vector<std::chrono::milliseconds> slept;
  SecConfig c = fast_config();
  c.sleep = [&](std::chrono::milliseconds d) { slept.push_back(d); };
  int calls = 0;
  SecClient cl(c, [&](const std::string&, const std::string&, const std::string&) -> HttpResponse {
    ++calls;
    if (calls == 1) return {429, "", 7};
    if (calls == 2) return {429, "", 500};
    return {200, "ok"};
  });
  CHECK(cl.get("h", "/p") == "ok");
  REQUIRE(slept.size() == 2);
  CHECK(slept[0] == std::chrono::milliseconds(7000));
  CHECK(slept[1] == std::chrono::milliseconds(60000));
}

TEST_CASE("User-Agent is passed to every request and the limiter spaces requests") {
  using clock = std::chrono::steady_clock;
  clock::time_point fake_now{};
  std::vector<std::string> agents;
  SecConfig c;
  c.user_agent = "Agent Smith smith@example.com";
  c.backoff_initial_ms = 0;
  CHECK(c.min_request_interval_ms == 150);
  c.now = [&] { return fake_now; };
  c.sleep = [&](std::chrono::milliseconds d) { fake_now += d; };
  SecClient cl(c, [&](const std::string&, const std::string&, const std::string& ua) -> HttpResponse {
    agents.push_back(ua);
    return {200, "x"};
  });
  const auto start = fake_now;
  constexpr int kN = 6;
  for (int i = 0; i < kN; ++i) cl.get("h", "/p");
  CHECK(fake_now - start >= (kN - 1) * std::chrono::milliseconds(150));
  REQUIRE(agents.size() == kN);
  for (const auto& a : agents) CHECK(a == "Agent Smith smith@example.com");
}

TEST_CASE("class-share ticker syncs end to end via the SEC spelling") {
  auto dir = test::temp_dir("sec");
  const auto path = dir / "c.csv";
  std::vector<std::string> paths;
  SecSyncOptions opt;
  opt.now = kNow;
  auto factory = [&] {
    return SecClient(fast_config(), [&](const std::string&, const std::string& p, const std::string&) -> HttpResponse {
      paths.push_back(p);
      if (p == "/files/company_tickers.json") return {200, kTickers};
      return {200, R"({"sic":"6331","sicDescription":"Fire, Marine & Casualty Insurance"})"};
    });
  };
  const auto stats = sync_sec_sectors({"BRK.B"}, path, factory, opt);
  CHECK(stats.fetched == 1);
  REQUIRE(paths.size() == 2);
  CHECK(paths[1] == "/submissions/CIK0001067983.json");
  const auto r = load_sec_cache(path);
  CHECK(r.at("BRK.B").sector == "Financials");
}

TEST_CASE("fill leaves the snapshot CSV untouched") {
  auto dir = test::temp_dir("sec");
  const std::string csv =
      "ticker,name,sector,exchange,median_dollar_volume\nAAPL,Apple,Unclassified,NASDAQ,1\n"
      "SPY,SPDR S&P 500 ETF Trust,Unclassified,ARCA,2\n";
  const auto snap = test::write_file(dir / "universe_2026-10-02_n10.csv", csv);
  const auto funds = test::write_file(dir / "funds.csv", "ticker,tracks\n");
  Universe u = load_snapshot(snap, funds);
  SecCache cache;
  cache["AAPL"] = {"AAPL", "1", "3571", "", "Information Technology", kNow};
  CHECK(apply_sector_fill(u, cache) == 2);
  CHECK(u.nodes()[0].sector == "Information Technology");
  CHECK(u.nodes()[1].sector == "ETF/Fund");
  std::ifstream in(snap);
  std::stringstream ss;
  ss << in.rdbuf();
  CHECK(ss.str() == csv);
}
