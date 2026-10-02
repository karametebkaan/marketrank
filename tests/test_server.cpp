#include <doctest/doctest.h>
#include <httplib.h>

#include <chrono>
#include <future>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "server/http_server.hpp"
#include "test_util.hpp"

using namespace fx;
using namespace std::chrono_literals;
using nlohmann::json;

namespace {
struct Fixture {
  std::filesystem::path web = test::temp_dir("web");
  std::unique_ptr<FrameStore> store;
  std::unique_ptr<FluxServer> server;
  std::thread th;
  int port = 0;
  explicit Fixture(bool start = true) {
    SyntheticConfig cfg;
    cfg.bars = 80;
    cfg.rotation_start = 40;
    BarStore bars(test::temp_dir("server_bars"));
    auto secs = generate_synthetic(cfg, bars);
    std::vector<std::string> tickers;
    for (auto& s : secs) tickers.push_back(s.ticker);
    store = std::make_unique<FrameStore>(build_panel(bars, tickers, cfg.tf), secs, CoreParams::money_flow(),
                                         LandscapeParams{}, 10);
    if (start) store->start();
    for (int k = 0; start && k < 600 && !store->status().ready; ++k) std::this_thread::sleep_for(50ms);
    test::write_file(web / "index.html", "hello");
    server = std::make_unique<FluxServer>(*store, std::nullopt, "synthetic 1d");
    ServerOptions o;
    o.port = 0;
    o.web_root = web;
    port = server->bind(o);
    th = std::thread([this] { server->listen(); });
  }
  ~Fixture() {
    server->stop();
    th.join();
  }
};
}  // namespace

TEST_CASE("server: status, times, frame and grid") {
  Fixture fx_;
  httplib::Client cli("127.0.0.1", fx_.port);
  auto st = cli.Get("/api/status");
  REQUIRE(st);
  CHECK(st->status == 200);
  CHECK(json::parse(st->body)["ready"] == true);
  auto times = json::parse(cli.Get("/api/times")->body);
  REQUIRE(times.size() == 10);
  auto fr = cli.Get("/api/frame");
  REQUIRE(fr->status == 200);
  auto f = json::parse(fr->body);
  CHECK(f["nodes"].size() > 0);
  CHECK(f["lattice"]["cols"].get<int>() > 0);
  const auto w = f["raster"]["w"].get<std::size_t>(), h = f["raster"]["h"].get<std::size_t>();
  auto grid = cli.Get("/api/frame/grid");
  REQUIRE(grid->status == 200);
  CHECK(grid->body.size() == w * h * 4);
  CHECK(cli.Get("/api/frame?t=1")->status == 404);
  CHECK(cli.Get("/api/frame?t=" + std::to_string(times[0].get<long long>()))->status == 200);
  CHECK(cli.Get("/")->body == "hello");
}

TEST_CASE("server: shock and params") {
  Fixture fx_;
  httplib::Client cli("127.0.0.1", fx_.port);
  auto f = json::parse(cli.Get("/api/frame")->body);
  const std::string ticker = f["nodes"][0][1];
  auto r = cli.Post("/api/shock", json{{"shocks", {{{"ticker", ticker}, {"size", -10}}}}}.dump(), "application/json");
  REQUIRE(r);
  CHECK(r->status == 200);
  auto body = json::parse(r->body);
  CHECK(body["receivers"].size() > 0);
  CHECK(body["shocked"][0]["dh"].get<double>() < 0);
  CHECK(cli.Get("/api/shock/grid")->status == 200);
  CHECK(cli.Post("/api/shock", R"({"shocks":[{"ticker":"NOPE","size":1}]})", "application/json")->status == 400);
  CHECK(cli.Post("/api/params", R"({"preset":"bogus"})", "application/json")->status == 400);
  const auto g = json::parse(cli.Get("/api/status")->body)["generation"].get<std::uint64_t>();
  auto p = cli.Post("/api/params", R"({"preset":"legacy","height":"linear"})", "application/json");
  CHECK(p->status == 202);
  CHECK(json::parse(cli.Get("/api/status")->body)["generation"].get<std::uint64_t>() > g);
}

TEST_CASE("server: stop releases an open SSE stream without hanging") {
  auto fx_ = std::make_unique<Fixture>();
  std::atomic<bool> got_status{false}, done{false};
  httplib::Client cli("127.0.0.1", fx_->port);
  cli.set_read_timeout(20, 0);
  std::thread reader([&] {
    cli.Get("/api/events", [&](const char* data, size_t n) {
      if (std::string(data, n).find("event: status") != std::string::npos) got_status = true;
      return true;
    });
    done = true;
  });
  for (int k = 0; k < 100 && !got_status; ++k) std::this_thread::sleep_for(50ms);
  CHECK(got_status.load());
  auto fut = std::async(std::launch::async, [&] { fx_.reset(); });  // stops, joins, destroys the store
  const bool finished = fut.wait_for(5s) == std::future_status::ready;
  CHECK(finished);
  if (!finished) {
    std::cerr << "shutdown hung; aborting\n";
    std::_Exit(1);  // the stuck future would otherwise block its destructor forever
  }
  reader.join();
  CHECK(done.load());
}

TEST_CASE("server: stop right after start is not lost") {
  for (int rep = 0; rep < 20; ++rep) {
    Fixture f;  // listen thread already started
    f.server->stop();
    f.th.join();
    f.th = std::thread([] {});  // destructor joins this
  }
  Fixture g;  // stop issued before listen even begins
  g.server->stop();
  g.th.join();
  auto sv = std::make_unique<FluxServer>(*g.store, std::nullopt, "x");
  ServerOptions o;
  o.port = 0;
  o.web_root = g.web;
  sv->bind(o);
  sv->stop();
  auto fut = std::async(std::launch::async, [&] { sv->listen(); });
  CHECK(fut.wait_for(5s) == std::future_status::ready);
  g.th = std::thread([] {});
}

TEST_CASE("server: 503 before ready, shock grid 404, node field order") {
  {
    Fixture f(false);
    httplib::Client cli("127.0.0.1", f.port);
    CHECK(cli.Get("/api/status")->status == 200);
    CHECK(cli.Get("/api/frame")->status == 503);
    CHECK(cli.Get("/api/frame/grid")->status == 503);
    CHECK(cli.Get("/api/frame?t=1")->status == 503);
    CHECK(cli.Post("/api/shock", R"({"shocks":[{"ticker":"S000","size":1}]})", "application/json")->status >= 400);
  }
  Fixture f;
  httplib::Client cli("127.0.0.1", f.port);
  CHECK(cli.Get("/api/shock/grid")->status == 404);
  auto fr = json::parse(cli.Get("/api/frame")->body);
  const auto& n0 = fr["nodes"][0];
  REQUIRE(n0.size() == 10);
  CHECK(n0[0].is_number_integer());
  CHECK(n0[1].is_string());
  CHECK(n0[2].is_string());
  CHECK(n0[3].is_number_integer());
  CHECK(n0[4].is_number());
  CHECK(n0[5].is_number());
  CHECK(fr["nodes"][0][1] == f.store->nodes()[n0[0].get<std::size_t>()].ticker);
  CHECK(fr["nodes"][0][2] == f.store->nodes()[n0[0].get<std::size_t>()].sector);
  const std::string ticker = n0[1];
  auto r = cli.Post("/api/shock", json{{"shocks", {{{"ticker", ticker}, {"size", -5}}}}}.dump(), "application/json");
  REQUIRE(r->status == 200);
  auto b = json::parse(r->body);
  auto g = cli.Get("/api/shock/grid");
  CHECK(g->status == 200);
  CHECK(g->body.size() == b["raster"]["w"].get<std::size_t>() * b["raster"]["h"].get<std::size_t>() * 4);
}

TEST_CASE("server: bad shock bodies are 400") {
  Fixture f;
  httplib::Client cli("127.0.0.1", f.port);
  auto post = [&](const std::string& body) { return cli.Post("/api/shock", body, "application/json")->status; };
  CHECK(post("{not json") == 400);
  CHECK(post(R"({"shocks":[]})") == 400);
  CHECK(post(R"({})") == 400);
  CHECK(post(R"({"shocks":[{"ticker":"X","size":1e999}]})") == 400);
  const std::string t = f.store->nodes()[0].ticker;
  CHECK(post(R"({"shocks":[{"ticker":")" + t + R"(","size":"big"}]})") == 400);
  CHECK(cli.Post("/api/params", "{oops", "application/json")->status == 400);
}

TEST_CASE("server: invalid params are 400 and change nothing") {
  Fixture f;
  httplib::Client cli("127.0.0.1", f.port);
  const auto before = json::parse(cli.Get("/api/status")->body);
  const std::vector<std::string> bad = {
      R"({"idw_radius":-1})",      R"({"idw_radius":17})",      R"({"idw_radius":2.5})",
      R"({"idw_power":0})",        R"({"idw_power":9})",        R"({"idw_power":-1})",
      R"({"idw_power":1e999})",    R"({"subdivision":0})",      R"({"subdivision":9})",
      R"({"subdivision":2.5})",    R"({"k_in":-1})",            R"({"k_out":-3})",
      R"({"k_out":1.5})",          R"({"k_in":"x"})",           R"({"retention":1e999})",
      R"({"lambda":1e999})",       R"({"retention":"a"})",      R"({"vol_scale":"yes"})",
      R"({"preset":"bogus"})",     R"({"height":"cubic"})",     R"({"h_ref":"nope"})",
      R"({"retention":-5})",       R"({"lambda":-1})",          R"({"k_out":0})"};
  for (const auto& b : bad) {
    INFO(b);
    CHECK(cli.Post("/api/params", b, "application/json")->status == 400);
  }
  const auto after = json::parse(cli.Get("/api/status")->body);
  CHECK(after["generation"] == before["generation"]);
  CHECK(after["params"] == before["params"]);
  CHECK(cli.Post("/api/params", R"({"subdivision":2,"idw_radius":4,"idw_power":3})", "application/json")->status == 202);
}

TEST_CASE("server: POST guards (content type, origin, size)") {
  Fixture f;
  httplib::Client cli("127.0.0.1", f.port);
  CHECK(cli.Post("/api/params", R"({"preset":"defaults"})", "text/plain")->status == 415);
  CHECK(cli.Post("/api/shock", "{}", "application/x-www-form-urlencoded")->status == 415);
  const std::string port = std::to_string(f.port);
  httplib::Headers evil{{"Origin", "http://evil.example"}};
  CHECK(cli.Post("/api/params", evil, R"({"preset":"defaults"})", "application/json")->status == 403);
  httplib::Headers ok1{{"Origin", "http://localhost:" + port}};
  CHECK(cli.Post("/api/params", ok1, R"({"preset":"money-flow"})", "application/json")->status == 202);
  httplib::Headers ok2{{"Origin", "http://127.0.0.1:" + port}};
  CHECK(cli.Post("/api/params", ok2, R"({"preset":"money-flow"})", "application/json")->status == 202);
  const std::string big = R"({"preset":"defaults","pad":")" + std::string(70 * 1024, 'a') + "\"}";
  auto r = cli.Post("/api/params", big, "application/json");
  REQUIRE(r);
  CHECK(r->status == 413);
}
