#include <doctest/doctest.h>
#include <httplib.h>

#include <chrono>
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
  Fixture() {
    SyntheticConfig cfg;
    cfg.bars = 80;
    cfg.rotation_start = 40;
    BarStore bars(test::temp_dir("server_bars"));
    auto secs = generate_synthetic(cfg, bars);
    std::vector<std::string> tickers;
    for (auto& s : secs) tickers.push_back(s.ticker);
    store = std::make_unique<FrameStore>(build_panel(bars, tickers, cfg.tf), secs, CoreParams::money_flow(),
                                         LandscapeParams{}, 10);
    store->start();
    for (int k = 0; k < 600 && !store->status().ready; ++k) std::this_thread::sleep_for(50ms);
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
  const auto t0 = std::chrono::steady_clock::now();
  fx_.reset();  // stops the server, joins the listener, then destroys the frame store
  CHECK(std::chrono::steady_clock::now() - t0 < 5s);
  reader.join();
  CHECK(done.load());
}
