#include <doctest/doctest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "market/bar_store.hpp"
#include "test_util.hpp"

using namespace mr;

namespace {
int free_port() {
  const int s = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  a.sin_port = 0;
  ::bind(s, reinterpret_cast<sockaddr*>(&a), sizeof a);
  socklen_t len = sizeof a;
  ::getsockname(s, reinterpret_cast<sockaddr*>(&a), &len);
  const int port = ntohs(a.sin_port);
  ::close(s);
  return port;
}

std::string slurp(const std::filesystem::path& p) {
  std::ifstream in(p);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}
}  // namespace

// The serve process must not hold the lake's DuckDB lock once its panel is built: a replay CLI run alongside
// it has to succeed. Runs the real binary on a small replay data dir.
TEST_CASE("serve mode releases the lake: a concurrent replay CLI succeeds") {
  const auto dir = test::temp_dir("serve_lock");
  const std::vector<std::string> tickers = {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF"};
  std::string sp = "ticker,name,sector\n";
  for (std::size_t k = 0; k < tickers.size(); ++k) sp += tickers[k] + "," + tickers[k] + " Inc," + (k % 2 ? "Energy" : "Utilities") + "\n";
  test::write_file(dir / "universe" / "sp500.csv", sp);
  test::write_file(dir / "universe" / "funds.csv", "ticker,tracks\n");
  test::write_file(dir / "portfolio.json",
                   R"({"initial_cash": 1000000, "inception": "2025-01-01", "holdings": [{"ticker": "AAA", "weight": 1.0}]})");
  {
    BarStore store(dir / "lake");
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const TimePoint day0 = (now / 86400 - 80) * 86400 + 4 * 3600;
    for (std::size_t k = 0; k < tickers.size(); ++k) {
      std::vector<Bar> bars;
      double c = 100.0;
      for (int d = 0; d < 80; ++d) {
        const TimePoint t = day0 + d * 86400;
        const auto wd = (t / 86400 + 4) % 7;  // 0 = Sunday
        if (wd == 0 || wd == 6) continue;
        const double o = c;
        c *= 1.0 + 0.01 * std::sin(0.3 * d + static_cast<double>(k));
        bars.push_back({t, o, std::max(o, c) * 1.01, std::min(o, c) * 0.99, c, 2e6, (o + c) / 2});
      }
      store.merge(tickers[k], Timeframe::Day, bars);
      store.save(tickers[k], Timeframe::Day);
    }
    store.flush();
  }
  const int port = free_port();
  const std::string bin = MR_BINARY, d = dir.string(), p = std::to_string(port);
  const std::string common = " --mode replay --universe sp500 --data '" + d + "'";
  test::write_file(dir / "run.sh",
                   "'" + bin + "' --serve" + common + " --port " + p + " --web '" + d + "/web' >'" + d + "/serve.log' 2>&1 &\n"
                   "S=$!\n"
                   "for i in $(seq 1 300); do grep -q serving '" + d + "/serve.log' && break; kill -0 $S 2>/dev/null || break; sleep 0.1; done\n"
                   "grep -q serving '" + d + "/serve.log' || { kill $S 2>/dev/null; exit 3; }\n"
                   "'" + bin + "'" + common + " >'" + d + "/cli.log' 2>&1\n"
                   "R=$?\nkill $S\nwait $S\nexit $R\n");
  const int rc = std::system(("bash '" + (dir / "run.sh").string() + "'").c_str());
  INFO("serve log:\n" << slurp(dir / "serve.log") << "\ncli log:\n" << slurp(dir / "cli.log"));
  REQUIRE(WIFEXITED(rc));
  CHECK(WEXITSTATUS(rc) == 0);
  CHECK(slurp(dir / "cli.log").find("HILLS") != std::string::npos);
}
