// M4: the walk-forward with external signals (--wf-external) and the bit-identity of runs without them.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "test_util.hpp"
#include "walkforward/report.hpp"
#include "walkforward/walkforward.hpp"

using namespace mr;
namespace fs = std::filesystem;

namespace {

// Same market as test_wf_driver.cpp: ~400 daily synthetic bars, node 0 renamed "VOO".
const Panel& ext_panel() {
  static const Panel p = [] {
    SyntheticConfig cfg;
    cfg.bars = 400;
    cfg.size_sigma = 0.5;
    BarStore store(test::temp_dir("wf_ext"));
    auto secs = generate_synthetic(cfg, store);
    std::vector<std::string> tickers;
    for (auto& s : secs) tickers.push_back(s.ticker);
    Panel q = build_panel(store, tickers, cfg.tf);
    q.tickers[0] = "VOO";
    return q;
  }();
  return p;
}

WalkForwardParams ext_params() {
  WalkForwardParams p;
  p.rebalance = Rebalance::Weekly;
  p.warmup_bars = 100;
  p.min_dollar_volume = 0;
  const Panel& panel = ext_panel();
  p.bt.base = {{"VOO", 0.5}, {panel.tickers[1], 0.3}, {panel.tickers[2], 0.2}};
  p.bt.k = 5;
  p.blend.train_months = 12;
  p.blend.embargo = 1;
  p.blend.gate_months = 12;
  p.blend.gate_min = 4;
  p.blend.gate_t = -1e9;
  return p;
}

std::string slurp(const fs::path& f) {
  std::ifstream in(f);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

const fs::path kFixture = fs::path(MR_SOURCE_DIR) / "tests" / "fixtures" / "wf_m3a";

}  // namespace

// The fixture was written by the M3a code (before the runtime signal list); MR_WRITE_WF_FIXTURE=1 rewrites it.
TEST_CASE("walkforward: a run without external signals reproduces the stored M3a outputs byte for byte") {
  const Panel& panel = ext_panel();
  const WalkForwardParams p = ext_params();
  const WalkForwardResult r = run_walkforward(panel, p);
  const auto out = test::temp_dir("wf_fixture");
  const auto dir = write_report(r, p, panel, out, "fixture");
  if (const char* w = std::getenv("MR_WRITE_WF_FIXTURE"); w && std::string(w) == "1") {
    fs::create_directories(kFixture);
    for (const char* f : {"results.json", "report.md", "equity.csv"}) fs::copy_file(dir / f, kFixture / f, fs::copy_options::overwrite_existing);
    fs::copy_file(out / "registry.csv", kFixture / "registry.csv", fs::copy_options::overwrite_existing);
  }
  for (const char* f : {"results.json", "report.md", "equity.csv"}) {
    CAPTURE(f);
    REQUIRE(fs::exists(kFixture / f));
    CHECK(slurp(dir / f) == slurp(kFixture / f));
  }
  CHECK(slurp(out / "registry.csv") == slurp(kFixture / "registry.csv"));
}
