#include <doctest/doctest.h>
#include <omp.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "cli/args.hpp"
#include "core/time.hpp"
#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "test_util.hpp"
#include "walkforward/report.hpp"
#include "walkforward/walkforward.hpp"

using namespace mr;

namespace {

// ~400 daily synthetic bars; node 0 is renamed "VOO" so the single-ticker benchmark exists.
Panel wf_panel() {
  SyntheticConfig cfg;
  cfg.bars = 400;
  cfg.size_sigma = 0.5;
  BarStore store(test::temp_dir("wf_driver"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  Panel p = build_panel(store, tickers, cfg.tf);
  p.tickers[0] = "VOO";
  return p;
}

const Panel& shared_panel() {
  static const Panel p = wf_panel();
  return p;
}

WalkForwardParams wf_params(Rebalance r) {
  WalkForwardParams p;
  p.rebalance = r;
  p.warmup_bars = 100;
  p.min_dollar_volume = 0;
  const Panel& panel = shared_panel();
  p.bt.base = {{"VOO", 0.5}, {panel.tickers[1], 0.3}, {panel.tickers[2], 0.2}};
  p.bt.k = 5;
  // Short windows so weights form and (gate_t very low) the gate opens inside ~60 weekly periods.
  p.blend.train_months = 12;
  p.blend.embargo = 1;
  p.blend.gate_months = 12;
  p.blend.gate_min = 4;
  p.blend.gate_t = -1e9;
  return p;
}

bool same_mask(const std::vector<bool>& a, const std::vector<bool>& b) { return a == b; }

void check_same_results(const WalkForwardResult& a, const WalkForwardResult& b) {
  REQUIRE(a.dates == b.dates);
  REQUIRE(a.months.size() == b.months.size());
  for (std::size_t m = 0; m < a.months.size(); ++m) {
    for (std::size_t s = 0; s < kSignals; ++s) CHECK(test::same_values(a.months[m].z[s], b.months[m].z[s]));
    CHECK(test::same_values(a.months[m].label, b.months[m].label));
    CHECK(same_mask(a.eligible[m], b.eligible[m]));
    CHECK(a.blend[m].w == b.blend[m].w);
    CHECK(a.blend[m].gate_open == b.blend[m].gate_open);
    CHECK(test::same_values(a.blend[m].score, b.blend[m].score));
  }
  REQUIRE(a.ic_table.size() == b.ic_table.size());
  for (std::size_t k = 0; k < a.ic_table.size(); ++k) {
    const auto &x = a.ic_table[k], &y = b.ic_table[k];
    CHECK(x.s == y.s);
    CHECK(x.h == y.h);
    CHECK(x.all.n == y.all.n);
    CHECK((x.all.mean == y.all.mean || (std::isnan(x.all.mean) && std::isnan(y.all.mean))));
    CHECK((x.all.t == y.all.t || (std::isnan(x.all.t) && std::isnan(y.all.t))));
    REQUIRE(x.by_year.size() == y.by_year.size());
    for (std::size_t j = 0; j < x.by_year.size(); ++j) {
      CHECK(x.by_year[j].first == y.by_year[j].first);
      CHECK(test::same_values({x.by_year[j].second}, {y.by_year[j].second}));
    }
  }
  REQUIRE(a.curves.size() == b.curves.size());
  for (std::size_t c = 0; c < a.curves.size(); ++c) {
    CHECK(a.curves[c].first == b.curves[c].first);
    CHECK(a.curves[c].second.t == b.curves[c].second.t);
    CHECK(test::same_values(a.curves[c].second.value, b.curves[c].second.value));
  }
}

std::size_t count_lines(const std::filesystem::path& p) {
  std::ifstream in(p);
  std::size_t n = 0;
  for (std::string line; std::getline(in, line);)
    if (!line.empty()) ++n;
  return n;
}

}  // namespace

TEST_CASE("walkforward: monthly dates are month-end bars and every curve key exists") {
  const Panel& panel = shared_panel();
  const WalkForwardParams p = wf_params(Rebalance::Monthly);
  const WalkForwardResult r = run_walkforward(panel, p);
  auto month_of = [&](std::size_t t) {
    const Civil c = civil_from_days(floor_div(panel.times[t], 86400));
    return c.y * 12 + static_cast<int>(c.m);
  };
  std::vector<std::size_t> expect;
  for (std::size_t t = p.warmup_bars; t + 2 <= panel.T(); ++t)
    if (month_of(t + 1) != month_of(t)) expect.push_back(t);
  REQUIRE(expect.size() >= 6);
  CHECK(r.dates == expect);
  CHECK(r.months.size() == r.dates.size());
  CHECK(r.eligible.size() == r.dates.size());
  CHECK(r.blend.size() == r.dates.size());
  for (std::size_t m = 0; m < r.months.size(); ++m) {
    CHECK(r.months[m].label.size() == panel.N());
    CHECK(r.eligible[m].size() == panel.N());
  }
  // Last month has no next rebalance: label all NaN.
  for (double x : r.months.back().label) CHECK(std::isnan(x));
  std::set<std::string> keys;
  for (const auto& [k, c] : r.curves) {
    keys.insert(k);
    CHECK_FALSE(c.value.empty());
  }
  std::set<std::string> want{"blend", "bench:buyhold", "bench:rebalanced", "bench:VOO"};
  for (std::size_t s = 0; s < kSignals; ++s) want.insert("sig:" + std::string(to_string(static_cast<Signal>(s))));
  CHECK(keys == want);
  // IC table: one row per signal x horizon, with samples.
  CHECK(r.ic_table.size() == kSignals * p.ic_horizons.size());
  bool any_n = false;
  for (const auto& row : r.ic_table) any_n = any_n || row.all.n > 0;
  CHECK(any_n);
}

TEST_CASE("walkforward: causality - bars after d_k + 1 do not change decisions at months <= k") {
  const Panel& panel = shared_panel();
  const WalkForwardParams p = wf_params(Rebalance::Weekly);
  const WalkForwardResult a = run_walkforward(panel, p);
  REQUIRE(a.dates.size() >= 30);
  bool any_gate = false;
  for (const auto& b : a.blend) any_gate = any_gate || b.gate_open;
  REQUIRE(any_gate);  // the score path is exercised
  for (std::size_t k : {a.dates.size() / 3, a.dates.size() / 2, a.dates.size() - 3}) {
    Panel q = panel;
    const std::size_t cut = a.dates[k] + 1;
    for (std::size_t t = cut + 1; t < q.T(); ++t)
      for (std::size_t i = 0; i < q.N(); ++i) {
        const double f = 1.0 + 0.03 * std::sin(static_cast<double>(t * 31 + i * 7));
        for (auto* v : {&q.open, &q.close, &q.vwap, &q.high, &q.low})
          if (std::isfinite((*v)[q.idx(t, i)])) (*v)[q.idx(t, i)] *= f;
        if (std::isfinite(q.volume[q.idx(t, i)])) q.volume[q.idx(t, i)] *= 2.0 - f;
      }
    const WalkForwardResult b = run_walkforward(q, p);
    REQUIRE(b.dates.size() == a.dates.size());
    for (std::size_t m = 0; m <= k; ++m) {
      CAPTURE(k);
      CAPTURE(m);
      for (std::size_t s = 0; s < kSignals; ++s) CHECK(test::same_values(a.months[m].z[s], b.months[m].z[s]));
      CHECK(a.eligible[m] == b.eligible[m]);
      CHECK(a.blend[m].w == b.blend[m].w);
      CHECK(a.blend[m].gate_open == b.blend[m].gate_open);
      CHECK(test::same_values(a.blend[m].score, b.blend[m].score));
    }
  }
}

TEST_CASE("walkforward: deterministic across thread counts") {
  const int saved = omp_get_max_threads();
  const WalkForwardParams p = wf_params(Rebalance::Weekly);
  omp_set_num_threads(1);
  const WalkForwardResult a = run_walkforward(shared_panel(), p);
  omp_set_num_threads(8);
  const WalkForwardResult b = run_walkforward(shared_panel(), p);
  omp_set_num_threads(saved);
  check_same_results(a, b);
}

TEST_CASE("walkforward: write_report creates the files and registry grows per strategy") {
  const Panel& panel = shared_panel();
  const WalkForwardParams p = wf_params(Rebalance::Monthly);
  const WalkForwardResult r = run_walkforward(panel, p);
  const auto out = test::temp_dir("wf_report");
  const std::size_t strategies = 1 + kSignals;
  const auto dir1 = write_report(r, p, panel, out, "run1");
  CHECK(dir1 == out / "run1");
  for (const char* f : {"results.json", "equity.csv", "trades.csv", "report.md"})
    CHECK(std::filesystem::exists(dir1 / f));
  REQUIRE(std::filesystem::exists(out / "registry.csv"));
  CHECK(count_lines(out / "registry.csv") == 1 + strategies);
  {
    std::ifstream in(out / "registry.csv");
    std::string header;
    std::getline(in, header);
    CHECK(header == "run_id,strategy,params_hash,cost_bps,top_n,sharpe_daily,T,ann_excess");
  }
  write_report(r, p, panel, out, "run2");
  CHECK(count_lines(out / "registry.csv") == 1 + 2 * strategies);
  std::ifstream md(dir1 / "report.md");
  const std::string text((std::istreambuf_iterator<char>(md)), std::istreambuf_iterator<char>());
  CHECK(text.find("pending") != std::string::npos);  // no large-cap sibling run
  CHECK(text.find("pi_rel_size") != std::string::npos);
  CHECK(params_hash(p).size() == 8);
}

TEST_CASE("cli: --walkforward and --wf-* flags") {
  CliArgs d = parse_cli({});
  CHECK_FALSE(d.walkforward);
  CHECK(d.wf_rebalance == Rebalance::Weekly);
  CliArgs a = parse_cli({"--mode", "replay", "--walkforward", "--wf-rebalance", "monthly", "--wf-top-n", "500",
                         "--wf-cost-bps", "25", "--wf-tilt", "0.3", "--wf-k", "20", "--wf-out", "/tmp/wfx",
                         "--wf-run-id", "abc", "--wf-largecap-run", "lc1", "--wf-warmup", "60"});
  CHECK(a.walkforward);
  CHECK(a.wf_rebalance == Rebalance::Monthly);
  CHECK(a.wf_top_n == 500);
  CHECK(a.wf_cost_bps == 25.0);
  CHECK(a.wf_tilt == 0.3);
  CHECK(a.wf_k == 20);
  CHECK(a.wf_out == "/tmp/wfx");
  CHECK(a.wf_run_id == "abc");
  CHECK(a.wf_largecap_run == "lc1");
  CHECK(a.wf_warmup == 60);
  CHECK(d.wf_warmup == 252);
  CHECK_THROWS_AS(parse_cli({"--walkforward"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--mode", "alpaca", "--walkforward"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--mode", "replay", "--walkforward", "--wf-rebalance", "daily"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--mode", "replay", "--walkforward", "--wf-tilt", "1.5"}), std::invalid_argument);
  CHECK(cli_usage().find("--walkforward") != std::string::npos);
  CHECK(cli_usage().find("--wf-largecap-run") != std::string::npos);
}
