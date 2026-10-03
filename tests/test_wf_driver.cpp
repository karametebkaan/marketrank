#include <doctest/doctest.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

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
    CHECK(x.signal == y.signal);
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
  std::set<std::string> want{"blend", "bench:buyhold", "bench:rebalanced", "bench:VOO", "sleeve:blend",
                             "bench:ew_eligible"};
  for (std::size_t s = 0; s < kSignals; ++s) {
    want.insert("sig:" + std::string(to_string(static_cast<Signal>(s))));
    want.insert("sleeve:" + std::string(to_string(static_cast<Signal>(s))));
  }
  CHECK(keys == want);
  // IC table: one row per signal x horizon, with samples.
  CHECK(r.ic_table.size() == kSignals * p.ic_horizons.size());
  bool any_n = false;
  for (const auto& row : r.ic_table) any_n = any_n || row.all.n > 0;
  CHECK(any_n);
}

TEST_CASE("walkforward: causality - bars after d_k do not change decisions at months <= k") {
  const Panel& panel = shared_panel();
  WalkForwardParams p = wf_params(Rebalance::Weekly);
  // The mask must depend on the (perturbed) dollar volumes so eligibility leaks show: top third by a 3-bar median,
  // where one strongly perturbed bar moves the median.
  p.top_n = panel.N() / 3;
  p.elig_window = 3;
  const WalkForwardResult a = run_walkforward(panel, p);
  REQUIRE(a.dates.size() >= 30);
  {
    std::size_t changes = 0;
    for (std::size_t m = 1; m < a.eligible.size(); ++m) changes += a.eligible[m] != a.eligible[m - 1] ? 1 : 0;
    REQUIRE(changes > 0);
    for (const auto& e : a.eligible) CHECK(static_cast<std::size_t>(std::count(e.begin(), e.end(), true)) <= p.top_n);
  }
  bool any_gate = false;
  for (const auto& b : a.blend) any_gate = any_gate || b.gate_open;
  REQUIRE(any_gate);  // the score path is exercised
  for (std::size_t k : {a.dates.size() / 3, a.dates.size() / 2, a.dates.size() - 3}) {
    Panel q = panel;
    const std::size_t cut = a.dates[k] + 1;
    for (std::size_t t = cut; t < q.T(); ++t)  // every bar after d_k, including the execution bar d_k + 1
      for (std::size_t i = 0; i < q.N(); ++i) {
        const double f = 1.0 + 0.03 * std::sin(static_cast<double>(t * 31 + i * 7));
        for (auto* v : {&q.open, &q.close, &q.vwap, &q.high, &q.low})
          if (std::isfinite((*v)[q.idx(t, i)])) (*v)[q.idx(t, i)] *= f;
        if (std::isfinite(q.volume[q.idx(t, i)])) q.volume[q.idx(t, i)] *= (i + t) % 2 ? 50.0 : 0.02;
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
    CHECK(header == "run_id,strategy,params_hash,cost_bps,top_n,sharpe_daily,T,ann_excess,ir_daily");
  }
  write_report(r, p, panel, out, "run2");
  CHECK(count_lines(out / "registry.csv") == 1 + 2 * strategies);
  std::ifstream md(dir1 / "report.md");
  const std::string text((std::istreambuf_iterator<char>(md)), std::istreambuf_iterator<char>());
  CHECK(text.find("pending") != std::string::npos);  // no large-cap sibling run
  CHECK(text.find("pi_rel_size") != std::string::npos);
  CHECK(params_hash(p).size() == 8);
  // c3 is the deflated IR of the excess; DSR(total) is informational only.
  CHECK(text.find("| c3: deflated IR (excess vs buy-and-hold) > 0.95 |") != std::string::npos);
  CHECK(text.find("DSR(total), informational") != std::string::npos);
  CHECK(text.find("DSR_excess") != std::string::npos);
  // Positive years are counts ("8 of 10"), not shares ("0.80 of 10").
  CHECK_FALSE(std::regex_search(text, std::regex("[0-9]\\.[0-9]+ of [0-9]")));
  CHECK(std::regex_search(text, std::regex("\\| [0-9]+ of [0-9]+ \\|")));
  std::ifstream js(dir1 / "results.json");
  const auto j = nlohmann::json::parse(js);
  for (const auto& s : j.at("strategies")) {
    CHECK(s.contains("ir_daily"));
    CHECK(s.contains("dsr_excess"));
    CHECK(s.contains("dsr"));
    CHECK(s.contains("skew_e"));
    CHECK(s.contains("kurt_e"));
  }
  CHECK(j.at("registry").contains("n_ir_trials"));
  CHECK(j.at("registry").contains("trial_ir_var"));
  // Registry rows carry ir_daily (9th column) and count as IR trials; sleeves are secondaries, not trials.
  const RegistryStats st = registry_stats(out / "registry.csv");
  CHECK(st.n_ir_trials == 2 * strategies);
  {
    std::ifstream in(out / "registry.csv");
    const std::string reg((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(reg.find("sleeve:") == std::string::npos);
  }
  // Secondaries (reported, not gated): every strategy against the rebalanced base, every sleeve against the
  // equal-weight eligible universe.
  const auto& sec = j.at("secondary");
  REQUIRE(sec.at("vs_rebalanced").size() == strategies);
  for (const auto& s : sec.at("vs_rebalanced"))
    for (const char* k : {"name", "ann_excess", "excess_ci95", "ir", "dsr_excess", "years", "year_hit_rate"})
      CHECK(s.contains(k));
  REQUIRE(sec.at("sleeves").size() == strategies);
  CHECK(sec.at("sleeves")[0].at("name") == "sleeve:blend");
  CHECK(sec.at("sleeves")[0].at("benchmark") == "bench:ew_eligible");
  for (const auto& s : j.at("strategies")) CHECK(s.at("name").get<std::string>().rfind("sleeve:", 0) != 0);
  CHECK(text.find("## Secondary (reported, not gated)") != std::string::npos);
  CHECK(text.find("against the base rebalanced on the same calendar") != std::string::npos);
  CHECK(text.find("| sleeve:blend |") != std::string::npos);
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

TEST_CASE("walkforward: IC table is non-overlapping per horizon, bucketed by UTC year") {
  const Panel& panel = shared_panel();
  const WalkForwardParams p = wf_params(Rebalance::Monthly);
  const WalkForwardResult r = run_walkforward(panel, p);
  const std::size_t T = panel.T();
  for (const auto& row : r.ic_table) {
    CAPTURE(row.signal);
    CAPTURE(row.h);
    const auto h = static_cast<std::size_t>(row.h);
    std::map<int, std::size_t> per_year;  // sample bars per UTC year
    std::size_t n = 0;
    for (std::size_t t = p.warmup_bars; t + 1 + h < T; t += h) {
      ++per_year[civil_from_days(floor_div(panel.times[t], 86400)).y];
      ++n;
    }
    CHECK(row.all.n == n);  // every sample bar t = warmup + j*h has a finite IC on this market
    REQUIRE(row.by_year.size() == per_year.size());
    if (row.h == 1) CHECK(per_year.size() == 2);  // daily samples span 2025 and 2026 (calendar-day synthetic bars)
    double weighted = 0;
    std::size_t j = 0;
    for (const auto& [year, count] : per_year) {
      CHECK(row.by_year[j].first == year);
      weighted += row.by_year[j].second * static_cast<double>(count);
      ++j;
    }
    CHECK(weighted / static_cast<double>(n) == doctest::Approx(row.all.mean).epsilon(1e-12));
  }
}

TEST_CASE("write_report: strategies without days are not registry trials; run ids are idempotent") {
  const Panel& panel = shared_panel();
  const WalkForwardParams p = wf_params(Rebalance::Monthly);
  WalkForwardResult r = run_walkforward(panel, p);
  for (auto& [k, c] : r.curves)
    if (k == "sig:pulse20") c = EquityCurve{};
  const auto out = test::temp_dir("wf_registry");
  write_report(r, p, panel, out, "a");
  CHECK(count_lines(out / "registry.csv") == 1 + kSignals);  // blend + 7 signals
  {
    std::ifstream in(out / "registry.csv");
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(text.find("sig:pulse20") == std::string::npos);
  }
  write_report(r, p, panel, out, "b");
  CHECK(count_lines(out / "registry.csv") == 1 + 2 * kSignals);
  write_report(r, p, panel, out, "a");  // same id, same params: its rows are replaced, not appended
  CHECK(count_lines(out / "registry.csv") == 1 + 2 * kSignals);
  CHECK(registry_stats(out / "registry.csv").n_trials == 2 * kSignals);
  WalkForwardParams other = p;
  other.bt.cost_bps = 25;
  CHECK_THROWS_AS(write_report(r, other, panel, out, "a"), std::invalid_argument);  // same id, other params
  CHECK(count_lines(out / "registry.csv") == 1 + 2 * kSignals);
  CHECK_FALSE(std::filesystem::exists(out / "registry.csv.tmp"));
}

TEST_CASE("registry_stats: finite sharpe rows only, sample variance") {
  const auto dir = test::temp_dir("wf_regstats");
  const auto path = test::write_file(dir / "registry.csv",
                                     "run_id,strategy,params_hash,cost_bps,top_n,sharpe_daily,T,ann_excess\n"
                                     "r1,blend,aaaaaaaa,10,0,0.01,100,0.1\n"
                                     "r1,sig:score,aaaaaaaa,10,0,0.03,100,0.1\n"
                                     "r1,sig:pulse1,aaaaaaaa,10,0,nan,100,0.1\n"
                                     "garbage\n"
                                     "\n"
                                     "r2,blend,bbbbbbbb,10,0,0.05,100,0.1\n");
  const RegistryStats st = registry_stats(path);
  CHECK(st.n_trials == 3);
  CHECK(st.trial_sr_var == doctest::Approx(0.0004).epsilon(1e-12));  // values .01 .03 .05: mean .03, ss 8e-4 / 2
  CHECK(st.n_ir_trials == 0);  // 8-column rows carry no IR
  CHECK(registry_stats(dir / "missing.csv").n_trials == 0);
}

TEST_CASE("registry_stats: mixed 8- and 9-column rows - only rows with a finite ir_daily are IR trials") {
  const auto dir = test::temp_dir("wf_regstats_ir");
  const auto path = test::write_file(dir / "registry.csv",
                                     "run_id,strategy,params_hash,cost_bps,top_n,sharpe_daily,T,ann_excess,ir_daily\n"
                                     "r1,blend,aaaaaaaa,10,0,0.01,100,0.1\n"          // old row: Sharpe trial only
                                     "r1,sig:score,aaaaaaaa,10,0,0.03,100,0.1,-0.02\n"
                                     "r1,sig:pulse1,aaaaaaaa,10,0,0.07,100,0.1,nan\n"  // Sharpe yes, IR no
                                     "r2,blend,bbbbbbbb,10,0,nan,100,0.1,0.04\n"       // IR yes, Sharpe no
                                     "r2,sig:score,bbbbbbbb,10,0,0.05,100,0.1,0.01\n");
  const RegistryStats st = registry_stats(path);
  CHECK(st.n_trials == 4);
  CHECK(st.n_ir_trials == 3);
  // IR values -0.02, 0.04, 0.01: mean 0.01, ss 0.0009 + 0.0009 + 0 = 0.0018, / 2
  CHECK(st.trial_ir_var == doctest::Approx(0.0009).epsilon(1e-12));
}

TEST_CASE("check_registry_conflict: read-only pre-pass check of the run id") {
  const auto dir = test::temp_dir("wf_precheck");
  CHECK_NOTHROW(check_registry_conflict(dir, "a", "aaaaaaaa"));  // no registry yet
  test::write_file(dir / "registry.csv",
                   "run_id,strategy,params_hash,cost_bps,top_n,sharpe_daily,T,ann_excess,ir_daily\n"
                   "a,blend,aaaaaaaa,10,0,0.01,100,0.1,0.02\n");
  CHECK_NOTHROW(check_registry_conflict(dir, "a", "aaaaaaaa"));  // same params: rows will be replaced
  CHECK_NOTHROW(check_registry_conflict(dir, "b", "bbbbbbbb"));
  CHECK_THROWS_AS(check_registry_conflict(dir, "a", "bbbbbbbb"), std::invalid_argument);
  CHECK_FALSE(std::filesystem::exists(dir / "registry.csv.lock"));  // read-only: no lock file, no rewrite
}

TEST_CASE("rereport: regenerating results.json and report.md from the stored files reproduces them") {
  const Panel& panel = shared_panel();
  const WalkForwardParams p = wf_params(Rebalance::Weekly);
  const WalkForwardResult r = run_walkforward(panel, p);
  const auto out = test::temp_dir("wf_rereport");
  const auto dir = write_report(r, p, panel, out, "rr");
  auto slurp = [](const std::filesystem::path& f) {
    std::ifstream in(f);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  };
  const std::string json0 = slurp(dir / "results.json"), md0 = slurp(dir / "report.md"),
                    reg0 = slurp(out / "registry.csv"), eq0 = slurp(dir / "equity.csv");
  std::filesystem::remove(dir / "report.md");
  CHECK(rereport(out, "rr") == dir);
  CHECK(slurp(dir / "results.json") == json0);
  CHECK(slurp(dir / "report.md") == md0);
  CHECK(slurp(out / "registry.csv") == reg0);  // same rows replaced in place
  CHECK(slurp(dir / "equity.csv") == eq0);      // stored curves are inputs, never rewritten
  CHECK_THROWS(rereport(out, "missing"));
}

TEST_CASE("walkforward_params: blend windows follow the rebalance calendar unless overridden") {
  auto blend_is = [](const BlendParams& b, std::size_t tr, std::size_t em, std::size_t g, std::size_t mn) {
    return b.train_months == tr && b.embargo == em && b.gate_months == g && b.gate_min == mn;
  };
  CHECK(blend_is(WalkForwardParams{}.blend, 156, 1, 104, 52));
  CHECK(blend_is(blend_defaults(Rebalance::Weekly), 156, 1, 104, 52));
  CHECK(blend_is(blend_defaults(Rebalance::Monthly), 36, 1, 24, 12));
  const std::vector<std::string> base{"--mode", "replay", "--walkforward"};
  auto with = [&](std::vector<std::string> extra) {
    auto v = base;
    v.insert(v.end(), extra.begin(), extra.end());
    return walkforward_params(parse_cli(v));
  };
  CHECK(blend_is(with({}).blend, 156, 1, 104, 52));
  CHECK(blend_is(with({"--wf-rebalance", "weekly"}).blend, 156, 1, 104, 52));
  CHECK(blend_is(with({"--wf-rebalance", "monthly"}).blend, 36, 1, 24, 12));
  CHECK(with({"--wf-rebalance", "monthly"}).rebalance == Rebalance::Monthly);
  CHECK(blend_is(with({"--wf-rebalance", "monthly", "--wf-blend", "10/2/8/4"}).blend, 10, 2, 8, 4));
  CHECK(blend_is(with({"--wf-blend", "10/2/8/4", "--wf-rebalance", "monthly"}).blend, 10, 2, 8, 4));
  const WalkForwardParams w = with({"--wf-top-n", "500", "--wf-cost-bps", "25", "--wf-tilt", "0.3", "--wf-k", "7",
                                    "--wf-warmup", "60", "--wf-largecap-run", "lc"});
  CHECK(w.top_n == 500);
  CHECK(w.bt.cost_bps == 25.0);
  CHECK(w.bt.tilt == 0.3);
  CHECK(w.bt.k == 7);
  CHECK(w.warmup_bars == 60);
  CHECK(w.largecap_run == "lc");
  CHECK_THROWS_AS(with({"--wf-blend", "10/2/8"}), std::invalid_argument);
  CHECK_THROWS_AS(with({"--wf-blend", "0/1/8/4"}), std::invalid_argument);
  CHECK_THROWS_AS(with({"--wf-blend", "10/0/8/4"}), std::invalid_argument);  // embargo >= 1: label(m) is future
}

TEST_CASE("cli: walk-forward flag validation") {
  CHECK(parse_cli({}).warnings.empty());
  CHECK(parse_cli({"--mode", "replay", "--walkforward", "--wf-k", "5"}).warnings.empty());
  const CliArgs w = parse_cli({"--mode", "replay", "--wf-k", "5", "--wf-out", "x"});
  REQUIRE(w.warnings.size() == 1);
  CHECK(w.warnings[0].find("--walkforward") != std::string::npos);
  CHECK_THROWS_AS(parse_cli({"--mode", "replay", "--walkforward", "--serve"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--mode", "replay", "--walkforward", "--export-slice"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--mode", "replay", "--walkforward", "--shock", "AAPL:-5"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--mode", "replay", "--walkforward", "--wf-largecap-run", "a/b"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--mode", "replay", "--walkforward", "--wf-largecap-run", ".."}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--mode", "replay", "--walkforward", "--wf-run-id", "a,b"}), std::invalid_argument);
  CHECK_NOTHROW(check_run_id("2026-10-02T1200Z-abcd1234", "run id"));
  CHECK(parse_cli({"--wf-rereport", "weekly-10bps", "--wf-out", "x"}).wf_rereport == "weekly-10bps");
  CHECK(parse_cli({"--wf-rereport", "weekly-10bps"}).warnings.empty());
  CHECK_THROWS_AS(parse_cli({"--wf-rereport", "a/b"}), std::invalid_argument);
  CHECK_THROWS_AS(check_run_id("", "run id"), std::invalid_argument);
}
