// M4: the walk-forward with external signals (--wf-external) and the bit-identity of runs without them.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "market/panel.hpp"
#include "pipeline/core_pipeline.hpp"
#include "market/synthetic_market.hpp"
#include "test_util.hpp"
#include "core/time.hpp"
#include "walkforward/external.hpp"
#include "walkforward/metrics.hpp"
#include "walkforward/report.hpp"
#include "walkforward/stats.hpp"
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

namespace {

// Raw values of built-in signal s at every bar (the walk-forward's own pass), keyed by bar.
std::map<std::size_t, std::vector<double>> raw_signal(const Panel& panel, const WalkForwardParams& p, Signal s) {
  CorePipeline pipe(panel.N(), p.core);
  SignalTracker tracker(panel.N());
  std::map<std::size_t, std::vector<double>> out;
  for (std::size_t t = 1; t < panel.T(); ++t) {
    const Frame f = pipe.step(panel, t);
    auto raw = tracker.update(f);
    out[t] = std::move(raw[static_cast<std::size_t>(s)]);
  }
  return out;
}

// CSV text "t,ticker,score" of the given bars (17 significant digits: doubles round-trip exactly).
std::string to_csv(const Panel& panel, const std::map<std::size_t, std::vector<double>>& raw,
                   const std::vector<std::size_t>& bars) {
  std::string s = "t,ticker,score\n";
  char buf[64];
  for (std::size_t t : bars)
    for (std::size_t i = 0; i < panel.N(); ++i) {
      const double v = raw.at(t)[i];
      if (std::isfinite(v)) std::snprintf(buf, sizeof buf, "%.17g", v);
      else std::snprintf(buf, sizeof buf, "nan");
      s += std::to_string(panel.times[t]) + "," + panel.tickers[i] + "," + buf + "\n";
    }
  return s;
}

const IcRow* find_ic(const WalkForwardResult& r, const std::string& name, int h) {
  for (const auto& row : r.ic_table)
    if (row.signal == name && row.h == h) return &row;
  return nullptr;
}

const EquityCurve* find_curve(const WalkForwardResult& r, const std::string& key) {
  for (const auto& [k, c] : r.curves)
    if (k == key) return &c;
  return nullptr;
}

bool same_row(const IcRow& a, const IcRow& b) {
  if (a.h != b.h || a.all.n != b.all.n || a.by_year.size() != b.by_year.size()) return false;
  if (!test::same_values({a.all.mean, a.all.t}, {b.all.mean, b.all.t})) return false;
  for (std::size_t j = 0; j < a.by_year.size(); ++j)
    if (a.by_year[j].first != b.by_year[j].first ||
        !test::same_values({a.by_year[j].second}, {b.by_year[j].second}))
      return false;
  return true;
}

}  // namespace

TEST_CASE("wf-external: a built-in signal's own scores at every bar reproduce its IC rows exactly") {
  const Panel& panel = ext_panel();
  WalkForwardParams p = ext_params();
  const auto raw = raw_signal(panel, p, Signal::Pulse5);
  std::vector<std::size_t> bars;
  for (std::size_t t = p.warmup_bars; t < panel.T(); ++t) bars.push_back(t);
  const ExternalSignal e = parse_external_signal("ext_pulse5", to_csv(panel, raw, bars), panel);
  CHECK(e.unknown_tickers == 0);
  p.externals = {external_tag(e)};
  const WalkForwardResult r = run_walkforward(panel, p, {e});
  REQUIRE(r.ic_table.size() == (kSignals + 1) * p.ic_horizons.size());
  for (int h : p.ic_horizons) {
    CAPTURE(h);
    const IcRow* a = find_ic(r, "pulse5", h);
    const IcRow* b = find_ic(r, "ext_pulse5", h);
    REQUIRE(a);
    REQUIRE(b);
    CHECK(a->all.n > 0);
    CHECK(same_row(*a, *b));
  }
  // Externals come after every built-in row.
  CHECK(r.ic_table[kSignals * p.ic_horizons.size()].signal == "ext_pulse5");
  // The CSV holds RAW scores: the harness z-scores (and winsorizes at +-3) externals exactly as built-ins. Feeding
  // already winsorized z-scores is not exact in general: re-standardizing can push values past +-3 (new ties).
}

TEST_CASE("wf-external: rebalance-date scores give the built-in's curves, rebalance ICs and leave built-ins alone") {
  const Panel& panel = ext_panel();
  WalkForwardParams p = ext_params();
  const WalkForwardResult base = run_walkforward(panel, p);
  const auto raw = raw_signal(panel, p, Signal::Score);
  // learned: scored span = rebalances 2 .. M-5, with a hole at 10 (held at the base inside the span).
  const std::size_t M = base.dates.size();
  REQUIRE(M > 20);
  auto in_learned = [&](std::size_t j) { return j >= 2 && j <= M - 5 && j != 10; };
  std::vector<std::size_t> bars;
  for (std::size_t j = 0; j < M; ++j)
    if (in_learned(j)) bars.push_back(base.dates[j]);
  const ExternalSignal e = parse_external_signal("learned", to_csv(panel, raw, bars), panel);
  const ExternalSignal e0 = parse_external_signal("B0", to_csv(panel, raw, base.dates), panel);
  p.externals = {external_tag(e), external_tag(e0)};
  const WalkForwardResult r = run_walkforward(panel, p, {e, e0});
  REQUIRE(r.externals.size() == 2);

  // Built-in results unchanged.
  REQUIRE(r.dates == base.dates);
  for (std::size_t k = 0; k < base.ic_table.size(); ++k) CHECK(same_row(r.ic_table[k], base.ic_table[k]));
  for (const auto& [k, c] : base.curves) {
    const EquityCurve* x = find_curve(r, k);
    REQUIRE(x);
    CHECK(test::same_values(x->value, c.value));
  }
  // Curve order: built-in sig: curves, then the externals, then the benchmarks; sleeves likewise.
  std::vector<std::string> keys;
  for (const auto& [k, c] : r.curves) keys.push_back(k);
  const auto pos = [&](const std::string& k) { return std::find(keys.begin(), keys.end(), k) - keys.begin(); };
  CHECK(pos("sig:learned") == 1 + static_cast<long>(kSignals));
  CHECK(pos("sig:B0") == 2 + static_cast<long>(kSignals));
  CHECK(pos("bench:buyhold") == 3 + static_cast<long>(kSignals));
  CHECK(pos("sleeve:B0") + 1 == pos("bench:ew_eligible"));

  // B0 (scores everywhere) = the built-in score signal, exactly.
  CHECK(test::same_values(find_curve(r, "sig:B0")->value, find_curve(r, "sig:score")->value));
  CHECK(test::same_values(find_curve(r, "sleeve:B0")->value, find_curve(r, "sleeve:score")->value));
  // learned is simulated over its scored span only: decisions 2 .. M-5 (the hole holds the base), cut at the close
  // of d_{M-4}; no burn-in periods.
  {
    std::vector<Decision> d;
    for (std::size_t j = 2; j <= M - 5; ++j)
      d.push_back(Decision{base.dates[j], in_learned(j) ? base.months[j].z[0] : std::vector<double>{}, base.eligible[j]});
    EquityCurve want = simulate(panel, p.bt, d);
    const TimePoint end = panel.times[base.dates[M - 4]];
    std::size_t keep = 0;
    while (keep < want.t.size() && want.t[keep] <= end) ++keep;
    REQUIRE(keep < want.t.size());
    want.t.resize(keep);
    want.value.resize(keep);
    const EquityCurve* got = find_curve(r, "sig:learned");
    CHECK(got->t == want.t);
    CHECK(test::same_values(got->value, want.value));
    CHECK(got->t.front() == panel.times[base.dates[2] + 1]);
    CHECK(got->t.back() == end);
    const EquityCurve* sl = find_curve(r, "sleeve:learned");
    CHECK(sl->t.front() == panel.times[base.dates[2] + 1]);
    CHECK(sl->t.back() == end);
  }
  CHECK(r.externals[0].scored_rebalances == M - 7);
  CHECK(r.externals[0].scored_bars == M - 7);
  CHECK(external_coverage_warning(r.externals[0]).empty());
  // Rebalance ICs against the period label.
  CHECK(r.externals[0].name == "learned");
  for (std::size_t j = 0; j < base.dates.size(); ++j) {
    const double want = spearman(base.months[j].z[0], base.months[j].label);
    CHECK(test::same_values({r.externals[1].rebalance_ic[j]}, {std::isfinite(want) ? want : NAN}));
    CHECK(test::same_values({r.externals[0].rebalance_ic[j]}, {!in_learned(j) || !std::isfinite(want) ? NAN : want}));
    CHECK(r.externals[0].scored[j] == in_learned(j));
    CHECK(r.externals[1].scored[j]);
  }
  // Rebalance-only IC samples are non-overlapping per horizon.
  for (int h : p.ic_horizons) {
    const IcRow* row = find_ic(r, "B0", h);
    REQUIRE(row);
    std::size_t n = 0, last = 0;
    bool have = false;
    for (std::size_t t : base.dates) {
      if (t + 1 + static_cast<std::size_t>(h) >= panel.T() || (have && t < last + static_cast<std::size_t>(h))) continue;
      have = true, last = t, ++n;
    }
    CAPTURE(h);
    CHECK(row->all.n == n);  // every sample has a finite IC on this market
    CHECK(n > 0);
  }
  CHECK_THROWS_AS(run_walkforward(panel, ext_params(), {e}), std::invalid_argument);  // tags missing from params
}

TEST_CASE("wf-external: report section, learned vs B0, registry rows, hash and rereport") {
  const Panel& panel = ext_panel();
  WalkForwardParams p = ext_params();
  const std::string hash0 = params_hash(p);
  const auto dates = walkforward_dates(panel, p);
  const auto score = raw_signal(panel, p, Signal::Score);
  const auto pulse = raw_signal(panel, p, Signal::Pulse5);
  const ExternalSignal e = parse_external_signal("learned", to_csv(panel, score, dates), panel);
  const ExternalSignal e0 = parse_external_signal("B0", to_csv(panel, pulse, dates), panel);
  p.externals = {external_tag(e), external_tag(e0)};
  CHECK(params_hash(p) != hash0);
  CHECK(describe(p).find(" ext=learned:" + e.digest + ";B0:" + e0.digest + ";") != std::string::npos);
  const WalkForwardResult r = run_walkforward(panel, p, {e, e0});
  const auto out = test::temp_dir("wf_ext_report");
  const auto dir = write_report(r, p, panel, out, "m4");
  const std::string md = slurp(dir / "report.md"), js = slurp(dir / "results.json"), reg = slurp(out / "registry.csv");
  CHECK(md.find("## External signals") != std::string::npos);
  CHECK(md.find("### learned vs B0") != std::string::npos);
  CHECK(md.find("| IC(learned) - IC(B0) |") != std::string::npos);
  CHECK(md.find("| tilt period return, learned - B0 |") != std::string::npos);
  CHECK(md.find("| learned | 1 |") != std::string::npos);  // IC row in the main table
  CHECK(reg.find(",sig:learned,") != std::string::npos);
  CHECK(reg.find(",sig:B0,") != std::string::npos);
  CHECK(reg.find("sleeve:") == std::string::npos);
  const auto j = nlohmann::json::parse(js);
  REQUIRE(j.at("external").size() == 2);
  CHECK(j.at("external")[0].at("rebalance_ic").size() == r.dates.size());
  CHECK(j.at("blend")[0].at("w").size() == kSignals);  // the blend stays over the built-ins
  // The paired IC difference equals the mean of the per-rebalance differences.
  {
    std::vector<double> d;
    for (std::size_t m = 0; m < r.dates.size(); ++m)
      if (std::isfinite(r.externals[0].rebalance_ic[m]) && std::isfinite(r.externals[1].rebalance_ic[m]))
        d.push_back(r.externals[0].rebalance_ic[m] - r.externals[1].rebalance_ic[m]);
    const MeanT mt = mean_t(d);
    char buf[64];
    std::snprintf(buf, sizeof buf, "| IC(learned) - IC(B0) | %.4f |", mt.mean);
    CHECK(md.find(buf) != std::string::npos);
  }
  fs::remove(dir / "report.md");
  rereport(out, "m4");
  CHECK(slurp(dir / "report.md") == md);
  CHECK(slurp(dir / "results.json") == js);
  CHECK(slurp(out / "registry.csv") == reg);
}

TEST_CASE("wf-external: CSV parsing") {
  const Panel& panel = ext_panel();
  const std::string t5 = std::to_string(panel.times[5]), t6 = std::to_string(panel.times[6]);
  const std::string a = panel.tickers[1], b = panel.tickers[2];
  const ExternalSignal e =
      parse_external_signal("x", "t,ticker,score\r\n" + t5 + "," + a + ",1.5\r\n" + t5 + "," + b + ",nan\n" + t5 +
                                     ",NOPE,2\n\n" + t6 + "," + a + ",-3\n",
                            panel);
  CHECK(e.unknown_tickers == 1);
  REQUIRE(e.scores.size() == 2);
  CHECK(e.scores.at(5)[1] == 1.5);
  CHECK(std::isnan(e.scores.at(5)[2]));
  CHECK(std::isnan(e.scores.at(5)[0]));
  CHECK(e.scores.at(6)[1] == -3.0);
  CHECK(e.digest.size() == 16);
  CHECK_THROWS(parse_external_signal("x", "t,ticker,value\n", panel));
  CHECK_THROWS(parse_external_signal("x", "t,ticker,score\n12345," + a + ",1\n", panel));  // not a bar time
  CHECK_THROWS(parse_external_signal("x", "t,ticker,score\n" + t5 + "," + a + ",1\n" + t5 + "," + a + ",2\n", panel));
  CHECK_THROWS(parse_external_signal("x", "t,ticker,score\n" + t5 + "," + b + ",nan\n" + t5 + "," + b + ",2\n", panel));
  CHECK_THROWS(parse_external_signal("x", "t,ticker,score\n" + t5 + "," + a + ",abc\n", panel));
  CHECK_THROWS(parse_external_signal("x", "t,ticker,score\n" + t5 + "," + a + ",inf\n", panel));
  CHECK_THROWS(parse_external_signal("x", "t,ticker,score\n" + t5 + "," + a + "\n", panel));
  CHECK_THROWS_AS(parse_external_signal("score", "t,ticker,score\n", panel), std::invalid_argument);
  CHECK_THROWS_AS(parse_external_signal("blend", "t,ticker,score\n", panel), std::invalid_argument);
  CHECK_THROWS_AS(parse_external_signal("a,b", "t,ticker,score\n", panel), std::invalid_argument);
  CHECK_THROWS_AS(load_external_signal("x", "/nonexistent/x.csv", panel), std::runtime_error);
  CHECK_NOTHROW(check_external_name("B0"));
  CHECK_NOTHROW(check_external_name("learned-v2.1"));
}

TEST_CASE("wf-external: scored span drives registry rows and the per-external M3a gate; paired periods need both") {
  const Panel& panel = ext_panel();
  WalkForwardParams p = ext_params();
  const auto dates = walkforward_dates(panel, p);
  const std::size_t M = dates.size();
  REQUIRE(M > 20);
  auto in_learned = [&](std::size_t j) { return j >= 2 && j <= M - 5 && j != 10; };
  std::vector<std::size_t> lb;
  for (std::size_t j = 0; j < M; ++j)
    if (in_learned(j)) lb.push_back(dates[j]);
  const ExternalSignal e = parse_external_signal("learned", to_csv(panel, raw_signal(panel, p, Signal::Score), lb), panel);
  const ExternalSignal e0 = parse_external_signal("B0", to_csv(panel, raw_signal(panel, p, Signal::Pulse5), dates), panel);
  p.externals = {external_tag(e), external_tag(e0)};
  const WalkForwardResult r = run_walkforward(panel, p, {e, e0});
  const auto out = test::temp_dir("wf_ext_gate");
  const auto dir = write_report(r, p, panel, out, "g");
  const std::string md = slurp(dir / "report.md");
  const auto j = nlohmann::json::parse(slurp(dir / "results.json"));

  const EquityCurve& curve = *find_curve(r, "sig:learned");
  const EquityCurve& bh = *find_curve(r, "bench:buyhold");
  const Perf perf = performance(curve, bh);
  CHECK(perf.T == curve.t.size());
  CHECK(perf.T < performance(*find_curve(r, "blend"), bh).T);
  // Registry row of sig:learned: the span's T and IR.
  {
    std::ifstream in(out / "registry.csv");
    bool found = false;
    for (std::string line; std::getline(in, line);)
      if (line.rfind("g,sig:learned,", 0) == 0) {
        found = true;
        CHECK(line.find("," + std::to_string(perf.T) + ",") != std::string::npos);
      }
    CHECK(found);
  }
  // Gate per external in results.json.
  const auto& g = j.at("external")[0].at("gate");
  CHECK(j.at("external")[0].at("name") == "learned");
  CHECK(g.at("span")[0] == format_rfc3339(curve.t.front()));
  CHECK(g.at("span")[1] == format_rfc3339(curve.t.back()));
  CHECK(g.at("c5") == "pending");
  CHECK(g.at("pass") == "pending");
  CHECK(g.at("c1").get<bool>() == (perf.ann_excess > 0 && perf.excess_ci95.lo > 0));
  CHECK(g.at("c2").get<bool>() == (perf.year_hit_rate >= 0.6));
  for (const char* k : {"c3", "c4", "core_pass", "c3_dsr_excess", "base_max_drawdown", "reason"}) CHECK(g.contains(k));
  CHECK(j.at("external")[0].at("scored_rebalances") == M - 7);
  CHECK(j.at("external")[0].at("scored").size() == M);
  CHECK(md.find("**Span rule.**") != std::string::npos);
  CHECK(md.find("### M3a gate per external") != std::string::npos);
  CHECK(md.find("| learned | " + std::string(g.at("c1").get<bool>() ? "pass" : "FAIL") + " | ") != std::string::npos);
  CHECK(md.find("scored by BOTH externals") != std::string::npos);

  // Paired periods: only decisions scored by both (learned's span minus its hole), each with a finite return.
  {
    std::size_t n_tilt = 0, n_ic = 0;
    for (std::size_t m = 0; m + 1 < M; ++m)
      if (in_learned(m)) ++n_tilt;
    for (std::size_t m = 0; m < M; ++m)
      if (in_learned(m) && std::isfinite(r.externals[0].rebalance_ic[m]) && std::isfinite(r.externals[1].rebalance_ic[m]))
        ++n_ic;
    CHECK(n_tilt == M - 7);
    const auto row_n = [&](const std::string& label) {
      const auto k = md.find("| " + label + " |");
      REQUIRE(k != std::string::npos);
      const std::string line = md.substr(k, md.find('\n', k) - k);
      // "| label | mean | t | n | years |": the 4th cell is n.
      std::vector<std::string> cells;
      std::size_t a = 1;
      for (std::size_t b; (b = line.find('|', a)) != std::string::npos; a = b + 1) cells.push_back(line.substr(a, b - a));
      return std::stoul(cells.at(3));
    };
    CHECK(row_n("tilt period return, learned - B0") == n_tilt);
    CHECK(row_n("sleeve period return, learned - B0") == n_tilt);
    CHECK(row_n("IC(learned) - IC(B0)") == n_ic);
  }
  fs::remove(dir / "report.md");
  rereport(out, "g");
  CHECK(slurp(dir / "report.md") == md);
}

TEST_CASE("wf-external: coverage of rebalance dates - none fails, partial warns") {
  const Panel& panel = ext_panel();
  WalkForwardParams p = ext_params();
  const auto dates = walkforward_dates(panel, p);
  const auto raw = raw_signal(panel, p, Signal::Score);
  // Off by one bar: no scored bar is a rebalance date.
  std::vector<std::size_t> off;
  for (std::size_t d : dates)
    if (d + 1 < panel.T() && std::find(dates.begin(), dates.end(), d + 1) == dates.end()) off.push_back(d + 1);
  const ExternalSignal bad = parse_external_signal("bad", to_csv(panel, raw, off), panel);
  p.externals = {external_tag(bad)};
  try {
    run_walkforward(panel, p, {bad});
    FAIL("expected a throw");
  } catch (const std::invalid_argument& ex) {
    CHECK(std::string(ex.what()).find("none of its") != std::string::npos);
  }
  // Every rebalance plus one stray bar: runs, with a warning naming 1 stray bar.
  std::vector<std::size_t> part = dates;
  part.push_back(off.front());
  std::sort(part.begin(), part.end());
  const ExternalSignal some = parse_external_signal("some", to_csv(panel, raw, part), panel);
  p.externals = {external_tag(some)};
  const WalkForwardResult r = run_walkforward(panel, p, {some});
  REQUIRE(r.externals.size() == 1);
  CHECK(r.externals[0].scored_bars == dates.size() + 1);
  CHECK(r.externals[0].scored_rebalances == dates.size());
  const std::string w = external_coverage_warning(r.externals[0]);
  CHECK(w.find("1 of its " + std::to_string(dates.size() + 1) + " scored bars") != std::string::npos);
}
