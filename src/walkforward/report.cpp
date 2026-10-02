#include "walkforward/report.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "cli/args.hpp"
#include "core/time.hpp"
#include "walkforward/metrics.hpp"

namespace mr {
namespace fs = std::filesystem;

std::string describe(const WalkForwardParams& p) {
  std::ostringstream s;
  s.precision(17);
  s << "core{" << describe(p.core) << " horizons=";
  for (int h : p.core.horizons) s << h << ';';
  s << "} rebalance=" << (p.rebalance == Rebalance::Weekly ? "weekly" : "monthly") << " warmup=" << p.warmup_bars
    << " elig_window=" << p.elig_window << " top_n=" << p.top_n << " min_dv=" << p.min_dollar_volume << " ic_h=";
  for (int h : p.ic_horizons) s << h << ';';
  s << " base=";
  for (const auto& b : p.bt.base) s << b.ticker << ':' << b.weight << ';';
  s << " tilt=" << p.bt.tilt << " k=" << p.bt.k << " max_name_tilt=" << p.bt.max_name_tilt
    << " cost_bps=" << p.bt.cost_bps << " max_turnover=" << p.bt.max_turnover << " blend=" << p.blend.train_months
    << '/' << p.blend.embargo << '/' << p.blend.gate_months << '/' << p.blend.gate_min << " gate_t=" << p.blend.gate_t;
  return s.str();
}

std::string params_hash(const WalkForwardParams& p) {
  std::uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : describe(p)) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  char buf[16];
  std::snprintf(buf, sizeof buf, "%08" PRIx32, static_cast<std::uint32_t>(h ^ (h >> 32)));
  return buf;
}

namespace {

std::string num(double x, int prec = 4) {
  if (!std::isfinite(x)) return "n/a";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*f", prec, x);
  return buf;
}
std::string pct(double x, int prec = 2) { return std::isfinite(x) ? num(100 * x, prec) + "%" : "n/a"; }
std::string full(double x) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.17g", x);
  return buf;
}
std::string date_of(TimePoint t) { return format_rfc3339(t).substr(0, 10); }
nlohmann::json jnum(double x) { return std::isfinite(x) ? nlohmann::json(x) : nlohmann::json(nullptr); }

const EquityCurve* find_curve(const WalkForwardResult& r, const std::string& key) {
  for (const auto& [k, c] : r.curves)
    if (k == key) return &c;
  return nullptr;
}

std::size_t common_days(const EquityCurve& a, const EquityCurve& b) {
  std::size_t i = 0, j = 0, n = 0;
  while (i < a.t.size() && j < b.t.size()) {
    if (a.t[i] < b.t[j]) ++i;
    else if (a.t[i] > b.t[j]) ++j;
    else ++n, ++i, ++j;
  }
  return n;
}

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == ',') out.push_back(cur), cur.clear();
    else cur += c;
  }
  out.push_back(cur);
  return out;
}

struct StratRow {
  std::string name;
  const EquityCurve* curve;
  Perf perf;
  std::size_t T;
  double dsr = std::nan("");
};

constexpr const char* kRegistryHeader = "run_id,strategy,params_hash,cost_bps,top_n,sharpe_daily,T,ann_excess";

}  // namespace

fs::path write_report(const WalkForwardResult& r, const WalkForwardParams& p, const Panel& panel,
                      const fs::path& out_dir, const std::string& run_id) {
  if (run_id.empty() || run_id.find_first_of(",/\\ \t\n") != std::string::npos || run_id == "." || run_id == "..")
    throw std::invalid_argument("write_report: run id must be non-empty without , / \\ or whitespace: '" + run_id + "'");
  // Sibling large-cap run first, so a bad id writes nothing.
  std::optional<bool> largecap;
  if (!p.largecap_run.empty()) {
    const fs::path sib = out_dir / p.largecap_run / "results.json";
    std::ifstream in(sib);
    if (!in) throw std::runtime_error("--wf-largecap-run: cannot read " + sib.string());
    try {
      largecap = nlohmann::json::parse(in).at("gate").at("core_pass").get<bool>();
    } catch (const std::exception& e) {
      throw std::runtime_error("--wf-largecap-run: bad " + sib.string() + ": " + e.what());
    }
  }

  const std::string hash = params_hash(p);
  const fs::path dir = out_dir / run_id;
  fs::create_directories(dir);
  static const EquityCurve kEmpty;
  const EquityCurve* bh_ptr = find_curve(r, "bench:buyhold");
  const EquityCurve& bh = bh_ptr ? *bh_ptr : kEmpty;
  const double base_mdd = performance(bh, bh).max_drawdown;

  std::vector<StratRow> strat, bench;
  for (const auto& [k, c] : r.curves) {
    StratRow row{k, &c, performance(c, bh), common_days(c, bh)};
    (k.rfind("bench:", 0) == 0 ? bench : strat).push_back(std::move(row));
  }

  // Registry: append this run's rows, then deflate against every row.
  const fs::path reg = out_dir / "registry.csv";
  {
    const bool fresh = !fs::exists(reg) || fs::file_size(reg) == 0;
    std::ofstream out(reg, std::ios::app);
    if (!out) throw std::runtime_error("cannot write " + reg.string());
    if (fresh) out << kRegistryHeader << "\n";
    for (const auto& s : strat)
      if (s.T > 0)  // a strategy with no simulated days is not a trial
        out << run_id << ',' << s.name << ',' << hash << ',' << full(p.bt.cost_bps) << ',' << p.top_n << ','
          << full(s.perf.sharpe_daily) << ',' << s.T << ',' << full(s.perf.ann_excess) << "\n";
  }
  std::size_t n_trials = 0;
  std::vector<double> srs;
  {
    std::ifstream in(reg);
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      ++n_trials;
      const auto f = split(line);
      if (f.size() < 8) continue;
      try {
        const double v = std::stod(f[5]);
        if (std::isfinite(v)) srs.push_back(v);
      } catch (const std::exception&) {
      }
    }
  }
  double trial_sr_var = 0;
  if (srs.size() >= 2) {
    double m = 0;
    for (double v : srs) m += v;
    m /= static_cast<double>(srs.size());
    for (double v : srs) trial_sr_var += (v - m) * (v - m);
    trial_sr_var /= static_cast<double>(srs.size() - 1);
  }
  for (auto& s : strat)
    if (std::isfinite(s.perf.sharpe_daily))
      s.dsr = deflated_sharpe(s.perf.sharpe_daily, s.T, s.perf.skew, s.perf.kurt, trial_sr_var,
                              std::max<std::size_t>(n_trials, 1));

  const StratRow* blend = nullptr;
  for (const auto& s : strat)
    if (s.name == "blend") blend = &s;
  const GateResult gate =
      decision_gate(blend ? blend->perf : Perf{}, base_mdd, blend ? blend->dsr : std::nan(""), largecap.value_or(false));
  const bool core_pass = gate.c1 && gate.c2 && gate.c3 && gate.c4;
  std::string reason = gate.reason;  // c5 is "pending", not a failure, without a sibling run
  if (const auto k = reason.find("c5:"); !largecap && k != std::string::npos) reason.erase(k >= 2 ? k - 2 : k);

  // Blend summary.
  const std::size_t M = r.dates.size();
  std::size_t open = 0, weighted = 0;
  std::array<double, kSignals> wsum{};
  for (const auto& b : r.blend) {
    open += b.gate_open ? 1 : 0;
    if (std::any_of(b.w.begin(), b.w.end(), [](double w) { return w > 0; })) {
      ++weighted;
      for (std::size_t s = 0; s < kSignals; ++s) wsum[s] += b.w[s];
    }
  }
  const double open_share = M ? static_cast<double>(open) / static_cast<double>(M) : 0.0;
  auto mean_w = [&](std::size_t s) { return weighted ? wsum[s] / static_cast<double>(weighted) : 0.0; };
  auto pos_year_share = [](const IcRow& row) {
    if (row.by_year.empty()) return std::nan("");
    std::size_t k = 0;
    for (const auto& [y, v] : row.by_year) k += v > 0 ? 1 : 0;
    return static_cast<double>(k) / static_cast<double>(row.by_year.size());
  };

  // results.json
  {
    nlohmann::json j;
    j["run_id"] = run_id;
    j["params_hash"] = hash;
    j["params"] = describe(p);
    j["largecap_run"] = p.largecap_run;
    j["panel"] = {{"nodes", panel.N()}, {"bars", panel.T()},
                  {"first", panel.T() ? format_rfc3339(panel.times.front()) : ""},
                  {"last", panel.T() ? format_rfc3339(panel.times.back()) : ""}};
    auto& ic = j["ic_table"] = nlohmann::json::array();
    for (const auto& row : r.ic_table) {
      nlohmann::json y = nlohmann::json::object();
      for (const auto& [yr, v] : row.by_year) y[std::to_string(yr)] = jnum(v);
      ic.push_back({{"signal", std::string(to_string(row.s))}, {"h", row.h}, {"mean", jnum(row.all.mean)},
                    {"t", jnum(row.all.t)}, {"n", row.all.n}, {"by_year", y}});
    }
    auto& bl = j["blend"] = nlohmann::json::array();
    for (std::size_t m = 0; m < M; ++m) {
      nlohmann::json w = nlohmann::json::object();
      for (std::size_t s = 0; s < kSignals; ++s) w[std::string(to_string(static_cast<Signal>(s)))] = r.blend[m].w[s];
      bl.push_back({{"date", format_rfc3339(panel.times[r.dates[m]])}, {"bar", r.dates[m]}, {"w", w},
                    {"gate_open", r.blend[m].gate_open}, {"oos_ic", jnum(r.blend[m].oos_ic)}});
    }
    j["gate_open_share"] = open_share;
    auto perf_json = [&](const StratRow& s) {
      const Perf& q = s.perf;
      return nlohmann::json{{"name", s.name}, {"T", s.T}, {"cum_return", jnum(q.cum_return)},
                            {"ann_return", jnum(q.ann_return)}, {"ann_vol", jnum(q.ann_vol)},
                            {"sharpe", jnum(q.sharpe)}, {"sharpe_daily", jnum(q.sharpe_daily)},
                            {"max_drawdown", jnum(q.max_drawdown)}, {"ann_excess", jnum(q.ann_excess)},
                            {"excess_ci95", {jnum(q.excess_ci95.lo), jnum(q.excess_ci95.hi)}}, {"ir", jnum(q.ir)},
                            {"year_hit_rate", jnum(q.year_hit_rate)}, {"years", q.years}, {"skew", jnum(q.skew)},
                            {"kurt", jnum(q.kurt)}, {"dsr", jnum(s.dsr)}, {"costs", jnum(s.curve->costs)},
                            {"turnover", jnum(s.curve->turnover)}};
    };
    auto& st = j["strategies"] = nlohmann::json::array();
    for (const auto& s : strat) st.push_back(perf_json(s));
    auto& be = j["benchmarks"] = nlohmann::json::array();
    for (const auto& s : bench) be.push_back(perf_json(s));
    j["base_max_drawdown"] = jnum(base_mdd);
    j["registry"] = {{"n_trials", n_trials}, {"trial_sr_var", jnum(trial_sr_var)}};
    j["gate"] = {{"c1", gate.c1}, {"c2", gate.c2}, {"c3", gate.c3}, {"c4", gate.c4},
                 {"c5", largecap ? nlohmann::json(gate.c5) : nlohmann::json("pending")},
                 {"core_pass", core_pass},
                 {"pass", largecap ? nlohmann::json(gate.pass) : nlohmann::json("pending")},
                 {"reason", reason}};
    std::ofstream(dir / "results.json") << j.dump(2) << "\n";
  }

  // equity.csv: wide, one column per curve, on the union of dates (empty cell = curve not started).
  {
    std::vector<TimePoint> ts;
    for (const auto& [k, c] : r.curves) ts.insert(ts.end(), c.t.begin(), c.t.end());
    std::sort(ts.begin(), ts.end());
    ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
    std::ofstream out(dir / "equity.csv");
    out << "date";
    for (const auto& [k, c] : r.curves) out << ',' << k;
    out << "\n";
    std::vector<std::size_t> pos(r.curves.size(), 0);
    for (TimePoint t : ts) {
      out << format_rfc3339(t);
      for (std::size_t c = 0; c < r.curves.size(); ++c) {
        const EquityCurve& e = r.curves[c].second;
        out << ',';
        if (pos[c] < e.t.size() && e.t[pos[c]] == t) out << full(e.value[pos[c]++]);
      }
      out << "\n";
    }
  }

  // trades.csv
  {
    std::ofstream out(dir / "trades.csv");
    out << "strategy,t,ticker,delta_weight,price\n";
    for (const auto& [k, c] : r.curves)
      for (const auto& line : c.trades_csv) out << k << ',' << line << "\n";
  }

  // report.md
  {
    std::ostringstream md;
    md << "# Walk-forward run `" << run_id << "`\n\n";
    md << "- params hash `" << hash << "`: `" << describe(p) << "`\n";
    md << "- panel: " << panel.N() << " nodes, " << panel.T() << " bars";
    if (panel.T()) md << " (" << date_of(panel.times.front()) << " to " << date_of(panel.times.back()) << ")";
    md << "\n- rebalances: " << M << " (" << (p.rebalance == Rebalance::Weekly ? "weekly" : "monthly");
    if (M) md << ", " << date_of(panel.times[r.dates.front()]) << " to " << date_of(panel.times[r.dates.back()]);
    md << "), warm-up " << p.warmup_bars << " bars\n";
    md << "- registry: " << n_trials << " trials, variance of daily Sharpe " << num(trial_sr_var, 6) << "\n\n";

    md << "## IC research table\n\n"
          "Spearman IC of each signal's z-score (eligible names) against the open-to-open return over h bars, on "
          "non-overlapping samples. h = 20 is a decay diagnostic.\n\n"
          "| signal | h | mean IC | t | n | positive years |\n|---|---:|---:|---:|---:|---:|\n";
    for (const auto& row : r.ic_table) {
      const double share = pos_year_share(row);
      md << "| " << to_string(row.s) << " | " << row.h << " | " << num(row.all.mean, 4) << " | "
         << num(row.all.t, 2) << " | " << row.all.n << " | "
         << (std::isfinite(share) ? num(share, 2) + " of " + std::to_string(row.by_year.size()) : "n/a") << " |\n";
    }

    md << "\n## Strategies against buy-and-hold of the base\n\n"
       << "Costs " << num(p.bt.cost_bps, 1) << " bps per side, tilt " << num(p.bt.tilt, 2) << " over the top "
       << p.bt.k << ". DSR deflates over all " << n_trials << " registry trials.\n\n"
       << "| curve | days | cum | ann | vol | Sharpe | max DD | ann excess | excess CI95 | IR | years + | "
          "turnover | costs | DSR |\n|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n";
    auto line = [&](const StratRow& s, bool is_strat) {
      const Perf& q = s.perf;
      md << "| " << s.name << " | " << s.T << " | " << pct(q.cum_return) << " | " << pct(q.ann_return) << " | "
         << pct(q.ann_vol) << " | " << num(q.sharpe, 2) << " | " << pct(q.max_drawdown) << " | "
         << pct(q.ann_excess) << " | " << pct(q.excess_ci95.lo) << " .. " << pct(q.excess_ci95.hi) << " | "
         << num(q.ir, 2) << " | " << num(q.year_hit_rate, 2) << " of " << q.years << " | "
         << num(s.curve->turnover, 2) << " | " << pct(s.curve->costs, 3) << " | "
         << (is_strat ? num(s.dsr, 3) : std::string("-")) << " |\n";
    };
    for (const auto& s : strat) line(s, true);
    for (const auto& s : bench) line(s, false);

    md << "\n## Blend\n\n"
       << "Gate open in " << open << " of " << M << " periods (" << pct(open_share, 1) << "); weights formed in "
       << weighted << " periods. Mean weight over those periods:\n\n| signal | mean w |\n|---|---:|\n";
    for (std::size_t s = 0; s < kSignals; ++s)
      md << "| " << to_string(static_cast<Signal>(s)) << " | " << num(mean_w(s), 3) << " |\n";
    md << "\nThe full weight history is in results.json (`blend`).\n";

    auto mark = [](bool ok) { return ok ? "pass" : "FAIL"; };
    md << "\n## Decision gate (blend)\n\n"
       << "| criterion | result |\n|---|---|\n"
       << "| c1: annual excess > 0 and CI95 lower bound > 0 | " << mark(gate.c1) << " |\n"
       << "| c2: excess positive in >= 60% of years | " << mark(gate.c2) << " |\n"
       << "| c3: deflated Sharpe > 0.95 | " << mark(gate.c3) << " |\n"
       << "| c4: max drawdown <= base " << pct(base_mdd) << " + 5% | " << mark(gate.c4) << " |\n"
       << "| c5: holds on the large-cap sub-universe | "
       << (largecap ? std::string(mark(gate.c5)) + " (run `" + p.largecap_run + "`)"
                    : std::string("pending (run again with --wf-top-n 500 and pass --wf-largecap-run)"))
       << " |\n\n";
    if (!core_pass) md << "**Verdict: FAIL.** " << reason << "\n";
    else if (!largecap) md << "**Verdict: pending** (c1 to c4 pass; c5 not yet run).\n";
    else md << "**Verdict: " << (gate.pass ? "PASS" : "FAIL") << ".** " << gate.reason << "\n";
    md << "\nAdvisory only. The universe is today's snapshot, so delisted names are missing (survivorship bias).\n";
    std::ofstream(dir / "report.md") << md.str();
  }
  return dir;
}

}  // namespace mr
