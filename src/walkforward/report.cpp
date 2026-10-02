#include "walkforward/report.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
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
nlohmann::json jnum(double x) { return std::isfinite(x) ? nlohmann::json(x) : nlohmann::json(nullptr); }
double unjnum(const nlohmann::json& j) { return j.is_number() ? j.get<double>() : std::nan(""); }

using Curves = std::vector<std::pair<std::string, EquityCurve>>;

const EquityCurve* find_curve(const Curves& curves, const std::string& key) {
  for (const auto& [k, c] : curves)
    if (k == key) return &c;
  return nullptr;
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
  double dsr = std::nan("");         // DSR(total): informational
  double dsr_excess = std::nan("");  // deflated IR of the excess: gate c3
};

constexpr const char* kRegistryHeader =
    "run_id,strategy,params_hash,cost_bps,top_n,sharpe_daily,T,ann_excess,ir_daily";

// Data rows of a registry (header and blank lines dropped); none for a missing file.
std::vector<std::string> read_registry_rows(const fs::path& path) {
  std::vector<std::string> rows;
  std::ifstream in(path);
  std::string line;
  if (!std::getline(in, line)) return rows;  // header
  while (std::getline(in, line))
    if (!line.empty()) rows.push_back(line);
  return rows;
}

// Sample variance (0 with fewer than 2 values).
double sample_var(const std::vector<double>& v) {
  if (v.size() < 2) return 0;
  double m = 0, ss = 0;
  for (double x : v) m += x;
  m /= static_cast<double>(v.size());
  for (double x : v) ss += (x - m) * (x - m);
  return ss / static_cast<double>(v.size() - 1);
}

// Finite value of field k, if the row has it.
std::optional<double> field(const std::vector<std::string>& f, std::size_t k) {
  if (f.size() <= k) return std::nullopt;
  try {
    const double v = std::stod(f[k]);
    if (std::isfinite(v)) return v;
  } catch (const std::exception&) {
  }
  return std::nullopt;
}

RegistryStats stats_of(const std::vector<std::string>& rows) {
  std::vector<double> srs, irs;
  for (const auto& line : rows) {
    const auto f = split(line);
    if (f.size() < 8) continue;
    if (auto v = field(f, 5)) srs.push_back(*v);
    if (auto v = field(f, 8)) irs.push_back(*v);  // 8-column (pre-2026-10-02) rows are not IR trials
  }
  RegistryStats st;
  st.n_trials = srs.size();
  st.trial_sr_var = sample_var(srs);
  st.n_ir_trials = irs.size();
  st.trial_ir_var = sample_var(irs);
  return st;
}

// Throws if run_id is in `rows` with a params hash other than `hash`.
void check_conflict(const std::vector<std::string>& rows, const fs::path& reg, const std::string& run_id,
                    const std::string& hash) {
  for (const auto& line : rows) {
    const auto f = split(line);
    if (f[0] != run_id) continue;
    if (f.size() < 3 || f[2] != hash)
      throw std::invalid_argument("run id '" + run_id + "' is already in " + reg.string() + " with params hash " +
                                  (f.size() >= 3 ? f[2] : std::string("?")) + " (this run: " + hash +
                                  "); choose another --wf-run-id");
  }
}

// Advisory exclusive lock on a side file (not registry.csv itself: the rename replaces that inode).
class FileLock {
 public:
  explicit FileLock(const fs::path& path) : fd_(::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644)) {
    if (fd_ < 0) throw std::runtime_error("cannot open lock file " + path.string());
    while (::flock(fd_, LOCK_EX) != 0)
      if (errno != EINTR) {
        ::close(fd_);
        throw std::runtime_error("cannot lock " + path.string());
      }
  }
  ~FileLock() {
    ::flock(fd_, LOCK_UN);
    ::close(fd_);
  }
  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;

 private:
  int fd_;
};

// Everything results.json and report.md need besides the IC table, the blend and the curves. Built from the
// parameters and the panel by write_report, or from a stored results.json by rereport.
struct Meta {
  std::string run_id, hash, params, largecap_run;
  std::size_t panel_n = 0, panel_t = 0;
  std::string panel_first, panel_last;  // RFC 3339, empty for an empty panel
  bool weekly = true;
  std::size_t warmup = 0, top_n = 0, k = 0;
  double cost_bps = 0, tilt = 0;
  std::vector<std::size_t> bars;   // rebalance bar indices
  std::vector<std::string> dates;  // rebalance times, RFC 3339
};

std::string day(const std::string& rfc3339) { return rfc3339.substr(0, 10); }

// Registry update, results.json and report.md (equity.csv and trades.csv are written by write_report only).
fs::path write_outputs(const Meta& mt, const std::vector<IcRow>& ic_table, const std::vector<BlendStep>& blend,
                       const Curves& curves, const fs::path& out_dir) {
  check_run_id(mt.run_id, "run id");
  if (!mt.largecap_run.empty()) check_run_id(mt.largecap_run, "large-cap run id");
  // Sibling large-cap run first, so a bad id writes nothing.
  std::optional<bool> largecap;
  if (!mt.largecap_run.empty()) {
    const fs::path sib = out_dir / mt.largecap_run / "results.json";
    std::ifstream in(sib);
    if (!in) throw std::runtime_error("--wf-largecap-run: cannot read " + sib.string());
    try {
      largecap = nlohmann::json::parse(in).at("gate").at("core_pass").get<bool>();
    } catch (const std::exception& e) {
      throw std::runtime_error("--wf-largecap-run: bad " + sib.string() + ": " + e.what());
    }
  }

  const fs::path dir = out_dir / mt.run_id;
  static const EquityCurve kEmpty;
  const EquityCurve* bh_ptr = find_curve(curves, "bench:buyhold");
  const EquityCurve& bh = bh_ptr ? *bh_ptr : kEmpty;
  const double base_mdd = performance(bh, bh).max_drawdown;

  std::vector<StratRow> strat, bench;
  for (const auto& [k, c] : curves) {
    StratRow row{k, &c, performance(c, bh)};
    (k.rfind("bench:", 0) == 0 ? bench : strat).push_back(std::move(row));
  }

  // Registry: under the lock, replace this run id's rows (same hash) or reject it (other hash), rewrite atomically,
  // then deflate against every trial.
  fs::create_directories(out_dir);
  const fs::path reg = out_dir / "registry.csv";
  RegistryStats trials;
  {
    FileLock lock(out_dir / "registry.csv.lock");
    std::vector<std::string> rows = read_registry_rows(reg);
    check_conflict(rows, reg, mt.run_id, mt.hash);  // authoritative (the pre-pass check is advisory)
    std::erase_if(rows, [&](const std::string& line) { return split(line)[0] == mt.run_id; });
    for (const auto& s : strat) {
      if (s.perf.T == 0) continue;  // a strategy with no simulated days is not a trial
      std::ostringstream row;
      row << mt.run_id << ',' << s.name << ',' << mt.hash << ',' << full(mt.cost_bps) << ',' << mt.top_n << ','
          << full(s.perf.sharpe_daily) << ',' << s.perf.T << ',' << full(s.perf.ann_excess) << ','
          << full(s.perf.ir_daily);
      rows.push_back(row.str());
    }
    const fs::path tmp = out_dir / "registry.csv.tmp";
    {
      std::ofstream out(tmp, std::ios::trunc);
      if (!out) throw std::runtime_error("cannot write " + tmp.string());
      out << kRegistryHeader << "\n";
      for (const auto& row : rows) out << row << "\n";
      out.flush();
      if (!out) throw std::runtime_error("cannot write " + tmp.string());
    }
    fs::rename(tmp, reg);
    trials = stats_of(rows);
  }
  fs::create_directories(dir);
  for (auto& s : strat) {
    s.dsr = dsr_total(s.perf, trials.trial_sr_var, trials.n_trials);
    s.dsr_excess = dsr_excess(s.perf, trials.trial_ir_var, trials.n_ir_trials);
  }

  const StratRow* blend_row = nullptr;
  for (const auto& s : strat)
    if (s.name == "blend") blend_row = &s;
  const GateResult gate = decision_gate(blend_row ? blend_row->perf : Perf{}, base_mdd,
                                        blend_row ? blend_row->dsr_excess : std::nan(""), largecap.value_or(false));
  const bool core_pass = gate.c1 && gate.c2 && gate.c3 && gate.c4;
  std::string reason = gate.reason;  // c5 is "pending", not a failure, without a sibling run
  if (const auto k = reason.find("c5:"); !largecap && k != std::string::npos) reason.erase(k >= 2 ? k - 2 : k);

  // Blend summary.
  const std::size_t M = mt.bars.size();
  std::size_t open = 0, weighted = 0;
  std::array<double, kSignals> wsum{};
  for (const auto& b : blend) {
    open += b.gate_open ? 1 : 0;
    if (std::any_of(b.w.begin(), b.w.end(), [](double w) { return w > 0; })) {
      ++weighted;
      for (std::size_t s = 0; s < kSignals; ++s) wsum[s] += b.w[s];
    }
  }
  const double open_share = M ? static_cast<double>(open) / static_cast<double>(M) : 0.0;
  auto mean_w = [&](std::size_t s) { return weighted ? wsum[s] / static_cast<double>(weighted) : 0.0; };
  auto pos_years = [](const IcRow& row) {
    std::size_t k = 0;
    for (const auto& [y, v] : row.by_year) k += v > 0 ? 1 : 0;
    return k;
  };

  // results.json
  {
    nlohmann::json j;
    j["run_id"] = mt.run_id;
    j["params_hash"] = mt.hash;
    j["params"] = mt.params;
    j["largecap_run"] = mt.largecap_run;
    j["panel"] = {{"nodes", mt.panel_n}, {"bars", mt.panel_t}, {"first", mt.panel_first}, {"last", mt.panel_last}};
    auto& ic = j["ic_table"] = nlohmann::json::array();
    for (const auto& row : ic_table) {
      nlohmann::json y = nlohmann::json::object();
      for (const auto& [yr, v] : row.by_year) y[std::to_string(yr)] = jnum(v);
      ic.push_back({{"signal", std::string(to_string(row.s))}, {"h", row.h}, {"mean", jnum(row.all.mean)},
                    {"t", jnum(row.all.t)}, {"n", row.all.n}, {"by_year", y}});
    }
    auto& bl = j["blend"] = nlohmann::json::array();
    for (std::size_t m = 0; m < M; ++m) {
      nlohmann::json w = nlohmann::json::object();
      for (std::size_t s = 0; s < kSignals; ++s) w[std::string(to_string(static_cast<Signal>(s)))] = blend[m].w[s];
      bl.push_back({{"date", mt.dates[m]}, {"bar", mt.bars[m]}, {"w", w}, {"gate_open", blend[m].gate_open},
                    {"oos_ic", jnum(blend[m].oos_ic)}});
    }
    j["gate_open_share"] = open_share;
    auto perf_json = [&](const StratRow& s) {
      const Perf& q = s.perf;
      return nlohmann::json{{"name", s.name}, {"T", q.T}, {"cum_return", jnum(q.cum_return)},
                            {"ann_return", jnum(q.ann_return)}, {"ann_vol", jnum(q.ann_vol)},
                            {"sharpe", jnum(q.sharpe)}, {"sharpe_daily", jnum(q.sharpe_daily)},
                            {"max_drawdown", jnum(q.max_drawdown)}, {"ann_excess", jnum(q.ann_excess)},
                            {"excess_ci95", {jnum(q.excess_ci95.lo), jnum(q.excess_ci95.hi)}}, {"ir", jnum(q.ir)},
                            {"ir_daily", jnum(q.ir_daily)}, {"year_hit_rate", jnum(q.year_hit_rate)},
                            {"years", q.years}, {"skew", jnum(q.skew)}, {"kurt", jnum(q.kurt)},
                            {"skew_e", jnum(q.skew_e)}, {"kurt_e", jnum(q.kurt_e)}, {"dsr", jnum(s.dsr)},
                            {"dsr_excess", jnum(s.dsr_excess)}, {"costs", jnum(s.curve->costs)},
                            {"turnover", jnum(s.curve->turnover)}};
    };
    auto& st = j["strategies"] = nlohmann::json::array();
    for (const auto& s : strat) st.push_back(perf_json(s));
    auto& be = j["benchmarks"] = nlohmann::json::array();
    for (const auto& s : bench) be.push_back(perf_json(s));
    j["base_max_drawdown"] = jnum(base_mdd);
    j["registry"] = {{"n_trials", trials.n_trials}, {"trial_sr_var", jnum(trials.trial_sr_var)},
                     {"n_ir_trials", trials.n_ir_trials}, {"trial_ir_var", jnum(trials.trial_ir_var)}};
    j["gate"] = {{"c1", gate.c1}, {"c2", gate.c2}, {"c3", gate.c3}, {"c4", gate.c4},
                 {"c5", largecap ? nlohmann::json(gate.c5) : nlohmann::json("pending")},
                 {"c3_dsr_excess", jnum(blend_row ? blend_row->dsr_excess : std::nan(""))},
                 {"core_pass", core_pass},
                 {"pass", largecap ? nlohmann::json(gate.pass) : nlohmann::json("pending")},
                 {"reason", reason}};
    std::ofstream(dir / "results.json") << j.dump(2) << "\n";
  }

  // report.md
  {
    std::ostringstream md;
    md << "# Walk-forward run `" << mt.run_id << "`\n\n";
    md << "- params hash `" << mt.hash << "`: `" << mt.params << "`\n";
    md << "- panel: " << mt.panel_n << " nodes, " << mt.panel_t << " bars";
    if (mt.panel_t) md << " (" << day(mt.panel_first) << " to " << day(mt.panel_last) << ")";
    md << "\n- rebalances: " << M << " (" << (mt.weekly ? "weekly" : "monthly");
    if (M) md << ", " << day(mt.dates.front()) << " to " << day(mt.dates.back());
    md << "), warm-up " << mt.warmup << " bars\n";
    md << "- registry: " << trials.n_trials << " trials, variance of daily Sharpe " << num(trials.trial_sr_var, 6)
       << "; " << trials.n_ir_trials << " IR trials, variance of daily IR " << num(trials.trial_ir_var, 6) << "\n\n";

    md << "## IC research table\n\n"
          "Spearman IC of each signal's z-score (eligible names) against the open-to-open return over h bars, on "
          "non-overlapping samples. h = 20 is a decay diagnostic.\n\n"
          "| signal | h | mean IC | t | n | positive years |\n|---|---:|---:|---:|---:|---:|\n";
    for (const auto& row : ic_table) {
      md << "| " << to_string(row.s) << " | " << row.h << " | " << num(row.all.mean, 4) << " | "
         << num(row.all.t, 2) << " | " << row.all.n << " | "
         << (row.by_year.empty() ? std::string("n/a")
                                 : std::to_string(pos_years(row)) + " of " + std::to_string(row.by_year.size()))
         << " |\n";
    }

    md << "\n## Strategies against buy-and-hold of the base\n\n"
       << "Costs " << num(mt.cost_bps, 1) << " bps per side, tilt " << num(mt.tilt, 2) << " over the top " << mt.k
       << ". Ann. excess is the mean daily excess return times 252 (not the gap between annualized returns). "
       << "DSR_excess deflates the daily IR of the excess over buy-and-hold across all " << trials.n_ir_trials
       << " IR trials in the registry (gate c3). DSR(total) deflates the total-return Sharpe across all "
       << trials.n_trials << " trials; it is informational only, since a high-Sharpe base passes it without any "
          "excess.\n\n"
       << "| curve | days | cum | ann | vol | Sharpe | max DD | ann excess | excess CI95 | IR | years + | "
          "turnover | costs | DSR_excess | DSR(total), informational |\n"
          "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|\n";
    auto line = [&](const StratRow& s, bool is_strat) {
      const Perf& q = s.perf;
      const auto hits = static_cast<long long>(std::llround(q.year_hit_rate * static_cast<double>(q.years)));
      md << "| " << s.name << " | " << q.T << " | " << pct(q.cum_return) << " | " << pct(q.ann_return) << " | "
         << pct(q.ann_vol) << " | " << num(q.sharpe, 2) << " | " << pct(q.max_drawdown) << " | "
         << pct(q.ann_excess) << " | " << pct(q.excess_ci95.lo) << " .. " << pct(q.excess_ci95.hi) << " | "
         << num(q.ir, 2) << " | " << hits << " of " << q.years << " | " << num(s.curve->turnover, 2) << " | "
         << pct(s.curve->costs, 3) << " | " << (is_strat ? num(s.dsr_excess, 3) : std::string("-")) << " | "
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
       << "| c3: deflated IR (excess vs buy-and-hold) > 0.95 | " << mark(gate.c3) << " (DSR_excess "
       << num(blend_row ? blend_row->dsr_excess : std::nan(""), 3) << "; DSR(total) "
       << num(blend_row ? blend_row->dsr : std::nan(""), 3) << ", informational) |\n"
       << "| c4: max drawdown <= base " << pct(base_mdd) << " + 5% | " << mark(gate.c4) << " |\n"
       << "| c5: holds on the large-cap sub-universe | "
       << (largecap ? std::string(mark(gate.c5)) + " (run `" + mt.largecap_run + "`)"
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

// "key=value" from a describe() string (value up to the next space or '}'); throws if missing.
std::string param_value(const std::string& params, const std::string& key) {
  const std::string pat = " " + key + "=";
  const auto k = params.find(pat);
  if (k == std::string::npos) throw std::runtime_error("rereport: params lack " + key);
  const auto b = k + pat.size();
  return params.substr(b, params.find_first_of(" }", b) - b);
}

}  // namespace

void check_run_id(const std::string& id, const std::string& what) {
  const bool ok = !id.empty() && id.size() <= 128 && id != "." && id != ".." &&
                  std::all_of(id.begin(), id.end(), [](char c) {
                    return std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == ':' ||
                           c == '+' || c == '-';
                  });
  if (!ok)
    throw std::invalid_argument(what + " must be 1-128 characters from A-Z a-z 0-9 . _ : + - (not . or ..), got '" +
                                id + "'");
}

RegistryStats registry_stats(const fs::path& registry_csv) { return stats_of(read_registry_rows(registry_csv)); }

void check_registry_conflict(const fs::path& out_dir, const std::string& run_id, const std::string& hash) {
  const fs::path reg = out_dir / "registry.csv";
  check_conflict(read_registry_rows(reg), reg, run_id, hash);
}

fs::path write_report(const WalkForwardResult& r, const WalkForwardParams& p, const Panel& panel,
                      const fs::path& out_dir, const std::string& run_id) {
  Meta mt;
  mt.run_id = run_id;
  mt.hash = params_hash(p);
  mt.params = describe(p);
  mt.largecap_run = p.largecap_run;
  mt.panel_n = panel.N();
  mt.panel_t = panel.T();
  if (panel.T()) mt.panel_first = format_rfc3339(panel.times.front()), mt.panel_last = format_rfc3339(panel.times.back());
  mt.weekly = p.rebalance == Rebalance::Weekly;
  mt.warmup = p.warmup_bars;
  mt.top_n = p.top_n;
  mt.k = p.bt.k;
  mt.cost_bps = p.bt.cost_bps;
  mt.tilt = p.bt.tilt;
  mt.bars = r.dates;
  for (std::size_t d : r.dates) mt.dates.push_back(format_rfc3339(panel.times[d]));
  const fs::path dir = write_outputs(mt, r.ic_table, r.blend, r.curves, out_dir);

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
  return dir;
}

fs::path rereport(const fs::path& out_dir, const std::string& run_id) {
  check_run_id(run_id, "run id");
  const fs::path dir = out_dir / run_id;
  nlohmann::json j;
  {
    std::ifstream in(dir / "results.json");
    if (!in) throw std::runtime_error("rereport: cannot read " + (dir / "results.json").string());
    j = nlohmann::json::parse(in);
  }
  if (j.at("run_id").get<std::string>() != run_id)
    throw std::runtime_error("rereport: " + (dir / "results.json").string() + " belongs to another run id");
  Meta mt;
  mt.run_id = run_id;
  mt.hash = j.at("params_hash").get<std::string>();
  mt.params = j.at("params").get<std::string>();
  mt.largecap_run = j.value("largecap_run", std::string());
  const auto& pj = j.at("panel");
  mt.panel_n = pj.at("nodes").get<std::size_t>();
  mt.panel_t = pj.at("bars").get<std::size_t>();
  mt.panel_first = pj.at("first").get<std::string>();
  mt.panel_last = pj.at("last").get<std::string>();
  mt.weekly = param_value(mt.params, "rebalance") == "weekly";
  mt.warmup = std::stoul(param_value(mt.params, "warmup"));
  mt.top_n = std::stoul(param_value(mt.params, "top_n"));
  mt.k = std::stoul(param_value(mt.params, "k"));
  mt.cost_bps = std::stod(param_value(mt.params, "cost_bps"));
  mt.tilt = std::stod(param_value(mt.params, "tilt"));

  std::vector<IcRow> ic;
  for (const auto& row : j.at("ic_table")) {
    IcRow r{parse_signal(row.at("signal").get<std::string>()), row.at("h").get<int>(), {}, {}};
    r.all.mean = unjnum(row.at("mean"));
    r.all.t = unjnum(row.at("t"));
    r.all.n = row.at("n").get<std::size_t>();
    for (const auto& [yr, v] : row.at("by_year").items()) r.by_year.emplace_back(std::stoi(yr), unjnum(v));
    std::sort(r.by_year.begin(), r.by_year.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    ic.push_back(std::move(r));
  }
  std::vector<BlendStep> blend;
  for (const auto& b : j.at("blend")) {
    BlendStep st;
    for (const auto& [name, w] : b.at("w").items()) st.w[static_cast<std::size_t>(parse_signal(name))] = w.get<double>();
    st.gate_open = b.at("gate_open").get<bool>();
    st.oos_ic = unjnum(b.at("oos_ic"));
    blend.push_back(std::move(st));
    mt.bars.push_back(b.at("bar").get<std::size_t>());
    mt.dates.push_back(b.at("date").get<std::string>());
  }

  // Curves from equity.csv (values are written with 17 significant digits, so they round-trip exactly); costs and
  // turnover from results.json.
  Curves curves;
  {
    std::ifstream in(dir / "equity.csv");
    if (!in) throw std::runtime_error("rereport: cannot read " + (dir / "equity.csv").string());
    std::string line;
    std::getline(in, line);
    const auto header = split(line);
    for (std::size_t c = 1; c < header.size(); ++c) curves.emplace_back(header[c], EquityCurve{});
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      const auto f = split(line);
      const TimePoint t = parse_rfc3339(f[0]);
      for (std::size_t c = 1; c < f.size() && c <= curves.size(); ++c)
        if (!f[c].empty()) {
          curves[c - 1].second.t.push_back(t);
          curves[c - 1].second.value.push_back(std::stod(f[c]));
        }
    }
  }
  std::map<std::string, std::pair<double, double>> cost_turn;
  for (const char* key : {"strategies", "benchmarks"})
    for (const auto& s : j.at(key))
      cost_turn[s.at("name").get<std::string>()] = {unjnum(s.at("costs")), unjnum(s.at("turnover"))};
  for (auto& [name, c] : curves)
    if (auto it = cost_turn.find(name); it != cost_turn.end()) c.costs = it->second.first, c.turnover = it->second.second;
  return write_outputs(mt, ic, blend, curves, out_dir);
}

}  // namespace mr
