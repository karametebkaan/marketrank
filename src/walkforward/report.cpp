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
  if (!p.externals.empty()) {  // M4: absent without externals, so M3a hashes are unchanged
    s << " ext=";
    for (const auto& e : p.externals) s << e << ';';
  }
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

// Per-period returns of a curve indexed by DECISION date: out[m] = V(d_{m+1}) / V(d_m) - 1, the return of the
// period decided at d_m (close of d_m to close of d_{m+1}), with V(d_m) = 1.0 (the starting capital) for the
// curve's first period; NaN where a close is not on the curve or m is last.
std::vector<double> period_returns(const EquityCurve& c, const std::vector<TimePoint>& closes) {
  std::map<TimePoint, double> at;
  for (std::size_t k = 0; k < c.t.size(); ++k) at.emplace(c.t[k], c.value[k]);
  std::vector<double> out(closes.size(), std::nan(""));
  for (std::size_t m = 0; m + 1 < closes.size(); ++m) {
    const auto b = at.find(closes[m + 1]);
    if (b == at.end()) continue;
    // The curve's first period: funded from 1.0 in cash at the open after d_m, so it starts from 1.0.
    const bool first = !c.t.empty() && c.t.front() > closes[m] && c.t.front() <= closes[m + 1];
    const auto a = at.find(closes[m]);
    const double v0 = first ? 1.0 : a != at.end() ? a->second : std::nan("");
    if (v0 > 0) out[m] = b->second / v0 - 1.0;
  }
  return out;
}

// Paired difference x[m] - y[m] over the decision dates m with use[m] where both are finite: mean, t, n, and the UTC
// years (of the decision date) whose mean difference is positive, of the years with any pair.
struct Paired {
  MeanT all;
  std::size_t years_pos = 0, years = 0;
};
Paired paired_diff(const std::vector<double>& x, const std::vector<double>& y, const std::vector<bool>& use,
                   const std::vector<std::string>& dates) {
  std::vector<double> d;
  std::map<std::string, std::pair<double, std::size_t>> by_year;
  for (std::size_t m = 0; m < x.size() && m < y.size() && m < dates.size() && m < use.size(); ++m)
    if (use[m] && std::isfinite(x[m]) && std::isfinite(y[m])) {
      d.push_back(x[m] - y[m]);
      auto& [sum, n] = by_year[dates[m].substr(0, 4)];
      sum += x[m] - y[m];
      ++n;
    }
  Paired p;
  p.all = mean_t(d);
  p.years = by_year.size();
  for (const auto& [yr, sn] : by_year) p.years_pos += sn.first / static_cast<double>(sn.second) > 0 ? 1 : 0;
  return p;
}

// The M3a gate (c1..c4, c5 from a large-cap sibling) applied to one external's tilt over its scored span.
struct ExtGate {
  std::string name, first, last;  // span: first and last day of the sig:<name> curve (RFC 3339), empty if none
  bool have = false;              // the curve has days
  double base_mdd = std::nan("");  // buy-and-hold's max drawdown over the span
  double dsr_excess = std::nan("");
  GateResult gate{};
  bool core_pass = false;
  std::optional<bool> largecap;  // the sibling run's core_pass for this external
  std::string reason;
};

// Buy-and-hold over [from, to], rebased to 1.0 at the close before `from` (as the external's tilt starts from it).
EquityCurve span_of(const EquityCurve& c, TimePoint from, TimePoint to) {
  EquityCurve out;
  double base = 1.0;
  for (std::size_t k = 0; k < c.t.size(); ++k) {
    if (c.t[k] < from) base = c.value[k];
    else if (c.t[k] <= to) out.t.push_back(c.t[k]), out.value.push_back(c.value[k]);
  }
  for (double& v : out.value) v /= base;
  return out;
}

// report.md section for external signals (M4): span rule, their IC rows, their tilt and sleeve curves, the M3a gate
// per external, and the decisive paired comparison "learned" vs "B0" when both are present.
void external_section(std::ostringstream& md, const Meta& mt, const std::vector<IcRow>& ic_table,
                      const std::vector<StratRow>& strat, const std::vector<StratRow>& vs_reb,
                      const std::vector<StratRow>& sleeves, const Curves& curves,
                      const std::vector<ExternalResult>& externals, const std::vector<ExtGate>& gates) {
  auto find_row = [](const std::vector<StratRow>& v, const std::string& name) -> const StratRow* {
    for (const auto& s : v)
      if (s.name == name) return &s;
    return nullptr;
  };
  md << "\n## External signals\n\n"
     << "Scores computed outside this binary (--wf-external), evaluated standalone: never part of the blend (the "
        "pre-registered M3a protocol blends the built-in signals only). The IC rows sample only the bars a signal "
        "has scores at (its rebalance dates), non-overlapping per horizon (a scored bar is taken when it is at "
        "least h bars after the previous sample), against eligible_at. Missing tickers or dates are not eligible "
        "for the signal.\n\n"
     << "**Span rule.** Each external is evaluated over its scored span only: its `sig:` and `sleeve:` curves "
        "start with the decision at its first scored rebalance d_a and end at the close of d_{b+1}, the end of the "
        "period decided at its last scored rebalance d_b (no burn-in periods holding the base). Inside the span "
        "the scores are z-scored under each rebalance's mask; an unscored rebalance inside the span holds the base "
        "(sleeves: the equal-weight universe). Its registry rows, DSR_excess, years and gate criteria are computed "
        "on that span against the benchmarks over the same days; c4 compares with buy-and-hold's drawdown over the "
        "span.\n\n"
     << "| signal | h | mean IC | t | n | positive years |\n|---|---:|---:|---:|---:|---:|\n";
  for (const auto& e : externals)
    for (const auto& row : ic_table) {
      if (row.signal != e.name) continue;
      std::size_t pos = 0;
      for (const auto& [y, v] : row.by_year) pos += v > 0 ? 1 : 0;
      md << "| " << row.signal << " | " << row.h << " | " << num(row.all.mean, 4) << " | " << num(row.all.t, 2)
         << " | " << row.all.n << " | "
         << (row.by_year.empty() ? std::string("n/a")
                                 : std::to_string(pos) + " of " + std::to_string(row.by_year.size()))
         << " |\n";
    }
  md << "\n| signal | scored bars | of them rebalances | span | rebalances with an IC | tilt ann excess vs "
        "buy-and-hold | IR | DSR_excess | tilt ann excess vs rebalanced base | IR | sleeve ann excess vs EW eligible "
        "| IR |\n|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n";
  for (std::size_t x = 0; x < externals.size(); ++x) {
    const auto& e = externals[x];
    std::size_t n_ic = 0;
    for (double v : e.rebalance_ic) n_ic += std::isfinite(v) ? 1 : 0;
    const StratRow* a = find_row(strat, "sig:" + e.name);
    const StratRow* b = find_row(vs_reb, "sig:" + e.name);
    const StratRow* c = find_row(sleeves, "sleeve:" + e.name);
    const ExtGate& g = gates[x];
    md << "| " << e.name << " | " << e.scored_bars << " | " << e.scored_rebalances << " | "
       << (g.have ? day(g.first) + " to " + day(g.last) : std::string("n/a")) << " | " << n_ic << " | "
       << (a ? pct(a->perf.ann_excess) : "n/a") << " | " << (a ? num(a->perf.ir, 2) : "n/a") << " | "
       << (a ? num(a->dsr_excess, 3) : "n/a") << " | " << (b ? pct(b->perf.ann_excess) : "n/a") << " | "
       << (b ? num(b->perf.ir, 2) : "n/a") << " | " << (c ? pct(c->perf.ann_excess) : "n/a") << " | "
       << (c ? num(c->perf.ir, 2) : "n/a") << " |\n";
  }
  auto mark = [](bool ok) { return ok ? "pass" : "FAIL"; };
  md << "\n### M3a gate per external (tilt over its scored span)\n\n"
     << "Same thresholds as the blend's gate: c1 annual excess > 0 with CI95 lower bound > 0; c2 excess positive in "
        ">= 60% of years; c3 DSR_excess > 0.95; c4 max drawdown <= buy-and-hold's over the span + 5%; c5 from the "
        "large-cap sibling run's gate for the same external (pending without one).\n\n"
     << "| signal | c1 | c2 | c3 | c4 | c5 | verdict | reason |\n|---|---|---|---|---|---|---|---|\n";
  for (const auto& g : gates) {
    if (!g.have) {
      md << "| " << g.name << " | n/a | n/a | n/a | n/a | n/a | FAIL | no simulated days |\n";
      continue;
    }
    const std::string verdict = !g.core_pass ? "FAIL" : !g.largecap ? "pending" : g.gate.pass ? "PASS" : "FAIL";
    md << "| " << g.name << " | " << mark(g.gate.c1) << " | " << mark(g.gate.c2) << " | " << mark(g.gate.c3) << " ("
       << num(g.dsr_excess, 3) << ") | " << mark(g.gate.c4) << " | "
       << (g.largecap ? std::string(mark(g.gate.c5)) : std::string("pending")) << " | " << verdict << " | "
       << (g.reason.empty() ? std::string("-") : g.reason) << " |\n";
  }

  const ExternalResult *learned = nullptr, *b0 = nullptr;
  for (const auto& e : externals) {
    if (e.name == "learned") learned = &e;
    if (e.name == "B0") b0 = &e;
  }
  if (!learned || !b0) return;
  std::vector<bool> both(learned->scored.size(), false);
  for (std::size_t m = 0; m < both.size() && m < b0->scored.size(); ++m) both[m] = learned->scored[m] && b0->scored[m];
  md << "\n### learned vs B0 (the decisive M4 comparison)\n\n"
     << "Paired per rebalance, using only the periods whose decision date d_m is scored by BOTH externals. IC "
        "difference: IC(learned) - IC(B0) at d_m against the period's open-to-open label. Tilt and sleeve "
        "differences: the return of the period decided at d_m (close of d_m to close of d_{m+1}) of `sig:learned` "
        "minus `sig:B0` (resp. the sleeves); the shared benchmark cancels, so this is the paired difference of their "
        "weekly excesses. Positive years: UTC years (of d_m) whose mean difference over those periods is "
        "positive.\n\n"
     << "| comparison | mean | t | n | positive years |\n|---|---:|---:|---:|---:|\n";
  auto line = [&](const std::string& what, const Paired& p, bool as_pct) {
    md << "| " << what << " | " << (as_pct ? pct(p.all.mean, 4) : num(p.all.mean, 4)) << " | " << num(p.all.t, 2)
       << " | " << p.all.n << " | " << p.years_pos << " of " << p.years << " |\n";
  };
  line("IC(learned) - IC(B0)", paired_diff(learned->rebalance_ic, b0->rebalance_ic, both, mt.dates), false);
  std::vector<TimePoint> closes;
  for (const auto& d : mt.dates) closes.push_back(parse_rfc3339(d));
  for (const char* kind : {"sig:", "sleeve:"}) {
    const EquityCurve* x = find_curve(curves, std::string(kind) + "learned");
    const EquityCurve* y = find_curve(curves, std::string(kind) + "B0");
    if (!x || !y) continue;
    line(std::string(kind) == "sig:" ? "tilt period return, learned - B0" : "sleeve period return, learned - B0",
         paired_diff(period_returns(*x, closes), period_returns(*y, closes), both, mt.dates), true);
  }
}

// Registry update, results.json and report.md (equity.csv and trades.csv are written by write_report only).
fs::path write_outputs(const Meta& mt, const std::vector<IcRow>& ic_table, const std::vector<BlendStep>& blend,
                       const Curves& curves, const std::vector<ExternalResult>& externals, const fs::path& out_dir) {
  check_run_id(mt.run_id, "run id");
  if (!mt.largecap_run.empty()) check_run_id(mt.largecap_run, "large-cap run id");
  // Sibling large-cap run first, so a bad id writes nothing.
  std::optional<bool> largecap;
  std::map<std::string, bool> largecap_ext;  // M4: the sibling's per-external core_pass
  if (!mt.largecap_run.empty()) {
    const fs::path sib = out_dir / mt.largecap_run / "results.json";
    std::ifstream in(sib);
    if (!in) throw std::runtime_error("--wf-largecap-run: cannot read " + sib.string());
    try {
      const auto sj = nlohmann::json::parse(in);
      largecap = sj.at("gate").at("core_pass").get<bool>();
      if (sj.contains("external"))
        for (const auto& e : sj.at("external"))
          if (e.contains("gate") && e.at("gate").contains("core_pass"))
            largecap_ext[e.at("name").get<std::string>()] = e.at("gate").at("core_pass").get<bool>();
    } catch (const std::exception& e) {
      throw std::runtime_error("--wf-largecap-run: bad " + sib.string() + ": " + e.what());
    }
  }

  const fs::path dir = out_dir / mt.run_id;
  static const EquityCurve kEmpty;
  const EquityCurve* bh_ptr = find_curve(curves, "bench:buyhold");
  const EquityCurve& bh = bh_ptr ? *bh_ptr : kEmpty;
  const double base_mdd = performance(bh, bh).max_drawdown;

  std::vector<StratRow> strat, bench, sleeves;
  for (const auto& [k, c] : curves) {
    if (k.rfind("sleeve:", 0) == 0) continue;
    StratRow row{k, &c, performance(c, bh)};
    (k.rfind("bench:", 0) == 0 ? bench : strat).push_back(std::move(row));
  }
  // Secondaries (reported, not gated): every strategy against the base rebalanced on the same calendar, and the
  // base-free sleeves against the equal-weight eligible universe.
  const EquityCurve* reb = find_curve(curves, "bench:rebalanced");
  const EquityCurve* ew = find_curve(curves, "bench:ew_eligible");
  std::vector<StratRow> vs_reb;
  if (reb)
    for (const auto& s : strat) vs_reb.push_back(StratRow{s.name, s.curve, performance(*s.curve, *reb)});
  if (ew)
    for (const auto& [k, c] : curves)
      if (k.rfind("sleeve:", 0) == 0) sleeves.push_back(StratRow{k, &c, performance(c, *ew)});

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
  for (auto* v : {&vs_reb, &sleeves})  // deflated with the same IR trials (an approximation: other benchmarks)
    for (auto& s : *v) s.dsr_excess = dsr_excess(s.perf, trials.trial_ir_var, trials.n_ir_trials);

  // M4: the M3a gate per external, on its tilt over its scored span.
  std::vector<ExtGate> ext_gates;
  for (const auto& e : externals) {
    ExtGate g;
    g.name = e.name;
    const StratRow* row = nullptr;
    for (const auto& s : strat)
      if (s.name == "sig:" + e.name) row = &s;
    if (row && !row->curve->t.empty() && row->perf.T > 0) {
      g.have = true;
      g.first = format_rfc3339(row->curve->t.front());
      g.last = format_rfc3339(row->curve->t.back());
      const EquityCurve bh_span = span_of(bh, row->curve->t.front(), row->curve->t.back());
      g.base_mdd = performance(bh_span, bh_span).max_drawdown;
      g.dsr_excess = row->dsr_excess;
      if (const auto it = largecap_ext.find(e.name); largecap && it != largecap_ext.end()) g.largecap = it->second;
      g.gate = decision_gate(row->perf, g.base_mdd, g.dsr_excess, g.largecap.value_or(false));
      g.core_pass = g.gate.c1 && g.gate.c2 && g.gate.c3 && g.gate.c4;
      g.reason = g.gate.reason;
      if (const auto k = g.reason.find("c5:"); !g.largecap && k != std::string::npos) g.reason.erase(k >= 2 ? k - 2 : k);
    }
    ext_gates.push_back(std::move(g));
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
      ic.push_back({{"signal", row.signal}, {"h", row.h}, {"mean", jnum(row.all.mean)},
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
    auto sec_json = [&](const StratRow& s, const char* benchmark) {
      const Perf& q = s.perf;
      return nlohmann::json{{"name", s.name}, {"benchmark", benchmark}, {"T", q.T},
                            {"cum_return", jnum(q.cum_return)}, {"ann_return", jnum(q.ann_return)},
                            {"ann_vol", jnum(q.ann_vol)}, {"sharpe", jnum(q.sharpe)},
                            {"max_drawdown", jnum(q.max_drawdown)}, {"ann_excess", jnum(q.ann_excess)},
                            {"excess_ci95", {jnum(q.excess_ci95.lo), jnum(q.excess_ci95.hi)}}, {"ir", jnum(q.ir)},
                            {"ir_daily", jnum(q.ir_daily)}, {"year_hit_rate", jnum(q.year_hit_rate)},
                            {"years", q.years}, {"dsr_excess", jnum(s.dsr_excess)}, {"costs", jnum(s.curve->costs)},
                            {"turnover", jnum(s.curve->turnover)}};
    };
    auto& sec = j["secondary"] = nlohmann::json::object();
    sec["vs_rebalanced"] = nlohmann::json::array();
    for (const auto& s : vs_reb) sec["vs_rebalanced"].push_back(sec_json(s, "bench:rebalanced"));
    sec["sleeves"] = nlohmann::json::array();
    for (const auto& s : sleeves) sec["sleeves"].push_back(sec_json(s, "bench:ew_eligible"));
    j["base_max_drawdown"] = jnum(base_mdd);
    j["registry"] = {{"n_trials", trials.n_trials}, {"trial_sr_var", jnum(trials.trial_sr_var)},
                     {"n_ir_trials", trials.n_ir_trials}, {"trial_ir_var", jnum(trials.trial_ir_var)}};
    j["gate"] = {{"c1", gate.c1}, {"c2", gate.c2}, {"c3", gate.c3}, {"c4", gate.c4},
                 {"c5", largecap ? nlohmann::json(gate.c5) : nlohmann::json("pending")},
                 {"c3_dsr_excess", jnum(blend_row ? blend_row->dsr_excess : std::nan(""))},
                 {"core_pass", core_pass},
                 {"pass", largecap ? nlohmann::json(gate.pass) : nlohmann::json("pending")},
                 {"reason", reason}};
    if (!externals.empty()) {  // M4; absent without externals (M3a outputs unchanged)
      auto& ex = j["external"] = nlohmann::json::array();
      for (std::size_t x = 0; x < externals.size(); ++x) {
        const auto& e = externals[x];
        const ExtGate& g = ext_gates[x];
        auto ics = nlohmann::json::array();
        for (double v : e.rebalance_ic) ics.push_back(jnum(v));
        nlohmann::json gate = nullptr;
        if (g.have)
          gate = {{"span", {g.first, g.last}}, {"base_max_drawdown", jnum(g.base_mdd)}, {"c1", g.gate.c1},
                  {"c2", g.gate.c2}, {"c3", g.gate.c3}, {"c4", g.gate.c4},
                  {"c5", g.largecap ? nlohmann::json(g.gate.c5) : nlohmann::json("pending")},
                  {"c3_dsr_excess", jnum(g.dsr_excess)}, {"core_pass", g.core_pass},
                  {"pass", g.largecap ? nlohmann::json(g.gate.pass) : nlohmann::json("pending")},
                  {"reason", g.reason}};
        ex.push_back({{"name", e.name}, {"rebalance_ic", ics}, {"scored", e.scored},
                      {"scored_bars", e.scored_bars}, {"scored_rebalances", e.scored_rebalances}, {"gate", gate}});
      }
    }
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
      md << "| " << row.signal << " | " << row.h << " | " << num(row.all.mean, 4) << " | "
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

    // Secondaries.
    auto sec_line = [&](const StratRow& s) {
      const Perf& q = s.perf;
      const auto hits = static_cast<long long>(std::llround(q.year_hit_rate * static_cast<double>(q.years)));
      md << "| " << s.name << " | " << q.T << " | " << pct(q.ann_return) << " | " << pct(q.max_drawdown) << " | "
         << pct(q.ann_excess) << " | " << pct(q.excess_ci95.lo) << " .. " << pct(q.excess_ci95.hi) << " | "
         << num(q.ir, 2) << " | " << hits << " of " << q.years << " | " << num(s.curve->turnover, 2) << " | "
         << num(s.dsr_excess, 3) << " |\n";
    };
    const char* kSecHeader =
        "| curve | days | ann | max DD | ann excess | excess CI95 | IR | years + | turnover | DSR_excess |\n"
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|\n";
    md << "\n## Secondary (reported, not gated)\n\n"
       << "Pre-registered on 2026-10-02 as secondaries for the next experiment (spec amendment). DSR_excess here uses "
          "the same registry IR trials as c3, an approximation since the benchmark differs.\n\n"
       << "### Strategies against the base rebalanced on the same calendar\n\n"
       << "This removes the rebalancing drag of the hindsight-selected base from the comparison.\n\n";
    if (vs_reb.empty()) md << "No `bench:rebalanced` curve in this run.\n";
    else {
      md << kSecHeader;
      for (const auto& s : vs_reb) sec_line(s);
    }
    md << "\n### Base-free sleeves against the equal-weight eligible universe\n\n"
       << "Each sleeve holds the top " << mt.k << " names at 1/" << mt.k
       << " each (tilt 1, no base; a flat blend holds the benchmark), with the same costs and turnover cap, "
          "against equal weight over each rebalance's eligible names rebalanced on the same calendar "
          "(`bench:ew_eligible`). Free of the hindsight-selected base.\n\n";
    if (sleeves.empty()) md << "No sleeves in this run (they were added on 2026-10-02; runs from then on carry them).\n";
    else {
      md << kSecHeader;
      for (const auto& s : sleeves) sec_line(s);
      if (ew) {
        const Perf q = performance(*ew, *ew);
        md << "\n`bench:ew_eligible` itself: ann " << pct(q.ann_return) << ", max DD " << pct(q.max_drawdown)
           << ", turnover " << num(ew->turnover, 2) << ", costs " << pct(ew->costs, 3) << ".\n";
      }
    }

    if (!externals.empty()) external_section(md, mt, ic_table, strat, vs_reb, sleeves, curves, externals, ext_gates);

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
  const fs::path dir = write_outputs(mt, r.ic_table, r.blend, r.curves, r.externals, out_dir);

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
    IcRow r{row.at("signal").get<std::string>(), row.at("h").get<int>(), {}, {}};
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
  auto add_costs = [&](const nlohmann::json& arr) {
    for (const auto& s : arr) cost_turn[s.at("name").get<std::string>()] = {unjnum(s.at("costs")), unjnum(s.at("turnover"))};
  };
  add_costs(j.at("strategies"));
  add_costs(j.at("benchmarks"));
  if (j.contains("secondary")) add_costs(j.at("secondary").at("sleeves"));
  for (auto& [name, c] : curves)
    if (auto it = cost_turn.find(name); it != cost_turn.end()) c.costs = it->second.first, c.turnover = it->second.second;
  std::vector<ExternalResult> externals;
  if (j.contains("external"))
    for (const auto& e : j.at("external")) {
      ExternalResult er{e.at("name").get<std::string>(), {}, {}, 0, 0};
      for (const auto& v : e.at("rebalance_ic")) er.rebalance_ic.push_back(unjnum(v));
      for (const auto& v : e.at("scored")) er.scored.push_back(v.get<bool>());
      er.scored_bars = e.at("scored_bars").get<std::size_t>();
      er.scored_rebalances = e.at("scored_rebalances").get<std::size_t>();
      externals.push_back(std::move(er));
    }
  return write_outputs(mt, ic, blend, curves, externals, out_dir);
}

}  // namespace mr
