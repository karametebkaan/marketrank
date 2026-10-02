#include "flows13f/compare.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <ostream>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "core/time.hpp"
#include "graph/csr.hpp"
#include "graph/markov_solver.hpp"
#include "graph/sparse_flux.hpp"
#include "pipeline/evaluation.hpp"  // spearman (average ranks; NaN below 3 finite pairs)

namespace mr {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kEstimateHalflife = 1e9;
constexpr double kSplitSnap = 1.25;   // |log ratio| < log(1.25): no split
constexpr double kSplitConfirm = 1.1;  // the median holder share ratio must be within 10% of the price-factor ratio
constexpr std::size_t kEdgeUnionTop = 5000;
constexpr std::uint32_t kShuffleSeed = 13;
constexpr double kAlpha = 0.85;       // p = 0.15, as the market_rank() preset

double median(std::vector<double> v) {
  if (v.empty()) return kNaN;
  const std::size_t m = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m), v.end());
  const double hi = v[m];
  if (v.size() % 2) return hi;
  const double lo = *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m));
  return 0.5 * (lo + hi);
}

// Aggregated edge list: ascending unique keys from * n + to, summed positive weights.
struct Agg {
  std::vector<std::uint64_t> key;
  std::vector<double> w;
  double at(std::uint64_t k) const {
    auto it = std::lower_bound(key.begin(), key.end(), k);
    return it != key.end() && *it == k ? w[static_cast<std::size_t>(it - key.begin())] : 0.0;
  }
};

Agg aggregate(const std::vector<FlowEdge>& edges, std::size_t n) {
  std::vector<std::pair<std::uint64_t, double>> kv;
  kv.reserve(edges.size());
  for (const auto& e : edges) {
    if (e.from >= n || e.to >= n) throw std::invalid_argument("compare_flows: edge index out of range");
    if (!(e.dollars > 0) || !std::isfinite(e.dollars)) continue;
    kv.emplace_back(static_cast<std::uint64_t>(e.from) * n + e.to, e.dollars);
  }
  std::sort(kv.begin(), kv.end());
  Agg a;
  for (const auto& [k, w] : kv) {
    if (!a.key.empty() && a.key.back() == k) a.w.back() += w;
    else a.key.push_back(k), a.w.push_back(w);
  }
  return a;
}

// Keys of the k heaviest edges (weight desc, key asc on ties), returned ascending.
std::vector<std::uint64_t> top_keys(const Agg& a, std::size_t k) {
  std::vector<std::size_t> idx(a.key.size());
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  k = std::min(k, idx.size());
  auto heavier = [&](std::size_t x, std::size_t y) { return a.w[x] > a.w[y] || (a.w[x] == a.w[y] && x < y); };
  std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(), heavier);
  std::vector<std::uint64_t> out;
  out.reserve(k);
  for (std::size_t i = 0; i < k; ++i) out.push_back(a.key[idx[i]]);
  std::sort(out.begin(), out.end());
  return out;
}

std::size_t intersection_size(const std::vector<std::uint64_t>& a, const std::vector<std::uint64_t>& b) {
  std::size_t c = 0;
  for (std::size_t i = 0, j = 0; i < a.size() && j < b.size();) {
    if (a[i] < b[j]) ++i;
    else if (b[j] < a[i]) ++j;
    else ++c, ++i, ++j;
  }
  return c;
}

double edge_spearman_of(const Agg& o, const Agg& e) {
  const auto to = top_keys(o, kEdgeUnionTop), te = top_keys(e, kEdgeUnionTop);
  std::vector<std::uint64_t> u;
  std::set_union(to.begin(), to.end(), te.begin(), te.end(), std::back_inserter(u));
  std::vector<double> x, y;
  x.reserve(u.size());
  y.reserve(u.size());
  for (auto k : u) x.push_back(o.at(k)), y.push_back(e.at(k));
  return spearman(x, y);
}

Csr csr_of(const Agg& a, std::size_t n) {
  Csr P;
  P.n = n;
  P.row_ptr.assign(n + 1, 0);
  std::size_t e = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t begin = e;
    double sum = 0;
    while (e < a.key.size() && a.key[e] / n == i) sum += a.w[e++];
    for (std::size_t k = begin; k < e && sum > 0; ++k) {
      P.col.push_back(static_cast<std::uint32_t>(a.key[k] % n));
      P.val.push_back(a.w[k] / sum);
      P.raw.push_back(a.w[k]);
    }
    P.row_ptr[i + 1] = P.col.size();
  }
  return P;
}

std::vector<double> pi_of_agg(const Agg& a, std::size_t n) {
  if (n == 0) return {};
  return stationary(csr_of(a, n), kAlpha, {}).pi;
}

std::vector<std::size_t> top_nodes(const std::vector<double>& v, std::size_t k) {
  std::vector<std::size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  k = std::min(k, idx.size());
  std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                    [&](std::size_t a, std::size_t b) { return v[a] > v[b] || (v[a] == v[b] && a < b); });
  idx.resize(k);
  std::sort(idx.begin(), idx.end());
  return idx;
}

std::vector<FlowEdge> to_edges(const FluxAccumulator& acc) {
  std::vector<FlowEdge> out;
  out.reserve(acc.edge_count());
  for (std::size_t i = 0; i < acc.size(); ++i)
    for (const auto& e : acc.rows()[i])
      if (e.w > 0) out.push_back({static_cast<std::uint32_t>(i), e.j, e.w});
  return out;
}

// One pipeline pass; `emit(k, edges)` is called once per range as soon as the range's last bar is in (ranges in
// order of their last bar), and the range's accumulator is released.
void for_each_range_flow(const Panel& panel, const CoreParams& params,
                         const std::vector<std::pair<std::size_t, std::size_t>>& ranges,
                         const std::function<void(std::size_t, std::vector<FlowEdge>)>& emit) {
  if (ranges.empty()) return;
  std::size_t end = 0;
  for (const auto& [a, b] : ranges) {
    if (a > b || b >= panel.T()) throw std::invalid_argument("estimated flows: bad bar range");
    end = std::max(end, b);
  }
  const std::size_t n = panel.N();
  std::vector<std::optional<FluxAccumulator>> acc(ranges.size());
  std::vector<bool> done(ranges.size(), false);
  CorePipeline pipe(n, params);
  for (std::size_t t = 1; t <= end; ++t) {
    pipe.step(panel, t);
    const BarFlux& bar = pipe.last_bar_flux();
    for (std::size_t k = 0; k < ranges.size(); ++k) {
      if (t < ranges[k].first || t > ranges[k].second) continue;
      if (!acc[k]) acc[k].emplace(n, kEstimateHalflife, std::max<std::size_t>(n, 1));  // no row cap
      acc[k]->add(bar);
    }
    for (std::size_t k = 0; k < ranges.size(); ++k)
      if (!done[k] && ranges[k].second == t) {
        done[k] = true;
        emit(k, acc[k] ? to_edges(*acc[k]) : std::vector<FlowEdge>{});
        acc[k].reset();
      }
  }
  for (std::size_t k = 0; k < ranges.size(); ++k)  // ranges ending at bar 0: no stepped bar
    if (!done[k]) emit(k, {});
}

}  // namespace

// ---- quarters ----

bool is_quarter(const std::string& q) {
  return q.size() == 6 && std::all_of(q.begin(), q.begin() + 4, [](char c) { return c >= '0' && c <= '9'; }) &&
         q[4] == 'Q' && q[5] >= '1' && q[5] <= '4';
}

std::pair<TimePoint, TimePoint> quarter_bounds(const std::string& q) {
  if (!is_quarter(q)) throw std::invalid_argument("bad quarter '" + q + "' (expected YYYYQn)");
  const int y = std::stoi(q.substr(0, 4));
  const int k = q[5] - '1';
  const auto m = static_cast<unsigned>(1 + 3 * k);
  const TimePoint s = utc_seconds(y, m, 1);
  const TimePoint e = k == 3 ? utc_seconds(y + 1, 1, 1) : utc_seconds(y, m + 3, 1);
  return {s, e};
}

std::string previous_quarter(const std::string& q) {
  quarter_bounds(q);  // validates
  int y = std::stoi(q.substr(0, 4));
  int k = q[5] - '0';
  if (--k == 0) k = 4, --y;
  std::ostringstream s;
  s << std::setw(4) << std::setfill('0') << y << 'Q' << k;
  return s.str();
}

std::optional<std::pair<std::size_t, std::size_t>> quarter_bar_range(const Panel& panel, const std::string& q) {
  const auto [s, e] = quarter_bounds(q);
  const auto first = std::lower_bound(panel.times.begin(), panel.times.end(), s);
  const auto last = std::lower_bound(panel.times.begin(), panel.times.end(), e);
  if (first == last) return std::nullopt;
  return std::make_pair(static_cast<std::size_t>(first - panel.times.begin()),
                        static_cast<std::size_t>(last - panel.times.begin()) - 1);
}

// ---- estimated flows ----

std::vector<std::vector<FlowEdge>> estimated_range_flows(const Panel& panel, const CoreParams& params,
                                                         const std::vector<std::pair<std::size_t, std::size_t>>& ranges) {
  std::vector<std::vector<FlowEdge>> out(ranges.size());
  for_each_range_flow(panel, params, ranges, [&](std::size_t k, std::vector<FlowEdge> e) { out[k] = std::move(e); });
  return out;
}

std::vector<FlowEdge> estimated_quarter_flows(const Panel& panel, const CoreParams& params, std::size_t first_bar,
                                              std::size_t last_bar) {
  return estimated_range_flows(panel, params, {{first_bar, last_bar}}).front();
}

// ---- pricing ----

QuarterPricing quarter_pricing(const Panel& panel, const QuarterHoldings& prev, const QuarterHoldings& cur) {
  const std::size_t n = panel.N();
  std::unordered_map<std::string, std::size_t> idx;
  for (std::size_t i = 0; i < n; ++i) idx.emplace(panel.tickers[i], i);

  // raw/adjusted factor at the end of a quarter, per ticker
  auto factors = [&](const QuarterHoldings& h) {
    std::vector<std::vector<double>> raw(n);
    for (const auto& r : h.rows) {
      auto it = idx.find(r.ticker);
      if (it == idx.end() || !(r.shares > 0) || !(r.value_usd > 0)) continue;
      const double px = r.value_usd / r.shares;
      if (std::isfinite(px)) raw[it->second].push_back(px);
    }
    std::vector<double> f(n, kNaN);
    const auto range = is_quarter(h.quarter) ? quarter_bar_range(panel, h.quarter) : std::nullopt;
    if (!range) return f;
    for (std::size_t i = 0; i < n; ++i) {
      if (raw[i].empty()) continue;
      double adj = kNaN;
      for (std::size_t t = range->second + 1; t-- > range->first;) {
        const double c = panel.close[panel.idx(t, i)];
        if (std::isfinite(c) && c > 0) { adj = c; break; }
      }
      const double m = median(std::move(raw[i]));
      if (std::isfinite(adj) && m > 0) f[i] = m / adj;
    }
    return f;
  };

  QuarterPricing q;
  q.factor_prev = factors(prev);
  q.factor_cur = factors(cur);
  q.mean_close.assign(n, kNaN);
  q.prices.assign(n, kNaN);
  q.ratio.assign(n, 1.0);
  if (const auto range = is_quarter(cur.quarter) ? quarter_bar_range(panel, cur.quarter) : std::nullopt) {
    for (std::size_t i = 0; i < n; ++i) {
      double s = 0;
      std::size_t c = 0;
      for (std::size_t t = range->first; t <= range->second; ++t) {
        const double x = panel.close[panel.idx(t, i)];
        if (std::isfinite(x) && x > 0) s += x, ++c;
      }
      if (c) q.mean_close[i] = s / static_cast<double>(c);
    }
  }
  // Share-count confirmation: per ticker, the median over managers holding it in both quarters of
  // shares_q / shares_{q-1} (holders who did not trade show exactly the split ratio).
  std::vector<double> share_ratio(n, kNaN);
  {
    auto per_manager = [&](const QuarterHoldings& h) {
      std::unordered_map<std::uint64_t, double> m;  // key cik * n + ticker index
      for (const auto& r : h.rows)
        if (auto it = idx.find(r.ticker); it != idx.end() && r.shares > 0)
          m[r.cik * static_cast<std::uint64_t>(n) + it->second] += r.shares;
      return m;
    };
    const auto mp = per_manager(prev), mc = per_manager(cur);
    std::vector<std::vector<double>> sr(n);
    for (const auto& [k, sc] : mc)
      if (auto it = mp.find(k); it != mp.end()) sr[k % n].push_back(sc / it->second);
    for (std::size_t i = 0; i < n; ++i) share_ratio[i] = median(std::move(sr[i]));
  }
  for (std::size_t i = 0; i < n; ++i) {
    const double fp = q.factor_prev[i], fc = q.factor_cur[i];
    if (std::isfinite(fp) && std::isfinite(fc)) {
      const double r = fp / fc, sr = share_ratio[i];
      const bool confirmed = std::isfinite(sr) && sr > 0 && std::abs(std::log(sr / r)) < std::log(kSplitConfirm);
      if (std::abs(std::log(r)) >= std::log(kSplitSnap) && confirmed) q.ratio[i] = r, ++q.splits;
    }
    // A full exit has no q rows: assume no split in q and take q-1's factor.
    const double f = std::isfinite(fc) ? fc : fp;
    if (std::isfinite(f) && std::isfinite(q.mean_close[i])) q.prices[i] = q.mean_close[i] * f;
  }
  return q;
}

std::vector<FlowEdge> restrict_to(const std::vector<FlowEdge>& edges, const std::vector<std::uint32_t>& nodes,
                                  std::size_t n_total) {
  std::vector<std::int64_t> local(n_total, -1);
  for (std::size_t k = 0; k < nodes.size(); ++k) {
    if (nodes[k] >= n_total) throw std::invalid_argument("restrict_to: node out of range");
    local[nodes[k]] = static_cast<std::int64_t>(k);
  }
  std::vector<FlowEdge> out;
  for (const auto& e : edges) {
    if (e.from >= n_total || e.to >= n_total) continue;
    const auto a = local[e.from], b = local[e.to];
    if (a >= 0 && b >= 0) out.push_back({static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(b), e.dollars});
  }
  return out;
}

// ---- agreement ----

std::vector<double> pi_of(const std::vector<FlowEdge>& edges, std::size_t n) { return pi_of_agg(aggregate(edges, n), n); }

Agreement compare_flows(const std::vector<FlowEdge>& observed, const std::vector<FlowEdge>& estimated, std::size_t n) {
  Agreement r{};
  const Agg o = aggregate(observed, n), e = aggregate(estimated, n);
  r.edge_spearman = edge_spearman_of(o, e);
  const auto o500 = top_keys(o, 500), e500 = top_keys(e, 500), o2k = top_keys(o, 2000), e2k = top_keys(e, 2000);
  r.top500_overlap = static_cast<double>(intersection_size(o500, e500)) / 500.0;
  r.top2000_overlap = static_cast<double>(intersection_size(o2k, e2k)) / 2000.0;

  std::vector<double> oin(n, 0), oout(n, 0), ein(n, 0), eout(n, 0);
  for (std::size_t k = 0; k < o.key.size(); ++k) oout[o.key[k] / n] += o.w[k], oin[o.key[k] % n] += o.w[k];
  for (std::size_t k = 0; k < e.key.size(); ++k) eout[e.key[k] / n] += e.w[k], ein[e.key[k] % n] += e.w[k];
  std::vector<double> a_in, b_in, a_out, b_out;
  for (std::size_t i = 0; i < n; ++i) {
    if (oin[i] + oout[i] + ein[i] + eout[i] <= 0) continue;
    a_in.push_back(oin[i]), b_in.push_back(ein[i]), a_out.push_back(oout[i]), b_out.push_back(eout[i]);
  }
  r.in_spearman = spearman(a_in, b_in);
  r.out_spearman = spearman(a_out, b_out);

  const auto po = pi_of_agg(o, n), pe = pi_of_agg(e, n);
  r.pi_spearman = spearman(po, pe);
  r.top50_pi_overlap = [&] {
    const auto a = top_nodes(po, 50), b = top_nodes(pe, 50);
    std::vector<std::size_t> c;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(c));
    return static_cast<double>(c.size()) / 50.0;
  }();

  // Baseline: permute the estimated matrix's node labels (Fisher-Yates on mt19937, fully specified).
  std::vector<std::uint32_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0u);
  std::mt19937 rng(kShuffleSeed);
  for (std::size_t i = n; i > 1; --i) std::swap(perm[i - 1], perm[rng() % i]);
  std::vector<FlowEdge> shuffled;
  shuffled.reserve(e.key.size());
  for (std::size_t k = 0; k < e.key.size(); ++k)
    shuffled.push_back({perm[e.key[k] / n], perm[e.key[k] % n], e.w[k]});
  r.shuf_edge_spearman = edge_spearman_of(o, aggregate(shuffled, n));
  std::vector<double> ps(n, 0);
  for (std::size_t i = 0; i < n; ++i) ps[perm[i]] = pe[i];  // pi of a relabelled chain = relabelled pi
  r.shuf_pi_spearman = spearman(po, ps);
  return r;
}

// ---- the --compare-13f run ----

namespace {

using nlohmann::json;

json agreement_json(const Agreement& a) {
  return {{"edge_spearman", a.edge_spearman},     {"top500_overlap", a.top500_overlap},
          {"top2000_overlap", a.top2000_overlap}, {"in_spearman", a.in_spearman},
          {"out_spearman", a.out_spearman},       {"pi_spearman", a.pi_spearman},
          {"top50_pi_overlap", a.top50_pi_overlap}, {"shuf_edge_spearman", a.shuf_edge_spearman},
          {"shuf_pi_spearman", a.shuf_pi_spearman}};
}

const char* const kMetrics[] = {"edge_spearman",   "top500_overlap",   "top2000_overlap",
                                "in_spearman",     "out_spearman",     "pi_spearman",
                                "top50_pi_overlap", "shuf_edge_spearman", "shuf_pi_spearman"};

struct Config {
  std::string name;
  CoreParams params;
  bool is_base = false;
};

std::string config_name(const CoreParams& p) {
  std::ostringstream s;
  s << "lambda=" << p.flux.lambda << " pressure=" << to_string(p.pressure);
  return s.str();
}

std::vector<Config> calibration_configs(const CoreParams& base) {
  std::vector<Config> out;
  bool base_in_grid = false;
  for (const double lambda : {0.0, 0.5, 1.0})
    for (const PressureMode pm : {PressureMode::Dollar, PressureMode::Sqrt}) {
      CoreParams p = base;
      p.flux.lambda = lambda;
      p.pressure = pm;
      const bool is_base = p == base;
      base_in_grid = base_in_grid || is_base;
      out.push_back({config_name(p), p, is_base});
    }
  if (!base_in_grid) out.insert(out.begin(), Config{"base " + config_name(base), base, true});
  return out;
}

double num(const json& v) { return v.is_number() ? v.get<double>() : kNaN; }

std::string fmt(double x, int prec = 3) {
  if (!std::isfinite(x)) return "n/a";
  std::ostringstream s;
  s << std::fixed << std::setprecision(prec) << x;
  return s.str();
}

std::string money(double x) {
  if (!std::isfinite(x)) return "n/a";
  std::ostringstream s;
  s << "$" << std::fixed << std::setprecision(2) << x / 1e9 << "B";
  return s.str();
}

}  // namespace

nlohmann::json run_compare_13f(const Panel& panel, const Compare13fOptions& opt, std::ostream& log) {
  const auto dir = opt.data / "13f";
  if (!std::filesystem::is_directory(dir)) throw std::runtime_error("13f: no directory " + dir.string());
  const std::regex pat("holdings_(\\d{4}Q[1-4])\\.csv");
  std::vector<std::string> found;
  for (const auto& ent : std::filesystem::directory_iterator(dir)) {
    std::smatch m;
    const std::string name = ent.path().filename().string();
    if (std::regex_match(name, m, pat)) found.push_back(m[1]);
  }
  std::sort(found.begin(), found.end());
  log << "13f quarters found (" << found.size() << "):";
  for (const auto& q : found) log << " " << q;
  log << "\n";
  const std::vector<std::string> wanted = opt.quarters.empty() ? found : opt.quarters;

  json report;
  report["found_quarters"] = found;
  report["preset"] = opt.preset;
  report["top_n"] = opt.observed.top_n;
  report["panel"] = {{"nodes", panel.N()},
                     {"bars", panel.T()},
                     {"first", panel.T() ? format_rfc3339(panel.times.front()) : ""},
                     {"last", panel.T() ? format_rfc3339(panel.times.back()) : ""}};
  json skipped = json::array();

  struct Planned {
    std::string q, prev;
    std::pair<std::size_t, std::size_t> bars;
    std::vector<std::uint32_t> nodes;
    std::vector<FlowEdge> observed;  // local indices into nodes
    json info;
  };
  std::vector<Planned> plan;
  for (const auto& q : wanted) {
    auto skip = [&](const std::string& why) {
      log << "  skip " << q << ": " << why << "\n";
      skipped.push_back({{"quarter", q}, {"reason", why}});
    };
    if (!std::binary_search(found.begin(), found.end(), q)) { skip("no holdings file"); continue; }
    const std::string prev = previous_quarter(q);
    if (!std::binary_search(found.begin(), found.end(), prev)) { skip("no previous quarter " + prev); continue; }
    const auto [qs, qe] = quarter_bounds(q);
    const auto bars = quarter_bar_range(panel, q);
    if (!bars) { skip("no panel bars in the quarter"); continue; }
    if (panel.times.front() >= qs) { skip("panel starts inside the quarter (extend --lookback-days)"); continue; }
    if (panel.times.back() < qe) { skip("quarter not complete in the panel"); continue; }
    plan.push_back({q, prev, *bars, {}, {}, {}});
  }
  report["skipped"] = skipped;

  // Observed matrices (kept for every config).
  for (auto& p : plan) {
    const auto t0 = std::chrono::steady_clock::now();
    const QuarterHoldings prev = load_quarter(opt.data, p.prev), cur = load_quarter(opt.data, p.q);
    const QuarterPricing pr = quarter_pricing(panel, prev, cur);
    std::unordered_map<std::string, double> ratio;
    for (std::size_t i = 0; i < panel.N(); ++i) ratio.emplace(panel.tickers[i], pr.ratio[i]);
    const ObservedFlows f = observed_flows(prev, cur, panel.tickers, pr.prices, [&](const std::string& t) {
      auto it = ratio.find(t);
      return it == ratio.end() ? 1.0 : it->second;
    }, opt.observed);
    p.nodes = f.nodes;
    p.observed = restrict_to(f.edges, f.nodes, panel.N());
    double obs_total = 0;
    for (const auto& e : p.observed) obs_total += e.dollars;
    p.info = {{"quarter", p.q},
              {"previous", p.prev},
              {"bars", p.bars.second - p.bars.first + 1},
              {"first_bar", format_rfc3339(panel.times[p.bars.first])},
              {"last_bar", format_rfc3339(panel.times[p.bars.second])},
              {"managers", f.managers},
              {"nodes", f.nodes.size()},
              {"observed_edges", p.observed.size()},
              {"observed_dollars", obs_total},
              {"paired", f.paired},
              {"unpaired_in", f.unpaired_in},
              {"unpaired_out", f.unpaired_out},
              {"outside_in", f.outside_in},
              {"outside_out", f.outside_out},
              {"skipped_value", f.skipped_value},
              {"holdings_value_prev", prev.total_value},
              {"holdings_value_cur", cur.total_value},
              {"unmapped_value_cur", cur.dropped_value},
              {"splits", pr.splits}};
    log << "  observed " << p.q << ": " << f.managers << " managers, " << f.nodes.size() << " nodes, "
        << p.observed.size() << " edges, paired " << money(f.paired) << ", " << pr.splits << " splits ("
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s)\n";
  }

  // Calibration grid: one pipeline pass per config, every quarter compared as its last bar is reached.
  const auto configs = calibration_configs(opt.base);
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  for (const auto& p : plan) ranges.push_back(p.bars);
  json cfg_json = json::array();
  for (const auto& c : configs) {
    const auto t0 = std::chrono::steady_clock::now();
    json per_q = json::array();
    std::vector<json> rows(plan.size());
    for_each_range_flow(panel, c.params, ranges, [&](std::size_t k, std::vector<FlowEdge> est) {
      double est_all = 0, est_in = 0;
      for (const auto& e : est) est_all += e.dollars;
      const auto local = restrict_to(est, plan[k].nodes, panel.N());
      for (const auto& e : local) est_in += e.dollars;
      const Agreement a = compare_flows(plan[k].observed, local, plan[k].nodes.size());
      json row = agreement_json(a);
      row["quarter"] = plan[k].q;
      row["estimated_edges"] = local.size();
      row["estimated_flow_all"] = est_all;
      row["estimated_flow_in_nodes"] = est_in;
      rows[k] = row;
    });
    json mean;
    for (const char* m : kMetrics) {
      double s = 0;
      std::size_t cnt = 0;
      for (const auto& r : rows)
        if (const double x = num(r[m]); std::isfinite(x)) s += x, ++cnt;
      mean[m] = cnt ? s / static_cast<double>(cnt) : kNaN;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    log << "  config " << c.name << (c.is_base ? " (base)" : "") << ": mean pi Spearman " << fmt(num(mean["pi_spearman"]))
        << ", edge Spearman " << fmt(num(mean["edge_spearman"])) << " (" << fmt(secs, 1) << " s)\n";
    cfg_json.push_back({{"name", c.name},
                        {"lambda", c.params.flux.lambda},
                        {"pressure", std::string(to_string(c.params.pressure))},
                        {"base", c.is_base},
                        {"quarters", rows},
                        {"mean", mean}});
  }
  json quarters = json::array();
  for (const auto& p : plan) quarters.push_back(p.info);
  report["quarters"] = quarters;
  report["configs"] = cfg_json;

  // Best setting by mean pi Spearman (MarketRank agreement) and by mean edge Spearman.
  auto best_by = [&](const char* metric) -> json {
    std::vector<std::pair<double, std::string>> v;
    for (const auto& c : cfg_json)
      if (const double x = num(c["mean"][metric]); std::isfinite(x)) v.emplace_back(x, c["name"].get<std::string>());
    if (v.empty()) return nullptr;
    std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    json j = {{"metric", metric}, {"config", v[0].second}, {"value", v[0].first}};
    const std::string shuf = std::string("shuf_") + metric;
    for (const auto& c : cfg_json)
      if (c["name"] == v[0].second) j["over_shuffled"] = v[0].first - num(c["mean"][shuf]);
    j["margin_over_next"] = v.size() > 1 ? v[0].first - v[1].first : kNaN;
    j["runner_up"] = v.size() > 1 ? json(v[1].second) : json(nullptr);
    return j;
  };
  report["best"] = {{"pi_spearman", best_by("pi_spearman")}, {"edge_spearman", best_by("edge_spearman")}};

  std::ofstream(dir / "report.json") << report.dump(2) << "\n";
  std::ofstream(dir / "report.md") << compare_report_md(report);
  log << "wrote " << (dir / "report.md").string() << " and " << (dir / "report.json").string() << "\n";
  return report;
}

std::string compare_report_md(const nlohmann::json& r) {
  std::ostringstream s;
  s << "# 13F observed flows vs MarketRank estimated flows\n\n";
  s << "Panel: " << r["panel"]["nodes"].get<std::size_t>() << " nodes, " << r["panel"]["bars"].get<std::size_t>()
    << " bars, " << r["panel"]["first"].get<std::string>() << " .. " << r["panel"]["last"].get<std::string>()
    << ". Preset: " << r["preset"].get<std::string>() << ".\n\n";
  s << "## Method\n\n"
       "- **Observed T_q** (13F): per manager, d = (shares_q - ratio * shares_{q-1}) * P_q; sources d < 0, sinks d > 0;\n"
       "  F_ij = out_i * in_j / sum(in) * min(1, sum(in)/sum(out)); summed over managers. The node set is the top "
    << r["top_n"].get<std::size_t>()
    << " tickers by 13F value (q-1 and q), accumulated densely, no cap.\n"
       "- **Prices and splits**: lake bars are adjustment=all. Per ticker and quarter end, f = median(13F value/shares) /\n"
       "  last adjusted close of that quarter; P_q = mean adjusted close over q's bars * f_q (q's raw basis);\n"
       "  ratio = f_{q-1} / f_q, snapped to 1 within 25% (dividend drift, price noise); a split also needs the median\n"
       "  holder share ratio shares_q / shares_{q-1} within 10% of it.\n"
       "- **Estimated T^_q**: the pipeline (warm from the panel's first bar) is stepped through q's last bar; the exact\n"
       "  BarFlux of every bar in q (UTC calendar quarter [start, next start)) is summed in a separate FluxAccumulator\n"
       "  (half-life 1e9, no row cap). Not the pipeline's pruned, row-capped cumulative accumulator.\n"
       "- **Same node set**: the estimated flows are restricted to the observed node set before comparing.\n"
       "- **Metrics**: edge Spearman over the union of each matrix's top-5000 edges (missing = 0); top-k edge overlap\n"
       "  |A n B| / k; node in/out Spearman over nodes with any flow; pi = damped power iteration on row-normalised\n"
       "  out-shares (p = 0.15, dangling -> teleport), Spearman over the node set and top-50 overlap.\n"
       "  Shuffled baselines permute the estimated node labels (seed 13). Note: under independence the top-5000-union\n"
       "  edge Spearman is biased negative (selection on either side), so compare edge Spearman to its baseline.\n\n";
  s << "## Quarters\n\nFound:";
  for (const auto& q : r["found_quarters"]) s << " " << q.get<std::string>();
  s << "\n\n";
  for (const auto& k : r["skipped"])
    s << "- skipped " << k["quarter"].get<std::string>() << ": " << k["reason"].get<std::string>() << "\n";
  if (!r["skipped"].empty()) s << "\n";
  s << "| quarter | bars | managers | nodes | obs edges | paired | unpaired in | unpaired out | outside in/out | "
       "splits |\n|---|---|---|---|---|---|---|---|---|---|\n";
  for (const auto& q : r["quarters"])
    s << "| " << q["quarter"].get<std::string>() << " | " << q["bars"] << " | " << q["managers"] << " | " << q["nodes"]
      << " | " << q["observed_edges"] << " | " << money(num(q["paired"])) << " | " << money(num(q["unpaired_in"]))
      << " | " << money(num(q["unpaired_out"])) << " | " << money(num(q["outside_in"])) << " / "
      << money(num(q["outside_out"])) << " | " << q["splits"] << " |\n";
  s << "\n## Calibration grid (mean over quarters)\n\n"
       "| config | edge rho | shuf | top500 | top2000 | in rho | out rho | pi rho | shuf | top50 pi |\n"
       "|---|---|---|---|---|---|---|---|---|---|\n";
  for (const auto& c : r["configs"]) {
    const auto& m = c["mean"];
    s << "| " << c["name"].get<std::string>() << (c["base"].get<bool>() ? " (base)" : "") << " | "
      << fmt(num(m["edge_spearman"])) << " | " << fmt(num(m["shuf_edge_spearman"])) << " | "
      << fmt(num(m["top500_overlap"])) << " | " << fmt(num(m["top2000_overlap"])) << " | "
      << fmt(num(m["in_spearman"])) << " | " << fmt(num(m["out_spearman"])) << " | " << fmt(num(m["pi_spearman"]))
      << " | " << fmt(num(m["shuf_pi_spearman"])) << " | " << fmt(num(m["top50_pi_overlap"])) << " |\n";
  }
  s << "\n## Best setting\n\n";
  for (const char* k : {"pi_spearman", "edge_spearman"}) {
    const auto& b = r["best"][k];
    if (b.is_null()) { s << "- " << k << ": n/a\n"; continue; }
    s << "- by mean " << k << ": **" << b["config"].get<std::string>() << "** (" << fmt(num(b["value"]))
      << "), ahead of " << (b["runner_up"].is_null() ? std::string("none") : b["runner_up"].get<std::string>())
      << " by " << fmt(num(b["margin_over_next"])) << "; " << fmt(num(b["over_shuffled"]))
      << " above its shuffled baseline\n";
  }
  for (const auto& c : r["configs"]) {
    if (!c["base"].get<bool>()) continue;
    s << "\n## Per quarter (" << c["name"].get<std::string>() << ", base)\n\n"
         "| quarter | edge rho | shuf | top500 | top2000 | in rho | out rho | pi rho | shuf | top50 pi |\n"
         "|---|---|---|---|---|---|---|---|---|---|\n";
    for (const auto& q : c["quarters"])
      s << "| " << q["quarter"].get<std::string>() << " | " << fmt(num(q["edge_spearman"])) << " | "
        << fmt(num(q["shuf_edge_spearman"])) << " | " << fmt(num(q["top500_overlap"])) << " | "
        << fmt(num(q["top2000_overlap"])) << " | " << fmt(num(q["in_spearman"])) << " | "
        << fmt(num(q["out_spearman"])) << " | " << fmt(num(q["pi_spearman"])) << " | "
        << fmt(num(q["shuf_pi_spearman"])) << " | " << fmt(num(q["top50_pi_overlap"])) << " |\n";
  }
  s << "\n## Honest limits\n\n"
       "- 13F is quarterly, long-only and institutional, with a 45-day lag; it omits shorts, retail and intra-quarter\n"
       "  round trips. Fund inflows/outflows appear as unpaired cash, not pairs.\n"
       "- Trade prices are unknown: d uses the quarter's mean lake close.\n";
  return s.str();
}

}  // namespace mr
