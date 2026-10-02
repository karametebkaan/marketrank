#include "flows13f/compare.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
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
constexpr double kSplitSnap = 1.08;   // |log ratio| < log(1.08): no split (dividend drift is ~1-2%/quarter)
constexpr double kSplitConfirm = 1.1;  // the median holder share ratio must be within 10% of the price-factor ratio
constexpr double kSplitExact = 1.01;    // ... or this share of holders sits within 1% of it (non-traders show r exactly)
constexpr double kSplitHolderShare = 0.30;
constexpr std::size_t kSplitMinHolders = 4;  // the fraction rule needs >= 4 holders present in both quarters ...
constexpr std::size_t kSplitMinExact = 2;    // ... and >= 2 of them within 1% of r
constexpr std::size_t kTopK = 5000;   // observed top-K edges (and the legacy union-top-K)
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
// Aggregated edge list: ascending unique keys from * n + to (self-pairs dropped), summed positive weights.
struct Agg {
  std::vector<std::uint64_t> key;
  std::vector<double> w;
};

Agg aggregate(const std::vector<FlowEdge>& edges, std::size_t n) {
  std::vector<std::pair<std::uint64_t, double>> kv;
  kv.reserve(edges.size());
  for (const auto& e : edges) {
    if (e.from >= n || e.to >= n) throw std::invalid_argument("compare_flows: edge index out of range");
    if (e.from == e.to || !(e.dollars > 0) || !std::isfinite(e.dollars)) continue;
    kv.emplace_back(static_cast<std::uint64_t>(e.from) * n + e.to, e.dollars);
  }
  std::sort(kv.begin(), kv.end());
  Agg a;
  a.key.reserve(kv.size());
  a.w.reserve(kv.size());
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

// Average ranks over all N = n(n-1) ordered pairs of a sparse matrix: the Z = N - nnz zero pairs share rank
// r0 = (Z + 1) / 2, entry e has rank Z + (its average rank among the nonzero weights).
struct PairRanks {
  std::vector<double> r;  // per entry
  double r0 = 0, mean = 0, ss = 0;  // ss = sum over all N pairs of (rank - mean)^2
};
PairRanks pair_ranks(const Agg& a, double N) {
  PairRanks p;
  const std::size_t m = a.key.size();
  const double Z = N - static_cast<double>(m);
  p.r0 = (Z + 1) / 2;
  p.mean = (N + 1) / 2;
  std::vector<std::size_t> idx(m);
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  std::sort(idx.begin(), idx.end(), [&](std::size_t x, std::size_t y) { return a.w[x] < a.w[y]; });
  p.r.assign(m, 0);
  for (std::size_t i = 0; i < m;) {
    std::size_t j = i;
    while (j + 1 < m && a.w[idx[j + 1]] == a.w[idx[i]]) ++j;
    const double avg = Z + 0.5 * static_cast<double>(i + j) + 1.0;
    for (std::size_t k = i; k <= j; ++k) p.r[idx[k]] = avg;
    i = j + 1;
  }
  p.ss = Z * (p.r0 - p.mean) * (p.r0 - p.mean);
  for (double r : p.r) p.ss += (r - p.mean) * (r - p.mean);
  return p;
}

// Row-normalised shares per entry and the squared norm of the share matrix.
std::vector<double> row_shares(const Agg& a, std::size_t n, double& ss) {
  std::vector<double> rowsum(n, 0.0), s(a.key.size());
  for (std::size_t e = 0; e < a.key.size(); ++e) rowsum[a.key[e] / n] += a.w[e];
  ss = 0;
  for (std::size_t e = 0; e < a.key.size(); ++e) {
    s[e] = a.w[e] / rowsum[a.key[e] / n];
    ss += s[e] * s[e];
  }
  return s;
}

void margins(const Agg& a, std::size_t n, std::vector<double>& in, std::vector<double>& out) {
  in.assign(n, 0.0);
  out.assign(n, 0.0);
  for (std::size_t e = 0; e < a.key.size(); ++e) out[a.key[e] / n] += a.w[e], in[a.key[e] % n] += a.w[e];
}

// The observed side, precomputed once: dense weights, centred pair ranks and row shares (n x n), top-k sets, pi.
struct ObsSide {
  std::size_t n = 0;
  Agg agg;
  std::vector<double> dense, crank, share;  // n*n; crank = rank - mean over off-diagonal pairs (0 on the diagonal)
  double rank_ss = 0, share_ss = 0;
  std::vector<std::uint64_t> t500, t2000, t5000;  // ascending keys
  std::vector<double> t5000_w;
  std::vector<double> in, out, pi;
};

ObsSide obs_side(const std::vector<FlowEdge>& edges, std::size_t n) {
  ObsSide o;
  o.n = n;
  o.agg = aggregate(edges, n);
  const double N = static_cast<double>(n) * static_cast<double>(n > 0 ? n - 1 : 0);
  o.dense.assign(n * n, 0.0);
  o.share.assign(n * n, 0.0);
  const PairRanks pr = pair_ranks(o.agg, N);
  o.rank_ss = pr.ss;
  o.crank.assign(n * n, pr.r0 - pr.mean);
  for (std::size_t i = 0; i < n; ++i) o.crank[i * n + i] = 0;
  const auto sh = row_shares(o.agg, n, o.share_ss);
  for (std::size_t e = 0; e < o.agg.key.size(); ++e) {
    const auto k = o.agg.key[e];
    o.dense[k] = o.agg.w[e];
    o.crank[k] = pr.r[e] - pr.mean;
    o.share[k] = sh[e];
  }
  o.t500 = top_keys(o.agg, 500);
  o.t2000 = top_keys(o.agg, 2000);
  o.t5000 = top_keys(o.agg, kTopK);
  for (auto k : o.t5000) o.t5000_w.push_back(o.dense[k]);
  margins(o.agg, n, o.in, o.out);
  o.pi = pi_of_agg(o.agg, n);
  return o;
}

// An estimated side: sparse entries with their pair ranks (minus r0) and row shares, a dense lookup, top-k lists, pi.
struct EstSide {
  std::size_t n = 0;
  Agg agg;
  std::vector<double> dense;  // n*n
  std::vector<double> rr;     // per entry: rank - r0
  double rank_ss = 0, share_ss = 0;
  std::vector<double> share;  // per entry
  std::vector<std::uint64_t> t500, t2000, t5000;
  std::vector<double> in, out, pi;
};

EstSide est_side(Agg agg, std::size_t n) {
  EstSide s;
  s.n = n;
  s.agg = std::move(agg);
  const double N = static_cast<double>(n) * static_cast<double>(n > 0 ? n - 1 : 0);
  const PairRanks pr = pair_ranks(s.agg, N);
  s.rank_ss = pr.ss;
  s.rr.resize(pr.r.size());
  for (std::size_t e = 0; e < pr.r.size(); ++e) s.rr[e] = pr.r[e] - pr.r0;
  s.share = row_shares(s.agg, n, s.share_ss);
  s.dense.assign(n * n, 0.0);
  for (std::size_t e = 0; e < s.agg.key.size(); ++e) s.dense[s.agg.key[e]] = s.agg.w[e];
  s.t500 = top_keys(s.agg, 500);
  s.t2000 = top_keys(s.agg, 2000);
  s.t5000 = top_keys(s.agg, kTopK);
  margins(s.agg, n, s.in, s.out);
  s.pi = pi_of_agg(s.agg, n);
  return s;
}

Agg gravity_agg(const Agg& a, std::size_t n) {
  std::vector<double> in, out;
  margins(a, n, in, out);
  double total = 0;
  for (double x : out) total += x;
  Agg g;
  if (!(total > 0)) return g;
  for (std::size_t i = 0; i < n; ++i) {
    if (!(out[i] > 0)) continue;
    for (std::size_t j = 0; j < n; ++j)
      if (j != i && in[j] > 0) g.key.push_back(static_cast<std::uint64_t>(i) * n + j), g.w.push_back(out[i] * in[j] / total);
  }
  return g;
}

// Support-matched gravity: out_i * in_j / total on a's own nonzero pairs only, each row rescaled to a's row sum.
// It keeps a's support (density) and out-margins and drops only a's pairing within each row.
Agg gravity_support_agg(const Agg& a, std::size_t n) {
  std::vector<double> in, out;
  margins(a, n, in, out);
  Agg g;
  g.key = a.key;
  g.w.assign(a.key.size(), 0.0);
  std::size_t e = 0;
  while (e < a.key.size()) {
    const std::uint64_t i = a.key[e] / n;
    std::size_t f = e;
    double row = 0, mass = 0;
    for (; f < a.key.size() && a.key[f] / n == i; ++f) row += a.w[f], mass += in[a.key[f] % n];
    for (std::size_t x = e; x < f; ++x) g.w[x] = mass > 0 ? row * in[a.key[x] % n] / mass : 0.0;
    e = f;
  }
  return g;
}

double overlap(const std::vector<std::uint64_t>& a, const std::vector<std::uint64_t>& b, std::size_t k) {
  const std::size_t d = std::min({k, a.size(), b.size()});
  return d ? static_cast<double>(intersection_size(a, b)) / static_cast<double>(d) : kNaN;
}

// Metrics of O against E with E's node i relabelled perm[i] (perm empty = identity; inv = inverse permutation).
FlowMetrics evaluate(const ObsSide& o, const EstSide& e, const std::vector<std::uint32_t>& perm,
                     const std::vector<std::uint32_t>& inv) {
  const std::size_t n = o.n;
  const bool id = perm.empty();
  auto P = [&](std::uint64_t i) -> std::uint64_t { return id ? i : perm[i]; };
  auto I = [&](std::uint64_t i) -> std::uint64_t { return id ? i : inv[i]; };
  auto fwd = [&](std::uint64_t k) { return P(k / n) * n + P(k % n); };     // E key -> O space
  auto est_at = [&](std::uint64_t k) { return e.dense[I(k / n) * n + I(k % n)]; };  // O-space key -> E weight
  auto mapped = [&](const std::vector<std::uint64_t>& keys) {
    std::vector<std::uint64_t> m;
    m.reserve(keys.size());
    for (auto k : keys) m.push_back(fwd(k));
    std::sort(m.begin(), m.end());
    return m;
  };
  FlowMetrics f{};
  double num = 0, dot = 0;
  for (std::size_t x = 0; x < e.agg.key.size(); ++x) {
    const auto k = fwd(e.agg.key[x]);
    num += o.crank[k] * e.rr[x];
    dot += o.share[k] * e.share[x];
  }
  f.full_spearman = o.rank_ss > 0 && e.rank_ss > 0 ? num / std::sqrt(o.rank_ss * e.rank_ss) : kNaN;
  f.row_cosine = o.share_ss > 0 && e.share_ss > 0 ? dot / std::sqrt(o.share_ss * e.share_ss) : kNaN;
  {
    std::vector<double> y;
    y.reserve(o.t5000.size());
    for (auto k : o.t5000) y.push_back(est_at(k));
    f.obs_topk_spearman = spearman(o.t5000_w, y);
  }
  const auto m500 = mapped(e.t500), m2000 = mapped(e.t2000), m5000 = mapped(e.t5000);
  f.top500_overlap = overlap(o.t500, m500, 500);
  f.top2000_overlap = overlap(o.t2000, m2000, 2000);
  {
    std::vector<std::uint64_t> u;
    std::set_union(o.t5000.begin(), o.t5000.end(), m5000.begin(), m5000.end(), std::back_inserter(u));
    std::vector<double> x, y;
    for (auto k : u) x.push_back(o.dense[k]), y.push_back(est_at(k));
    f.union_spearman = spearman(x, y);
  }
  std::vector<double> ein(n), eout(n), epi(n);
  for (std::size_t i = 0; i < n; ++i) ein[P(i)] = e.in[i], eout[P(i)] = e.out[i], epi[P(i)] = e.pi[i];
  std::vector<double> a_in, b_in, a_out, b_out, a_pi, b_pi;
  for (std::size_t i = 0; i < n; ++i) {
    if (o.in[i] + o.out[i] + ein[i] + eout[i] <= 0) continue;
    a_in.push_back(o.in[i]), b_in.push_back(ein[i]), a_out.push_back(o.out[i]), b_out.push_back(eout[i]);
    a_pi.push_back(o.pi[i]), b_pi.push_back(epi[i]);
  }
  f.in_spearman = spearman(a_in, b_in);
  f.out_spearman = spearman(a_out, b_out);
  f.pi_spearman = spearman(o.pi, epi);
  f.pi_spearman_flow = spearman(a_pi, b_pi);
  {
    const auto a = top_nodes(o.pi, 50), b = top_nodes(epi, 50);
    std::vector<std::size_t> c;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(c));
    const std::size_t d = std::min<std::size_t>(50, n);
    f.top50_pi_overlap = d ? static_cast<double>(c.size()) / static_cast<double>(d) : kNaN;
  }
  return f;
}

std::vector<std::uint32_t> permutation(std::size_t n, std::uint32_t seed) {
  std::vector<std::uint32_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0u);
  std::mt19937 rng(seed);
  for (std::size_t i = n; i > 1; --i) std::swap(perm[i - 1], perm[rng() % i]);
  return perm;
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
    const BarFlux& bar = pipe.last_bar();
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

std::string next_quarter(const std::string& q) {
  quarter_bounds(q);  // validates
  int y = std::stoi(q.substr(0, 4));
  int k = q[5] - '0';
  if (++k == 5) k = 1, ++y;
  std::ostringstream s;
  s << std::setw(4) << std::setfill('0') << y << 'Q' << k;
  return s.str();
}

std::string placebo_quarter(const std::string& q, const std::function<bool(const std::string&)>& available) {
  const std::string back4 = previous_quarter(previous_quarter(previous_quarter(previous_quarter(q))));
  for (const auto& c : {back4, next_quarter(q), previous_quarter(q)})
    if (available(c)) return c;
  return "";
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
  std::vector<double> share_ratio(n, kNaN), cur_value(n, 0.0);
  std::vector<std::vector<double>> sr(n);  // per ticker: holder share ratios
  {
    auto per_manager = [&](const QuarterHoldings& h) {
      std::unordered_map<std::uint64_t, double> m;  // key cik * n + ticker index
      for (const auto& r : h.rows)
        if (auto it = idx.find(r.ticker); it != idx.end() && r.shares > 0)
          m[r.cik * static_cast<std::uint64_t>(n) + it->second] += r.shares;
      return m;
    };
    const auto mp = per_manager(prev), mc = per_manager(cur);
    for (const auto& [k, sc] : mc)
      if (auto it = mp.find(k); it != mp.end()) sr[k % n].push_back(sc / it->second);
    for (std::size_t i = 0; i < n; ++i) share_ratio[i] = median(sr[i]);
    for (const auto& r : cur.rows)
      if (auto it = idx.find(r.ticker); it != idx.end() && std::isfinite(r.value_usd)) cur_value[it->second] += r.value_usd;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const double fp = q.factor_prev[i], fc = q.factor_cur[i];
    if (std::isfinite(fp) && std::isfinite(fc)) {
      const double r = fp / fc, med = share_ratio[i];
      if (std::abs(std::log(r)) >= std::log(kSplitSnap)) {
        // (a) the median holder ratio is within 10% of r and closer to r than to 1 (unchanged holdings are not a
        // confirmation), or (b) at least 4 holders are present in both quarters and at least 30% of them (and at
        // least 2) sit within 1% of r.
        const bool by_median = std::isfinite(med) && med > 0 &&
                               std::abs(std::log(med / r)) < std::min(std::log(kSplitConfirm), std::abs(std::log(r)) / 2);
        std::size_t exact = 0;
        for (double x : sr[i]) exact += x > 0 && std::abs(std::log(x / r)) < std::log(kSplitExact);
        const bool by_holders =
            sr[i].size() >= kSplitMinHolders && exact >= kSplitMinExact &&
            static_cast<double>(exact) >= kSplitHolderShare * static_cast<double>(sr[i].size());
        const bool confirmed = by_median || by_holders;
        if (confirmed) q.ratio[i] = r, ++q.splits;
        else q.unconfirmed.push_back({i, r, med, cur_value[i]});
      }
    }
    // A full exit has no q rows: assume no split in q and take q-1's factor.
    const double f = std::isfinite(fc) ? fc : fp;
    if (std::isfinite(f) && std::isfinite(q.mean_close[i])) q.prices[i] = q.mean_close[i] * f;
  }
  std::stable_sort(q.unconfirmed.begin(), q.unconfirmed.end(),
                   [](const SplitCandidate& a, const SplitCandidate& b) { return a.value_usd > b.value_usd; });
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

namespace {
std::vector<FlowEdge> to_edges(const Agg& g, std::size_t n) {
  std::vector<FlowEdge> out;
  out.reserve(g.key.size());
  for (std::size_t e = 0; e < g.key.size(); ++e)
    if (g.w[e] > 0) out.push_back({static_cast<std::uint32_t>(g.key[e] / n), static_cast<std::uint32_t>(g.key[e] % n), g.w[e]});
  return out;
}
}  // namespace

std::vector<FlowEdge> gravity_support_null(const std::vector<FlowEdge>& edges, std::size_t n) {
  return to_edges(gravity_support_agg(aggregate(edges, n), n), n);
}

std::vector<FlowEdge> gravity_null(const std::vector<FlowEdge>& edges, std::size_t n) {
  const Agg g = gravity_agg(aggregate(edges, n), n);
  std::vector<FlowEdge> out;
  out.reserve(g.key.size());
  for (std::size_t e = 0; e < g.key.size(); ++e)
    out.push_back({static_cast<std::uint32_t>(g.key[e] / n), static_cast<std::uint32_t>(g.key[e] % n), g.w[e]});
  return out;
}

Agreement compare_flows(const std::vector<FlowEdge>& observed, const std::vector<FlowEdge>& estimated, std::size_t n,
                        std::size_t perms, const std::vector<FlowEdge>* placebo) {
  Agreement r;
  const ObsSide o = obs_side(observed, n);
  const std::vector<std::uint32_t> none;
  {
    Agg ea = aggregate(estimated, n);
    Agg ga = gravity_agg(ea, n);
    Agg gs = gravity_support_agg(ea, n);
    r.obs_edges = o.agg.key.size();
    r.est_edges = ea.key.size();
    r.gravity_edges = ga.key.size();
    r.gravity_support_edges = gs.key.size();
    const EstSide e = est_side(std::move(ea), n);
    r.est = evaluate(o, e, none, none);
    const double N = static_cast<double>(n) * static_cast<double>(n > 0 ? n - 1 : 0);
    auto expected = [&](std::size_t k) {
      const double a = static_cast<double>(std::min(k, o.agg.key.size())), b = static_cast<double>(std::min(k, e.agg.key.size()));
      return N > 0 && a > 0 && b > 0 ? std::max(a, b) / N : kNaN;
    };
    r.top500_expected = expected(500);
    r.top2000_expected = expected(2000);
    // Permutation null: R relabellings of the estimate, seeds 1..R.
    FlowMetrics sum{}, sq{};
    std::vector<std::size_t> cnt(std::size(kFlowMetricFields), 0);
    for (std::size_t s = 1; s <= perms; ++s) {
      const auto perm = permutation(n, static_cast<std::uint32_t>(s));
      std::vector<std::uint32_t> inv(n);
      for (std::size_t i = 0; i < n; ++i) inv[perm[i]] = static_cast<std::uint32_t>(i);
      const FlowMetrics f = evaluate(o, e, perm, inv);
      for (std::size_t m = 0; m < std::size(kFlowMetricFields); ++m) {
        const auto mp = kFlowMetricFields[m].second;
        if (!std::isfinite(f.*mp)) continue;
        sum.*mp += f.*mp, sq.*mp += f.*mp * f.*mp, ++cnt[m];
      }
    }
    r.perm.draws = perms;
    for (std::size_t m = 0; m < std::size(kFlowMetricFields); ++m) {
      const auto mp = kFlowMetricFields[m].second;
      const double c = static_cast<double>(cnt[m]);
      r.perm.mean.*mp = cnt[m] ? sum.*mp / c : kNaN;
      r.perm.sd.*mp = cnt[m] > 1 ? std::sqrt(std::max(0.0, (sq.*mp - c * r.perm.mean.*mp * r.perm.mean.*mp) / (c - 1))) : kNaN;
    }
    r.gravity = evaluate(o, est_side(std::move(ga), n), none, none);
    r.gravity_support = evaluate(o, est_side(std::move(gs), n), none, none);
  }
  if (placebo) {
    r.has_placebo = true;
    r.placebo = evaluate(o, est_side(aggregate(*placebo, n), n), none, none);
  }
  return r;
}

// ---- the --compare-13f run ----

namespace {

using nlohmann::json;

json metrics_json(const FlowMetrics& f) {
  json j = json::object();
  for (const auto& [name, mp] : kFlowMetricFields) j[name] = f.*mp;
  return j;
}

FlowMetrics minus(const FlowMetrics& a, const FlowMetrics& b) {
  FlowMetrics d{};
  for (const auto& [name, mp] : kFlowMetricFields) d.*mp = a.*mp - b.*mp;
  return d;
}

FlowMetrics nan_metrics() {
  FlowMetrics f{};
  for (const auto& [name, mp] : kFlowMetricFields) f.*mp = kNaN;
  return f;
}

// Headline metrics: the primary edge metrics and MarketRank (pi) agreement.
const char* const kPrimary[] = {"obs_topk_spearman", "full_spearman", "row_cosine", "pi_spearman", "pi_spearman_flow"};

json params_json(const CoreParams& p) {
  return {{"pressure", std::string(to_string(p.pressure))},
          {"adv_window", p.adv_window},
          {"corr_window", p.corr_window},
          {"min_dollar_volume", p.min_dollar_volume},
          {"max_volume_ratio", p.max_volume_ratio},
          {"stale_bars", p.stale_bars},
          {"flux", {{"lambda", p.flux.lambda}, {"sink_candidates", p.flux.sink_candidates},
                    {"sinks_per_source", p.flux.sinks_per_source}}},
          {"halflife_slow", p.halflife_slow},
          {"halflife_fast", p.halflife_fast},
          {"halflife_long", p.halflife_long},
          {"row_cap", p.row_cap},
          {"transition", {{"lift", std::string(to_string(p.transition.lift))},
                          {"k_out", p.transition.k_out},
                          {"k_in", p.transition.k_in},
                          {"retention", p.transition.retention},
                          {"dangling", p.transition.dangling == DanglingMode::Teleport ? "teleport" : "self-loop"}}},
          {"h_ref", std::string(to_string(p.h_ref))},
          {"alpha", p.alpha},
          {"beta", p.beta},
          {"vol_scale", p.vol_scale},
          {"vol_window", p.vol_window},
          {"horizons", p.horizons}};
}

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

// Per metric: mean, sd, min, max, n over the finite values of rows[*][field][metric].
json density_json(const Agreement& a, std::size_t m) {
  const double pairs = static_cast<double>(m) * static_cast<double>(m > 0 ? m - 1 : 0);
  auto d = [&](std::size_t k) { return json{{"edges", k}, {"fraction", pairs > 0 ? static_cast<double>(k) / pairs : kNaN}}; };
  return {{"pairs", pairs},
          {"observed", d(a.obs_edges)},
          {"estimate", d(a.est_edges)},
          {"gravity", d(a.gravity_edges)},
          {"gravity_support", d(a.gravity_support_edges)}};
}

json spread(const std::vector<json>& rows, const char* field) {
  json out = json::object();
  for (const auto& [name, mp] : kFlowMetricFields) {
    std::vector<double> v;
    for (const auto& r : rows)
      if (r.contains(field) && r[field].is_object())
        if (const double x = num(r[field][name]); std::isfinite(x)) v.push_back(x);
    json s = {{"n", v.size()}, {"positive", std::count_if(v.begin(), v.end(), [](double x) { return x > 0; })}};
    if (v.empty()) {
      s["mean"] = s["sd"] = s["min"] = s["max"] = nullptr;
    } else {
      double m = 0;
      for (double x : v) m += x;
      m /= static_cast<double>(v.size());
      double ss = 0;
      for (double x : v) ss += (x - m) * (x - m);
      s["mean"] = m;
      s["sd"] = v.size() > 1 ? json(std::sqrt(ss / static_cast<double>(v.size() - 1))) : json(nullptr);
      s["min"] = *std::min_element(v.begin(), v.end());
      s["max"] = *std::max_element(v.begin(), v.end());
    }
    out[name] = s;
  }
  return out;
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
  const std::size_t warm = std::max({opt.base.corr_window, opt.base.adv_window, opt.base.vol_window});

  json report;
  report["found_quarters"] = found;
  report["preset"] = opt.preset;
  report["base_params"] = params_json(opt.base);
  report["observed_params"] = {{"top_n", opt.observed.top_n}, {"prune_rel", opt.observed.prune_rel}};
  report["lookback_days"] = opt.lookback_days;
  report["timeframe"] = opt.timeframe;
  report["source_tree_git_sha_at_run_time"] = opt.git_sha;
  report["perms"] = opt.perms;
  report["warmup_returns_required"] = warm;
  report["panel"] = {{"nodes", panel.N()},
                     {"bars", panel.T()},
                     {"first", panel.T() ? format_rfc3339(panel.times.front()) : ""},
                     {"last", panel.T() ? format_rfc3339(panel.times.back()) : ""}};
  json skipped = json::array();

  // A quarter whose estimate is usable: bars inside it, complete in the panel, and warmed up.
  auto coverage = [&](const std::string& q) -> std::string {
    const auto [qs, qe] = quarter_bounds(q);
    const auto bars = quarter_bar_range(panel, q);
    if (!bars) return "no panel bars in the quarter";
    if (panel.times.front() >= qs) return "panel starts inside the quarter (extend --lookback-days)";
    if (panel.times.back() < qe) return "quarter not complete in the panel";
    // Bar 0 has no return, so the quarter's first bar has bars->first - 1 prior returns.
    if (bars->first < warm + 1)
      return "warm-up: " + std::to_string(bars->first == 0 ? 0 : bars->first - 1) +
             " returns before the quarter, need " + std::to_string(warm) + " (extend --lookback-days)";
    return "";
  };
  auto usable = [&](const std::string& q) { return coverage(q).empty(); };

  struct Planned {
    std::string q, prev, placebo;
    std::pair<std::size_t, std::size_t> bars;
    std::vector<std::uint32_t> nodes;
    std::vector<FlowEdge> observed;  // local indices into nodes
    std::vector<double> adv;         // local: mean close * volume over q's bars
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
    if (const std::string why = coverage(q); !why.empty()) { skip(why); continue; }
    plan.push_back({q, prev, placebo_quarter(q, usable), *quarter_bar_range(panel, q), {}, {}, {}, {}});
  }
  report["skipped"] = skipped;

  // Observed matrices (kept for every config) and the config-independent size baselines.
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
    const std::size_t m = p.nodes.size();
    double obs_total = 0;
    for (const auto& e : p.observed) obs_total += e.dollars;
    p.adv.assign(m, 0.0);
    std::vector<double> value(m, 0.0);
    {
      std::unordered_map<std::string, std::size_t> local;
      for (std::size_t k = 0; k < m; ++k) local.emplace(panel.tickers[p.nodes[k]], k);
      for (const auto& h : cur.rows)
        if (auto it = local.find(h.ticker); it != local.end() && std::isfinite(h.value_usd)) value[it->second] += h.value_usd;
      for (std::size_t k = 0; k < m; ++k) {
        double s = 0;
        std::size_t c = 0;
        for (std::size_t t = p.bars.first; t <= p.bars.second; ++t) {
          const double x = panel.close[panel.idx(t, p.nodes[k])] * panel.volume[panel.idx(t, p.nodes[k])];
          if (std::isfinite(x)) s += x, ++c;
        }
        p.adv[k] = c ? s / static_cast<double>(c) : 0.0;
      }
    }
    const auto pi_obs = pi_of(p.observed, m);
    json cands = json::array();
    for (const auto& c : pr.unconfirmed)
      cands.push_back({{"ticker", panel.tickers[c.node]},
                       {"price_ratio", c.price_ratio},
                       {"share_ratio", c.share_ratio},
                       {"value_usd", c.value_usd}});
    p.info = {{"quarter", p.q},
              {"previous", p.prev},
              {"placebo_quarter", p.placebo.empty() ? json(nullptr) : json(p.placebo)},
              {"bars", p.bars.second - p.bars.first + 1},
              {"warmup_bars", p.bars.first},
              {"warmup_returns", p.bars.first - 1},
              {"first_bar", format_rfc3339(panel.times[p.bars.first])},
              {"last_bar", format_rfc3339(panel.times[p.bars.second])},
              {"managers", f.managers},
              {"nodes", m},
              {"observed_edges", p.observed.size()},
              {"observed_dollars", obs_total},
              {"paired", f.paired},
              {"unpaired_in", f.unpaired_in},
              {"unpaired_out", f.unpaired_out},
              {"outside_in", f.outside_in},
              {"outside_out", f.outside_out},
              {"skipped_value", f.skipped_value},
              {"inconsistent_positions", f.inconsistent_positions},
              {"inconsistent_value", f.inconsistent_value},
              {"holdings_value_prev", prev.total_value},
              {"holdings_value_cur", cur.total_value},
              {"unmapped_value_cur", cur.dropped_value},
              {"splits", pr.splits},
              {"split_candidates_unconfirmed", cands},
              {"pi_obs_vs_adv", spearman(pi_obs, p.adv)},
              {"pi_obs_vs_13f_value", spearman(pi_obs, value)}};
    log << "  observed " << p.q << ": " << f.managers << " managers, " << m << " nodes, " << p.observed.size()
        << " edges, paired " << money(f.paired) << ", " << f.inconsistent_positions
        << " inconsistent positions dropped, " << pr.splits << " splits, " << pr.unconfirmed.size()
        << " unconfirmed split candidates, placebo " << (p.placebo.empty() ? "none" : p.placebo) << " ("
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s)\n";
  }

  // Estimates needed: every planned quarter and every placebo quarter, restricted to the union of the node sets.
  std::vector<std::string> needed;
  for (const auto& p : plan) {
    needed.push_back(p.q);
    if (!p.placebo.empty()) needed.push_back(p.placebo);
  }
  std::sort(needed.begin(), needed.end());
  needed.erase(std::unique(needed.begin(), needed.end()), needed.end());
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  for (const auto& q : needed) ranges.push_back(*quarter_bar_range(panel, q));
  std::vector<bool> in_union(panel.N(), false);
  for (const auto& p : plan)
    for (auto v : p.nodes) in_union[v] = true;

  const auto configs = calibration_configs(opt.base);
  json cfg_json = json::array();
  for (const auto& c : configs) {
    const auto t0 = std::chrono::steady_clock::now();
    std::map<std::string, std::vector<FlowEdge>> est;
    for_each_range_flow(panel, c.params, ranges, [&](std::size_t k, std::vector<FlowEdge> e) {
      std::erase_if(e, [&](const FlowEdge& x) { return !in_union[x.from] || !in_union[x.to]; });
      est[needed[k]] = std::move(e);
    });
    std::vector<json> rows;
    for (const auto& p : plan) {
      const std::size_t m = p.nodes.size();
      const auto local = restrict_to(est[p.q], p.nodes, panel.N());
      std::vector<FlowEdge> placebo;
      if (!p.placebo.empty()) placebo = restrict_to(est[p.placebo], p.nodes, panel.N());
      const Agreement a = compare_flows(p.observed, local, m, opt.perms, p.placebo.empty() ? nullptr : &placebo);
      double est_in = 0;
      for (const auto& e : local) est_in += e.dollars;
      json row = {{"quarter", p.q},
                  {"placebo_quarter", p.placebo.empty() ? json(nullptr) : json(p.placebo)},
                  {"estimate", metrics_json(a.est)},
                  {"gravity", metrics_json(a.gravity)},
                  {"placebo", a.has_placebo ? metrics_json(a.placebo) : json(nullptr)},
                  {"perm_mean", metrics_json(a.perm.mean)},
                  {"perm_sd", metrics_json(a.perm.sd)},
                  {"lift_placebo", metrics_json(a.has_placebo ? minus(a.est, a.placebo) : nan_metrics())},
                  {"lift_gravity", metrics_json(minus(a.est, a.gravity))},
                  {"gravity_support", metrics_json(a.gravity_support)},
                  {"lift_gravity_support", metrics_json(minus(a.est, a.gravity_support))},
                  {"lift_perm", metrics_json(minus(a.est, a.perm.mean))},
                  {"density", density_json(a, m)},
                  {"top500_expected", a.top500_expected},
                  {"top2000_expected", a.top2000_expected},
                  {"pi_est_vs_adv", spearman(pi_of(local, m), p.adv)},
                  {"estimated_edges", local.size()},
                  {"estimated_flow_in_nodes", est_in}};
      rows.push_back(row);
    }
    json summary = {{"estimate", spread(rows, "estimate")},
                    {"lift_placebo", spread(rows, "lift_placebo")},
                    {"lift_gravity", spread(rows, "lift_gravity")},
                    {"lift_gravity_support", spread(rows, "lift_gravity_support")},
                    {"lift_perm", spread(rows, "lift_perm")},
                    {"perm_mean", spread(rows, "perm_mean")}};
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    log << "  config " << c.name << (c.is_base ? " (base)" : "") << ": pi Spearman "
        << fmt(num(summary["estimate"]["pi_spearman"]["mean"])) << " (placebo lift "
        << fmt(num(summary["lift_placebo"]["pi_spearman"]["mean"])) << "), obs-topK edge Spearman "
        << fmt(num(summary["estimate"]["obs_topk_spearman"]["mean"])) << " (placebo lift "
        << fmt(num(summary["lift_placebo"]["obs_topk_spearman"]["mean"])) << ") (" << fmt(secs, 1) << " s)\n";
    cfg_json.push_back({{"name", c.name},
                        {"lambda", c.params.flux.lambda},
                        {"pressure", std::string(to_string(c.params.pressure))},
                        {"base", c.is_base},
                        {"quarters", rows},
                        {"summary", summary}});
  }
  json quarters = json::array();
  for (const auto& p : plan) quarters.push_back(p.info);
  report["quarters"] = quarters;
  report["configs"] = cfg_json;

  // Best setting per headline metric: highest mean lift over the temporal placebo.
  json best = json::object();
  for (const char* metric : kPrimary) {
    std::vector<std::pair<double, std::size_t>> v;
    for (std::size_t k = 0; k < cfg_json.size(); ++k)
      if (const double x = num(cfg_json[k]["summary"]["lift_placebo"][metric]["mean"]); std::isfinite(x))
        v.emplace_back(x, k);
    if (v.empty()) { best[metric] = nullptr; continue; }
    std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    const auto& c = cfg_json[v[0].second];
    const auto& lp = c["summary"]["lift_placebo"][metric];
    best[metric] = {{"config", c["name"]},
                    {"lift_placebo_mean", v[0].first},
                    {"lift_placebo_sd", lp["sd"]},
                    {"lift_placebo_min", lp["min"]},
                    {"lift_placebo_max", lp["max"]},
                    {"quarters", lp["n"]},
                    {"estimate_mean", c["summary"]["estimate"][metric]["mean"]},
                    {"lift_gravity_mean", c["summary"]["lift_gravity"][metric]["mean"]},
                    {"runner_up", v.size() > 1 ? cfg_json[v[1].second]["name"] : json(nullptr)},
                    {"margin_over_next", v.size() > 1 ? json(v[0].first - v[1].first) : json(nullptr)}};
  }
  report["best"] = best;
  report["best_config"] = best["pi_spearman"].is_object() ? best["pi_spearman"]["config"] : json(nullptr);

  std::ofstream(dir / "report.json") << report.dump(2) << "\n";
  std::ofstream(dir / "report.md") << compare_report_md(report);
  log << "wrote " << (dir / "report.md").string() << " and " << (dir / "report.json").string() << "\n";
  return report;
}

std::string compare_report_md(const nlohmann::json& r) {
  auto sv = [](const json& j) { return j.is_string() ? j.get<std::string>() : std::string("n/a"); };
  std::ostringstream s;
  s << "# 13F observed flows vs MarketRank estimated flows\n\n";
  s << "Panel: " << r["panel"]["nodes"].get<std::size_t>() << " nodes, " << r["panel"]["bars"].get<std::size_t>()
    << " bars, " << sv(r["panel"]["first"]) << " .. " << sv(r["panel"]["last"]) << " (" << sv(r["timeframe"])
    << ", lookback " << r["lookback_days"] << " days). Preset: " << sv(r["preset"]) << ". Source tree at run time (git): " << sv(r["source_tree_git_sha_at_run_time"])
    << ". Permutation draws: " << r["perms"] << ". Warm-up required: " << r["warmup_returns_required"] << " returns before a quarter's first bar.\n\n";
  s << "## Method\n\n"
       "- **Observed T_q** (13F): per manager, d = (shares_q - ratio * shares_{q-1}) * P_q; sources d < 0, sinks d > 0;\n"
       "  F_ij = out_i * in_j / sum(in) * min(1, sum(in)/sum(out)); summed over managers. Node set: the top "
    << r["observed_params"]["top_n"]
    << " tickers by 13F value (q-1 and q), accumulated densely, no cap.\n"
       "- **Dropped positions**: a position is dropped when either side's value/shares is 100x or more off P_q (the\n"
       "  13F-median price moved through lake returns; q-1 at P_q * ratio), or a side filed at value 0 holds over $1M\n"
       "  of shares at that price: a SHARES or VALUE filing error. Counted per quarter; the dropped value is the 13F\n"
       "  value of both quarters' rows, as filed (it can include the bogus values).\n"
       "- **Prices and splits**: lake bars are adjustment=all. Per ticker and quarter end, f = median(13F value/shares) /\n"
       "  last adjusted close of that quarter; P_q = mean adjusted close over q's bars * f_q (q's raw basis);\n"
       "  ratio = f_{q-1} / f_q, snapped to 1 within 8% (dividend drift); a split also needs the median holder share\n"
       "  ratio shares_q / shares_{q-1} within 10% of it (and closer to it than to 1), or, with at least 4 holders\n"
       "  present in both quarters, at least 30% of them (and at least 2) within 1% of it; otherwise it is listed as an\n"
       "  unconfirmed candidate.\n"
       "- **Estimated T^_q**: the pipeline (warm from the panel's first bar; a quarter needs at least the warm-up count of\n"
       "  prior returns before it) is stepped through q's last bar; the exact BarFlux of every bar in q (UTC calendar quarter) is\n"
       "  summed in a separate FluxAccumulator (half-life 1e9, no row cap).\n"
       "- **Same node set**: the estimated flows are restricted to the observed node set before comparing.\n"
       "- **Edge metrics** (primary): obs-topK = Spearman over the observed top-5000 edges with the estimate looked up\n"
       "  (0 if missing); full = Spearman over all n(n-1) ordered pairs; row cosine = cosine of the row-normalised share\n"
       "  matrices. Top-k overlaps |A n B| / min(k, |A|, |B|) are descriptive, shown with the hypergeometric\n"
       "  expectation. The union-top-5000 Spearman is kept in the JSON only (it rises with estimator density).\n"
       "- **MarketRank agreement**: pi by damped power iteration on row-normalised out-shares (p = 0.15, dangling ->\n"
       "  teleport); Spearman over all nodes and over nodes with any flow; top-50 overlap.\n"
       "- **Nulls**: lift = metric - null against (1) the **temporal placebo** - the same config's estimate of another\n"
       "  quarter (q-4, else q+1, else q-1), which cancels density and size effects and is the fair null across\n"
       "  configs; (2) the **gravity null** G = out * in^T / total from the estimate's own margins, dense over every\n"
       "  pair with margins (missing pairs count as 0 in the edge metrics, so it is favoured by density); (2b) the\n"
       "  **support-matched gravity null**: G masked to the estimate's nonzero pairs, each row rescaled to the\n"
       "  estimate's row sum - same support (density) and out-margins, so its lift isolates the estimate's pairing\n"
       "  within rows; (3) label permutations of the estimate (seeds 1..R, mean and sd). Size baseline for pi:\n"
       "  Spearman of observed pi against quarter ADV and against 13F value.\n"
       "- **Best config**: highest mean lift over the placebo.\n\n";
  s << "## Quarters\n\nFound:";
  for (const auto& q : r["found_quarters"]) s << " " << q.get<std::string>();
  s << "\n\n";
  for (const auto& k : r["skipped"]) s << "- skipped " << sv(k["quarter"]) << ": " << sv(k["reason"]) << "\n";
  if (!r["skipped"].empty()) s << "\n";
  s << "| quarter | bars | warm-up | placebo | managers | nodes | obs edges | paired | unpaired in | unpaired out | "
       "dropped (price 100x off) | splits | unconfirmed | pi_obs vs ADV | pi_obs vs 13F value |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n";
  for (const auto& q : r["quarters"])
    s << "| " << sv(q["quarter"]) << " | " << q["bars"] << " | " << q["warmup_bars"] << " | " << sv(q["placebo_quarter"])
      << " | " << q["managers"] << " | " << q["nodes"] << " | " << q["observed_edges"] << " | "
      << money(num(q["paired"])) << " | " << money(num(q["unpaired_in"])) << " | " << money(num(q["unpaired_out"]))
      << " | " << q["inconsistent_positions"] << " (" << money(num(q["inconsistent_value"])) << ") | " << q["splits"]
      << " | " << q["split_candidates_unconfirmed"].size() << " | "
      << fmt(num(q["pi_obs_vs_adv"])) << " | " << fmt(num(q["pi_obs_vs_13f_value"])) << " |\n";
  for (const auto& q : r["quarters"]) {
    const auto& c = q["split_candidates_unconfirmed"];
    if (c.empty()) continue;
    s << "\nUnconfirmed split candidates " << sv(q["quarter"]) << " (" << c.size() << ", largest 13F value first):";
    for (std::size_t k = 0; k < std::min<std::size_t>(10, c.size()); ++k)
      s << " " << sv(c[k]["ticker"]) << " (r " << fmt(num(c[k]["price_ratio"]), 2) << ", shares "
        << fmt(num(c[k]["share_ratio"]), 2) << ", " << money(num(c[k]["value_usd"])) << ")";
    s << "\n";
  }
  s << "\n## Calibration grid (mean over quarters; lift = metric - null, placebo lift as mean +- sd [min, max])\n";
  for (const char* metric : kPrimary) {
    s << "\n### " << metric
      << "\n\n| config | estimate | lift vs placebo | lift vs gravity (dense) [q>0] | lift vs support-matched gravity "
         "+- sd [q>0] | perm null |\n|---|---|---|---|---|---|\n";
    for (const auto& c : r["configs"]) {
      const auto& sm = c["summary"];
      const auto& lp = sm["lift_placebo"][metric];
      s << "| " << sv(c["name"]) << (c["base"].get<bool>() ? " (base)" : "") << " | "
        << fmt(num(sm["estimate"][metric]["mean"])) << " | " << fmt(num(lp["mean"])) << " +- " << fmt(num(lp["sd"]))
        << " [" << fmt(num(lp["min"])) << ", " << fmt(num(lp["max"])) << "] | "
        << fmt(num(sm["lift_gravity"][metric]["mean"])) << " [" << sm["lift_gravity"][metric]["positive"] << "/"
        << sm["lift_gravity"][metric]["n"] << "] | " << fmt(num(sm["lift_gravity_support"][metric]["mean"])) << " +- "
        << fmt(num(sm["lift_gravity_support"][metric]["sd"])) << " [" << sm["lift_gravity_support"][metric]["positive"]
        << "/" << sm["lift_gravity_support"][metric]["n"] << "] | " << fmt(num(sm["perm_mean"][metric]["mean"]))
        << " |\n";
    }
  }
  s << "\n## Best setting (by mean lift over the placebo)\n\n";
  for (const char* k : kPrimary) {
    const auto& b = r["best"][k];
    if (!b.is_object()) { s << "- " << k << ": n/a (no quarter with a placebo)\n"; continue; }
    s << "- " << k << ": **" << sv(b["config"]) << "**, placebo lift " << fmt(num(b["lift_placebo_mean"])) << " +- "
      << fmt(num(b["lift_placebo_sd"])) << " [" << fmt(num(b["lift_placebo_min"])) << ", "
      << fmt(num(b["lift_placebo_max"])) << "] over " << b["quarters"] << " quarters; estimate "
      << fmt(num(b["estimate_mean"])) << ", gravity lift " << fmt(num(b["lift_gravity_mean"])) << "; ahead of "
      << sv(b["runner_up"]) << " by " << fmt(num(b["margin_over_next"])) << "\n";
  }
  for (const auto& c : r["configs"]) {
    if (!c["base"].get<bool>()) continue;
    s << "\n## Per quarter (" << sv(c["name"]) << ", base): estimate / placebo / gravity / perm mean +- sd\n\n"
         "| quarter | obs-topK rho | full rho | row cosine | pi rho | pi rho (flow) | top500 (exp) | top2000 (exp) | "
         "pi_est vs ADV | support-matched gravity lift (topK / full / cosine) | density obs / est / dense gravity |\n"
         "|---|---|---|---|---|---|---|---|---|---|---|\n";
    for (const auto& q : c["quarters"]) {
      s << "| " << sv(q["quarter"]);
      for (const char* m : kPrimary) {
        const auto pl = q["placebo"].is_object() ? num(q["placebo"][m]) : kNaN;
        s << " | " << fmt(num(q["estimate"][m])) << " / " << fmt(pl) << " / " << fmt(num(q["gravity"][m])) << " / "
          << fmt(num(q["perm_mean"][m])) << " +- " << fmt(num(q["perm_sd"][m]));
      }
      s << " | " << fmt(num(q["estimate"]["top500_overlap"])) << " (" << fmt(num(q["top500_expected"]), 5) << ") | "
        << fmt(num(q["estimate"]["top2000_overlap"])) << " (" << fmt(num(q["top2000_expected"]), 5) << ") | "
        << fmt(num(q["pi_est_vs_adv"])) << " | " << fmt(num(q["lift_gravity_support"]["obs_topk_spearman"])) << " / "
        << fmt(num(q["lift_gravity_support"]["full_spearman"])) << " / " << fmt(num(q["lift_gravity_support"]["row_cosine"]))
        << " | " << fmt(num(q["density"]["observed"]["fraction"])) << " / " << fmt(num(q["density"]["estimate"]["fraction"]))
        << " / " << fmt(num(q["density"]["gravity"]["fraction"])) << " |\n";
    }
  }
  s << "\n## Honest limits\n\n"
       "- 13F is quarterly, long-only and institutional, with a 45-day lag; it omits shorts, retail and intra-quarter\n"
       "  round trips. Fund inflows/outflows appear as unpaired cash, not pairs.\n"
       "- Proportional pairing makes T_q a sum of per-manager rank-1 matrices (out_m in_m^T / sum in_m), so edge\n"
       "  agreement is largely agreement on marginals; the support-matched gravity and placebo lifts measure what is\n"
       "  left (the dense gravity null is also favoured by density).\n"
       "- Trade prices are unknown: d uses the quarter's mean lake close.\n";
  return s.str();
}

}  // namespace mr
