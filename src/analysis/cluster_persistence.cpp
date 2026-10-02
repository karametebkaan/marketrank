#include "analysis/cluster_persistence.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

#include "analysis/partition_metrics.hpp"
#include "core/time.hpp"
#include "geom/community.hpp"
#include "market/sec_sectors.hpp"

namespace mr {

namespace {

constexpr int kGraphs = 3;
const std::array<const char*, kGraphs> kGraphName{"cum", "fast", "bar"};
constexpr int kMinSize = 8;  // the landscape's CommunityTracker min_size
constexpr double kResolution = 1.0;
constexpr std::uint64_t kSeed2 = 0x9e3779b97f4a7c15ULL;  // second Louvain visiting order (noise ceiling)

using Part = std::vector<std::int32_t>;  // per node: -1 inactive, -2 loose, else block id

Part to_part(const CommunityResult& r) {
  Part p(r.id.size(), -1);
  for (std::size_t i = 0; i < r.id.size(); ++i)
    if (r.id[i] >= 0) p[i] = r.id[i] == r.loose_id ? -2 : r.id[i];
  return p;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct Job {
  std::size_t t = 0;
  int g = 0;
  Csr W;
  std::vector<bool> act;
  bool second_seed = false, spectral = false;
  // results
  Part p0, p1, sp;
  double louvain_ms = 0, spectral_ms = 0, modularity = 0;
  std::size_t nodes = 0, edges = 0;
  int communities = 0;
  double loose_frac = 0;
};

void run_job(Job& j) {
  const auto t0 = std::chrono::steady_clock::now();
  const CommunityResult r = louvain_seeded(j.W, j.act, 0, kResolution, kMinSize);
  j.louvain_ms = ms_since(t0);
  j.p0 = to_part(r);
  j.modularity = r.modularity;
  j.communities = r.count - (r.loose_id >= 0 ? 1 : 0);
  std::size_t loose = 0;
  for (std::size_t i = 0; i < j.act.size(); ++i) {
    j.nodes += j.act[i] ? 1 : 0;
    loose += j.p0[i] == -2 ? 1 : 0;
  }
  j.loose_frac = j.nodes ? static_cast<double>(loose) / static_cast<double>(j.nodes) : 0.0;
  j.edges = j.W.col.size() / 2;
  if (j.second_seed) j.p1 = to_part(louvain_seeded(j.W, j.act, kSeed2, kResolution, kMinSize));
  if (j.spectral) {
    const auto t1 = std::chrono::steady_clock::now();
    j.sp = spectral_bisection(j.W, j.act, std::max(1, j.communities), kMinSize);
    j.spectral_ms = ms_since(t1);
  }
  j.W = Csr{};
}

struct Stat {
  double mean = NAN, median = NAN, q25 = NAN, q75 = NAN;
  std::size_t n = 0;
};

Stat stat(std::vector<double> v) {
  Stat s;
  v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return !std::isfinite(x); }), v.end());
  s.n = v.size();
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / static_cast<double>(v.size());
  auto q = [&](double p) {
    const double pos = p * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (pos - static_cast<double>(lo)) * (v[hi] - v[lo]);
  };
  s.median = q(0.5);
  s.q25 = q(0.25);
  s.q75 = q(0.75);
  return s;
}

// (graph, metric, lag) -> values with the anchor's year.
using Key = std::tuple<std::string, std::string, std::size_t>;
struct Series {
  std::vector<double> v;
  std::vector<std::string> year;
};

}  // namespace

void run_cluster_persistence(const Panel& panel, const std::vector<Security>& nodes, const CoreParams& params,
                             const ClusterPersistenceOptions& opt) {
  const std::size_t T = panel.T(), N = panel.N();
  if (nodes.size() != N) throw std::invalid_argument("cluster persistence: nodes size != panel N");
  if (opt.stride == 0 || opt.spectral_stride == 0) throw std::invalid_argument("cluster persistence: stride must be >= 1");
  if (T < opt.warmup + 2) throw std::runtime_error("cluster persistence: not enough bars");
  std::vector<char> stock(N, 0);
  for (std::size_t i = 0; i < N; ++i) stock[i] = nodes[i].sector != kSectorEtfFund;

  std::vector<std::size_t> anchors, sp_anchors;
  std::set<std::size_t> need, need_seed2, need_sp;
  for (std::size_t a = std::max<std::size_t>(1, opt.warmup); a < T; a += opt.stride) {
    anchors.push_back(a);
    need.insert(a);
    need_seed2.insert(a);
    for (auto k : opt.lags)
      if (a + k < T) need.insert(a + k);
    if ((a - std::max<std::size_t>(1, opt.warmup)) % opt.spectral_stride == 0) {
      sp_anchors.push_back(a);
      for (std::size_t k : {std::size_t{0}, std::size_t{1}, std::size_t{5}})
        if (a + k < T) need_sp.insert(a + k);
    }
  }
  std::fprintf(stderr,
               "cluster persistence: %zu bars, %zu nodes, %zu anchors (stride %zu from bar %zu), %zu clustered bars, "
               "%zu spectral bars, threads=%d\n",
               T, N, anchors.size(), opt.stride, opt.warmup, need.size(), need_sp.size(), omp_get_max_threads());

  std::array<std::map<std::size_t, Part>, kGraphs> p0, p1;
  std::map<std::size_t, Part> sp;
  std::map<Key, Series> series;
  auto add = [&](int g, const std::string& metric, std::size_t lag, double x, std::size_t t) {
    auto& s = series[{g < kGraphs ? kGraphName[static_cast<std::size_t>(g)] : "fast_spectral", metric, lag}];
    s.v.push_back(x);
    s.year.push_back(format_rfc3339(panel.times[t]).substr(0, 4));
  };

  std::vector<Job> batch;
  const std::size_t batch_max = static_cast<std::size_t>(std::max(1, omp_get_max_threads())) * 2;
  auto flush = [&] {
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t b = 0; b < batch.size(); ++b) run_job(batch[b]);
    for (auto& j : batch) {
      const auto gi = static_cast<std::size_t>(j.g);
      if (need_seed2.count(j.t)) {  // descriptive statistics at the anchors
        add(j.g, "louvain_ms", 0, j.louvain_ms, j.t);
        add(j.g, "nodes", 0, static_cast<double>(j.nodes), j.t);
        add(j.g, "edges", 0, static_cast<double>(j.edges), j.t);
        add(j.g, "communities", 0, j.communities, j.t);
        add(j.g, "loose_frac", 0, j.loose_frac, j.t);
        add(j.g, "modularity", 0, j.modularity, j.t);
      }
      if (j.spectral) add(kGraphs, "spectral_ms", 0, j.spectral_ms, j.t);
      p0[gi][j.t] = std::move(j.p0);
      if (j.second_seed) p1[gi][j.t] = std::move(j.p1);
      if (j.spectral) sp[j.t] = std::move(j.sp);
    }
    batch.clear();
  };

  // The final clustered bar's graphs, kept for a serial timing pass.
  std::array<Csr, kGraphs> last_W;
  std::vector<bool> last_act;
  const std::size_t last_need = *need.rbegin();

  CorePipeline pipe(N, params);
  const auto t_start = std::chrono::steady_clock::now();
  for (std::size_t t = 1; t <= last_need; ++t) {
    const Frame f = pipe.step(panel, t);
    if (!need.count(t)) continue;
    std::vector<bool> act(N, false);
    for (std::size_t i = 0; i < N; ++i) act[i] = f.active[i] && stock[i];
    const std::array<Csr, kGraphs> W{symmetric_flux_graph(f.P, act), symmetric_flux_graph(f.P_fast, act),
                                     symmetric_flux_graph(bar_flux_csr(pipe.last_bar()), act)};
    if (t == last_need) {
      last_W = W;
      last_act = act;
    }
    for (int g = 0; g < kGraphs; ++g) {
      Job j;
      j.t = t;
      j.g = g;
      j.W = W[static_cast<std::size_t>(g)];
      j.act = act;
      j.second_seed = true;  // every clustered bar: the cross-seed lag agreement needs seed 2 at t + lag
      j.spectral = g == 1 && need_sp.count(t) > 0;
      batch.push_back(std::move(j));
    }
    if (batch.size() >= batch_max) flush();
    if (t % 100 == 0)
      std::fprintf(stderr, "  bar %zu/%zu (%s) %.0f s\n", t, last_need, format_rfc3339(f.t).substr(0, 10).c_str(),
                   ms_since(t_start) / 1000);
  }
  flush();

  // Serial timing on the final bar (one thread, no contention): 3 repetitions each.
  for (int g = 0; g < kGraphs; ++g) {
    for (int rep = 0; rep < 3; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      const CommunityResult r = louvain(last_W[static_cast<std::size_t>(g)], last_act, kResolution, kMinSize);
      add(g, "louvain_ms_serial", 0, ms_since(t0), last_need);
      const auto t1 = std::chrono::steady_clock::now();
      spectral_bisection(last_W[static_cast<std::size_t>(g)], last_act, std::max(1, r.count - (r.loose_id >= 0 ? 1 : 0)),
                         kMinSize);
      add(g, "spectral_ms_serial", 0, ms_since(t1), last_need);
    }
  }

  // Sector labels (ETF/Fund and Unclassified left out).
  std::map<std::string, std::int64_t> sector_id;
  std::vector<std::int64_t> sector(N, -1);
  for (std::size_t i = 0; i < N; ++i) {
    const auto& s = nodes[i].sector;
    if (!stock[i] || s.empty() || s == kSectorUnclassified) continue;
    sector[i] = sector_id.emplace(s, static_cast<std::int64_t>(sector_id.size())).first->second;
  }

  // Agreement of two partitions over the nodes active (labelled) in both; `core` also drops the loose pool.
  auto agree = [&](int g, const Part& a, const Part& b, std::size_t lag, std::size_t t, const std::string& suffix,
                   bool null_and_core) {
    std::vector<std::int64_t> x, y, xc, yc;
    for (std::size_t i = 0; i < N; ++i) {
      if (a[i] == -1 || b[i] == -1) continue;
      x.push_back(a[i]);
      y.push_back(b[i]);
      if (a[i] != -2 && b[i] != -2) {
        xc.push_back(a[i]);
        yc.push_back(b[i]);
      }
    }
    if (x.empty()) return;
    add(g, "nmi" + suffix, lag, partition_nmi(x, y), t);
    add(g, "ari" + suffix, lag, partition_ari(x, y), t);
    if (!null_and_core) return;
    const auto yn = permuted_labels(y, t * 1000003ULL + lag * 101ULL + static_cast<std::uint64_t>(g) + 1);
    add(g, "nmi_null" + suffix, lag, partition_nmi(x, yn), t);
    add(g, "ari_null" + suffix, lag, partition_ari(x, yn), t);
    if (!xc.empty()) {
      add(g, "nmi_core" + suffix, lag, partition_nmi(xc, yc), t);
      add(g, "ari_core" + suffix, lag, partition_ari(xc, yc), t);
    }
  };

  for (std::size_t a : anchors) {
    for (int g = 0; g < kGraphs; ++g) {
      const auto gi = static_cast<std::size_t>(g);
      const Part& pa = p0[gi].at(a);
      for (auto k : opt.lags)
        if (a + k <= last_need) {
          agree(g, pa, p0[gi].at(a + k), k, a, "", true);
          // Seed 1 at t vs seed 2 at t + lag: no shared visiting order, so a merely order-locked optimum does not
          // count as persistence. Compare with the same-bar two-seed agreement (lag 0, "_seed").
          agree(g, pa, p1[gi].at(a + k), k, a, "_xseed", false);
        }
      agree(g, pa, p1[gi].at(a), 0, a, "_seed", false);  // noise ceiling
      std::vector<std::int64_t> x, s;
      for (std::size_t i = 0; i < N; ++i)
        if (pa[i] != -1 && sector[i] >= 0) {
          x.push_back(pa[i]);
          s.push_back(sector[i]);
        }
      if (!x.empty()) {
        add(g, "nmi_sector", 0, partition_nmi(x, s), a);
        add(g, "ari_sector", 0, partition_ari(x, s), a);
        const auto sn = permuted_labels(s, a * 7919ULL + static_cast<std::uint64_t>(g) + 17);
        add(g, "nmi_sector_null", 0, partition_nmi(x, sn), a);
        add(g, "ari_sector_null", 0, partition_ari(x, sn), a);
      }
    }
  }
  for (std::size_t a : sp_anchors) {
    if (!sp.count(a)) continue;
    for (std::size_t k : {std::size_t{1}, std::size_t{5}})
      if (sp.count(a + k)) agree(kGraphs, sp.at(a), sp.at(a + k), k, a, "", true);
    agree(kGraphs, sp.at(a), p0[1].at(a), 0, a, "_vs_louvain", false);
  }

  std::filesystem::create_directories(opt.out_dir);
  const auto csv_path = opt.out_dir / "cluster_persistence.csv";
  const auto year_path = opt.out_dir / "cluster_persistence_by_year.csv";
  std::ofstream csv(csv_path), ycsv(year_path);
  if (!csv || !ycsv) throw std::runtime_error("cluster persistence: cannot write " + opt.out_dir.string());
  csv << "lag,graph,metric,mean,median,q25,q75,n\n";
  ycsv << "year,lag,graph,metric,mean,n\n";
  char buf[256];
  for (const auto& [key, s] : series) {
    const auto& [graph, metric, lag] = key;
    const Stat st = stat(s.v);
    std::snprintf(buf, sizeof buf, "%zu,%s,%s,%.6g,%.6g,%.6g,%.6g,%zu\n", lag, graph.c_str(), metric.c_str(), st.mean,
                  st.median, st.q25, st.q75, st.n);
    csv << buf;
    std::map<std::string, std::vector<double>> by_year;
    for (std::size_t i = 0; i < s.v.size(); ++i) by_year[s.year[i]].push_back(s.v[i]);
    for (const auto& [y, v] : by_year) {
      const Stat ys = stat(v);
      std::snprintf(buf, sizeof buf, "%s,%zu,%s,%s,%.6g,%zu\n", y.c_str(), lag, graph.c_str(), metric.c_str(), ys.mean,
                    ys.n);
      ycsv << buf;
    }
  }
  std::printf("cluster persistence over %zu anchors in %.0f s -> %s, %s\n", anchors.size(), ms_since(t_start) / 1000,
              csv_path.string().c_str(), year_path.string().c_str());
  std::printf("%-14s %-22s %4s %9s %9s %9s %9s %6s\n", "graph", "metric", "lag", "mean", "median", "q25", "q75", "n");
  for (const auto& [key, s] : series) {
    const auto& [graph, metric, lag] = key;
    const Stat st = stat(s.v);
    std::printf("%-14s %-22s %4zu %9.4f %9.4f %9.4f %9.4f %6zu\n", graph.c_str(), metric.c_str(), lag, st.mean,
                st.median, st.q25, st.q75, st.n);
  }
}

}  // namespace mr
