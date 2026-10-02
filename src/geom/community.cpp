#include "geom/community.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace fx {

namespace {

using Nbr = std::vector<std::pair<std::uint32_t, double>>;

// One Louvain level: adjacency without self-loops (sorted by id), plus the diagonal weight A_ii.
struct Level {
  std::vector<Nbr> nbr;
  std::vector<double> self;
};

// Phase 1. comm[i] starts as i. Returns whether any node moved.
bool move_nodes(const Level& g, double m2, double res, std::vector<std::uint32_t>& comm) {
  const std::size_t n = g.nbr.size();
  comm.resize(n);
  std::vector<double> k(n), tot(n), w_to(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    comm[i] = static_cast<std::uint32_t>(i);
    double s = g.self[i];
    for (auto& e : g.nbr[i]) s += e.second;
    k[i] = tot[i] = s;
  }
  if (m2 <= 0) return false;
  bool any = false;
  std::vector<std::uint32_t> touched;
  for (int pass = 0; pass < 50; ++pass) {
    bool moved = false;
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t ci = comm[i];
      touched.clear();
      for (auto& e : g.nbr[i]) {
        const std::uint32_t c = comm[e.first];
        if (w_to[c] == 0.0) touched.push_back(c);
        w_to[c] += e.second;
      }
      tot[ci] -= k[i];
      auto gain = [&](std::uint32_t c) { return w_to[c] - res * tot[c] * k[i] / m2; };
      const double gain_own = gain(ci);
      std::sort(touched.begin(), touched.end());
      touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
      std::uint32_t best = ci;
      double best_gain = gain_own;
      for (auto c : touched) {
        const double gc = gain(c);
        if (gc > best_gain + 1e-12) {  // strict: ties keep the lowest community id (candidates ascend)
          best = c;
          best_gain = gc;
        }
      }
      if (best != ci && best_gain > gain_own + 1e-12) {
        comm[i] = best;
        moved = true;
      } else {
        best = ci;
      }
      tot[best] += k[i];
      for (auto c : touched) w_to[c] = 0.0;
    }
    if (!moved) break;
    any = true;
  }
  return any;
}

}  // namespace

Csr symmetric_flux_graph(const Csr& P, const std::vector<bool>& active) {
  if (active.size() != P.n) throw std::invalid_argument("symmetric_flux_graph: active size != P.n");
  std::vector<std::vector<std::pair<std::uint32_t, double>>> rows(P.n);
  for (std::size_t i = 0; i < P.n; ++i) {
    if (!active[i]) continue;
    for (std::size_t e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) {
      const std::uint32_t j = P.col[e];
      if (j == i || !active[j] || e >= P.raw.size()) continue;
      const double w = P.raw[e];
      if (!(w > 0) || !std::isfinite(w)) continue;
      rows[i].push_back({j, w});
      rows[j].push_back({static_cast<std::uint32_t>(i), w});
    }
  }
  Csr W;
  W.n = P.n;
  W.row_ptr.assign(P.n + 1, 0);
  for (std::size_t i = 0; i < P.n; ++i) {
    auto& r = rows[i];
    std::sort(r.begin(), r.end());  // by column, then weight: a canonical summation order
    std::size_t out = 0;
    for (std::size_t a = 0; a < r.size(); ++a) {
      if (out && r[out - 1].first == r[a].first)
        r[out - 1].second += r[a].second;
      else
        r[out++] = r[a];
    }
    r.resize(out);
    for (auto& e : r) {
      W.col.push_back(e.first);
      W.val.push_back(e.second);
    }
    W.row_ptr[i + 1] = W.col.size();
  }
  W.raw = W.val;
  return W;
}

CommunityResult louvain(const Csr& W, const std::vector<bool>& active, double resolution, int min_size,
                        int max_communities) {
  if (active.size() != W.n) throw std::invalid_argument("louvain: active size != W.n");
  CommunityResult out;
  out.id.assign(W.n, -1);
  std::vector<std::uint32_t> compact(W.n, 0), members;  // node -> compact index; compact -> node
  for (std::size_t i = 0; i < W.n; ++i)
    if (active[i]) {
      compact[i] = static_cast<std::uint32_t>(members.size());
      members.push_back(static_cast<std::uint32_t>(i));
    }
  const std::size_t na = members.size();
  if (na == 0) return out;

  // Level 0: canonical (column-sorted, duplicate-merged) adjacency over active nodes.
  Level g;
  g.nbr.resize(na);
  g.self.assign(na, 0.0);
  double m2 = 0;
  for (std::size_t a = 0; a < na; ++a) {
    const std::size_t i = members[a];
    Nbr& r = g.nbr[a];
    for (std::size_t e = W.row_ptr[i]; e < W.row_ptr[i + 1]; ++e) {
      const std::uint32_t j = W.col[e];
      const double w = W.val[e];
      if (!active[j] || !(w > 0) || !std::isfinite(w)) continue;
      if (j == i)
        g.self[a] += w;
      else
        r.push_back({compact[j], w});
    }
    std::sort(r.begin(), r.end());
    std::size_t o = 0;
    for (std::size_t x = 0; x < r.size(); ++x) {
      if (o && r[o - 1].first == r[x].first)
        r[o - 1].second += r[x].second;
      else
        r[o++] = r[x];
    }
    r.resize(o);
    m2 += g.self[a];
    for (auto& e : r) m2 += e.second;
  }
  const Level g0 = g;

  // Phases 1 and 2.
  std::vector<std::uint32_t> assign(na);  // original compact node -> current level node
  for (std::size_t a = 0; a < na; ++a) assign[a] = static_cast<std::uint32_t>(a);
  for (;;) {
    std::vector<std::uint32_t> comm;
    if (!move_nodes(g, m2, resolution, comm)) break;
    std::vector<std::int64_t> renum(g.nbr.size(), -1);  // renumber by first appearance (smallest member)
    std::uint32_t K = 0;
    for (std::size_t i = 0; i < comm.size(); ++i)
      if (renum[comm[i]] < 0) renum[comm[i]] = K++;
    Level h;
    h.nbr.resize(K);
    h.self.assign(K, 0.0);
    for (std::size_t i = 0; i < g.nbr.size(); ++i) {
      const auto ci = static_cast<std::uint32_t>(renum[comm[i]]);
      h.self[ci] += g.self[i];
      for (auto& e : g.nbr[i]) {
        const auto cj = static_cast<std::uint32_t>(renum[comm[e.first]]);
        if (cj == ci)
          h.self[ci] += e.second;
        else
          h.nbr[ci].push_back({cj, e.second});
      }
    }
    for (auto& r : h.nbr) {
      std::sort(r.begin(), r.end());
      std::size_t o = 0;
      for (std::size_t x = 0; x < r.size(); ++x) {
        if (o && r[o - 1].first == r[x].first)
          r[o - 1].second += r[x].second;
        else
          r[o++] = r[x];
      }
      r.resize(o);
    }
    for (std::size_t a = 0; a < na; ++a) assign[a] = static_cast<std::uint32_t>(renum[comm[assign[a]]]);
    if (K == g.nbr.size()) break;
    g = std::move(h);
  }

  // Community-level merging of small communities and the cap.
  std::uint32_t K0 = 0;
  for (auto c : assign) K0 = std::max<std::uint32_t>(K0, c + 1);
  std::vector<std::map<std::uint32_t, double>> cadj(K0);
  std::vector<std::size_t> size(K0, 0);
  for (std::size_t a = 0; a < na; ++a) {
    ++size[assign[a]];
    for (auto& e : g0.nbr[a])
      if (assign[e.first] != assign[a]) cadj[assign[a]][assign[e.first]] += e.second;
  }
  std::vector<std::uint32_t> parent(K0);
  for (std::uint32_t c = 0; c < K0; ++c) parent[c] = c;
  std::vector<char> alive(K0, 1), loose(K0, 0);
  std::set<std::pair<std::size_t, std::uint32_t>> by_size;
  for (std::uint32_t c = 0; c < K0; ++c) by_size.insert({size[c], c});
  std::size_t alive_count = K0;
  while (!by_size.empty()) {
    const auto [sz, c] = *by_size.begin();
    if (static_cast<int>(sz) >= min_size && static_cast<int>(alive_count) <= max_communities) break;
    by_size.erase(by_size.begin());
    std::uint32_t best = c;
    double bw = 0;
    for (auto& e : cadj[c])
      if (e.second > bw) {
        bw = e.second;
        best = e.first;
      }
    if (best == c) {  // no neighbour: pool into loose
      loose[c] = 1;
      for (auto& e : cadj[c]) cadj[e.first].erase(c);
      cadj[c].clear();
      --alive_count;
      continue;
    }
    by_size.erase({size[best], best});
    for (auto& e : cadj[c]) {
      cadj[e.first].erase(c);
      if (e.first == best) continue;
      cadj[e.first][best] += e.second;
      cadj[best][e.first] += e.second;
    }
    cadj[best].erase(c);
    cadj[c].clear();
    size[best] += size[c];
    parent[c] = best;
    alive[c] = 0;
    --alive_count;
    by_size.insert({size[best], best});
  }
  auto find = [&](std::uint32_t c) {
    while (parent[c] != c) c = parent[c];
    return c;
  };
  // Final ids: surviving communities by smallest member, the loose pool last.
  std::map<std::uint32_t, std::int32_t> final_id;
  std::int32_t K = 0;
  bool any_loose = false;
  for (std::size_t a = 0; a < na; ++a) {
    const std::uint32_t r = find(assign[a]);
    if (loose[r]) {
      any_loose = true;
      continue;
    }
    if (!final_id.count(r)) final_id[r] = K++;
  }
  out.count = K + (any_loose ? 1 : 0);
  out.loose_id = any_loose ? K : -1;
  for (std::size_t a = 0; a < na; ++a) {
    const std::uint32_t r = find(assign[a]);
    out.id[members[a]] = loose[r] ? K : final_id[r];
  }
  // Modularity of the final partition on the input graph.
  std::vector<double> in(static_cast<std::size_t>(out.count), 0.0), tot(static_cast<std::size_t>(out.count), 0.0);
  for (std::size_t a = 0; a < na; ++a) {
    const auto ca = static_cast<std::size_t>(out.id[members[a]]);
    double k = g0.self[a];
    in[ca] += g0.self[a];
    for (auto& e : g0.nbr[a]) {
      k += e.second;
      if (out.id[members[e.first]] == out.id[members[a]]) in[ca] += e.second;
    }
    tot[ca] += k;
  }
  double Q = 0;
  if (m2 > 0)
    for (std::size_t c = 0; c < in.size(); ++c) Q += in[c] / m2 - resolution * (tot[c] / m2) * (tot[c] / m2);
  out.modularity = Q;
  return out;
}

std::vector<double> community_graph(const Csr& W, const CommunityResult& r) {
  const auto K = static_cast<std::size_t>(r.count);
  std::vector<double> cw(K * K, 0.0);
  for (std::size_t i = 0; i < W.n; ++i) {
    if (r.id[i] < 0) continue;
    for (std::size_t e = W.row_ptr[i]; e < W.row_ptr[i + 1]; ++e) {
      const std::int32_t cj = r.id[W.col[e]];
      if (cj < 0 || cj == r.id[i]) continue;
      cw[static_cast<std::size_t>(r.id[i]) * K + static_cast<std::size_t>(cj)] += W.val[e];
    }
  }
  return cw;
}

namespace {

void order_set(const std::vector<double>& cw, std::size_t K, const std::vector<std::size_t>& cnt,
               const std::vector<int>& S, std::vector<int>& out) {
  const std::size_t m = S.size();
  if (m == 1) {
    out.push_back(S[0]);
    return;
  }
  auto w = [&](std::size_t a, std::size_t b) {
    return cw[static_cast<std::size_t>(S[a]) * K + static_cast<std::size_t>(S[b])] +
           cw[static_cast<std::size_t>(S[b]) * K + static_cast<std::size_t>(S[a])];
  };
  // connected components of the induced subgraph
  std::vector<int> comp(m, -1);
  int nc = 0;
  for (std::size_t s = 0; s < m; ++s) {
    if (comp[s] >= 0) continue;
    std::vector<std::size_t> stack{s};
    comp[s] = nc;
    while (!stack.empty()) {
      const std::size_t a = stack.back();
      stack.pop_back();
      for (std::size_t b = 0; b < m; ++b)
        if (comp[b] < 0 && a != b && w(a, b) > 0) {
          comp[b] = nc;
          stack.push_back(b);
        }
    }
    ++nc;
  }
  if (nc > 1) {
    std::vector<std::vector<int>> parts(static_cast<std::size_t>(nc));
    for (std::size_t a = 0; a < m; ++a) parts[static_cast<std::size_t>(comp[a])].push_back(S[a]);
    std::vector<std::size_t> total(parts.size(), 0);
    for (std::size_t p = 0; p < parts.size(); ++p)
      for (int c : parts[p]) total[p] += cnt[static_cast<std::size_t>(c)];
    std::vector<std::size_t> idx(parts.size());
    for (std::size_t p = 0; p < idx.size(); ++p) idx[p] = p;
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
      return total[a] > total[b] || (total[a] == total[b] && parts[a].front() < parts[b].front());
    });
    for (auto p : idx) order_set(cw, K, cnt, parts[p], out);
    return;
  }
  // Fiedler vector by power iteration on B = I + D^-1/2 W D^-1/2 with the trivial vector deflated.
  std::vector<double> d(m, 0.0), v0(m);
  std::vector<double> A(m * m, 0.0);
  for (std::size_t a = 0; a < m; ++a)
    for (std::size_t b = 0; b < m; ++b)
      if (a != b) {
        A[a * m + b] = w(a, b);
        d[a] += A[a * m + b];
      }
  double n0 = 0;
  for (std::size_t a = 0; a < m; ++a) n0 += d[a];
  for (std::size_t a = 0; a < m; ++a) v0[a] = std::sqrt(d[a] / n0);
  for (std::size_t a = 0; a < m; ++a)
    for (std::size_t b = 0; b < m; ++b) A[a * m + b] = A[a * m + b] / std::sqrt(d[a] * d[b]);
  auto deflate_normalize = [&](std::vector<double>& x) {
    double dot = 0, nn = 0;
    for (std::size_t a = 0; a < m; ++a) dot += x[a] * v0[a];
    for (std::size_t a = 0; a < m; ++a) x[a] -= dot * v0[a];
    for (double e : x) nn += e * e;
    nn = std::sqrt(nn);
    if (nn > 0)
      for (double& e : x) e /= nn;
    return nn;
  };
  std::vector<double> x(m), y(m);
  for (std::size_t a = 0; a < m; ++a) x[a] = std::sin(static_cast<double>(a) + 1.0);
  deflate_normalize(x);
  for (int it = 0; it < 500; ++it) {
    for (std::size_t a = 0; a < m; ++a) {
      double s = x[a];
      for (std::size_t b = 0; b < m; ++b) s += A[a * m + b] * x[b];
      y[a] = s;
    }
    if (deflate_normalize(y) == 0) break;
    x = y;
  }
  std::size_t big = 0;
  for (std::size_t a = 1; a < m; ++a)
    if (std::fabs(x[a]) > std::fabs(x[big]) + 1e-15) big = a;
  if (x[big] < 0)
    for (double& e : x) e = -e;
  std::vector<std::size_t> idx(m);
  for (std::size_t a = 0; a < m; ++a) idx[a] = a;
  std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
    return x[a] < x[b] || (x[a] == x[b] && S[a] < S[b]);
  });
  double total = 0;
  for (int c : S) total += static_cast<double>(cnt[static_cast<std::size_t>(c)]);
  std::size_t split = 1;
  double cum = 0, best = 1e300;
  for (std::size_t k = 1; k < m; ++k) {
    cum += static_cast<double>(cnt[static_cast<std::size_t>(S[idx[k - 1]])]);
    const double dist = std::fabs(cum - total / 2);
    if (dist < best - 1e-12) {
      best = dist;
      split = k;
    }
  }
  std::vector<int> left, right;
  for (std::size_t k = 0; k < m; ++k) (k < split ? left : right).push_back(S[idx[k]]);
  std::sort(left.begin(), left.end());
  std::sort(right.begin(), right.end());
  std::vector<int> lo, ro;
  order_set(cw, K, cnt, left, lo);
  order_set(cw, K, cnt, right, ro);
  // Orient each half so that its end nearer the other half is the one more strongly connected to it.
  auto tie = [&](int c, const std::vector<int>& other) {
    double t = 0;
    for (int o : other)
      t += cw[static_cast<std::size_t>(c) * K + static_cast<std::size_t>(o)] +
           cw[static_cast<std::size_t>(o) * K + static_cast<std::size_t>(c)];
    return t;
  };
  if (lo.size() > 1 && tie(lo.front(), right) > tie(lo.back(), right)) std::reverse(lo.begin(), lo.end());
  if (ro.size() > 1 && tie(ro.back(), left) > tie(ro.front(), left)) std::reverse(ro.begin(), ro.end());
  out.insert(out.end(), lo.begin(), lo.end());
  out.insert(out.end(), ro.begin(), ro.end());
}

}  // namespace

std::vector<int> spectral_order(const std::vector<double>& cw, const std::vector<std::size_t>& count, int loose_id) {
  const std::size_t K = count.size();
  if (cw.size() != K * K) throw std::invalid_argument("spectral_order: matrix size mismatch");
  std::vector<int> S, out;
  for (std::size_t c = 0; c < K; ++c)
    if (static_cast<int>(c) != loose_id) S.push_back(static_cast<int>(c));
  if (!S.empty()) order_set(cw, K, count, S, out);
  if (loose_id >= 0) out.push_back(loose_id);
  return out;
}

std::vector<std::int64_t> match_labels(const std::vector<std::int32_t>& new_id, int count, int loose_id,
                                       const std::vector<std::int64_t>& old_label, std::int64_t& next_label,
                                       double min_jaccard) {
  const auto K = static_cast<std::size_t>(count);
  std::vector<std::int64_t> label(K, -1);
  std::vector<std::size_t> cnt(K, 0);
  std::map<std::int64_t, std::size_t> old_size;
  std::map<std::pair<int, std::int64_t>, std::size_t> overlap;
  for (std::size_t i = 0; i < new_id.size(); ++i) {
    const bool has_new = new_id[i] >= 0 && new_id[i] != loose_id;
    if (has_new) ++cnt[static_cast<std::size_t>(new_id[i])];
    if (i < old_label.size() && old_label[i] >= 0) {
      ++old_size[old_label[i]];
      if (has_new) ++overlap[{new_id[i], old_label[i]}];
    }
  }
  std::vector<std::tuple<std::size_t, int, std::int64_t>> pairs;
  for (auto& kv : overlap) pairs.emplace_back(kv.second, kv.first.first, kv.first.second);
  std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
    if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) > std::get<0>(b);
    if (std::get<1>(a) != std::get<1>(b)) return std::get<1>(a) < std::get<1>(b);
    return std::get<2>(a) < std::get<2>(b);
  });
  std::set<std::int64_t> used_old;
  std::vector<bool> matched(K, false);
  for (auto& [ov, nw, old] : pairs) {
    if (matched[static_cast<std::size_t>(nw)] || used_old.count(old)) continue;
    const double jac = static_cast<double>(ov) / static_cast<double>(cnt[static_cast<std::size_t>(nw)] + old_size[old] - ov);
    if (jac < min_jaccard) continue;
    matched[static_cast<std::size_t>(nw)] = true;
    used_old.insert(old);
    label[static_cast<std::size_t>(nw)] = old;
  }
  for (std::size_t c = 0; c < K; ++c)
    if (static_cast<int>(c) != loose_id && !matched[c]) label[c] = next_label++;
  return label;
}

std::vector<std::int64_t> arrange_order(const std::vector<int>& spectral, int loose_id,
                                        const std::vector<std::int64_t>& new_label,
                                        const std::vector<std::int64_t>& old_order) {
  std::map<std::int64_t, std::size_t> old_pos;
  for (std::size_t k = 0; k < old_order.size(); ++k) old_pos[old_order[k]] = k;
  std::vector<std::int64_t> ord;
  for (int c : spectral)
    if (c != loose_id) ord.push_back(new_label[static_cast<std::size_t>(c)]);
  std::vector<std::size_t> slots;
  std::vector<std::int64_t> kept;
  for (std::size_t k = 0; k < ord.size(); ++k)
    if (old_pos.count(ord[k])) {
      slots.push_back(k);
      kept.push_back(ord[k]);
    }
  std::stable_sort(kept.begin(), kept.end(), [&](auto a, auto b) { return old_pos[a] < old_pos[b]; });
  for (std::size_t k = 0; k < slots.size(); ++k) ord[slots[k]] = kept[k];
  return ord;
}

CommunityTracker::CommunityTracker(std::size_t n, int recluster_bars, int min_size)
    : n_(n), bars_(std::max(1, recluster_bars)), min_size_(min_size), label_(n, -1), group_(n, 0), node_group_(n, -1) {}

const std::vector<std::uint32_t>& CommunityTracker::update(const Csr& P, const std::vector<bool>& active) {
  if (active.size() != n_ || P.n != n_) throw std::invalid_argument("CommunityTracker: size mismatch");
  for (std::size_t i = 0; i < n_; ++i)
    if (!active[i]) label_[i] = -1;
  reclustered_ = !have_ || since_ >= bars_;
  if (reclustered_) {
    const auto t0 = std::chrono::steady_clock::now();
    Csr W = symmetric_flux_graph(P, active);
    CommunityResult r = louvain(W, active, 1.0, min_size_);
    const auto K = static_cast<std::size_t>(r.count);
    std::vector<std::size_t> cnt(K, 0);
    for (std::size_t i = 0; i < n_; ++i)
      if (r.id[i] >= 0) ++cnt[static_cast<std::size_t>(r.id[i])];
    const std::vector<int> spectral = spectral_order(community_graph(W, r), cnt, r.loose_id);
    std::vector<std::int64_t> new_label = match_labels(r.id, r.count, r.loose_id, label_, next_label_);
    order_ = arrange_order(spectral, r.loose_id, new_label, order_);
    for (std::size_t i = 0; i < n_; ++i) {
      if (!active[i])
        label_[i] = -1;
      else if (r.id[i] == r.loose_id)
        label_[i] = -2;
      else
        label_[i] = new_label[static_cast<std::size_t>(r.id[i])];
    }
    modularity_ = r.modularity;
    cluster_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    have_ = true;
    since_ = 1;
  } else {
    ++since_;
    bool fresh = false;
    for (std::size_t i = 0; i < n_; ++i) fresh = fresh || (active[i] && label_[i] == -1);
    if (fresh) {
      Csr W = symmetric_flux_graph(P, active);
      for (std::size_t i = 0; i < n_; ++i) {
        if (!active[i] || label_[i] != -1) continue;
        std::int64_t best = -2;
        double bw = 0;
        std::map<std::int64_t, double> acc;
        for (std::size_t e = W.row_ptr[i]; e < W.row_ptr[i + 1]; ++e) {
          const std::int64_t l = label_[W.col[e]];
          if (l >= 0) acc[l] += W.val[e];
        }
        for (auto& kv : acc)
          if (kv.second > bw) {
            bw = kv.second;
            best = kv.first;
          }
        label_[i] = best;
      }
    }
  }
  std::int64_t max_label = -1;
  for (auto l : order_) max_label = std::max(max_label, l);
  std::vector<std::uint32_t> pos(static_cast<std::size_t>(max_label + 1), 0);
  for (std::size_t k = 0; k < order_.size(); ++k) pos[static_cast<std::size_t>(order_[k])] = static_cast<std::uint32_t>(k);
  std::set<std::int64_t> present;
  loose_nodes_ = 0;
  for (std::size_t i = 0; i < n_; ++i) {
    if (!active[i]) {
      group_[i] = 0;
      node_group_[i] = -1;
    } else if (label_[i] == -2) {
      group_[i] = static_cast<std::uint32_t>(order_.size());
      node_group_[i] = -1;
      ++loose_nodes_;
    } else {
      group_[i] = pos[static_cast<std::size_t>(label_[i])];
      node_group_[i] = static_cast<std::int32_t>(label_[i]);
      present.insert(label_[i]);
    }
  }
  communities_ = static_cast<int>(present.size());
  return group_;
}

}  // namespace fx
