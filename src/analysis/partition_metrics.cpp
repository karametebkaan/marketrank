#include "analysis/partition_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace mr {

namespace {

// Dense ids 0..K-1 for the labels (order of first appearance).
std::vector<std::uint32_t> dense(const std::vector<std::int64_t>& a, std::size_t& K) {
  std::unordered_map<std::int64_t, std::uint32_t> id;
  std::vector<std::uint32_t> out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    auto it = id.find(a[i]);
    if (it == id.end()) it = id.emplace(a[i], static_cast<std::uint32_t>(id.size())).first;
    out[i] = it->second;
  }
  K = id.size();
  return out;
}

struct Contingency {
  double n = 0;
  std::vector<double> rows, cols, cells;  // marginals and the non-zero cell counts
};

Contingency contingency(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b) {
  if (a.size() != b.size()) throw std::invalid_argument("partition agreement: size mismatch");
  std::size_t Ka = 0, Kb = 0;
  const auto da = dense(a, Ka), db = dense(b, Kb);
  Contingency c;
  c.n = static_cast<double>(a.size());
  c.rows.assign(Ka, 0.0);
  c.cols.assign(Kb, 0.0);
  std::vector<std::uint64_t> key(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    c.rows[da[i]] += 1;
    c.cols[db[i]] += 1;
    key[i] = static_cast<std::uint64_t>(da[i]) * Kb + db[i];
  }
  std::sort(key.begin(), key.end());
  for (std::size_t i = 0; i < key.size();) {
    std::size_t j = i;
    while (j < key.size() && key[j] == key[i]) ++j;
    c.cells.push_back(static_cast<double>(j - i));
    i = j;
  }
  return c;
}

double entropy(const std::vector<double>& m, double n) {
  double h = 0;
  for (double x : m)
    if (x > 0) h -= (x / n) * std::log(x / n);
  return h;
}

}  // namespace

double partition_nmi(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b) {
  const Contingency c = contingency(a, b);
  if (c.n == 0) return std::numeric_limits<double>::quiet_NaN();
  const double ha = entropy(c.rows, c.n), hb = entropy(c.cols, c.n);
  if (ha + hb <= 0) return 1.0;  // both a single block
  // I = H(A) + H(B) - H(A,B)
  const double mi = ha + hb - entropy(c.cells, c.n);
  return std::clamp(2.0 * mi / (ha + hb), 0.0, 1.0);
}

double partition_ari(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b) {
  const Contingency c = contingency(a, b);
  if (c.n == 0) return std::numeric_limits<double>::quiet_NaN();
  auto c2 = [](double x) { return x * (x - 1) / 2; };
  double idx = 0, sa = 0, sb = 0;
  for (double x : c.cells) idx += c2(x);
  for (double x : c.rows) sa += c2(x);
  for (double x : c.cols) sb += c2(x);
  const double total = c2(c.n);
  if (total <= 0) return 1.0;
  const double expected = sa * sb / total, maxi = (sa + sb) / 2;
  if (maxi - expected == 0) return 1.0;
  return (idx - expected) / (maxi - expected);
}

std::vector<std::int64_t> permuted_labels(std::vector<std::int64_t> a, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::shuffle(a.begin(), a.end(), rng);
  return a;
}

CommunityResult louvain_seeded(const Csr& W, const std::vector<bool>& active, std::uint64_t seed, double resolution,
                               int min_size, int max_communities) {
  if (seed == 0) return louvain(W, active, resolution, min_size, max_communities);
  const std::size_t n = W.n;
  std::vector<std::uint32_t> perm(n);  // old index -> new index
  std::iota(perm.begin(), perm.end(), 0u);
  std::mt19937_64 rng(seed);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<std::uint32_t> inv(n);
  for (std::size_t i = 0; i < n; ++i) inv[perm[i]] = static_cast<std::uint32_t>(i);
  Csr V;
  V.n = n;
  V.row_ptr.assign(n + 1, 0);
  std::vector<bool> act(n);
  std::vector<std::pair<std::uint32_t, double>> row;
  for (std::size_t q = 0; q < n; ++q) {
    const std::size_t i = inv[q];
    act[q] = active[i];
    row.clear();
    for (std::size_t e = W.row_ptr[i]; e < W.row_ptr[i + 1]; ++e) row.push_back({perm[W.col[e]], W.val[e]});
    std::sort(row.begin(), row.end());
    for (auto& [j, w] : row) {
      V.col.push_back(j);
      V.val.push_back(w);
    }
    V.row_ptr[q + 1] = V.col.size();
  }
  V.raw = V.val;
  CommunityResult r = louvain(V, act, resolution, min_size, max_communities);
  CommunityResult out = r;
  for (std::size_t i = 0; i < n; ++i) out.id[i] = r.id[perm[i]];
  return out;
}

std::vector<std::int32_t> spectral_bisection(const Csr& W, const std::vector<bool>& active, int k, int min_size) {
  if (active.size() != W.n) throw std::invalid_argument("spectral_bisection: active size != W.n");
  const std::size_t n = W.n;
  std::vector<std::int32_t> out(n, -1);
  // Components of the active subgraph (positive finite weights).
  std::vector<std::int32_t> comp(n, -1);
  std::vector<std::vector<std::uint32_t>> blocks, pool(1);
  for (std::size_t s = 0; s < n; ++s) {
    if (!active[s] || comp[s] >= 0) continue;
    std::vector<std::uint32_t> members{static_cast<std::uint32_t>(s)}, stack{static_cast<std::uint32_t>(s)};
    comp[s] = 1;
    while (!stack.empty()) {
      const std::uint32_t a = stack.back();
      stack.pop_back();
      for (std::size_t e = W.row_ptr[a]; e < W.row_ptr[a + 1]; ++e) {
        const std::uint32_t b = W.col[e];
        if (!active[b] || comp[b] >= 0 || !(W.val[e] > 0) || !std::isfinite(W.val[e])) continue;
        comp[b] = 1;
        members.push_back(b);
        stack.push_back(b);
      }
    }
    std::sort(members.begin(), members.end());
    if (static_cast<int>(members.size()) < min_size)
      pool[0].insert(pool[0].end(), members.begin(), members.end());
    else
      blocks.push_back(std::move(members));
  }
  std::vector<std::int32_t> local(n, -1);  // node -> index within the block being split
  auto split = [&](const std::vector<std::uint32_t>& S) {
    const std::size_t m = S.size();
    for (std::size_t a = 0; a < m; ++a) local[S[a]] = static_cast<std::int32_t>(a);
    std::vector<std::uint32_t> rp(m + 1, 0), cj;
    std::vector<double> wv, d(m, 0.0);
    for (std::size_t a = 0; a < m; ++a) {
      const std::uint32_t i = S[a];
      for (std::size_t e = W.row_ptr[i]; e < W.row_ptr[i + 1]; ++e) {
        const std::int32_t b = local[W.col[e]];
        const double w = W.val[e];
        if (b < 0 || static_cast<std::size_t>(b) == a || !(w > 0) || !std::isfinite(w)) continue;
        cj.push_back(static_cast<std::uint32_t>(b));
        wv.push_back(w);
        d[a] += w;
      }
      rp[a + 1] = static_cast<std::uint32_t>(cj.size());
    }
    for (auto i : S) local[i] = -1;
    double dsum = 0;
    for (double x : d) dsum += x;
    std::vector<double> v0(m, 0.0), isd(m, 0.0);
    for (std::size_t a = 0; a < m; ++a) {
      v0[a] = dsum > 0 ? std::sqrt(d[a] / dsum) : 0.0;
      isd[a] = d[a] > 0 ? 1.0 / std::sqrt(d[a]) : 0.0;
    }
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
        double s = 0;
        for (std::uint32_t e = rp[a]; e < rp[a + 1]; ++e) s += wv[e] * isd[cj[e]] * x[cj[e]];
        y[a] = x[a] + isd[a] * s;
      }
      if (deflate_normalize(y) == 0) break;
      x.swap(y);
    }
    std::vector<std::size_t> idx(m);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return x[a] < x[b] || (x[a] == x[b] && a < b); });
    std::vector<std::uint32_t> lo, hi;
    for (std::size_t r = 0; r < m; ++r) (r < m / 2 ? lo : hi).push_back(S[idx[r]]);
    std::sort(lo.begin(), lo.end());
    std::sort(hi.begin(), hi.end());
    return std::make_pair(std::move(lo), std::move(hi));
  };
  while (static_cast<int>(blocks.size()) < k) {
    std::size_t big = 0;
    for (std::size_t b = 1; b < blocks.size(); ++b)
      if (blocks[b].size() > blocks[big].size()) big = b;
    if (blocks.empty() || static_cast<int>(blocks[big].size()) < 2 * min_size) break;
    auto [lo, hi] = split(blocks[big]);
    blocks[big] = std::move(lo);
    blocks.push_back(std::move(hi));
  }
  for (std::size_t b = 0; b < blocks.size(); ++b)
    for (auto i : blocks[b]) out[i] = static_cast<std::int32_t>(b);
  for (auto i : pool[0]) out[i] = -2;
  return out;
}

Csr bar_flux_csr(const BarFlux& bar) {
  Csr P;
  P.n = bar.rows.size();
  P.row_ptr.assign(P.n + 1, 0);
  for (std::size_t i = 0; i < P.n; ++i) {
    for (const auto& e : bar.rows[i]) {
      P.col.push_back(e.j);
      P.val.push_back(e.w);
    }
    P.row_ptr[i + 1] = P.col.size();
  }
  P.raw = P.val;
  return P;
}

}  // namespace mr
