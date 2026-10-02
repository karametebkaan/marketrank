#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "geom/community.hpp"
#include "geom/territory.hpp"

using namespace fx;

namespace {
using Edges = std::vector<std::tuple<std::uint32_t, std::uint32_t, double>>;

// Symmetric weighted graph from undirected edges; `shuffle_seed != 0` shuffles the entries within each row.
Csr make_w(std::size_t n, const Edges& edges, std::uint64_t shuffle_seed = 0) {
  std::vector<std::vector<std::pair<std::uint32_t, double>>> rows(n);
  for (auto& [a, b, w] : edges) {
    rows[a].push_back({b, w});
    rows[b].push_back({a, w});
  }
  std::mt19937_64 rng(shuffle_seed);
  Csr W;
  W.n = n;
  W.row_ptr.push_back(0);
  for (auto& r : rows) {
    if (shuffle_seed) std::shuffle(r.begin(), r.end(), rng);
    for (auto& e : r) {
      W.col.push_back(e.first);
      W.val.push_back(e.second);
    }
    W.row_ptr.push_back(W.col.size());
  }
  W.raw = W.val;
  return W;
}

Edges planted(std::size_t groups, std::size_t size, double inside, double between) {
  Edges e;
  for (std::size_t g = 0; g < groups; ++g)
    for (std::size_t a = 0; a < size; ++a)
      for (std::size_t b = a + 1; b < size; ++b)
        e.emplace_back(g * size + a, g * size + b, inside);
  for (std::size_t g = 0; g + 1 < groups; ++g) e.emplace_back(g * size, (g + 1) * size + 1, between);
  e.emplace_back(0, (groups - 1) * size + 2, between);
  return e;
}
}  // namespace

TEST_CASE("louvain recovers a planted partition") {
  const std::size_t n = 40;
  Csr W = make_w(n, planted(4, 10, 10.0, 0.1));
  std::vector<bool> active(n, true);
  auto r = louvain(W, active);
  CHECK(r.count == 4);
  CHECK(r.loose_id == -1);
  CHECK(r.modularity > 0.6);
  for (std::size_t g = 0; g < 4; ++g)
    for (std::size_t k = 0; k < 10; ++k) CHECK(r.id[g * 10 + k] == r.id[g * 10]);
  std::set<int> ids(r.id.begin(), r.id.end());
  CHECK(ids.size() == 4);
  for (std::size_t g = 0; g < 4; ++g) CHECK(r.id[g * 10] == static_cast<int>(g));  // ordered by smallest member
}

TEST_CASE("louvain is deterministic and independent of edge order within rows") {
  std::mt19937_64 rng(9);
  const std::size_t n = 300;
  Edges e;
  for (std::size_t k = 0; k < 1500; ++k) {
    auto a = static_cast<std::uint32_t>(rng() % n), b = static_cast<std::uint32_t>(rng() % n);
    if (a != b) e.emplace_back(a, b, 0.5 + static_cast<double>(rng() % 100) / 10.0);
  }
  std::vector<bool> active(n, true);
  auto a = louvain(make_w(n, e), active);
  auto b = louvain(make_w(n, e), active);
  auto c = louvain(make_w(n, e, 77), active);
  CHECK(a.id == b.id);
  CHECK(a.id == c.id);
  CHECK(a.modularity == b.modularity);
}

TEST_CASE("small communities merge into their strongest neighbour; isolated nodes become loose and last") {
  Edges e = planted(1, 20, 5.0, 0.0);
  e.pop_back();  // planted() adds one extra link; drop it (single group)
  for (std::uint32_t a = 20; a < 23; ++a)
    for (std::uint32_t b = a + 1; b < 23; ++b) e.emplace_back(a, b, 5.0);
  e.emplace_back(0, 20, 0.5);  // the 3-clique hangs off the big community
  Csr W = make_w(26, e);        // nodes 23..25 are isolated
  std::vector<bool> active(26, true);
  auto r = louvain(W, active);
  CHECK(r.loose_id >= 0);
  CHECK(r.loose_id == r.count - 1);
  for (int i = 20; i < 23; ++i) CHECK(r.id[static_cast<std::size_t>(i)] == r.id[0]);
  for (int i = 23; i < 26; ++i) CHECK(r.id[static_cast<std::size_t>(i)] == r.loose_id);
  CHECK(r.count == 2);
  std::vector<bool> some(26, true);
  some[24] = false;
  auto s = louvain(W, some);
  CHECK(s.id[24] == -1);
}

TEST_CASE("louvain caps the number of communities") {
  const std::size_t groups = 40, size = 9;
  Csr W = make_w(groups * size, planted(groups, size, 10.0, 0.5));
  std::vector<bool> active(groups * size, true);
  auto r = louvain(W, active, 1.0, 8, 10);
  CHECK(r.count <= 10);
  CHECK(r.loose_id == -1);
}

TEST_CASE("spectral order respects a chain of communities") {
  const std::size_t K = 4;
  for (int perm = 0; perm < 2; ++perm) {
    // communities 0..3 are the chain A-B-C-D in label order, or relabelled B-D-A-C
    std::vector<int> lab = perm ? std::vector<int>{1, 3, 0, 2} : std::vector<int>{0, 1, 2, 3};
    std::vector<double> cw(K * K, 0.0);
    for (std::size_t k = 0; k + 1 < K; ++k) {
      cw[static_cast<std::size_t>(lab[k]) * K + static_cast<std::size_t>(lab[k + 1])] = 3.0;
      cw[static_cast<std::size_t>(lab[k + 1]) * K + static_cast<std::size_t>(lab[k])] = 3.0;
    }
    auto o = spectral_order(cw, std::vector<std::size_t>(K, 10), -1);
    REQUIRE(o.size() == K);
    std::vector<int> expect = lab, rev(lab.rbegin(), lab.rend());
    CHECK((o == expect || o == rev));
  }
  // loose community goes last, disconnected sets fall back to components
  std::vector<double> cw(9, 0.0);
  cw[0 * 3 + 1] = cw[1 * 3 + 0] = 1.0;
  auto o = spectral_order(cw, {5, 5, 2}, 0);
  CHECK(o.back() == 0);
  CHECK(o.size() == 3);
}

namespace {
Csr chain_p(const std::vector<std::vector<std::pair<std::uint32_t, double>>>& rows) {
  Csr P;
  P.n = rows.size();
  P.row_ptr.push_back(0);
  for (auto& r : rows) {
    for (auto& e : r) {
      P.col.push_back(e.first);
      P.val.push_back(e.second);
      P.raw.push_back(e.second);
    }
    P.row_ptr.push_back(P.col.size());
  }
  return P;
}
}  // namespace

TEST_CASE("symmetric flux graph ignores self loops, non-positive and non-finite raw, and inactive nodes") {
  Csr P = chain_p({{{0, 5.0}, {1, 2.0}, {2, -1.0}}, {{0, 1.0}, {3, 4.0}}, {{1, 9.0}}, {{1, std::nan("")}}});
  std::vector<bool> active = {true, true, true, false};
  Csr W = symmetric_flux_graph(P, active);
  double w01 = 0;
  for (auto e = W.row_ptr[0]; e < W.row_ptr[1]; ++e) {
    CHECK(W.col[e] != 0);
    if (W.col[e] == 1) w01 += W.val[e];
  }
  CHECK(w01 == doctest::Approx(3.0));  // 2.0 + 1.0
  CHECK(W.row_ptr[4] - W.row_ptr[3] == 0);
  for (auto e = W.row_ptr[2]; e < W.row_ptr[3]; ++e) CHECK(W.col[e] == 1);
}

TEST_CASE("louvain clusters 10,000 nodes of degree ~30 quickly") {
  std::mt19937_64 rng(1);
  const std::size_t n = 10000;
  Edges e;
  for (std::size_t i = 0; i < n; ++i)
    for (int k = 0; k < 15; ++k) {
      auto j = static_cast<std::uint32_t>(rng() % n);
      if (j != i) e.emplace_back(static_cast<std::uint32_t>(i), j, 0.1 + static_cast<double>(rng() % 1000) / 100.0);
    }
  Csr W = make_w(n, e);
  std::vector<bool> active(n, true);
  const auto t0 = std::chrono::steady_clock::now();
  auto r = louvain(W, active);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  MESSAGE("louvain n=10000 deg~30: " << ms << " ms, " << r.count << " communities, Q=" << r.modularity);
#if defined(NDEBUG) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
  CHECK(ms < 400.0);  // optimized builds only; the time is printed above either way
#endif
}

TEST_CASE("tracker keeps membership between re-clusters and recovers the planted groups") {
  const std::size_t n = 60;
  // P with raw weights: a flux graph with three dense groups of 20
  std::vector<std::vector<std::pair<std::uint32_t, double>>> rows(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j)
      if (i != j && i / 20 == j / 20) rows[i].push_back({static_cast<std::uint32_t>(j), 1.0});
    rows[i].push_back({static_cast<std::uint32_t>((i + 20) % n), 0.01});
  }
  Csr P = chain_p(rows);
  std::vector<bool> active(n, true);
  CommunityTracker tr(n, 5);
  const auto g1 = tr.update(P, active);
  for (std::size_t i = 0; i < n; ++i) CHECK(g1[i] == g1[(i / 20) * 20]);
  CHECK(tr.communities() == 3);
  // frames 2..5 keep membership even when the flux changes completely
  Csr Q = chain_p([&] {
    std::vector<std::vector<std::pair<std::uint32_t, double>>> r(n);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        if (i != j && i % 3 == j % 3) r[i].push_back({static_cast<std::uint32_t>(j), 1.0});
    return r;
  }());
  for (int f = 2; f <= 5; ++f) {
    auto g = tr.update(Q, active);
    CHECK(g == g1);
  }
  auto g6 = tr.update(Q, active);  // frame 6 re-clusters
  CHECK(g6 != g1);
  // a node that becomes active joins its strongest neighbour's community
  std::vector<bool> fewer = active;
  fewer[5] = false;
  CommunityTracker t2(n, 5);
  t2.update(P, fewer);
  auto g = t2.update(P, active);
  CHECK(g[5] == g[0]);
}

// ---- label matching and layout order (pure functions behind CommunityTracker) ----

namespace {
// Per-node vectors from lists of members.
std::vector<std::int32_t> ids_from(std::size_t n, const std::vector<std::vector<int>>& comms) {
  std::vector<std::int32_t> v(n, -1);
  for (std::size_t c = 0; c < comms.size(); ++c)
    for (int i : comms[c]) v[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(c);
  return v;
}
std::vector<std::int64_t> labels_from(std::size_t n, const std::vector<std::pair<std::int64_t, std::vector<int>>>& comms) {
  std::vector<std::int64_t> v(n, -1);
  for (auto& [l, m] : comms)
    for (int i : m) v[static_cast<std::size_t>(i)] = l;
  return v;
}
std::vector<int> range(int a, int b) {
  std::vector<int> r;
  for (int i = a; i < b; ++i) r.push_back(i);
  return r;
}
}  // namespace

TEST_CASE("matching rejects a pair with Jaccard < 0.3 even when it has the largest overlap") {
  // old 7 = nodes 0..9. New 0 = nodes 0..9 plus 40 more (overlap 10, Jaccard 10/50 = 0.2): rejected.
  // New 1 = nodes 50..59 and old 8 = nodes 50..57 (overlap 8, Jaccard 0.8): accepted.
  auto nw = ids_from(100, {[&] { auto a = range(0, 10); auto b = range(10, 50); a.insert(a.end(), b.begin(), b.end()); return a; }(), range(50, 60)});
  auto old = labels_from(100, {{7, range(0, 10)}, {8, range(50, 58)}});
  std::int64_t next = 100;
  auto lab = match_labels(nw, 2, -1, old, next);
  CHECK(lab[0] == 100);  // fresh label
  CHECK(lab[1] == 8);
  CHECK(next == 101);
}

TEST_CASE("matching is greedy by overlap, then lower new id, then lower old id") {
  // overlap desc: new 1 (12 shared with old 5) beats new 0 (6 shared with old 5) for old 5.
  auto nw = ids_from(40, {range(0, 12), range(12, 24)});
  auto old = labels_from(40, {{5, range(6, 18)}});  // overlaps new0: 6, new1: 6 -> tie, lower new id wins
  std::int64_t next = 50;
  auto lab = match_labels(nw, 2, -1, old, next);
  CHECK(lab[0] == 5);
  CHECK(lab[1] == 50);
  // strictly larger overlap wins regardless of id
  auto old2 = labels_from(40, {{5, range(2, 20)}});  // new0: 10, new1: 8
  next = 50;
  CHECK(match_labels(nw, 2, -1, old2, next)[0] == 5);
  auto old3 = labels_from(40, {{5, range(8, 24)}});  // new0: 4, new1: 12
  next = 50;
  auto l3 = match_labels(nw, 2, -1, old3, next);
  CHECK(l3[1] == 5);
  CHECK(l3[0] == 50);
  // one new community overlapping two old ones equally: the lower old id is taken
  auto nw2 = ids_from(20, {range(0, 10)});
  auto old4 = labels_from(20, {{9, range(0, 5)}, {3, range(5, 10)}});
  next = 50;
  CHECK(match_labels(nw2, 1, -1, old4, next)[0] == 3);
}

TEST_CASE("labels persist when Louvain renumbers its communities") {
  auto old = labels_from(30, {{11, range(0, 10)}, {22, range(10, 20)}, {33, range(20, 30)}});
  for (auto perm : {std::vector<int>{0, 1, 2}, std::vector<int>{2, 0, 1}, std::vector<int>{1, 2, 0}}) {
    std::vector<std::vector<int>> comms(3);
    for (std::size_t g = 0; g < 3; ++g) comms[static_cast<std::size_t>(perm[g])] = range(static_cast<int>(g) * 10, static_cast<int>(g) * 10 + 10);
    auto nw = ids_from(30, comms);
    std::int64_t next = 100;
    auto lab = match_labels(nw, 3, -1, old, next);
    for (std::size_t g = 0; g < 3; ++g) CHECK(lab[static_cast<std::size_t>(perm[g])] == old[g * 10]);
    CHECK(next == 100);  // nothing fresh was needed
  }
  // with no previous state every community gets a fresh label, in ascending new id
  std::int64_t next = 0;
  auto lab = match_labels(ids_from(30, {range(0, 10), range(10, 20), range(20, 30)}), 3, -1, std::vector<std::int64_t>(30, -1), next);
  CHECK(lab == std::vector<std::int64_t>({0, 1, 2}));
}

TEST_CASE("layout order keeps the old relative order and puts unmatched communities at their spectral position") {
  // old order 1, 2, 3. New spectral order of new ids is [c, x, a, b] with labels {3, 9, 1, 2} (9 is unmatched).
  const std::vector<std::int64_t> old_order = {1, 2, 3};
  const std::vector<std::int64_t> new_label = {1, 2, 3, 9};  // new ids 0..3
  auto ord = arrange_order({2, 3, 0, 1}, -1, new_label, old_order);
  CHECK(ord == std::vector<std::int64_t>({1, 9, 2, 3}));  // matched ones re-sorted into old order, 9 stays at slot 1
  // loose community is not part of the order
  auto ord2 = arrange_order({1, 0, 2}, 2, std::vector<std::int64_t>({1, 2, -1}), {2, 1});
  CHECK(ord2 == std::vector<std::int64_t>({2, 1}));
  // no previous order: the spectral order is used as is
  CHECK(arrange_order({2, 0, 1}, -1, std::vector<std::int64_t>({5, 6, 7}), {}) == std::vector<std::int64_t>({7, 5, 6}));
}

TEST_CASE("tracker keeps labels stable across re-clusters when the communities are the same") {
  const std::size_t n = 60;
  std::vector<std::vector<std::pair<std::uint32_t, double>>> rows(n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      if (i != j && i / 20 == j / 20) rows[i].push_back({static_cast<std::uint32_t>(j), 1.0});
  Csr P = chain_p(rows);
  std::vector<bool> active(n, true);
  CommunityTracker tr(n, 1);  // re-cluster every frame
  tr.update(P, active);
  auto l1 = tr.node_group();
  // Swap the groups of nodes 0 and 59: the community holding node 0 is now the one with the smallest member, so
  // Louvain numbers its communities differently (ids rotate), but the labels must follow the node sets.
  auto grp = [](std::size_t i) { return i == 0 ? std::size_t{2} : (i == 59 ? std::size_t{0} : i / 20); };
  std::vector<std::vector<std::pair<std::uint32_t, double>>> mod(n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      if (i != j && grp(i) == grp(j)) mod[i].push_back({static_cast<std::uint32_t>(j), 1.0});
  tr.update(chain_p(mod), active);
  auto l2 = tr.node_group();
  CHECK(l2[1] == l1[1]);
  CHECK(l2[25] == l1[25]);
  CHECK(l2[40] == l1[40]);
  CHECK(l2[0] == l1[40]);
  CHECK(l2[59] == l1[1]);
}
