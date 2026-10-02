#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include "analysis/partition_metrics.hpp"
#include "geom/community.hpp"

using namespace mr;

namespace {
using L = std::vector<std::int64_t>;

Csr graph(std::size_t n, const std::vector<std::tuple<std::uint32_t, std::uint32_t, double>>& edges) {
  std::vector<std::vector<std::pair<std::uint32_t, double>>> rows(n);
  for (auto& [a, b, w] : edges) {
    rows[a].push_back({b, w});
    rows[b].push_back({a, w});
  }
  Csr W;
  W.n = n;
  W.row_ptr.push_back(0);
  for (auto& r : rows) {
    std::sort(r.begin(), r.end());
    for (auto& e : r) {
      W.col.push_back(e.first);
      W.val.push_back(e.second);
    }
    W.row_ptr.push_back(W.col.size());
  }
  W.raw = W.val;
  return W;
}

// `groups` cliques of `size` nodes (weight 10) in a ring of weak (0.1) links.
Csr cliques(std::size_t groups, std::size_t size) {
  std::vector<std::tuple<std::uint32_t, std::uint32_t, double>> e;
  for (std::size_t g = 0; g < groups; ++g) {
    for (std::size_t a = 0; a < size; ++a)
      for (std::size_t b = a + 1; b < size; ++b)
        e.emplace_back(static_cast<std::uint32_t>(g * size + a), static_cast<std::uint32_t>(g * size + b), 10.0);
    e.emplace_back(static_cast<std::uint32_t>(g * size), static_cast<std::uint32_t>(((g + 1) % groups) * size + 1), 0.1);
  }
  return graph(groups * size, e);
}
}  // namespace

TEST_CASE("NMI and ARI on hand-computed partitions") {
  CHECK(partition_nmi(L{0, 0, 1, 1}, L{0, 0, 1, 1}) == doctest::Approx(1.0));
  CHECK(partition_ari(L{0, 0, 1, 1}, L{0, 0, 1, 1}) == doctest::Approx(1.0));
  // Label values do not matter, only the blocks.
  CHECK(partition_nmi(L{5, 5, 9, 9}, L{1, 1, 0, 0}) == doctest::Approx(1.0));
  CHECK(partition_ari(L{5, 5, 9, 9}, L{1, 1, 0, 0}) == doctest::Approx(1.0));
  // Independent blocks: I = 0; ARI = (0 - 4/6) / (2 - 4/6) = -0.5.
  CHECK(partition_nmi(L{0, 0, 1, 1}, L{0, 1, 0, 1}) == doctest::Approx(0.0));
  CHECK(partition_ari(L{0, 0, 1, 1}, L{0, 1, 0, 1}) == doctest::Approx(-0.5));
  // {3,3} vs {2,2,2}: cells 2,1,1,2. H(A) = ln 2, H(B) = ln 3, H(A,B) = 2/3 ln 3 + 1/3 ln 6;
  // NMI = 2 I / (H(A) + H(B)) = 0.515804; ARI = (2 - 18/15) / (4.5 - 18/15) = 0.242424.
  const L a{0, 0, 0, 1, 1, 1}, b{0, 0, 1, 1, 2, 2};
  CHECK(partition_nmi(a, b) == doctest::Approx(0.5158037).epsilon(1e-6));
  CHECK(partition_ari(a, b) == doctest::Approx(0.2424242).epsilon(1e-6));
  CHECK(partition_nmi(a, b) == doctest::Approx(partition_nmi(b, a)));
  CHECK(partition_ari(a, b) == doctest::Approx(partition_ari(b, a)));
  // Trivial partitions.
  CHECK(partition_nmi(L{3, 3, 3}, L{7, 7, 7}) == doctest::Approx(1.0));
  CHECK(partition_ari(L{3, 3, 3}, L{7, 7, 7}) == doctest::Approx(1.0));
  CHECK(std::isnan(partition_nmi(L{}, L{})));
  CHECK_THROWS(partition_nmi(L{0, 1}, L{0}));
}

TEST_CASE("the permutation null preserves block sizes and sits at chance") {
  std::mt19937_64 rng(3);
  L x(4000), y(4000);
  for (auto& v : x) v = static_cast<std::int64_t>(rng() % 12);
  for (auto& v : y) v = static_cast<std::int64_t>(rng() % 9);
  const L yp = permuted_labels(y, 42);
  L s1 = y, s2 = yp;
  std::sort(s1.begin(), s1.end());
  std::sort(s2.begin(), s2.end());
  CHECK(s1 == s2);
  CHECK(yp != y);
  CHECK(permuted_labels(y, 42) == yp);  // seeded
  // A partition against its own permuted copy: chance level, ~0 for ARI, small for NMI.
  CHECK(std::fabs(partition_ari(y, yp)) < 0.01);
  CHECK(partition_nmi(y, yp) < 0.01);
  CHECK(std::fabs(partition_ari(x, permuted_labels(x, 7))) < 0.01);
  CHECK(partition_nmi(y, y) == doctest::Approx(1.0));
}

TEST_CASE("seeded Louvain: seed 0 is louvain(), other seeds find the same clear partition") {
  const Csr W = cliques(5, 12);
  std::vector<bool> act(W.n, true);
  act[3] = false;
  const CommunityResult base = louvain(W, act);
  const CommunityResult s0 = louvain_seeded(W, act, 0);
  CHECK(s0.id == base.id);
  const CommunityResult s7 = louvain_seeded(W, act, 7);
  CHECK(s7.id[3] == -1);
  L a, b;
  for (std::size_t i = 0; i < W.n; ++i)
    if (act[i]) {
      a.push_back(base.id[i]);
      b.push_back(s7.id[i]);
    }
  CHECK(partition_ari(a, b) == doctest::Approx(1.0));
  CHECK(s7.count == 5);
  CHECK(s7.modularity == doctest::Approx(base.modularity));
}

TEST_CASE("spectral bisection splits cliques and pools small components") {
  // Two 10-cliques joined by one weak edge, a 3-node component, an isolated and an inactive node.
  std::vector<std::tuple<std::uint32_t, std::uint32_t, double>> e;
  for (std::uint32_t g = 0; g < 2; ++g)
    for (std::uint32_t a = 0; a < 10; ++a)
      for (std::uint32_t b = a + 1; b < 10; ++b) e.emplace_back(g * 10 + a, g * 10 + b, 5.0);
  e.emplace_back(0, 10, 0.05);
  e.emplace_back(20, 21, 1.0);
  e.emplace_back(21, 22, 1.0);
  const Csr W = graph(25, e);
  std::vector<bool> act(25, true);
  act[24] = false;
  const auto p = spectral_bisection(W, act, 2, 8);
  CHECK(p[24] == -1);
  CHECK(p[23] == -2);
  for (std::size_t i = 20; i < 23; ++i) CHECK(p[i] == -2);
  for (std::size_t i = 1; i < 10; ++i) CHECK(p[i] == p[0]);
  for (std::size_t i = 11; i < 20; ++i) CHECK(p[i] == p[10]);
  CHECK(p[0] != p[10]);
  CHECK(p[0] >= 0);
  CHECK(p[10] >= 0);
  // k = 1 keeps the one block.
  const auto one = spectral_bisection(W, act, 1, 8);
  CHECK(one[0] == one[15]);
}

TEST_CASE("bar_flux_csr carries the bar's flux as raw weights") {
  BarFlux bf;
  bf.rows = {{{1, 2.0}, {2, 3.0}}, {}, {{0, 1.5}}};
  const Csr P = bar_flux_csr(bf);
  CHECK(P.n == 3);
  CHECK(P.row_ptr == std::vector<std::size_t>{0, 2, 2, 3});
  CHECK(P.raw == std::vector<double>{2.0, 3.0, 1.5});
  const Csr W = symmetric_flux_graph(P, {true, true, true});
  // W(0,2) = 3 + 1.5
  CHECK(W.val[W.row_ptr[0] + 1] == doctest::Approx(4.5));
}
