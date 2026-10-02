#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "graph/return_window.hpp"
#include "graph/sparse_flux.hpp"

using namespace fx;

namespace {
std::vector<double> dense_flux(const std::vector<double>& p, const std::vector<double>& u,
                               std::size_t w, double lambda) {
  const std::size_t n = p.size();
  auto a = [&](std::size_t i, std::size_t j) {
    if (u.empty()) return 1.0;
    double d = 0;
    for (std::size_t s = 0; s < w; ++s) d += u[i * w + s] * u[j * w + s];
    return 1.0 + lambda * d;
  };
  std::vector<double> F(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (!(p[i] < 0)) continue;
    double denom = 0;
    for (std::size_t k = 0; k < n; ++k)
      if (p[k] > 0) denom += p[k] * a(i, k);
    for (std::size_t j = 0; j < n; ++j)
      if (p[j] > 0) F[i * n + j] = -p[i] * p[j] * a(i, j) / denom;
  }
  return F;
}

struct Market {
  std::vector<double> p, u;
  std::size_t w = 8;
  explicit Market(std::size_t n) {
    std::mt19937 rng(11);
    std::normal_distribution<double> d(0.0, 1.0);
    ReturnWindow rw(n, w);
    for (std::size_t t = 0; t < w; ++t) {
      std::vector<double> r(n);
      for (auto& x : r) x = d(rng) * 0.01;
      rw.push(r);
    }
    u = rw.unit_vectors();
    p.resize(n);
    for (auto& x : p) x = d(rng);
  }
};
}  // namespace

TEST_CASE("sparse flux equals the dense formula when every sink is kept") {
  const std::size_t n = 12;
  Market m(n);
  SparseFluxParams sp;
  sp.lambda = 0.8;
  sp.sink_candidates = n;
  sp.sinks_per_source = n;
  BarFlux bf = bar_flux_sparse(m.p, m.u, m.w, sp);
  auto F = dense_flux(m.p, m.u, m.w, 0.8);
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<double> row(n, 0.0);
    for (const auto& e : bf.rows[i]) row[e.j] = e.w;
    const double out_expected = m.p[i] < 0 ? -m.p[i] : 0.0;
    double rs = 0, col = 0;
    for (std::size_t j = 0; j < n; ++j) {
      CHECK(row[j] == doctest::Approx(F[i * n + j]).epsilon(1e-12));
      rs += row[j];
      col += F[j * n + i];
    }
    CHECK(rs == doctest::Approx(out_expected));
    CHECK(bf.out[i] == doctest::Approx(out_expected));
    CHECK(bf.in[i] == doctest::Approx(col).epsilon(1e-12));
  }
}

TEST_CASE("truncated edges keep exact shares and exact column totals") {
  const std::size_t n = 30;
  Market m(n);
  SparseFluxParams sp;
  sp.lambda = 1.0;
  sp.sink_candidates = 8;
  sp.sinks_per_source = 3;
  BarFlux bf = bar_flux_sparse(m.p, m.u, m.w, sp);
  auto F = dense_flux(m.p, m.u, m.w, 1.0);
  double sum_in = 0, sum_out = 0;
  for (std::size_t i = 0; i < n; ++i) {
    CHECK(bf.rows[i].size() <= 3);
    for (std::size_t e = 1; e < bf.rows[i].size(); ++e) CHECK(bf.rows[i][e - 1].j < bf.rows[i][e].j);
    for (const auto& e : bf.rows[i]) CHECK(e.w == doctest::Approx(F[i * n + e.j]).epsilon(1e-12));
    double col = 0;
    for (std::size_t k = 0; k < n; ++k) col += F[k * n + i];
    CHECK(bf.in[i] == doctest::Approx(col).epsilon(1e-12));
    sum_in += bf.in[i];
    sum_out += bf.out[i];
  }
  CHECK(sum_in == doctest::Approx(sum_out));
}

TEST_CASE("edges go to the top candidate sinks by pressure (no affinity)") {
  std::vector<double> p = {-2.0, 5.0, 1.0, 3.0, 4.0};
  SparseFluxParams sp;
  sp.lambda = 0.0;
  sp.sink_candidates = 3;
  sp.sinks_per_source = 2;
  BarFlux bf = bar_flux_sparse(p, {}, 0, sp);
  REQUIRE(bf.rows[0].size() == 2);
  CHECK(bf.rows[0][0].j == 1);
  CHECK(bf.rows[0][1].j == 4);
  CHECK(bf.rows[0][0].w == doctest::Approx(2.0 * 5.0 / 13.0));
  CHECK(bf.in[2] == doctest::Approx(2.0 * 1.0 / 13.0));
  CHECK(bf.out[0] == 2.0);
}

TEST_CASE("no sinks or no sources gives no flux") {
  BarFlux a = bar_flux_sparse(std::vector<double>{-1, -2, 0}, {}, 0, SparseFluxParams{});
  for (double x : a.out) CHECK(x == 0.0);
  for (const auto& r : a.rows) CHECK(r.empty());
  BarFlux b = bar_flux_sparse(std::vector<double>{1, 2, 0}, {}, 0, SparseFluxParams{});
  for (double x : b.in) CHECK(x == 0.0);
}

TEST_CASE("accumulator decays by half-life, merges by column and caps rows") {
  BarFlux bar;
  bar.rows = {{{1, 3.0}, {2, 1.0}}, {}, {}};
  bar.out = {4.0, 0, 0};
  bar.in = {0, 3.0, 1.0};
  FluxAccumulator acc(3, 1.0, 256);
  acc.add(bar);
  acc.add(bar);
  REQUIRE(acc.rows()[0].size() == 2);
  CHECK(acc.rows()[0][0].j == 1);
  CHECK(acc.rows()[0][0].w == doctest::Approx(4.5));
  CHECK(acc.out()[0] == doctest::Approx(6.0));
  CHECK(acc.in()[2] == doctest::Approx(1.5));
  CHECK(acc.total() == doctest::Approx(6.0));
  CHECK(acc.edge_count() == 2);
  FluxAccumulator capped(3, 1.0, 1);
  capped.add(bar);
  REQUIRE(capped.rows()[0].size() == 1);
  CHECK(capped.rows()[0][0].j == 1);
}

TEST_CASE("infinite half-life never decays; invalid settings throw") {
  BarFlux bar;
  bar.rows = {{{1, 2.0}}, {}};
  bar.out = {2.0, 0};
  bar.in = {0, 2.0};
  FluxAccumulator acc(2, std::numeric_limits<double>::infinity());
  acc.add(bar);
  acc.add(bar);
  CHECK(acc.rows()[0][0].w == 4.0);
  CHECK_THROWS_AS(FluxAccumulator(2, 0.0), std::invalid_argument);
  CHECK_THROWS_AS(FluxAccumulator(2, 1.0, 0), std::invalid_argument);
}
