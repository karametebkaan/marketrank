#include <doctest/doctest.h>

#include "graph/csr.hpp"
#include "graph/markov_solver.hpp"

using namespace fx;

namespace {
// Builds a Csr directly from a dense row-stochastic matrix (k = n keeps all edges;
// diagonal entries are handled by putting them in explicitly).
Csr dense_to_csr(const std::vector<double>& M, std::size_t n) {
  Csr P;
  P.n = n;
  P.row_ptr.push_back(0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (M[i * n + j] != 0) {
        P.col.push_back(static_cast<std::uint32_t>(j));
        P.val.push_back(M[i * n + j]);
      }
    }
    P.row_ptr.push_back(P.col.size());
  }
  return P;
}
}  // namespace

TEST_CASE("undamped 3-state chain matches the analytic stationary distribution") {
  Csr P = dense_to_csr({0.5, 0.5, 0, 0.25, 0.5, 0.25, 0, 0.5, 0.5}, 3);
  SolveResult r = stationary(P, 1.0, {});
  CHECK(r.converged);
  CHECK(r.pi[0] == doctest::Approx(0.25));
  CHECK(r.pi[1] == doctest::Approx(0.5));
  CHECK(r.pi[2] == doctest::Approx(0.25));
}

TEST_CASE("damped 2-state chain matches the closed form") {
  Csr P = dense_to_csr({0.9, 0.1, 0.5, 0.5}, 2);
  SolveResult r = stationary(P, 0.85, {});
  CHECK(r.converged);
  CHECK(r.pi[0] == doctest::Approx(0.5 / (1 - 0.4 * 0.85)));
  CHECK(r.pi[0] + r.pi[1] == doctest::Approx(1.0));
}

TEST_CASE("warm start converges to the same answer in fewer iterations") {
  Csr P = dense_to_csr({0.5, 0.5, 0, 0.25, 0.5, 0.25, 0, 0.5, 0.5}, 3);
  SolveResult cold = stationary(P, 0.85, {});
  SolveResult warm = stationary(P, 0.85, cold.pi);
  CHECK(warm.converged);
  CHECK(warm.iterations < cold.iterations);
  for (int i = 0; i < 3; ++i) CHECK(warm.pi[i] == doctest::Approx(cold.pi[i]));
}

TEST_CASE("damping makes a reducible chain converge (two absorbing nodes)") {
  std::vector<double> F = {0, 1, 1, 0,  //
                           0, 0, 0, 0,  //
                           0, 0, 0, 0,  //
                           0, 1, 1, 0};
  Csr P = build_transition(F, 4, 4);
  SolveResult r = stationary(P, 0.85, {});
  CHECK(r.converged);
  CHECK(r.pi[1] == doctest::Approx(r.pi[2]));
  CHECK(r.pi[1] > r.pi[0]);
  auto h = hotness(r.pi);
  CHECK(h[1] > 0);
  CHECK(h[0] < 0);
  double sum_h = 0;
  for (double x : h) sum_h += x;
  CHECK(sum_h == doctest::Approx(0.0).epsilon(1e-9));
}

TEST_CASE("propagate applies k damped steps") {
  Csr P = dense_to_csr({0.9, 0.1, 0.5, 0.5}, 2);
  auto one = propagate(P, 1.0, std::vector<double>{0.5, 0.5}, 1);
  CHECK(one[0] == doctest::Approx(0.7));
  auto two = propagate(P, 1.0, std::vector<double>{0.5, 0.5}, 2);
  CHECK(two[0] == doctest::Approx(0.7 * 0.9 + 0.3 * 0.5));
}
