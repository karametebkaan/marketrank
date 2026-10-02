#include <doctest/doctest.h>

#include "graph/forecaster.hpp"
#include "graph/markov_solver.hpp"

using namespace mr;

namespace {
Csr two_state() {
  Csr P;
  P.n = 2;
  P.row_ptr = {0, 2, 4};
  P.col = {0, 1, 0, 1};
  P.val = {0.9, 0.1, 0.5, 0.5};
  return P;
}
}  // namespace

TEST_CASE("forecast score from propagation without drift") {
  Forecast f = forecast(two_state(), 1.0, std::vector<double>{0.5, 0.5}, {}, 1, 0.5);
  CHECK(f.k == 1);
  CHECK(f.pi_k[0] == doctest::Approx(0.7));
  CHECK(f.score[0] == doctest::Approx(2 * (0.7 - 0.5)));
  CHECK(f.score[1] == doctest::Approx(2 * (0.3 - 0.5)));
}

TEST_CASE("forecast of a vector already stationary for P_fast is zero") {
  Csr P = two_state();
  auto pi = stationary(P, 0.85, {}).pi;
  Forecast f = forecast(P, 0.85, pi, {}, 8, 0.5);
  CHECK(f.score[0] == doctest::Approx(0.0).epsilon(1e-8));
}

TEST_CASE("drift term adds beta * N * (pi_now - pi_prev)") {
  Csr P = two_state();
  auto pi = stationary(P, 0.85, {}).pi;
  std::vector<double> prev = {pi[0] - 0.1, pi[1] + 0.1};
  Forecast f = forecast(P, 0.85, pi, prev, 1, 0.5);
  CHECK(f.score[0] == doctest::Approx(0.5 * 2 * 0.1).epsilon(1e-6));
  CHECK(f.score[1] == doctest::Approx(-0.5 * 2 * 0.1).epsilon(1e-6));
}
