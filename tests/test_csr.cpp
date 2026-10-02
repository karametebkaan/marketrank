#include <doctest/doctest.h>

#include <vector>

#include "graph/csr.hpp"

using namespace fx;

TEST_CASE("left_multiply computes x * P") {
  Csr P;
  P.n = 2;
  P.row_ptr = {0, 1, 2};
  P.col = {1, 1};
  P.val = {1.0, 1.0};
  P.raw = {1.0, 0.0};
  auto y = left_multiply(P, std::vector<double>{0.3, 0.7});
  CHECK(y[0] == doctest::Approx(0.0));
  CHECK(y[1] == doctest::Approx(1.0));
}

TEST_CASE("transposed pull product equals left_multiply") {
  Csr P;
  P.n = 3;
  P.row_ptr = {0, 2, 3, 5};
  P.col = {0, 2, 1, 0, 1};
  P.val = {0.25, 0.75, 1.0, 0.5, 0.5};
  P.raw = {1, 3, 1, 1, 1};
  std::vector<double> x = {0.2, 0.3, 0.5};
  const Csr PT = transpose(P);
  CHECK(PT.n == 3);
  auto a = left_multiply(P, x);
  auto b = left_multiply_transposed(PT, x);
  for (std::size_t i = 0; i < 3; ++i) CHECK(b[i] == doctest::Approx(a[i]));
}
