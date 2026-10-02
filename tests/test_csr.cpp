#include <doctest/doctest.h>

#include "graph/csr.hpp"

using namespace fx;

TEST_CASE("build_transition keeps top-k, normalizes rows, adds self-loops") {
  // row 0: flows 5, 1, 3 to nodes 1, 2, 3; row 1: nothing; row 2: one flow; row 3: nothing
  std::vector<double> F = {0, 5, 1, 3,  //
                           0, 0, 0, 0,  //
                           2, 0, 0, 0,  //
                           0, 0, 0, 0};
  Csr P = build_transition(F, 4, 2);
  REQUIRE(P.row_ptr.size() == 5);
  // row 0 keeps 5 (col 1) and 3 (col 3), ascending columns
  REQUIRE(P.row_ptr[1] - P.row_ptr[0] == 2);
  CHECK(P.col[0] == 1);
  CHECK(P.val[0] == doctest::Approx(5.0 / 8.0));
  CHECK(P.col[1] == 3);
  CHECK(P.val[1] == doctest::Approx(3.0 / 8.0));
  // row 1 is an accumulator: self-loop
  REQUIRE(P.row_ptr[2] - P.row_ptr[1] == 1);
  CHECK(P.col[P.row_ptr[1]] == 1);
  CHECK(P.val[P.row_ptr[1]] == 1.0);
  for (std::size_t i = 0; i < 4; ++i) {
    double s = 0;
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) s += P.val[e];
    CHECK(s == doctest::Approx(1.0));
  }
}

TEST_CASE("k >= n keeps every positive edge") {
  std::vector<double> F = {0, 1, 1, 1, 0, 1, 1, 1, 0};
  Csr P = build_transition(F, 3, 10);
  CHECK(P.val.size() == 6);
}

TEST_CASE("left_multiply computes x * P") {
  std::vector<double> F = {0, 1, 0, 0};  // node0 -> node1, node1 self-loop
  Csr P = build_transition(F, 2, 5);
  auto y = left_multiply(P, std::vector<double>{0.3, 0.7});
  CHECK(y[0] == doctest::Approx(0.0));
  CHECK(y[1] == doctest::Approx(1.0));
}

TEST_CASE("raw holds the un-normalized pruned flux, 0 for synthesized self-loops") {
  std::vector<double> F = {0, 5, 1, 3,  //
                           0, 0, 0, 0,  //
                           2, 0, 0, 0,  //
                           0, 0, 0, 0};
  Csr P = build_transition(F, 4, 2);
  REQUIRE(P.raw.size() == P.val.size());
  CHECK(P.raw[0] == 5.0);  // row 0, col 1
  CHECK(P.raw[1] == 3.0);  // row 0, col 3
  CHECK(P.raw[P.row_ptr[1]] == 0.0);  // self-loop
  CHECK(P.raw[P.row_ptr[2]] == 2.0);
}

TEST_CASE("top-k ties are broken toward the lower column index") {
  std::vector<double> F = {0, 2, 2, 2, 2,  //
                           0, 0, 0, 0, 0,  //
                           0, 0, 0, 0, 0,  //
                           0, 0, 0, 0, 0,  //
                           0, 0, 0, 0, 0};
  Csr P = build_transition(F, 5, 2);
  REQUIRE(P.row_ptr[1] - P.row_ptr[0] == 2);
  CHECK(P.col[0] == 1);
  CHECK(P.col[1] == 2);
}

TEST_CASE("inactive nodes get only a self-loop and receive no edges") {
  std::vector<double> F = {0, 1, 2,  //
                           1, 0, 1,  //
                           1, 1, 0};
  Csr P = build_transition(F, 3, 5, {true, true, false});
  REQUIRE(P.row_ptr[3] - P.row_ptr[2] == 1);
  CHECK(P.col[P.row_ptr[2]] == 2);
  for (std::size_t e = 0; e < P.row_ptr[2]; ++e) CHECK(P.col[e] != 2);
}
