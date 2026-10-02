#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "flux_test_util.hpp"
#include "graph/transition.hpp"

using namespace mr;

namespace {
void check_stochastic(const Csr& P) {
  for (std::size_t i = 0; i < P.n; ++i) {
    double s = 0;
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) {
      s += P.val[e];
      if (e > P.row_ptr[i]) CHECK(P.col[e - 1] < P.col[e]);
    }
    CHECK(s == doctest::Approx(1.0));
  }
}
}  // namespace

TEST_CASE("legacy settings: top-k rows, normalized, self-loops for empty rows") {
  std::vector<double> F = {0, 5, 1, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0};
  Csr P = build_transition(test::acc_from_dense(F, 4), test::legacy_tp(2));
  REQUIRE(P.row_ptr.size() == 5);
  REQUIRE(P.row_ptr[1] == 2);
  CHECK(P.col[0] == 1);
  CHECK(P.val[0] == doctest::Approx(5.0 / 8.0));
  CHECK(P.raw[0] == 5.0);
  CHECK(P.col[1] == 3);
  CHECK(P.val[1] == doctest::Approx(3.0 / 8.0));
  REQUIRE(P.row_ptr[2] - P.row_ptr[1] == 1);
  CHECK(P.col[P.row_ptr[1]] == 1);
  CHECK(P.val[P.row_ptr[1]] == 1.0);
  CHECK(P.raw[P.row_ptr[1]] == 0.0);
  check_stochastic(P);
}

TEST_CASE("ties at the k-th weight go to the lower column") {
  std::vector<double> F = {0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Csr P = build_transition(test::acc_from_dense(F, 4), test::legacy_tp(2));
  REQUIRE(P.row_ptr[1] == 2);
  CHECK(P.col[0] == 1);
  CHECK(P.col[1] == 2);
}

TEST_CASE("inactive nodes get only a self-loop and receive no edges") {
  std::vector<double> F = {0, 1, 1, 1, 0, 1, 1, 1, 0};
  TransitionParams tp;  // defaults, including retention
  Csr P = build_transition(test::acc_from_dense(F, 3), tp, {true, true, false});
  REQUIRE(P.row_ptr[3] - P.row_ptr[2] == 1);
  CHECK(P.col[P.row_ptr[2]] == 2);
  CHECK(P.val[P.row_ptr[2]] == 1.0);
  CHECK(P.raw[P.row_ptr[2]] == 0.0);
  for (std::size_t e = 0; e < P.row_ptr[2]; ++e) CHECK(P.col[e] != 2);
  check_stochastic(P);
}

TEST_CASE("excess and ratio lift prefer above-gravity edges over big expected ones") {
  // out = [12, 2, 31], in = [2, 40, 3], total = 45.
  // F01 = 10 vs E01 = 10.667 (below gravity); F02 = 2 vs E02 = 0.8 (above gravity).
  std::vector<double> F = {0, 10, 2, 1, 0, 1, 1, 30, 0};
  TransitionParams off = test::legacy_tp(1);
  Csr Poff = build_transition(test::acc_from_dense(F, 3), off);
  CHECK(Poff.col[Poff.row_ptr[0]] == 1);
  for (LiftMode mode : {LiftMode::Excess, LiftMode::Ratio}) {
    TransitionParams tp = off;
    tp.lift = mode;
    Csr P = build_transition(test::acc_from_dense(F, 3), tp);
    REQUIRE(P.row_ptr[1] - P.row_ptr[0] == 1);
    CHECK(P.col[P.row_ptr[0]] == 2);
    CHECK(P.val[P.row_ptr[0]] == 1.0);
    CHECK(P.raw[P.row_ptr[0]] == 2.0);
    check_stochastic(P);
  }
}

TEST_CASE("k_in keeps a column's strongest inbound edge even when no row selects it") {
  std::vector<double> F = {0, 0, 0, 1, 5, 0, 0, 2, 5, 0, 0, 1, 5, 0, 0, 0};
  TransitionParams tp = test::legacy_tp(1);
  Csr without = build_transition(test::acc_from_dense(F, 4), tp);
  CHECK(without.row_ptr[2] - without.row_ptr[1] == 1);
  tp.k_in = 1;
  Csr with = build_transition(test::acc_from_dense(F, 4), tp);
  REQUIRE(with.row_ptr[2] - with.row_ptr[1] == 2);
  CHECK(with.col[with.row_ptr[1]] == 0);
  CHECK(with.col[with.row_ptr[1] + 1] == 3);
  check_stochastic(with);
}

TEST_CASE("retention gives a self-loop of in / (in + out)") {
  std::vector<double> F = {0, 1, 3, 0};  // out = [1, 3], in = [3, 1]
  TransitionParams tp = test::legacy_tp(5);
  tp.retention = 1.0;
  Csr P = build_transition(test::acc_from_dense(F, 2), tp);
  REQUIRE(P.row_ptr[1] == 2);
  CHECK(P.col[0] == 0);
  CHECK(P.val[0] == doctest::Approx(0.75));
  CHECK(P.raw[0] == doctest::Approx(3.0));
  CHECK(P.col[1] == 1);
  CHECK(P.val[1] == doctest::Approx(0.25));
  CHECK(P.val[P.row_ptr[1]] == doctest::Approx(0.75));      // row 1 -> node 0
  CHECK(P.val[P.row_ptr[1] + 1] == doctest::Approx(0.25));  // row 1 self
  check_stochastic(P);
}

TEST_CASE("lift mode strings") {
  for (auto m : {LiftMode::Off, LiftMode::Excess, LiftMode::Ratio})
    CHECK(parse_lift_mode(to_string(m)) == m);
  CHECK_THROWS_AS(parse_lift_mode("max"), std::invalid_argument);
}

TEST_CASE("an active mask of the wrong size throws") {
  std::vector<double> F = {0, 1, 1, 1, 0, 1, 1, 1, 0};
  const auto acc = test::acc_from_dense(F, 3);
  CHECK_THROWS_AS(build_transition(acc, TransitionParams{}, {true, true}), std::invalid_argument);
  CHECK_NOTHROW(build_transition(acc, TransitionParams{}, {}));
}
