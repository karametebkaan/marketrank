#include <doctest/doctest.h>

#include <stdexcept>
#include <vector>

#include "graph/hotness.hpp"

using namespace fx;

TEST_CASE("relative hotness against uniform, size and degenerate references") {
  std::vector<double> pi = {0.5, 0.25, 0.25};
  auto u = relative_hotness(pi, std::vector<double>{1, 1, 1});
  CHECK(u[0] == doctest::Approx(0.5));
  CHECK(u[1] == doctest::Approx(-0.25));
  auto s = relative_hotness(pi, std::vector<double>{2, 1, 1});  // ref = [0.5, 0.25, 0.25]
  for (double x : s) CHECK(x == doctest::Approx(0.0));
  auto z = relative_hotness(pi, std::vector<double>{0, 0, 0});  // uniform fallback
  CHECK(z[0] == doctest::Approx(0.5));
  auto f = relative_hotness(pi, std::vector<double>{1, 0, 1});  // floored, finite
  CHECK(f[1] > 1e6);
}

TEST_CASE("hot reference strings") {
  for (auto r : {HotRef::Uniform, HotRef::Size, HotRef::LongRun}) CHECK(parse_hot_ref(to_string(r)) == r);
  CHECK_THROWS_AS(parse_hot_ref("cap"), std::invalid_argument);
}
