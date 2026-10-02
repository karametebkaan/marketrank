#include <doctest/doctest.h>

#include <cmath>
#include <limits>

#include "graph/flux_builder.hpp"

using namespace fx;

TEST_CASE("bar_flux conserves each source's outflow (no affinity)") {
  std::vector<double> out;
  bar_flux(std::vector<double>{-3, 1, 2}, {}, 0.0, out);
  REQUIRE(out.size() == 9);
  CHECK(out[0 * 3 + 1] == doctest::Approx(1.0));
  CHECK(out[0 * 3 + 2] == doctest::Approx(2.0));
  for (std::size_t j = 0; j < 3; ++j) {
    CHECK(out[1 * 3 + j] == 0.0);  // sinks emit nothing
    CHECK(out[2 * 3 + j] == 0.0);
  }
}

TEST_CASE("bar_flux applies correlation affinity") {
  std::vector<double> corr = {1, 1, 0, 1, 1, 0, 0, 0, 1};
  std::vector<double> out;
  bar_flux(std::vector<double>{-1, 1, 1}, corr, 1.0, out);
  CHECK(out[1] == doctest::Approx(2.0 / 3.0));
  CHECK(out[2] == doctest::Approx(1.0 / 3.0));
  CHECK(out[1] + out[2] == doctest::Approx(1.0));
}

TEST_CASE("bar_flux with no sinks produces no flux") {
  std::vector<double> out;
  bar_flux(std::vector<double>{-1, -2, 0}, {}, 0.0, out);
  for (double v : out) CHECK(v == 0.0);
}

TEST_CASE("accumulators decay by half-life and NaN inputs are inert") {
  FluxParams p;
  p.lambda = 0.0;
  p.halflife_slow = 1.0;
  p.halflife_fast = 1.0;
  FluxBuilder fb(3, p);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> r = {-0.01, 0.01, nan}, dv = {300, 100, 100};
  fb.step(r, dv);
  fb.step(r, dv);
  // per-bar flux 0->1 = |-3| = 3; after two steps with halflife 1: 0.5*3 + 3 = 4.5
  CHECK(fb.flux_slow()[0 * 3 + 1] == doctest::Approx(4.5));
  CHECK(fb.flux_fast()[0 * 3 + 1] == doctest::Approx(4.5));
  CHECK(fb.flux_slow()[0 * 3 + 2] == 0.0);
}

TEST_CASE("rolling correlation tracks co-movement") {
  FluxParams p;
  p.corr_window = 20;
  FluxBuilder fb(3, p);
  std::vector<double> dv = {1, 1, 1};
  for (int t = 0; t < 50; ++t) {
    const double x = std::sin(t * 0.7) * 0.01;
    const double z = std::cos(t * 1.3) * 0.01;
    fb.step(std::vector<double>{x, -x, z}, dv);
  }
  CHECK(fb.correlation(0, 0) == doctest::Approx(1.0));
  CHECK(fb.correlation(0, 1) == doctest::Approx(-1.0).epsilon(1e-6));
  CHECK(std::abs(fb.correlation(0, 2)) < 0.6);
}
