#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "graph/pressure.hpp"

using namespace fx;

TEST_CASE("dollar and sqrt pressure") {
  PressureModel d(2, PressureMode::Dollar), s(2, PressureMode::Sqrt);
  std::vector<double> r = {0.02, -0.01}, v = {100, 400}, vw = {10, 25};
  auto pd = d.step(r, v, vw);
  CHECK(pd[0] == doctest::Approx(0.02 * 1000));
  CHECK(pd[1] == doctest::Approx(-0.01 * 10000));
  auto ps = s.step(r, v, vw);
  CHECK(ps[0] == doctest::Approx(0.02 * std::sqrt(1000.0)));
  CHECK(ps[1] == doctest::Approx(-0.01 * 100.0));
}

TEST_CASE("relative pressure uses the median volume of previous bars only") {
  PressureModel m(1, PressureMode::Relative, 3);
  auto bar = [&](double r, double v) {
    return m.step(std::vector<double>{r}, std::vector<double>{v}, std::vector<double>{10.0})[0];
  };
  CHECK(bar(0.01, 100) == 0.0);
  CHECK(bar(0.01, 300) == doctest::Approx(0.01 * 3.0));
  CHECK(bar(-0.02, 50) == doctest::Approx(-0.02 * 50 / 200.0));
  CHECK(bar(0.01, 1000) == doctest::Approx(0.01 * 1000 / 100.0));
  CHECK(bar(0.01, 10) == doctest::Approx(0.01 * 10 / 300.0));
}

TEST_CASE("missing data gives zero pressure and missing bars are not recorded") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  PressureModel m(2, PressureMode::Relative, 5);
  m.step(std::vector<double>{nan, 0.01}, std::vector<double>{100, nan},
         std::vector<double>{10, 10});
  auto p = m.step(std::vector<double>{0.01, 0.01}, std::vector<double>{200, 100},
                  std::vector<double>{10, 10});
  CHECK(p[0] == doctest::Approx(0.01 * 2.0));
  CHECK(p[1] == 0.0);
  auto mdv = m.median_dollar_volume();
  CHECK(mdv[0] == doctest::Approx(1500.0));
  CHECK(mdv[1] == doctest::Approx(1000.0));
}

TEST_CASE("pressure mode strings") {
  for (auto m : {PressureMode::Dollar, PressureMode::Sqrt, PressureMode::Relative})
    CHECK(parse_pressure_mode(to_string(m)) == m);
  CHECK_THROWS_AS(parse_pressure_mode("cap"), std::invalid_argument);
  CHECK_THROWS_AS(PressureModel(1, PressureMode::Relative, 0), std::invalid_argument);
}
