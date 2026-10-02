#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "graph/pressure.hpp"

using namespace mr;

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

TEST_CASE("relative pressure caps V/ADV at max_volume_ratio; 0 means uncapped") {
  auto run = [](double cap) {
    PressureModel m(1, PressureMode::Relative, 3, cap);
    m.step(std::vector<double>{0.01}, std::vector<double>{100}, std::vector<double>{10.0});
    return m.step(std::vector<double>{0.02}, std::vector<double>{5000}, std::vector<double>{10.0})[0];
  };
  CHECK(run(5.0) == doctest::Approx(0.02 * 5.0));
  CHECK(run(0.0) == doctest::Approx(0.02 * 50.0));
  // Below the cap the ratio is untouched, and dollar/sqrt ignore the cap.
  PressureModel lo(1, PressureMode::Relative, 3, 5.0);
  lo.step(std::vector<double>{0.01}, std::vector<double>{100}, std::vector<double>{10.0});
  CHECK(lo.step(std::vector<double>{0.02}, std::vector<double>{300}, std::vector<double>{10.0})[0] ==
        doctest::Approx(0.02 * 3.0));
  PressureModel sq(1, PressureMode::Sqrt, 3, 5.0);
  CHECK(sq.step(std::vector<double>{0.02}, std::vector<double>{100}, std::vector<double>{10.0})[0] ==
        doctest::Approx(0.02 * std::sqrt(1000.0)));
}

TEST_CASE("volatility-scaled pressure divides by the std of previous returns") {
  PressureModel m(1, PressureMode::Sqrt, 20, 0.0, 5);
  auto bar = [&](double r) {
    return m.step(std::vector<double>{r}, std::vector<double>{100.0}, std::vector<double>{4.0})[0];
  };
  const double prev[5] = {0.01, -0.01, 0.02, -0.02, 0.0};
  for (int k = 0; k < 5; ++k) CHECK(bar(prev[k]) == 0.0);  // fewer than 5 previous returns
  double mean = 0, ss = 0;
  for (double x : prev) mean += x / 5;
  for (double x : prev) ss += (x - mean) * (x - mean);
  const double sigma = std::sqrt(ss / 4);
  CHECK(bar(0.03) == doctest::Approx(0.03 / sigma * std::sqrt(400.0)));
}

TEST_CASE("the current return does not enter sigma") {
  auto run = [](double cur) {
    PressureModel m(1, PressureMode::Sqrt, 20, 0.0, 5);
    for (double x : {0.01, -0.01, 0.02, -0.02, 0.005})
      m.step(std::vector<double>{x}, std::vector<double>{100.0}, std::vector<double>{4.0});
    return m.step(std::vector<double>{cur}, std::vector<double>{100.0},
                  std::vector<double>{4.0})[0] / cur;
  };
  CHECK(run(0.01) == doctest::Approx(run(0.5)));
}

TEST_CASE("zero volatility uses the sigma floor; missing returns are not recorded") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  PressureModel m(1, PressureMode::Sqrt, 20, 0.0, 5);
  for (int k = 0; k < 5; ++k)
    m.step(std::vector<double>{0.0}, std::vector<double>{100.0}, std::vector<double>{4.0});
  m.step(std::vector<double>{nan}, std::vector<double>{100.0}, std::vector<double>{4.0});
  CHECK(m.step(std::vector<double>{0.001}, std::vector<double>{100.0},
               std::vector<double>{4.0})[0] == doctest::Approx(0.001 / 1e-4 * 20.0));
}
