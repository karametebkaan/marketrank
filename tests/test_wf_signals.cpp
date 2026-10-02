#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/core_pipeline.hpp"
#include "test_util.hpp"
#include "walkforward/signals.hpp"

using namespace mr;

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

Frame make_frame(std::vector<double> pi, std::vector<bool> active) {
  Frame f;
  f.pi = std::move(pi);
  f.active = std::move(active);
  const std::size_t n = f.pi.size();
  f.h.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) f.h[i] = 0.1 * static_cast<double>(i) - 0.15;
  f.inflow.assign(n, 1.0);
  return f;
}
std::size_t S(Signal s) { return static_cast<std::size_t>(s); }
}  // namespace

TEST_CASE("signal names round-trip") {
  for (std::size_t i = 0; i < kSignals; ++i) CHECK(parse_signal(to_string(static_cast<Signal>(i))) == static_cast<Signal>(i));
  CHECK(to_string(Signal::PiRelSize) == "pi_rel_size");
  CHECK_THROWS_AS(parse_signal("nope"), std::invalid_argument);
}

TEST_CASE("SignalTracker: pulses, NaN rules, neg hotness") {
  const std::size_t n = 4;
  // node 2 rises each frame, renormalised so the total is 1; node 3 is inactive in frame 2.
  Frame f1 = make_frame({0.4, 0.3, 0.2, 0.1}, {true, true, true, true});
  Frame f2 = make_frame({0.4, 0.3, 0.25, 0.0}, {true, true, true, false});
  Frame f3 = make_frame({0.3, 0.2, 0.3, 0.2}, {true, true, true, true});
  {
    double s = 0;
    for (double x : f2.pi) s += x;
    for (double& x : f2.pi) x /= s;
  }
  SignalTracker tr(n);
  auto s1 = tr.update(f1);
  auto s2 = tr.update(f2);
  auto s3 = tr.update(f3);
  for (double x : s1[S(Signal::Pulse1)]) CHECK(std::isnan(x));
  const double n3 = 3, n4 = 4;
  CHECK(s2[S(Signal::Pulse1)][2] == std::log(f2.pi[2] * n3) - std::log(f1.pi[2] * n4));
  CHECK(s2[S(Signal::Score)][2] == std::log(f2.pi[2] * n3));
  CHECK(std::isnan(s2[S(Signal::Pulse1)][3]));
  CHECK(std::isnan(s2[S(Signal::Score)][3]));
  CHECK(std::isnan(s3[S(Signal::Pulse1)][3]));  // inactive at t-1
  CHECK(std::isfinite(s3[S(Signal::Pulse1)][2]));
  for (const auto* s : {&s1, &s2, &s3})
    for (double x : (*s)[S(Signal::Pulse5)]) CHECK(std::isnan(x));
  for (std::size_t i = 0; i < n; ++i) CHECK(s1[S(Signal::NegHotness)][i] == -f1.h[i]);
  CHECK(std::isnan(s1[S(Signal::Forecast)][0]));  // no forecasts
  CHECK(std::isnan(s1[S(Signal::InflowMom5)][0]));
}

TEST_CASE("SignalTracker: Pulse5, Pulse20 and InflowMom5 appear after K steps") {
  const std::size_t n = 3;
  SignalTracker tr(n);
  std::array<std::vector<double>, kSignals> s;
  std::vector<Frame> fr;
  for (int t = 0; t < 22; ++t) {
    const double a = 0.3 + 0.01 * t;
    Frame f = make_frame({a, 0.5 - a / 2, 0.5 - a / 2}, {true, true, true});
    f.inflow.assign(n, 1.0 + t);
    fr.push_back(f);
    s = tr.update(f);
    if (t < 5) CHECK(std::isnan(s[S(Signal::Pulse5)][0]));
    if (t == 5) CHECK(s[S(Signal::Pulse5)][0] == std::log(fr[5].pi[0] * 3) - std::log(fr[0].pi[0] * 3));
    if (t < 20) CHECK(std::isnan(s[S(Signal::Pulse20)][0]));
    if (t == 20) CHECK(s[S(Signal::Pulse20)][0] == std::log(fr[20].pi[0] * 3) - std::log(fr[0].pi[0] * 3));
    if (t < 5) CHECK(std::isnan(s[S(Signal::InflowMom5)][0]));
    if (t == 5) CHECK(s[S(Signal::InflowMom5)][0] == std::log(6.0 / 1.0));
  }
  CHECK(s[S(Signal::Pulse20)][0] == std::log(fr[21].pi[0] * 3) - std::log(fr[1].pi[0] * 3));
}

TEST_CASE("zscore") {
  const std::vector<double> x{1, 2, 3, kNaN};
  const auto z = zscore(x, {true, true, true, true});
  CHECK(std::isnan(z[3]));
  double m = (z[0] + z[1] + z[2]) / 3, v = 0;
  for (int i = 0; i < 3; ++i) v += (z[i] - m) * (z[i] - m);
  CHECK(m == doctest::Approx(0).epsilon(1e-12));
  CHECK(std::sqrt(v / 3) == doctest::Approx(1.0));
  const auto zc = zscore(std::vector<double>{5, 5, 5}, {true, true, true});
  for (double e : zc) CHECK(e == 0.0);
  const auto zm = zscore(x, {true, false, true, true});
  CHECK(std::isnan(zm[1]));
  std::vector<double> big(101, 0.0);
  big[0] = 1000;
  const auto zb = zscore(big, std::vector<bool>(101, true));
  CHECK(zb[0] == 3.0);
}

TEST_CASE("Pulse1 equals Frame::pulse on a real pipeline") {
  SyntheticConfig cfg;
  BarStore store(test::temp_dir("wf_signals"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  Panel panel = build_panel(store, tickers, cfg.tf);
  CorePipeline pipe(panel.N(), CoreParams::market_rank());
  SignalTracker tr(panel.N());
  std::size_t finite = 0;
  for (std::size_t t = 1; t <= 8; ++t) {
    const Frame f = pipe.step(panel, t);
    const auto s = tr.update(f);
    REQUIRE(s[S(Signal::Pulse1)].size() == f.pulse.size());
    for (std::size_t i = 0; i < f.pulse.size(); ++i) {
      const double a = s[S(Signal::Pulse1)][i], b = f.pulse[i];
      CHECK((a == b || (std::isnan(a) && std::isnan(b))));
      finite += std::isfinite(a);
    }
  }
  CHECK(finite > 0);
}
