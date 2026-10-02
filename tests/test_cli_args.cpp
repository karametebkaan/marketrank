#include <doctest/doctest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "cli/args.hpp"

using namespace fx;

TEST_CASE("cli defaults") {
  CliArgs a = parse_cli({});
  CHECK(a.mode == "synthetic");
  CHECK(a.tf == Timeframe::Day);
  CHECK(a.lookback_days == 365);
  CHECK(a.universe == UniverseSource::Auto);
  CHECK(a.universe_size == 10000);
  CHECK_FALSE(a.eval);
  CHECK(a.threads == 0);
  CHECK(a.params.transition.lift == LiftMode::Excess);
  CHECK(describe(a.params).find("lift=excess") != std::string::npos);
}

TEST_CASE("--legacy applies first regardless of position") {
  CliArgs a = parse_cli({"--pressure", "sqrt", "--legacy", "--k-in", "5"});
  CHECK(a.params.pressure == PressureMode::Sqrt);
  CHECK(a.params.transition.lift == LiftMode::Off);
  CHECK(a.params.transition.k_in == 5);
  CHECK(a.params.transition.retention == 0.0);
}

TEST_CASE("model, universe and eval flags") {
  CliArgs a = parse_cli({"--mode", "replay", "--timeframe", "1h", "--lift", "ratio", "--h-ref",
                         "longrun", "--retention", "0.5", "--lambda", "0.3", "--k-out", "12",
                         "--universe", "snapshot", "--universe-size", "2000", "--refresh-universe",
                         "--eval", "--eval-bars", "60", "--threads", "4", "--top", "7"});
  CHECK(a.mode == "replay");
  CHECK(a.lookback_days == 60);
  CHECK(a.params.transition.lift == LiftMode::Ratio);
  CHECK(a.params.h_ref == HotRef::LongRun);
  CHECK(a.params.transition.retention == 0.5);
  CHECK(a.params.flux.lambda == 0.3);
  CHECK(a.params.transition.k_out == 12);
  CHECK(a.universe == UniverseSource::Snapshot);
  CHECK(a.universe_size == 2000);
  CHECK(a.refresh_universe);
  CHECK(a.eval);
  CHECK(a.eval_bars == 60);
  CHECK(a.threads == 4);
  CHECK(a.top == 7);
}

TEST_CASE("bad cli values throw") {
  CHECK_THROWS_AS(parse_cli({"--mode", "live"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--lift", "max"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--lambda", "2"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--k-out"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--top", "abc"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--universe", "world"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--bogus"}), std::invalid_argument);
  CHECK(parse_cli({"--help"}).help);
}
