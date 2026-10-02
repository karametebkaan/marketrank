#include <doctest/doctest.h>

#include <limits>
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

TEST_CASE("--money-flow applies first, later flags override, and conflicts with --legacy") {
  CliArgs a = parse_cli({"--k-in", "5", "--money-flow", "--h-ref", "netflow"});
  CHECK(a.params.pressure == PressureMode::Dollar);
  CHECK(a.params.transition.lift == LiftMode::Off);
  CHECK(a.params.transition.k_in == 5);
  CHECK(a.params.h_ref == HotRef::NetFlow);
  CHECK(parse_cli({"--money-flow"}).params.h_ref == HotRef::Size);
  CHECK_THROWS_AS(parse_cli({"--legacy", "--money-flow"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--money-flow", "--legacy"}), std::invalid_argument);
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

TEST_CASE("liquidity floor and volume cap flags") {
  CliArgs d = parse_cli({});
  CHECK(d.params.min_dollar_volume == 1e6);
  CHECK(d.params.max_volume_ratio == 5.0);
  CHECK(describe(d.params).find("min_dv=1e+06 max_vr=5") != std::string::npos);
  CliArgs a = parse_cli({"--min-dollar-volume", "250000", "--max-volume-ratio", "3.5"});
  CHECK(a.params.min_dollar_volume == 250000.0);
  CHECK(a.params.max_volume_ratio == 3.5);
  CliArgs l = parse_cli({"--legacy"});
  CHECK(l.params.min_dollar_volume == 0.0);
  CHECK(l.params.max_volume_ratio == 0.0);
  CHECK_THROWS_AS(parse_cli({"--min-dollar-volume", "abc"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--max-volume-ratio", "-1"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--max-volume-ratio"}), std::invalid_argument);
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

TEST_CASE("storage flags") {
  CliArgs a = parse_cli({"--migrate-cache"});
  CHECK(a.migrate_cache);
  CHECK(a.migrate_from == "data/cache");
  CliArgs b = parse_cli({"--migrate-cache", "/tmp/old", "--maintain"});
  CHECK(b.migrate_from == "/tmp/old");
  CHECK(b.maintain);
}

TEST_CASE("data_window: full range for synthetic, lookback window for replay and alpaca") {
  const TimePoint now = 1790812800;
  const auto syn = data_window(parse_cli({"--mode", "synthetic"}), now);
  CHECK(syn.first == std::numeric_limits<TimePoint>::min());
  CHECK(syn.second == std::numeric_limits<TimePoint>::max());
  const auto rep = data_window(parse_cli({"--mode", "replay", "--lookback-days", "10"}), now);
  CHECK(rep.second == now);
  CHECK(rep.first == now - 10 * 86400);
  const auto alp = data_window(parse_cli({"--mode", "alpaca", "--lookback-days", "10"}), now);
  CHECK(alp.second == now - 16 * 60);
  CHECK(alp.first == alp.second - 10 * 86400);
}

TEST_CASE("--shock is repeatable and parses signed sizes") {
  CliArgs a = parse_cli({"--shock", "NVDA:-10", "--shock", "AAPL:2.5", "--shock", "BRK.B:+1"});
  REQUIRE(a.shocks.size() == 3);
  CHECK(a.shocks[0].first == "NVDA");
  CHECK(a.shocks[0].second == -10.0);
  CHECK(a.shocks[1].second == 2.5);
  CHECK(a.shocks[2].first == "BRK.B");
  CHECK(a.shocks[2].second == 1.0);
  CHECK(parse_cli({}).shocks.empty());
}

TEST_CASE("malformed --shock values throw") {
  CHECK_THROWS_AS(parse_cli({"--shock", "NVDA"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--shock", ":5"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--shock", "NVDA:abc"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--shock", "NVDA:"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--shock", "NVDA:5x"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--shock"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--shock", "NVDA:nan"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--shock", "NVDA:inf"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--shock", "NVDA:-inf"}), std::invalid_argument);
}

TEST_CASE("--refetch-full is a sync-only flag") {
  CHECK_FALSE(parse_cli({"--mode", "alpaca"}).refetch_full);
  CHECK(parse_cli({"--mode", "alpaca", "--refetch-full"}).refetch_full);
  CHECK_THROWS_AS(parse_cli({"--mode", "replay", "--refetch-full"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--refetch-full"}), std::invalid_argument);
  CHECK(cli_usage().find("--refetch-full") != std::string::npos);
}

TEST_CASE("--sync-sectors parses and is documented") {
  CHECK_FALSE(parse_cli({}).sync_sectors);
  const CliArgs a = parse_cli({"--sync-sectors", "--universe", "snapshot", "--universe-size", "10000"});
  CHECK(a.sync_sectors);
  CHECK(a.universe == UniverseSource::Snapshot);
  CHECK(a.universe_size == 10000);
  CHECK(cli_usage().find("--sync-sectors") != std::string::npos);
}

TEST_CASE("serve flags default to the money-flow preset") {
  CliArgs a = parse_cli({"--serve", "--port", "9000", "--host", "0.0.0.0", "--web", "/tmp/w"});
  CHECK(a.serve);
  CHECK(a.port == 9000);
  CHECK(a.host == "0.0.0.0");
  CHECK(a.web == "/tmp/w");
  CHECK(a.params.pressure == PressureMode::Dollar);  // money_flow()
  CHECK(a.params.h_ref == HotRef::Size);
  CliArgs b = parse_cli({"--serve", "--h-ref", "netflow"});
  CHECK(b.params.h_ref == HotRef::NetFlow);
  CHECK(b.port == 8765);  // 8080 is taken by Kinetica on the dev machine
  CHECK(b.host == "127.0.0.1");
  CHECK(cli_usage().find("--port N (8765)") != std::string::npos);
  CliArgs c = parse_cli({"--serve", "--legacy"});
  CHECK(c.params.transition.lift == LiftMode::Off);
  CHECK(c.params.h_ref == HotRef::Uniform);
}
