// M3c Task 4: estimated-vs-observed flow comparison, MarketRank agreement, quarter pricing, CLI flags.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cli/args.hpp"
#include "core/time.hpp"
#include "flows13f/compare.hpp"
#include "flows13f/observed.hpp"
#include "graph/return_window.hpp"
#include "graph/sparse_flux.hpp"
#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/core_pipeline.hpp"
#include "test_util.hpp"

using namespace mr;

namespace {

// Complete directed graph on n nodes (no self-loops) with i.i.d. uniform weights.
std::vector<FlowEdge> complete_random(std::size_t n, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u(1.0, 1000.0);
  std::vector<FlowEdge> e;
  for (std::uint32_t i = 0; i < n; ++i)
    for (std::uint32_t j = 0; j < n; ++j)
      if (i != j) e.push_back({i, j, u(rng)});
  return e;
}

Panel synthetic_panel() {
  SyntheticConfig cfg;
  BarStore store(test::temp_dir("compare13f"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  return build_panel(store, tickers, cfg.tf);
}

// A panel of daily bars at 16:00 UTC with the given closes (row-major [t][i]).
Panel make_panel(const std::vector<TimePoint>& times, const std::vector<std::string>& tickers,
                 const std::vector<double>& close) {
  Panel p;
  p.times = times;
  p.tickers = tickers;
  p.close = close;
  p.open = p.high = p.low = p.vwap = close;
  p.volume.assign(close.size(), 1e6);
  return p;
}

}  // namespace

TEST_CASE("13f compare (a): identical inputs agree perfectly") {
  std::mt19937_64 rng(7);
  const std::size_t n = 60;  // 3540 edges: enough for the top-2000 overlap and the top-50 pi overlap
  const auto e = complete_random(n, rng);
  const Agreement a = compare_flows(e, e, n);
  CHECK(a.edge_spearman == doctest::Approx(1.0));
  CHECK(a.in_spearman == doctest::Approx(1.0));
  CHECK(a.out_spearman == doctest::Approx(1.0));
  CHECK(a.pi_spearman == doctest::Approx(1.0));
  CHECK(a.top500_overlap == 1.0);
  CHECK(a.top2000_overlap == 1.0);
  CHECK(a.top50_pi_overlap == 1.0);
  // the shuffled baseline is far from the identity
  CHECK(a.shuf_edge_spearman < 0.5);
  CHECK(a.shuf_pi_spearman < 0.5);
}

TEST_CASE("13f compare (b): independent random inputs give Spearman near 0") {
  // Complete graphs with n(n-1) <= 5000 edges: the top-5000 union is every pair, so independent weights
  // give no agreement. Node and pi Spearman over 70 nodes have sd ~0.12 per draw, so the bound is
  // checked on the mean over 20 draws (sd ~0.027); the well-powered edge Spearman is checked per draw.
  std::mt19937_64 rng(2026);
  const std::size_t n = 70, draws = 20;
  double edge = 0, in = 0, out = 0, pi = 0, se = 0, sp = 0;
  for (std::size_t d = 0; d < draws; ++d) {
    const auto o = complete_random(n, rng);
    const auto e = complete_random(n, rng);
    const Agreement a = compare_flows(o, e, n);
    CHECK(std::abs(a.edge_spearman) < 0.1);
    CHECK(std::abs(a.shuf_edge_spearman) < 0.1);
    edge += a.edge_spearman, in += a.in_spearman, out += a.out_spearman, pi += a.pi_spearman;
    se += a.shuf_edge_spearman, sp += a.shuf_pi_spearman;
  }
  const double k = static_cast<double>(draws);
  CHECK(std::abs(edge / k) < 0.1);
  CHECK(std::abs(in / k) < 0.1);
  CHECK(std::abs(out / k) < 0.1);
  CHECK(std::abs(pi / k) < 0.1);
  CHECK(std::abs(se / k) < 0.1);
  CHECK(std::abs(sp / k) < 0.1);
}

TEST_CASE("13f compare: the engine spearman uses average ranks and is NaN when undefined") {
  CHECK(spearman(std::vector<double>{1, 2, 3, 4}, std::vector<double>{10, 20, 30, 40}) == doctest::Approx(1.0));
  CHECK(spearman(std::vector<double>{1, 2, 3, 4}, std::vector<double>{4, 3, 2, 1}) == doctest::Approx(-1.0));
  // ties: ranks (1.5, 1.5, 3) vs (1, 2, 3) -> pearson = 0.866
  CHECK(spearman(std::vector<double>{5, 5, 9}, std::vector<double>{1, 2, 3}) == doctest::Approx(std::sqrt(3.0) / 2));
  CHECK(std::isnan(spearman(std::vector<double>{1, 1, 1}, std::vector<double>{1, 2, 3})));
  CHECK(std::isnan(spearman(std::vector<double>{1}, std::vector<double>{1})));
}

TEST_CASE("13f compare: top-k overlap is |A and B| / k, duplicates are summed") {
  // observed: 0->1 heavy, 1->2; estimated: same edges listed in two pieces
  const std::vector<FlowEdge> o = {{0, 1, 10}, {1, 2, 5}, {2, 0, 1}};
  const std::vector<FlowEdge> e = {{0, 1, 4}, {1, 2, 3}, {0, 1, 6}, {2, 0, 0.5}};
  const Agreement a = compare_flows(o, e, 3);
  CHECK(a.top500_overlap == doctest::Approx(3.0 / 500));
  CHECK(a.edge_spearman == doctest::Approx(1.0));
}

TEST_CASE("13f compare (c): pi_of reproduces the A/B/C example") {
  // Slide 3: A, B, C = 0, 1, 2; T in dollars, row = from, column = to.
  const std::vector<FlowEdge> e = {{0, 1, 500'000}, {0, 2, 100'000}, {1, 0, 200'000},
                                   {1, 2, 1'000'000}, {2, 0, 750'000}, {2, 1, 400'000}};
  const auto pi = pi_of(e, 3);
  REQUIRE(pi.size() == 3);
  CHECK(std::abs(pi[1] - 0.36016) < 1e-5);
  CHECK(std::abs(pi[2] - 0.34665) < 1e-5);
  CHECK(std::abs(pi[0] - 0.29319) < 1e-5);
  // a node with no outflow teleports: a 2-node chain 0 -> 1 with 1 dangling
  const auto d = pi_of({{0, 1, 1.0}}, 2);
  CHECK(d[0] + d[1] == doctest::Approx(1.0));
  CHECK(d[1] > d[0]);
  // exact: teleport t = (0.15 + 0.85*pi1)/2; pi0 = t, pi1 = 0.85*pi0 + t
  CHECK(d[0] == doctest::Approx((0.15 + 0.85 * d[1]) / 2).epsilon(1e-8));
}

TEST_CASE("13f compare (d): estimated_quarter_flows equals the summed bar_flux_sparse rows") {
  const Panel panel = synthetic_panel();
  REQUIRE(panel.T() > 30);
  for (const double lambda : {0.0, 1.0}) {
    CoreParams p = CoreParams::market_rank();
    p.flux.lambda = lambda;
    const std::size_t first = 10, last = 25;
    // reference: the pipeline's own pressure, the affinity of the return window before each bar
    CorePipeline pipe(panel.N(), p);
    ReturnWindow w(panel.N(), p.corr_window);
    std::map<std::pair<std::uint32_t, std::uint32_t>, double> ref;
    for (std::size_t t = 1; t <= last; ++t) {
      pipe.step(panel, t);
      std::vector<double> ret(panel.N(), std::numeric_limits<double>::quiet_NaN());
      for (std::size_t i = 0; i < panel.N(); ++i) {
        const double c = panel.close[panel.idx(t, i)], c0 = panel.close[panel.idx(t - 1, i)];
        if (std::isfinite(c) && std::isfinite(c0) && c0 > 0) ret[i] = c / c0 - 1.0;
      }
      if (t >= first) {
        std::span<const double> unit;
        if (lambda > 0) unit = w.unit_vectors();
        const BarFlux bar = bar_flux_sparse(pipe.last_pressure(), unit, w.window(), p.flux);
        for (std::uint32_t i = 0; i < bar.rows.size(); ++i)
          for (const auto& e : bar.rows[i]) ref[{i, e.j}] += e.w;
      }
      w.push(ret);
    }
    REQUIRE(!ref.empty());
    const auto est = estimated_quarter_flows(panel, p, first, last);
    std::map<std::pair<std::uint32_t, std::uint32_t>, double> got;
    for (const auto& e : est) got[{e.from, e.to}] += e.dollars;
    CHECK(got.size() == ref.size());
    double worst = 0;
    for (const auto& [k, v] : ref) {
      auto it = got.find(k);
      REQUIRE(it != got.end());
      worst = std::max(worst, std::abs(it->second - v) / v);
    }
    CHECK(worst < 1e-6);
    // the one-pass multi-range form gives the same as per-range calls
    const auto multi = estimated_range_flows(panel, p, {{first, last}, {3, 8}});
    REQUIRE(multi.size() == 2);
    CHECK(multi[0].size() == est.size());
    CHECK(multi[1].size() == estimated_quarter_flows(panel, p, 3, 8).size());
  }
}

TEST_CASE("13f compare: quarter bounds and bar ranges are UTC calendar quarters") {
  const auto [s, e] = quarter_bounds("2025Q4");
  CHECK(s == utc_seconds(2025, 10, 1));
  CHECK(e == utc_seconds(2026, 1, 1));
  CHECK(quarter_bounds("2024Q1").first == utc_seconds(2024, 1, 1));
  CHECK(previous_quarter("2025Q1") == "2024Q4");
  CHECK(previous_quarter("2025Q3") == "2025Q2");
  CHECK_THROWS(quarter_bounds("2025Q5"));
  CHECK_THROWS(quarter_bounds("25Q1"));
  const Panel p = make_panel({utc_seconds(2025, 9, 30, 16), utc_seconds(2025, 10, 1, 16), utc_seconds(2025, 12, 31, 16),
                              utc_seconds(2026, 1, 2, 16)},
                             {"A"}, {1, 2, 3, 4});
  const auto r = quarter_bar_range(p, "2025Q4");
  REQUIRE(r);
  CHECK(r->first == 1);
  CHECK(r->second == 2);
  CHECK_FALSE(quarter_bar_range(p, "2024Q1"));
}

TEST_CASE("13f compare: synthetic split is converted to the adjusted basis") {
  // Adjusted (adjustment=all) closes: SPL did a 2:1 split during 2025Q4, so the adjusted history is 50
  // throughout while the 13F raw price was 100 at the end of 2025Q3 and 50 at the end of 2025Q4. FLAT has
  // no split; its adjustment factor drifts by a dividend-sized 0.5%, which must not count as a split.
  const std::vector<TimePoint> times = {utc_seconds(2025, 9, 30, 16), utc_seconds(2025, 11, 14, 16),
                                        utc_seconds(2025, 12, 31, 16), utc_seconds(2026, 1, 2, 16)};
  const Panel p = make_panel(times, {"SPL", "FLAT"}, {50, 20, 50, 20, 50, 20, 50, 20});
  QuarterHoldings prev, cur;
  prev.quarter = "2025Q3";
  cur.quarter = "2025Q4";
  // manager 1 holds SPL through the split (10 raw shares -> 20) and buys FLAT with cash
  prev.rows = {{1, "SPL", 10, 1000}, {1, "FLAT", 100, 2000 * 1.005}, {2, "SPL", 4, 400}};
  cur.rows = {{1, "SPL", 20, 1000}, {1, "FLAT", 150, 3000}, {2, "SPL", 8, 400}};
  const QuarterPricing q = quarter_pricing(p, prev, cur);
  REQUIRE(q.ratio.size() == 2);
  CHECK(q.ratio[0] == doctest::Approx(2.0));
  CHECK(q.ratio[1] == 1.0);                         // 0.5% drift snapped to no split
  CHECK(q.prices[0] == doctest::Approx(50.0));      // mean adjusted close x (raw/adjusted at q end = 1)
  CHECK(q.prices[1] == doctest::Approx(20.0));
  CHECK(q.splits == 1);
  auto ratio = [&](const std::string& t) { return q.ratio[t == "SPL" ? 0 : 1]; };
  const ObservedFlows f = observed_flows(prev, cur, p.tickers, q.prices, ratio);
  // SPL: no trade after the split adjustment; FLAT: +50 shares x 20 = 1000 unpaired cash in
  CHECK(f.paired == 0.0);
  CHECK(f.unpaired_out == 0.0);
  CHECK(f.unpaired_in == doctest::Approx(1000.0));

  // A split after the quarter: adjusted closes are a tenth of the raw price in both quarters (factor 10).
  const Panel p10 = make_panel(times, {"SPL", "FLAT"}, {10, 20, 10, 20, 10, 20, 10, 20});
  QuarterHoldings a = prev, b = cur;
  a.rows = {{1, "SPL", 10, 1000}};
  b.rows = {{1, "SPL", 5, 500}};  // sold 5 raw shares at raw 100
  const QuarterPricing q10 = quarter_pricing(p10, a, b);
  CHECK(q10.ratio[0] == 1.0);
  CHECK(q10.prices[0] == doctest::Approx(100.0));  // raw basis of the quarter, matching the 13F shares
  const ObservedFlows g = observed_flows(a, b, p10.tickers, q10.prices,
                                         [&](const std::string& t) { return q10.ratio[t == "SPL" ? 0 : 1]; });
  CHECK(g.unpaired_out == doctest::Approx(500.0));
}

TEST_CASE("13f compare: restrict_to keeps edges inside the node set, remapped to local indices") {
  const std::vector<FlowEdge> e = {{0, 5, 1}, {5, 9, 2}, {9, 0, 3}, {2, 5, 4}};
  const auto r = restrict_to(e, {0, 5, 9}, 10);
  REQUIRE(r.size() == 3);
  CHECK((r[0].from == 0 && r[0].to == 1 && r[0].dollars == 1));
  CHECK((r[1].from == 1 && r[1].to == 2));
  CHECK((r[2].from == 2 && r[2].to == 0));
}

TEST_CASE("13f compare (e): CLI flags") {
  const CliArgs a = parse_cli({"--mode", "replay", "--compare-13f", "--13f-quarters", "2025Q3,2025Q4"});
  CHECK(a.compare_13f);
  CHECK(a.quarters_13f == std::vector<std::string>{"2025Q3", "2025Q4"});
  CHECK_FALSE(parse_cli({"--mode", "replay"}).compare_13f);
  CHECK(parse_cli({"--mode", "replay", "--compare-13f"}).quarters_13f.empty());
  CHECK_THROWS_AS(parse_cli({"--compare-13f"}), std::invalid_argument);  // needs --mode replay
  CHECK_THROWS_AS(parse_cli({"--mode", "replay", "--compare-13f", "--13f-quarters", "2025Q9"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--mode", "replay", "--13f-quarters", "2025Q1"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--mode", "replay", "--compare-13f", "--13f-quarters"}), std::invalid_argument);
  CHECK(cli_usage().find("--compare-13f") != std::string::npos);
}

TEST_CASE("13f compare: run_compare_13f end to end on synthetic holdings") {
  // Daily bars 2025-05-01 .. 2026-01-15 for 4 tickers with random-walk closes.
  const std::vector<std::string> tickers = {"AAA", "BBB", "CCC", "DDD"};
  std::vector<TimePoint> times;
  std::vector<double> close;
  std::mt19937_64 rng(5);
  std::normal_distribution<double> z(0.0, 0.02);
  std::vector<double> px = {50, 80, 20, 120};
  for (TimePoint t = utc_seconds(2025, 5, 1, 20); t < utc_seconds(2026, 1, 15); t += 86400) {
    times.push_back(t);
    for (auto& x : px) close.push_back(x *= std::exp(z(rng)));
  }
  const Panel panel = make_panel(times, tickers, close);
  const auto data = test::temp_dir("compare13f_e2e");
  test::write_file(data / "13f" / "cusip_map.csv", "cusip,ticker\nC1,AAA\nC2,BBB\nC3,CCC\nC4,DDD\n");
  const std::string hdr = "cik,cusip,issuer,shares,value_usd\n";
  test::write_file(data / "13f" / "holdings_2025Q3.csv",
                   hdr + "1,C1,a,100,5000\n1,C2,b,100,8000\n2,C3,c,500,10000\n2,C4,d,10,1200\n");
  test::write_file(data / "13f" / "holdings_2025Q4.csv",
                   hdr + "1,C1,a,50,2500\n1,C2,b,140,11200\n2,C3,c,300,6000\n2,C4,d,40,4800\n");
  test::write_file(data / "13f" / "holdings_2024Q1.csv", hdr + "1,C1,a,1,50\n");
  Compare13fOptions opt;
  opt.data = data;
  std::ostringstream log;
  const auto r = run_compare_13f(panel, opt, log);
  CHECK(r["found_quarters"].size() == 3);
  // 2024Q1 has no previous quarter; 2025Q3's previous (2025Q2) has no file
  CHECK(r["skipped"].size() == 2);
  REQUIRE(r["quarters"].size() == 1);
  CHECK(r["quarters"][0]["quarter"] == "2025Q4");
  CHECK(r["quarters"][0]["managers"] == 2);
  CHECK(r["configs"].size() == 6);  // the marketrank preset is a grid point (lambda 1, dollar)
  std::size_t bases = 0;
  for (const auto& c : r["configs"]) bases += c["base"].get<bool>();
  CHECK(bases == 1);
  CHECK(std::filesystem::exists(data / "13f" / "report.md"));
  CHECK(std::filesystem::exists(data / "13f" / "report.json"));
  CHECK(log.str().find("2024Q1") != std::string::npos);
  // a requested quarter limits the run
  opt.quarters = {"2025Q3"};
  const auto r2 = run_compare_13f(panel, opt, log);
  CHECK(r2["quarters"].empty());
  CHECK(r2["skipped"].size() == 1);
}

TEST_CASE("13f compare: a price-factor jump without matching share counts is not a split") {
  // MOV's 13F price halves against a flat adjusted close (a mispriced filing, or a big real move the
  // adjusted series somehow misses) but holders' share counts do not double: no split.
  const std::vector<TimePoint> times = {utc_seconds(2025, 9, 30, 16), utc_seconds(2025, 12, 31, 16)};
  const Panel p = make_panel(times, {"MOV"}, {50, 50});
  QuarterHoldings prev, cur;
  prev.quarter = "2025Q3";
  cur.quarter = "2025Q4";
  prev.rows = {{1, "MOV", 10, 1000}, {2, "MOV", 30, 3000}};
  cur.rows = {{1, "MOV", 10, 500}, {2, "MOV", 33, 1650}};
  const QuarterPricing q = quarter_pricing(p, prev, cur);
  CHECK(q.ratio[0] == 1.0);
  CHECK(q.splits == 0);
}
