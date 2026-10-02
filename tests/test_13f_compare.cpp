// M3c Task 4: estimated-vs-observed flow comparison, MarketRank agreement, quarter pricing, CLI flags.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
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
  const Agreement a = compare_flows(e, e, n, 10);
  CHECK(a.est.obs_topk_spearman == doctest::Approx(1.0));
  CHECK(a.est.full_spearman == doctest::Approx(1.0));
  CHECK(a.est.row_cosine == doctest::Approx(1.0));
  CHECK(a.est.union_spearman == doctest::Approx(1.0));
  CHECK(a.est.in_spearman == doctest::Approx(1.0));
  CHECK(a.est.out_spearman == doctest::Approx(1.0));
  CHECK(a.est.pi_spearman == doctest::Approx(1.0));
  CHECK(a.est.pi_spearman_flow == doctest::Approx(1.0));
  CHECK(a.est.top500_overlap == 1.0);
  CHECK(a.est.top2000_overlap == 1.0);
  CHECK(a.est.top50_pi_overlap == 1.0);
  CHECK(a.perm.draws == 10);
  CHECK(a.perm.mean.full_spearman < 0.5);  // the permuted labels are far from the identity
  CHECK(a.perm.mean.pi_spearman < 0.5);
  CHECK(a.perm.sd.full_spearman > 0);
  CHECK(a.top500_expected == doctest::Approx(500.0 / (60.0 * 59.0)));
  CHECK_FALSE(a.has_placebo);
}

TEST_CASE("13f compare (b): independent random inputs give Spearman near 0") {
  // Complete graphs: independent weights give no agreement. Node and pi Spearman over 70 nodes have sd ~0.12
  // per draw, so the bound is checked on the mean over 20 draws (sd ~0.027); the well-powered edge Spearmans
  // are checked per draw.
  std::mt19937_64 rng(2026);
  const std::size_t n = 70, draws = 20;
  double in = 0, out = 0, pi = 0, pin = 0;
  for (std::size_t d = 0; d < draws; ++d) {
    const auto o = complete_random(n, rng);
    const auto e = complete_random(n, rng);
    const Agreement a = compare_flows(o, e, n, 5);
    CHECK(std::abs(a.est.full_spearman) < 0.1);
    CHECK(std::abs(a.est.obs_topk_spearman) < 0.1);
    in += a.est.in_spearman, out += a.est.out_spearman, pi += a.est.pi_spearman, pin += a.perm.mean.pi_spearman;
  }
  const double k = static_cast<double>(draws);
  CHECK(std::abs(in / k) < 0.1);
  CHECK(std::abs(out / k) < 0.1);
  CHECK(std::abs(pi / k) < 0.1);
  CHECK(std::abs(pin / k) < 0.1);
}

TEST_CASE("13f compare: large sparse independent case has a permutation null near 0") {
  // n = 2000 nodes (the observed node-set size), 20,000 random edges each, one draw, 100 permutations.
  std::mt19937_64 rng(99);
  const std::uint32_t n = 2000;
  std::uniform_int_distribution<std::uint32_t> node(0, n - 1);
  std::lognormal_distribution<double> w(0.0, 1.5);
  auto sparse = [&] {
    std::vector<FlowEdge> e;
    while (e.size() < 20000) {
      const auto i = node(rng), j = node(rng);
      if (i != j) e.push_back({i, j, w(rng)});
    }
    return e;
  };
  const auto o = sparse(), e = sparse();
  const Agreement a = compare_flows(o, e, n);
  CHECK(a.perm.draws == 100);
  MESSAGE("perm null mean/sd: obs_topk ", a.perm.mean.obs_topk_spearman, "/", a.perm.sd.obs_topk_spearman, " full ",
          a.perm.mean.full_spearman, "/", a.perm.sd.full_spearman, " cosine ", a.perm.mean.row_cosine, "/",
          a.perm.sd.row_cosine, " union ", a.perm.mean.union_spearman, " top500 ", a.perm.mean.top500_overlap,
          " (expected ", a.top500_expected, ")");
  MESSAGE("estimate: obs_topk ", a.est.obs_topk_spearman, " full ", a.est.full_spearman, " cosine ", a.est.row_cosine);
  for (const double x : {a.perm.mean.obs_topk_spearman, a.perm.mean.full_spearman, a.perm.mean.row_cosine,
                         a.est.obs_topk_spearman, a.est.full_spearman, a.est.row_cosine})
    CHECK(std::abs(x) < 0.05);
  CHECK(a.perm.sd.full_spearman > 0);
  CHECK(a.top500_expected == doctest::Approx(500.0 / (2000.0 * 1999.0)));
}

TEST_CASE("13f compare: a size-only estimator does not beat the gravity null") {
  // Observed: managers hold size-weighted portfolios and trade at random (proportional pairing, rank-1 per
  // manager). Estimated: bars of size-scaled random pressure with the top 30 sinks, independent of the managers.
  // The estimate knows only node sizes, so it must not beat the gravity model of its own margins.
  std::mt19937_64 rng(11);
  const std::uint32_t n = 300;
  std::normal_distribution<double> z(0.0, 1.0);
  std::vector<double> size(n);
  for (auto& x : size) x = std::exp(1.5 * z(rng));
  std::discrete_distribution<std::uint32_t> pick(size.begin(), size.end());
  std::vector<double> O(n * n, 0.0), E(n * n, 0.0);
  for (int m = 0; m < 300; ++m) {
    std::vector<double> out(n, 0.0), in(n, 0.0);
    for (int k = 0; k < 40; ++k) {
      const auto i = pick(rng);
      const double d = z(rng) * size[i];
      (d < 0 ? out[i] : in[i]) += std::abs(d);
    }
    double so = 0, si = 0;
    for (std::uint32_t i = 0; i < n; ++i) so += out[i], si += in[i];
    if (so <= 0 || si <= 0) continue;
    const double scale = std::min(si, so) / (si * so);
    for (std::uint32_t i = 0; i < n; ++i)
      for (std::uint32_t j = 0; j < n; ++j)
        if (i != j) O[i * n + j] += out[i] * in[j] * scale;
  }
  for (int t = 0; t < 60; ++t) {
    std::vector<double> p(n);
    for (std::uint32_t i = 0; i < n; ++i) p[i] = z(rng) * size[i];
    std::vector<std::uint32_t> snk;
    for (std::uint32_t i = 0; i < n; ++i)
      if (p[i] > 0) snk.push_back(i);
    std::sort(snk.begin(), snk.end(), [&](auto a, auto b) { return p[a] > p[b]; });
    snk.resize(std::min<std::size_t>(30, snk.size()));
    double ps = 0;
    for (auto j : snk) ps += p[j];
    for (std::uint32_t i = 0; i < n; ++i)
      if (p[i] < 0)
        for (auto j : snk) E[i * n + j] += -p[i] * p[j] / ps;
  }
  auto edges = [&](const std::vector<double>& M) {
    std::vector<FlowEdge> e;
    for (std::uint32_t i = 0; i < n; ++i)
      for (std::uint32_t j = 0; j < n; ++j)
        if (i != j && M[i * n + j] > 0) e.push_back({i, j, M[i * n + j]});
    return e;
  };
  const Agreement a = compare_flows(edges(O), edges(E), n, 20);
  MESSAGE("size-only: obs_topk ", a.est.obs_topk_spearman, " vs gravity ", a.gravity.obs_topk_spearman, "; full ",
          a.est.full_spearman, " vs ", a.gravity.full_spearman, "; cosine ", a.est.row_cosine, " vs ",
          a.gravity.row_cosine, "; perm full ", a.perm.mean.full_spearman);
  CHECK(a.est.obs_topk_spearman - a.gravity.obs_topk_spearman <= 0.02);
  CHECK(a.est.full_spearman - a.gravity.full_spearman <= 0.02);
  CHECK(a.est.row_cosine - a.gravity.row_cosine <= 0.02);
  // ... while it does beat the label-permutation null (the reason that null is not enough)
  CHECK(a.est.obs_topk_spearman > a.perm.mean.obs_topk_spearman + 0.1);
}

TEST_CASE("13f compare: gravity null and placebo machinery") {
  // gravity of a matrix = out * in^T / total off the diagonal
  const std::vector<FlowEdge> e = {{0, 1, 6}, {0, 2, 2}, {1, 2, 4}};
  const auto g = gravity_null(e, 3);
  std::map<std::pair<std::uint32_t, std::uint32_t>, double> G;
  for (const auto& x : g) G[{x.from, x.to}] = x.dollars;
  // out = (8, 4, 0), in = (0, 6, 6), total 12
  CHECK(G.size() == 3);
  CHECK(G[{0, 1}] == doctest::Approx(4.0));
  CHECK(G[{0, 2}] == doctest::Approx(4.0));
  CHECK(G[{1, 2}] == doctest::Approx(2.0));
  // placebo equal to the estimate: zero lift; different placebo: metrics of that pair
  std::mt19937_64 rng(3);
  const auto o = complete_random(20, rng), est = complete_random(20, rng), other = complete_random(20, rng);
  const Agreement same = compare_flows(o, est, 20, 3, &est);
  REQUIRE(same.has_placebo);
  CHECK(same.placebo.full_spearman == doctest::Approx(same.est.full_spearman));
  CHECK(same.placebo.pi_spearman == doctest::Approx(same.est.pi_spearman));
  const Agreement diff = compare_flows(o, est, 20, 3, &other);
  CHECK(diff.placebo.full_spearman == doctest::Approx(compare_flows(o, other, 20, 1).est.full_spearman));
  // placebo quarter: q-4, else q+1, else q-1
  auto avail = [](std::set<std::string> s) { return [s](const std::string& q) { return s.count(q) > 0; }; };
  CHECK(placebo_quarter("2025Q4", avail({"2024Q4", "2026Q1", "2025Q3"})) == "2024Q4");
  CHECK(placebo_quarter("2025Q4", avail({"2026Q1", "2025Q3"})) == "2026Q1");
  CHECK(placebo_quarter("2025Q4", avail({"2025Q3"})) == "2025Q3");
  CHECK(placebo_quarter("2025Q4", avail({})).empty());
  CHECK(next_quarter("2025Q4") == "2026Q1");
}

TEST_CASE("13f compare: the engine spearman uses average ranks and is NaN when undefined") {
  CHECK(spearman(std::vector<double>{1, 2, 3, 4}, std::vector<double>{10, 20, 30, 40}) == doctest::Approx(1.0));
  CHECK(spearman(std::vector<double>{1, 2, 3, 4}, std::vector<double>{4, 3, 2, 1}) == doctest::Approx(-1.0));
  // ties: ranks (1.5, 1.5, 3) vs (1, 2, 3) -> pearson = 0.866
  CHECK(spearman(std::vector<double>{5, 5, 9}, std::vector<double>{1, 2, 3}) == doctest::Approx(std::sqrt(3.0) / 2));
  CHECK(std::isnan(spearman(std::vector<double>{1, 1, 1}, std::vector<double>{1, 2, 3})));
  CHECK(std::isnan(spearman(std::vector<double>{1}, std::vector<double>{1})));
}

TEST_CASE("13f compare: top-k overlap is |A and B| / min(k, |A|, |B|), duplicates are summed") {
  // observed: 0->1 heavy, 1->2; estimated: same edges listed in two pieces
  const std::vector<FlowEdge> o = {{0, 1, 10}, {1, 2, 5}, {2, 0, 1}};
  const std::vector<FlowEdge> e = {{0, 1, 4}, {1, 2, 3}, {0, 1, 6}, {2, 0, 0.5}, {2, 1, 0.1}};
  const Agreement a = compare_flows(o, e, 3, 1);
  CHECK(a.est.top500_overlap == doctest::Approx(1.0));  // 3 common of min(500, 3, 4)
  CHECK(a.est.obs_topk_spearman == doctest::Approx(1.0));
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
  for (const auto& extra : std::vector<std::vector<std::string>>{
           {"--eval"}, {"--shock", "AAA:5"}, {"--export-slice"}, {"--walkforward"}}) {
    std::vector<std::string> args = {"--mode", "replay", "--compare-13f"};
    args.insert(args.end(), extra.begin(), extra.end());
    CHECK_THROWS_WITH_AS(parse_cli(args), doctest::Contains("--compare-13f"), std::invalid_argument);
  }
}

TEST_CASE("13f compare: run_compare_13f end to end on synthetic holdings") {
  // Daily bars 2025-06-01 .. 2026-04-15 for 4 tickers with random-walk closes. Warm-up needs 60 bars
  // (corr_window): 2025Q3 starts at bar 30 (skipped), 2025Q4 at bar 122. 2025Q4's placebo is 2026Q1 (q-4 is
  // outside the panel; q+1 is complete).
  const std::vector<std::string> tickers = {"AAA", "BBB", "CCC", "DDD"};
  std::vector<TimePoint> times;
  std::vector<double> close;
  std::mt19937_64 rng(5);
  std::normal_distribution<double> z(0.0, 0.02);
  std::vector<double> px = {50, 80, 20, 120};
  for (TimePoint t = utc_seconds(2025, 6, 1, 20); t < utc_seconds(2026, 4, 15); t += 86400) {
    times.push_back(t);
    for (auto& x : px) close.push_back(x *= std::exp(z(rng)));
  }
  const Panel panel = make_panel(times, tickers, close);
  const auto data = test::temp_dir("compare13f_e2e");
  test::write_file(data / "13f" / "cusip_map.csv", "cusip,ticker\nC1,AAA\nC2,BBB\nC3,CCC\nC4,DDD\n");
  const std::string hdr = "cik,cusip,issuer,shares,value_usd\n";
  test::write_file(data / "13f" / "holdings_2025Q2.csv", hdr + "1,C1,a,100,5000\n2,C3,c,500,10000\n");
  test::write_file(data / "13f" / "holdings_2025Q3.csv",
                   hdr + "1,C1,a,100,5000\n1,C2,b,100,8000\n2,C3,c,500,10000\n2,C4,d,10,1200\n"
                         "3,C2,b,200,16000\n3,C3,c,100,2000\n3,C4,d,10,1200\n4,C4,d,50,6000\n4,C1,a,10,500\n");
  test::write_file(data / "13f" / "holdings_2025Q4.csv",
                   hdr + "1,C1,a,50,2500\n1,C2,b,140,11200\n2,C3,c,300,6000\n2,C4,d,40,4800\n"
                         "3,C2,b,100,8000\n3,C3,c,300,6000\n3,C4,d,30,3600\n4,C4,d,20,2400\n4,C1,a,80,4000\n");
  Compare13fOptions opt;
  opt.data = data;
  opt.lookback_days = 400;
  opt.timeframe = "1d";
  opt.git_sha = "abc123";
  opt.perms = 5;
  std::ostringstream log;
  const auto r = run_compare_13f(panel, opt, log);
  CHECK(r["found_quarters"].size() == 3);
  REQUIRE(r["skipped"].size() == 2);  // 2025Q2: no previous quarter; 2025Q3: warm-up
  CHECK(r["skipped"][1]["quarter"] == "2025Q3");
  CHECK(r["skipped"][1]["reason"].get<std::string>().find("warm-up") != std::string::npos);
  REQUIRE(r["quarters"].size() == 1);
  const auto& q = r["quarters"][0];
  CHECK(q["quarter"] == "2025Q4");
  CHECK(q["managers"] == 4);
  CHECK(q["inconsistent_positions"] == 0);
  CHECK(q["warmup_bars"] == 122);
  CHECK(q["placebo_quarter"] == "2026Q1");
  CHECK(q.contains("pi_obs_vs_adv"));
  CHECK(q.contains("pi_obs_vs_13f_value"));
  CHECK(q["split_candidates_unconfirmed"].is_array());
  CHECK(r["warmup_returns_required"] == 60);
  CHECK(r["base_params"]["corr_window"] == 60);
  CHECK(r["base_params"]["pressure"] == "dollar");
  CHECK(r["lookback_days"] == 400);
  CHECK(r["timeframe"] == "1d");
  CHECK(r["source_tree_git_sha_at_run_time"] == "abc123");
  CHECK_FALSE(r.contains("git_sha"));
  CHECK(q["warmup_returns"] == 121);
  CHECK(r["observed_params"]["top_n"] == 2000);
  CHECK(r["perms"] == 5);
  CHECK(r["configs"].size() == 6);  // the marketrank preset is a grid point (lambda 1, dollar)
  std::size_t bases = 0;
  for (const auto& c : r["configs"]) {
    bases += c["base"].get<bool>();
    const auto& row = c["quarters"][0];
    for (const char* k : {"estimate", "gravity", "placebo", "perm_mean", "perm_sd", "lift_placebo", "lift_gravity",
                          "lift_perm"})
      CHECK(row.contains(k));
    CHECK(c["summary"]["lift_placebo"]["pi_spearman"].contains("sd"));
    REQUIRE(row["placebo"].is_object());
    for (const auto& [name, mp] : kFlowMetricFields)
      if (std::string(name) == "obs_topk_spearman" || std::string(name) == "row_cosine" ||
          std::string(name) == "full_spearman")
        CHECK(std::isfinite(row["lift_placebo"][name].get<double>()));
  }
  CHECK(bases == 1);
  CHECK(r["best"].contains("pi_spearman"));
  CHECK(std::filesystem::exists(data / "13f" / "report.md"));
  CHECK(std::filesystem::exists(data / "13f" / "report.json"));
  const std::string md = compare_report_md(r);
  CHECK(md.find("rank-1") != std::string::npos);  // honest limit M8
  CHECK(md.find("placebo") != std::string::npos);
  CHECK(log.str().find("2025Q2") != std::string::npos);
  // a requested quarter limits the run
  opt.quarters = {"2025Q3"};
  const auto r2 = run_compare_13f(panel, opt, log);
  CHECK(r2["quarters"].empty());
  CHECK(r2["skipped"].size() == 1);
}

TEST_CASE("13f compare: reverse split, small split and unconfirmed split candidates") {
  const std::vector<TimePoint> times = {utc_seconds(2025, 9, 30, 16), utc_seconds(2025, 12, 31, 16)};
  // REV: 1:10 reverse split (raw 5 -> 50, shares / 10); SML: 11:10 split (ratio 1.1, above the 1.08 snap);
  // BUY: a real 2:1 split, but every holder also bought 50% more post-split shares (share ratio 3): unconfirmed.
  const Panel p = make_panel(times, {"REV", "SML", "BUY"}, {50, 10, 40, 50, 10, 40});
  QuarterHoldings prev, cur;
  prev.quarter = "2025Q3";
  cur.quarter = "2025Q4";
  prev.rows = {{1, "REV", 1000, 5000}, {2, "REV", 2000, 10000}, {1, "SML", 100, 1100}, {1, "BUY", 10, 800},
               {2, "BUY", 20, 1600}};
  cur.rows = {{1, "REV", 100, 5000}, {2, "REV", 200, 10000}, {1, "SML", 110, 1100}, {1, "BUY", 30, 1200},
              {2, "BUY", 60, 2400}};
  const QuarterPricing q = quarter_pricing(p, prev, cur);
  CHECK(q.ratio[0] == doctest::Approx(0.1));
  CHECK(q.ratio[1] == doctest::Approx(1.1));
  CHECK(q.ratio[2] == 1.0);
  CHECK(q.splits == 2);
  REQUIRE(q.unconfirmed.size() == 1);
  CHECK(q.unconfirmed[0].node == 2);
  CHECK(q.unconfirmed[0].price_ratio == doctest::Approx(2.0));
  CHECK(q.unconfirmed[0].share_ratio == doctest::Approx(3.0));
  CHECK(q.unconfirmed[0].value_usd == doctest::Approx(3600.0));  // q's 13F value
  auto ratio = [&](const std::string& t) { return q.ratio[t == "REV" ? 0 : t == "SML" ? 1 : 2]; };
  const ObservedFlows f = observed_flows(prev, cur, p.tickers, q.prices, ratio);
  // REV and SML: no trade after adjustment
  CHECK(f.unpaired_out == 0.0);
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
  REQUIRE(q.unconfirmed.size() == 1);  // listed for inspection
  CHECK(q.unconfirmed[0].value_usd == doctest::Approx(2150.0));
}

TEST_CASE("13f compare: a small price-factor shift with unchanged holder shares is not a split") {
  // r = 1.09 (above the 1.08 snap) but holders kept their share counts (share ratio 1.0): within 10% of r, yet
  // closer to 1 than to r, so no split; listed as a candidate.
  const std::vector<TimePoint> times = {utc_seconds(2025, 9, 30, 16), utc_seconds(2025, 12, 31, 16)};
  const Panel p = make_panel(times, {"DRF"}, {100, 100});
  QuarterHoldings prev, cur;
  prev.quarter = "2025Q3";
  cur.quarter = "2025Q4";
  prev.rows = {{1, "DRF", 10, 1090}, {2, "DRF", 20, 2180}};
  cur.rows = {{1, "DRF", 10, 1000}, {2, "DRF", 20, 2000}};
  const QuarterPricing q = quarter_pricing(p, prev, cur);
  CHECK(q.ratio[0] == 1.0);
  CHECK(q.splits == 0);
  CHECK(q.unconfirmed.size() == 1);
}

TEST_CASE("13f compare: split confirmed by the fraction of holders at exactly r") {
  // 2:1 split; 4 of 10 holders did not trade (share ratio exactly 2), 6 bought 20-25% more post-split shares, so the
  // median share ratio (2.4) misses r by more than 10%; 40% of holders at r (>= 30%) confirms it. Likewise a 1:10
  // reverse split where 6 of 10 holders sold 30%.
  const std::vector<TimePoint> times = {utc_seconds(2025, 9, 30, 16), utc_seconds(2025, 12, 31, 16)};
  const Panel p = make_panel(times, {"SPL", "REV"}, {50, 50, 50, 50});
  QuarterHoldings prev, cur;
  prev.quarter = "2025Q3";
  cur.quarter = "2025Q4";
  for (std::uint64_t m = 1; m <= 10; ++m) {
    const double buy = m <= 4 ? 1.0 : m == 10 ? 1.25 : 1.2;
    prev.rows.push_back({m, "SPL", 100, 100 * 100.0});  // raw 100
    cur.rows.push_back({m, "SPL", 200 * buy, 200 * buy * 50.0});
    const double sell = m <= 4 ? 1.0 : 0.7;
    prev.rows.push_back({m, "REV", 1000, 1000 * 5.0});  // raw 5
    cur.rows.push_back({m, "REV", 100 * sell, 100 * sell * 50.0});
  }
  const QuarterPricing q = quarter_pricing(p, prev, cur);
  CHECK(q.ratio[0] == doctest::Approx(2.0));
  CHECK(q.ratio[1] == doctest::Approx(0.1));
  CHECK(q.splits == 2);
  CHECK(q.unconfirmed.empty());
}

TEST_CASE("13f compare: the holder-fraction split rule needs at least 4 holders and 2 exact matches") {
  // Each ticker has a 2:1 price-factor ratio; holders' share ratios are listed below. The median never confirms
  // (it misses r by more than 10%), so only the fraction rule can.
  //   TWO: 2 holders, 1 exact (50%)          -> too few holders: unconfirmed
  //   THR: 3 holders, 1 exact (33%)          -> too few holders: unconfirmed
  //   FOR: 4 holders, 2 exact (50%), median 2.25 -> confirmed
  const std::vector<TimePoint> times = {utc_seconds(2025, 9, 30, 16), utc_seconds(2025, 12, 31, 16)};
  const std::vector<std::string> tk = {"TWO", "THR", "FOR"};
  const Panel p = make_panel(times, tk, {50, 50, 50, 50, 50, 50});
  const std::vector<std::vector<double>> ratios = {{2.0, 2.6}, {2.0, 2.5, 2.6}, {2.0, 2.0, 2.5, 2.6}};
  QuarterHoldings prev, cur;
  prev.quarter = "2025Q3";
  cur.quarter = "2025Q4";
  for (std::size_t i = 0; i < tk.size(); ++i)
    for (std::size_t m = 0; m < ratios[i].size(); ++m) {
      const std::uint64_t cik = m + 1;
      prev.rows.push_back({cik, tk[i], 100, 100 * 100.0});  // raw 100
      cur.rows.push_back({cik, tk[i], 100 * ratios[i][m], 100 * ratios[i][m] * 50.0});
    }
  const QuarterPricing q = quarter_pricing(p, prev, cur);
  CHECK(q.ratio[0] == 1.0);
  CHECK(q.ratio[1] == 1.0);
  CHECK(q.ratio[2] == doctest::Approx(2.0));
  CHECK(q.splits == 1);
  REQUIRE(q.unconfirmed.size() == 2);
}

TEST_CASE("13f compare: sparse full-pair Spearman and row cosine match a dense brute force") {
  // Random sparse matrices with zeros, duplicate pairs, self-pairs and tied weights (adapted from the reviewer's
  // cross-check): the sparse rank formula must equal Spearman over the explicit n(n-1) vectors.
  std::mt19937_64 rng(1);
  auto dense = [](const std::vector<FlowEdge>& e, std::size_t n) {
    std::vector<double> d(n * n, 0.0);
    for (const auto& x : e)
      if (x.from != x.to && x.dollars > 0) d[x.from * n + x.to] += x.dollars;
    return d;
  };
  double worst = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const std::size_t n = 5 + rng() % 60, m = rng() % (n * n / 2 + 1);
    auto gen = [&](bool ties) {
      std::vector<FlowEdge> e;
      for (std::size_t k = 0; k < m; ++k) {
        const auto i = static_cast<std::uint32_t>(rng() % n), j = static_cast<std::uint32_t>(rng() % n);
        e.push_back({i, j, ties ? double(1 + rng() % 4) : std::exp(static_cast<double>(rng() % 1000) / 100.0)});
      }
      return e;
    };
    const auto o = gen(trial % 2), e = gen(trial % 3 == 0);
    const Agreement a = compare_flows(o, e, n, 0);
    const auto O = dense(o, n), E = dense(e, n);
    std::vector<double> x, y;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        if (i != j) x.push_back(O[i * n + j]), y.push_back(E[i * n + j]);
    const double ref = spearman(x, y);
    auto shares = [n](std::vector<double> M) {
      for (std::size_t i = 0; i < n; ++i) {
        double s = 0;
        for (std::size_t j = 0; j < n; ++j) s += M[i * n + j];
        if (s > 0)
          for (std::size_t j = 0; j < n; ++j) M[i * n + j] /= s;
      }
      return M;
    };
    const auto So = shares(O), Se = shares(E);
    double d = 0, xx = 0, yy = 0;
    for (std::size_t k = 0; k < n * n; ++k) d += So[k] * Se[k], xx += So[k] * So[k], yy += Se[k] * Se[k];
    const double cref = xx > 0 && yy > 0 ? d / std::sqrt(xx * yy) : std::nan("");
    auto diff = [](double p, double q) { return std::isnan(p) && std::isnan(q) ? 0.0 : std::abs(p - q); };
    worst = std::max({worst, diff(a.est.full_spearman, ref), diff(a.est.row_cosine, cref)});
  }
  CHECK(worst < 1e-12);
}

TEST_CASE("13f compare: the warm-up counts prior returns, not prior bars") {
  // corr_window = 60: 2025Q4's first bar needs 60 returns before it, i.e. first bar index >= 61.
  auto run = [](TimePoint start) {
    const std::vector<std::string> tickers = {"AAA", "BBB", "CCC"};
    std::vector<TimePoint> times;
    std::vector<double> close;
    std::mt19937_64 rng(8);
    std::normal_distribution<double> z(0.0, 0.02);
    std::vector<double> px = {50, 80, 20};
    for (TimePoint t = start; t < utc_seconds(2026, 1, 3); t += 86400) {
      times.push_back(t);
      for (auto& x : px) close.push_back(x *= std::exp(z(rng)));
    }
    const auto data = test::temp_dir("compare13f_warm");
    test::write_file(data / "13f" / "cusip_map.csv", "cusip,ticker\nC1,AAA\nC2,BBB\nC3,CCC\n");
    const std::string hdr = "cik,cusip,issuer,shares,value_usd\n";
    test::write_file(data / "13f" / "holdings_2025Q3.csv", hdr + "1,C1,a,100,5000\n1,C2,b,100,8000\n");
    test::write_file(data / "13f" / "holdings_2025Q4.csv", hdr + "1,C1,a,50,2500\n1,C3,c,100,2000\n");
    Compare13fOptions opt;
    opt.data = data;
    opt.perms = 1;
    opt.quarters = {"2025Q4"};
    std::ostringstream log;
    return run_compare_13f(make_panel(times, tickers, close), opt, log);
  };
  const auto exact = run(utc_seconds(2025, 8, 2, 20));  // 2025-10-01 is bar 60: 59 prior returns
  CHECK(exact["quarters"].empty());
  REQUIRE(exact["skipped"].size() == 1);
  CHECK(exact["skipped"][0]["reason"].get<std::string>().find("warm-up") != std::string::npos);
  const auto ok = run(utc_seconds(2025, 8, 1, 20));  // bar 61: 60 prior returns
  REQUIRE(ok["quarters"].size() == 1);
  CHECK(ok["quarters"][0]["warmup_returns"] == 60);
}
