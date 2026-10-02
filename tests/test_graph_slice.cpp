// --export-slice: the README's real-data figure. Slice selection on a synthetic panel and the slice's own
// MarketRank against a direct dense power iteration.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "cli/args.hpp"
#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/core_pipeline.hpp"
#include "pipeline/graph_slice.hpp"
#include "test_util.hpp"

using namespace mr;

namespace {
// Damped power iteration on the dense row-normalized matrix, dangling rows teleporting uniformly.
std::vector<double> dense_power(std::size_t k, const std::vector<SliceEdge>& edges, double alpha) {
  std::vector<double> T(k * k, 0.0), row(k, 0.0);
  for (const auto& e : edges) {
    T[e.from * k + e.to] += e.dollars;
    row[e.from] += e.dollars;
  }
  std::vector<double> x(k, 1.0 / static_cast<double>(k));
  for (int it = 0; it < 5000; ++it) {
    std::vector<double> y(k, 0.0);
    double dangling = 0;
    for (std::size_t i = 0; i < k; ++i) {
      if (row[i] == 0) {
        dangling += x[i];
        continue;
      }
      for (std::size_t j = 0; j < k; ++j) y[j] += x[i] * T[i * k + j] / row[i];
    }
    for (double& v : y) v = alpha * v + (alpha * dangling + 1.0 - alpha) / static_cast<double>(k);
    x = y;
  }
  return x;
}
}  // namespace

TEST_CASE("export_slice: top-pi stock and its partners on a synthetic panel") {
  SyntheticConfig cfg;
  BarStore store(test::temp_dir("slice"));
  const auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (const auto& s : secs) tickers.push_back(s.ticker);
  const Panel panel = build_panel(store, tickers, cfg.tf);
  const CoreParams params = CoreParams::market_rank();
  const GraphSlice s = export_slice(panel, params, 6);
  const Frame f = run_panel_last(panel, params);

  REQUIRE(s.nodes.size() == 6);
  CHECK(std::set<std::size_t>(s.nodes.begin(), s.nodes.end()).size() == 6);
  // nodes[0] is the top-pi node of the latest frame
  const auto top = static_cast<std::size_t>(std::max_element(f.pi.begin(), f.pi.end()) - f.pi.begin());
  CHECK(s.nodes[0] == top);
  CHECK(s.t == panel.times.back());
  for (std::size_t k = 0; k < s.nodes.size(); ++k) {
    CHECK(f.active[s.nodes[k]]);
    CHECK(s.pi[k] == f.pi[s.nodes[k]]);
    CHECK(s.mr[k] == doctest::Approx(f.pi[s.nodes[k]] * static_cast<double>(s.n_active)));
  }
  REQUIRE_FALSE(s.edges.empty());
  bool touches_top = false;
  for (const auto& e : s.edges) {
    CHECK(e.from < 6);
    CHECK(e.to < 6);
    CHECK(e.from != e.to);
    CHECK(e.dollars > 0);
    touches_top = touches_top || e.from == 0 || e.to == 0;
  }
  CHECK(touches_top);

  // slice_pi sums to 1 and matches a direct power iteration on the slice
  REQUIRE(s.slice_pi.size() == 6);
  double sum = 0;
  for (double x : s.slice_pi) sum += x;
  CHECK(sum == doctest::Approx(1.0).epsilon(1e-12));
  const auto direct = dense_power(6, s.edges, 0.85);
  for (std::size_t k = 0; k < 6; ++k) CHECK(std::abs(s.slice_pi[k] - direct[k]) < 1e-9);

  const auto j = slice_json(s, secs, "1d", "marketrank");
  CHECK(j["tickers"].size() == 6);
  CHECK(j["sectors"].size() == 6);
  CHECK(j["edges"].size() == s.edges.size());
  CHECK(j["p"].get<double>() == doctest::Approx(0.15));
}

TEST_CASE("slice_market_rank reproduces the A/B/C worked example") {
  // A=0, B=1, C=2; T in dollars, row = from.
  const std::vector<SliceEdge> e = {{0, 1, 500e3}, {0, 2, 100e3}, {1, 0, 200e3},
                                    {1, 2, 1e6},   {2, 0, 750e3}, {2, 1, 400e3}};
  const auto pi = slice_market_rank(3, e, 0.85);
  CHECK(std::abs(pi[1] - 0.36016) < 5e-5);
  CHECK(std::abs(pi[2] - 0.34665) < 5e-5);
  CHECK(std::abs(pi[0] - 0.29319) < 5e-5);
}

TEST_CASE("--export-slice parsing") {
  CHECK(parse_cli({"--mode", "replay", "--export-slice"}).export_slice == 6);
  const CliArgs a = parse_cli({"--mode", "replay", "--export-slice", "4", "--slice-out", "x.json"});
  CHECK(a.export_slice == 4);
  CHECK(a.slice_out == "x.json");
  CHECK_THROWS(parse_cli({"--export-slice"}));  // synthetic mode
  CHECK_THROWS(parse_cli({"--mode", "replay", "--export-slice", "0"}));
}
