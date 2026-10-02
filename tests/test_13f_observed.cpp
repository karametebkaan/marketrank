#include <doctest/doctest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

#include "flows13f/observed.hpp"

using namespace mr;

namespace {
const double NaN = std::numeric_limits<double>::quiet_NaN();
const std::vector<std::string> T = {"A", "B", "C"};
const std::vector<double> P = {10.0, 10.0, 10.0};
double one(const std::string&) { return 1.0; }

QuarterHoldings Q(const std::string& q, std::vector<Holding> rows) {
  QuarterHoldings h;
  h.quarter = q;
  h.rows = std::move(rows);
  return h;
}
double edge(const ObservedFlows& f, std::uint32_t a, std::uint32_t b) {
  for (auto& e : f.edges) if (e.from == a && e.to == b) return e.dollars;
  return 0;
}
}  // namespace

TEST_CASE("observed: single manager balanced") {
  auto prev = Q("2024Q1", {{1, "A", 20, 200}, {1, "B", 0, 0}, {1, "C", 0, 0}});
  auto cur = Q("2024Q2", {{1, "A", 10, 100}, {1, "B", 6, 60}, {1, "C", 4, 40}});
  auto f = observed_flows(prev, cur, T, P, one);
  CHECK(f.edges.size() == 2);
  CHECK(edge(f, 0, 1) == doctest::Approx(60));
  CHECK(edge(f, 0, 2) == doctest::Approx(40));
  CHECK(f.paired == doctest::Approx(100));
  CHECK(f.unpaired_in == doctest::Approx(0));
  CHECK(f.unpaired_out == doctest::Approx(0));
  CHECK(f.managers == 1);
}

TEST_CASE("observed: adds exceed cuts") {
  auto prev = Q("p", {{1, "A", 10, 100}});
  auto cur = Q("c", {{1, "A", 5, 50}, {1, "B", 10, 100}});  // out 50, in 100
  auto f = observed_flows(prev, cur, T, P, one);
  CHECK(edge(f, 0, 1) == doctest::Approx(50));
  CHECK(f.paired == doctest::Approx(50));
  CHECK(f.unpaired_in == doctest::Approx(50));
  CHECK(f.unpaired_out == doctest::Approx(0));
}

TEST_CASE("observed: cuts exceed adds") {
  auto prev = Q("p", {{1, "A", 20, 200}});
  auto cur = Q("c", {{1, "A", 0, 0}, {1, "B", 5, 50}});
  auto f = observed_flows(prev, cur, T, P, one);
  CHECK(edge(f, 0, 1) == doctest::Approx(50));
  CHECK(f.unpaired_out == doctest::Approx(150));
}

TEST_CASE("observed: two managers summed") {
  auto prev = Q("p", {{1, "A", 10, 100}, {2, "A", 5, 50}});
  auto cur = Q("c", {{1, "B", 10, 100}, {2, "B", 5, 50}});
  auto f = observed_flows(prev, cur, T, P, one);
  CHECK(f.edges.size() == 1);
  CHECK(edge(f, 0, 1) == doctest::Approx(150));
  CHECK(f.managers == 2);
}

TEST_CASE("observed: split adjustment") {
  auto prev = Q("p", {{1, "A", 10, 100}});
  auto cur = Q("c", {{1, "A", 20, 100}});
  auto f = observed_flows(prev, cur, T, P, [](const std::string& t) { return t == "A" ? 2.0 : 1.0; });
  CHECK(f.edges.empty());
  CHECK(f.paired == doctest::Approx(0));
  CHECK(f.unpaired_in == doctest::Approx(0));
  CHECK(f.unpaired_out == doctest::Approx(0));
}

TEST_CASE("observed: new buy and full sell") {
  auto prev = Q("p", {{1, "A", 10, 100}});
  auto cur = Q("c", {{1, "C", 10, 100}});
  auto f = observed_flows(prev, cur, T, P, one);
  CHECK(edge(f, 0, 2) == doctest::Approx(100));
}

TEST_CASE("observed: deterministic order") {
  auto prev = Q("p", {{2, "C", 10, 100}, {1, "A", 10, 100}, {1, "B", 10, 100}});
  auto cur = Q("c", {{2, "A", 10, 100}, {1, "C", 20, 200}});
  auto f1 = observed_flows(prev, cur, T, P, one);
  auto f2 = observed_flows(prev, cur, T, P, one);
  REQUIRE(f1.edges.size() == f2.edges.size());
  for (std::size_t i = 0; i < f1.edges.size(); ++i) {
    CHECK(f1.edges[i].from == f2.edges[i].from);
    CHECK(f1.edges[i].to == f2.edges[i].to);
    CHECK(f1.edges[i].dollars == f2.edges[i].dollars);
    if (i) CHECK(std::make_pair(f1.edges[i - 1].from, f1.edges[i - 1].to) < std::make_pair(f1.edges[i].from, f1.edges[i].to));
  }
  CHECK(f1.paired == f2.paired);
}

TEST_CASE("observed: fallback price when lake price NaN") {
  std::vector<double> p = {NaN, NaN, 10.0};
  auto prev = Q("p", {{1, "A", 10, 200}});  // fallback 20/share
  auto cur = Q("c", {{1, "A", 5, 100}, {1, "B", 1, 7}, {1, "C", 3, 30}});
  auto f = observed_flows(prev, cur, T, p, one);
  // A: -5 * 20 = -100 ; B: +1*7 = 7 ; C: +3*10 = 30
  CHECK(f.paired == doctest::Approx(37));
  CHECK(f.unpaired_out == doctest::Approx(63));
  CHECK(edge(f, 0, 1) == doctest::Approx(7));
}

TEST_CASE("observed: large manager bounded") {
  std::vector<std::string> tk;
  std::vector<double> pr;
  QuarterHoldings prev, cur;
  for (int i = 0; i < 2500; ++i) { tk.push_back("S" + std::to_string(i)); pr.push_back(1.0); }
  for (int i = 0; i < 1200; ++i) prev.rows.push_back({1, tk[i], 100, 100});
  for (int i = 1200; i < 2500; ++i) cur.rows.push_back({1, tk[i], 10.0 + i, 0});
  auto f = observed_flows(prev, cur, tk, pr, one);
  CHECK(f.edges.size() <= 1200u * 1000u);
  CHECK(f.paired > 0);
  double s = 0;
  for (auto& e : f.edges) s += e.dollars;
  CHECK(s == doctest::Approx(f.paired));
}

TEST_CASE("load_quarter joins cusip map and drops unmapped") {
  auto dir = std::filesystem::temp_directory_path() / "mr_13f_obs_test";
  std::filesystem::create_directories(dir / "13f");
  {
    std::ofstream h(dir / "13f" / "holdings_2024Q1.csv");
    h << "cik,cusip,issuer,shares,value_usd\n1,111,AAA,10,100\n1,222,BBB,5,50\n2,111,AAA,1,10\n";
    std::ofstream m(dir / "13f" / "cusip_map.csv");
    m << "cusip,ticker,name,security_type,figi,fetched_at\n111,AAA,x,Common,f,t\n222,,,,,t\n";
  }
  auto q = load_quarter(dir, "2024Q1");
  CHECK(q.rows.size() == 2);
  CHECK(q.total_value == doctest::Approx(160));
  CHECK(q.dropped_value == doctest::Approx(50));
  std::filesystem::remove_all(dir);
}
