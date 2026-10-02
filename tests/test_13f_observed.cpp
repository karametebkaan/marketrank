#include <doctest/doctest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <random>

#include "flows13f/observed.hpp"

using namespace mr;

namespace {
const double NaN = std::numeric_limits<double>::quiet_NaN();
const std::vector<std::string> T = {"A", "B", "C"};
const std::vector<double> P = {10.0, 10.0, 10.0};
double one(const std::string&) { return 1.0; }

QuarterHoldings Q(const std::string& q, std::vector<Holding13F> rows) {
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

TEST_CASE("observed: shuffled input is bit-identical") {
  std::vector<std::string> tk;
  std::vector<double> pr;
  for (int i = 0; i < 30; ++i) { tk.push_back("S" + std::to_string(i)); pr.push_back(1.0 + i * 0.37); }
  std::mt19937 g(5);
  QuarterHoldings prev, cur;
  for (int c = 1; c <= 8; ++c)
    for (int i = 0; i < 30; ++i) {
      prev.rows.push_back({std::uint64_t(c), tk[i], double(g() % 100), 1.0});
      cur.rows.push_back({std::uint64_t(c), tk[i], double(g() % 100), 1.0});
      if (i % 7 == 0) cur.rows.push_back({std::uint64_t(c), tk[i], 0.1 * (g() % 10), 1.0});
    }
  auto a = observed_flows(prev, cur, tk, pr, one);
  std::shuffle(prev.rows.begin(), prev.rows.end(), g);
  std::shuffle(cur.rows.begin(), cur.rows.end(), g);
  auto b = observed_flows(prev, cur, tk, pr, one);
  REQUIRE(a.edges.size() == b.edges.size());
  for (std::size_t i = 0; i < a.edges.size(); ++i) {
    CHECK(a.edges[i].from == b.edges[i].from);
    CHECK(a.edges[i].to == b.edges[i].to);
    CHECK(a.edges[i].dollars == b.edges[i].dollars);
  }
  CHECK(a.paired == b.paired);
}

TEST_CASE("observed: top_n selection and outside tallies") {
  // values: A 1000, B 500, C 10 (total across prev+cur). top_n = 2 -> {A,B}
  auto prev = Q("p", {{1, "A", 10, 500}, {1, "C", 1, 5}});
  auto cur = Q("c", {{1, "A", 5, 500}, {1, "B", 5, 500}, {1, "C", 3, 5}});
  ObservedParams pp;
  pp.top_n = 2;
  auto f = observed_flows(prev, cur, T, P, one, pp);
  REQUIRE(f.nodes.size() == 2);
  CHECK(f.nodes[0] == 0);
  CHECK(f.nodes[1] == 1);
  CHECK(edge(f, 0, 1) == doctest::Approx(50));
  CHECK(f.outside_in == doctest::Approx(20));
  CHECK(f.outside_out == doctest::Approx(0));
  CHECK(f.paired == doctest::Approx(50));
}

TEST_CASE("observed: top_n ties go to lower index") {
  auto prev = Q("p", {{1, "A", 1, 10}, {1, "B", 1, 10}, {1, "C", 1, 10}});
  auto cur = Q("c", {});
  ObservedParams pp;
  pp.top_n = 2;
  auto f = observed_flows(prev, cur, T, P, one, pp);
  REQUIRE(f.nodes.size() == 2);
  CHECK(f.nodes[0] == 0);
  CHECK(f.nodes[1] == 1);
  CHECK(f.outside_out == doctest::Approx(10));
}

TEST_CASE("observed: no cap bias with 2000 small sinks") {
  const int n = 2001;
  std::vector<std::string> tk;
  std::vector<double> pr(n, 1.0);
  for (int i = 0; i < n; ++i) tk.push_back("S" + std::to_string(i));
  QuarterHoldings prev, cur;
  prev.rows.push_back({1, tk[0], 1000, 1000});
  double tot = 0;
  for (int i = 1; i < n; ++i) { cur.rows.push_back({1, tk[i], double(i), 0}); tot += i; }
  ObservedParams pp;
  pp.top_n = 5000;
  auto f = observed_flows(prev, cur, tk, pr, one, pp);
  REQUIRE(f.edges.size() == 2000);
  for (auto& e : f.edges) CHECK(e.dollars == doctest::Approx(1000.0 * e.to / tot).epsilon(1e-9));
}

TEST_CASE("observed: prune_rel drops tiny edges") {
  auto prev = Q("p", {{1, "A", 1000000, 1}});
  auto cur = Q("c", {{1, "B", 999999, 1}, {1, "C", 1, 1}});
  ObservedParams pp;
  pp.prune_rel = 1e-3;
  auto f = observed_flows(prev, cur, T, P, one, pp);
  CHECK(f.edges.size() == 1);
}

TEST_CASE("observed: skipped_value and fallback rules") {
  std::vector<double> p = {NaN, NaN, NaN};
  // A: no price, cur shares 0 -> prev fallback = prev_value/(ratio*prev_shares) = 100/(2*10)=5
  // B: no price and zero shares/value -> skipped. Z: not in tickers -> skipped.
  auto prev = Q("p", {{1, "A", 10, 100}, {1, "B", 5, 0}, {1, "Z", 1, 33}});
  auto cur = Q("c", {{1, "C", 4, 40}});
  auto f = observed_flows(prev, cur, T, p, [](const std::string& t) { return t == "A" ? 2.0 : 1.0; });
  // A: d = (0 - 2*10)*5 = -100 ; C: +40
  CHECK(f.paired == doctest::Approx(40));
  CHECK(f.unpaired_out == doctest::Approx(60));
  CHECK(f.skipped_value == doctest::Approx(33));  // Z (B has value 0)
}

TEST_CASE("load_quarter edge cases") {
  auto dir = std::filesystem::temp_directory_path() / "mr_13f_obs_test2";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir / "13f");
  {
    std::ofstream h(dir / "13f" / "holdings_2024Q1.csv");
    h << "cik,cusip,issuer,shares,value_usd\n1,111,AAA,10,100\n1,111,AAA,5,50\nabc,111,AAA,1,1\n"
         "2,111,AAA,12x,5\n3,111\n";
  }
  CHECK_THROWS_AS(load_quarter(dir, "2024Q1"), std::runtime_error);  // no cusip_map
  {
    std::ofstream m(dir / "13f" / "cusip_map.csv");
    m << "cusip,ticker,name,security_type,figi,fetched_at\n111,AAA,x,Common,f,t\n";
  }
  auto q = load_quarter(dir, "2024Q1");
  CHECK(q.rows.size() == 2);  // duplicates kept as rows
  CHECK(q.bad_rows == 3);
  CHECK(q.total_value == doctest::Approx(150));
  std::filesystem::remove_all(dir);
}

TEST_CASE("observed: perf 2e6 holdings, top_n 2000") {
  const int nt = 2000, nm = 500, per = 2000;
  std::vector<std::string> tk;
  std::vector<double> pr;
  for (int i = 0; i < nt; ++i) { tk.push_back("S" + std::to_string(i)); pr.push_back(5.0 + i % 50); }
  std::mt19937 g(1);
  QuarterHoldings prev, cur;
  for (int m = 0; m < nm; ++m)
    for (int i = 0; i < per; ++i) {
      prev.rows.push_back({std::uint64_t(m), tk[i], double(g() % 1000), double(g() % 5000)});
      cur.rows.push_back({std::uint64_t(m), tk[i], double(g() % 1000), double(g() % 5000)});
    }
  auto t0 = std::chrono::steady_clock::now();
  auto f = observed_flows(prev, cur, tk, pr, one);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("[perf] observed_flows 2e6 holdings: %.2f s, %zu edges\n", sec, f.edges.size());
  CHECK(f.paired > 0);
#if defined(NDEBUG) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
  CHECK(sec < 10.0);
#endif
}
