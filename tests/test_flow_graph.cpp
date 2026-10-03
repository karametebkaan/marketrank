#include <doctest/doctest.h>

#include "server/flow_graph.hpp"

using namespace mr;

namespace {
// Rows: 0 -> {1: 5, 2: 3, 3: 1}, 1 -> {0: 2, 2: 4}, 2 -> {0: 7}, 3 -> {} ; node 4 inactive with an edge 4 -> 0.
Csr toy() {
  Csr P;
  P.n = 5;
  P.row_ptr = {0, 3, 5, 6, 6, 7};
  P.col = {1, 2, 3, 0, 2, 0, 0};
  P.raw = {5, 3, 1, 2, 4, 7, 9};
  P.val.assign(P.raw.size(), 0.0);
  return P;
}
bool has(const std::vector<FlowNeighbours::Edge>& e, std::uint32_t a, std::uint32_t b) {
  for (const auto& x : e)
    if (x.a == a && x.b == b) return true;
  return false;
}
}  // namespace

TEST_CASE("flow_neighbours keeps each active node's top-k out- and in-edges, deduplicated and sorted") {
  const std::vector<bool> act{true, true, true, true, false};
  const FlowNeighbours g = flow_neighbours(toy(), act, 1);
  // top-1 out: 0->1 (5), 1->2 (4), 2->0 (7); top-1 in: into 0: 2->0 (7), into 1: 0->1, into 2: 1->2 (4), into 3: 0->3
  CHECK(g.edges.size() == 4);
  CHECK(has(g.edges, 0, 1));
  CHECK(has(g.edges, 1, 2));
  CHECK(has(g.edges, 2, 0));
  CHECK(has(g.edges, 0, 3));
  CHECK_FALSE(has(g.edges, 4, 0));  // inactive source
  for (std::size_t k = 1; k < g.edges.size(); ++k)
    CHECK((g.edges[k - 1].a < g.edges[k].a || (g.edges[k - 1].a == g.edges[k].a && g.edges[k - 1].b < g.edges[k].b)));
  CHECK(g.out_total[0] == doctest::Approx(9.0));
  CHECK(g.out_total[4] == 0.0f);
  CHECK(flow_neighbours(toy(), act, 10).edges.size() == 6);  // every active kept edge
}

TEST_CASE("flow_subgraph: focus nodes, their k strongest neighbours, and all stored edges among them") {
  const std::vector<bool> act{true, true, true, true, false};
  const FlowNeighbours g = flow_neighbours(toy(), act, 10);
  const std::vector<std::uint32_t> focus{3};
  const FlowSubgraph s = flow_subgraph(g, focus, 1);
  CHECK(s.nodes == std::vector<std::uint32_t>{0, 3});  // 3's only neighbour is 0
  REQUIRE(s.edges.size() == 1);
  CHECK((s.edges[0].a == 0 && s.edges[0].b == 3));
  const std::vector<std::uint32_t> f0{0};
  const FlowSubgraph s0 = flow_subgraph(g, f0, 2);  // 0's strongest: 2 (7), 1 (5)
  CHECK(s0.nodes == std::vector<std::uint32_t>{0, 1, 2});
  CHECK(s0.edges.size() == 5);  // 0->1, 0->2, 1->0, 1->2, 2->0
}
