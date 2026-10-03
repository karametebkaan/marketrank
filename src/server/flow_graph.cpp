#include "server/flow_graph.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace mr {

namespace {
// Strongest first; ties to the lower index, so the choice is deterministic.
bool stronger(double wa, std::uint32_t ia, double wb, std::uint32_t ib) { return wa != wb ? wa > wb : ia < ib; }
}  // namespace

FlowNeighbours flow_neighbours(const Csr& P, const std::vector<bool>& active, std::size_t k) {
  if (active.size() != P.n) throw std::invalid_argument("flow_neighbours: active size != P.n");
  FlowNeighbours g;
  g.out_total.assign(P.n, 0.0f);
  std::vector<std::vector<std::pair<double, std::uint32_t>>> in(P.n);  // per target: (raw, source)
  std::vector<FlowNeighbours::Edge> e;
  std::vector<std::pair<double, std::uint32_t>> row;
  for (std::size_t i = 0; i < P.n; ++i) {
    if (!active[i]) continue;
    row.clear();
    double tot = 0;
    for (auto p = P.row_ptr[i]; p < P.row_ptr[i + 1]; ++p) {
      const auto j = P.col[p];
      if (j == i || !active[j] || !(P.raw[p] > 0)) continue;
      tot += P.raw[p];
      row.push_back({P.raw[p], j});
      in[j].push_back({P.raw[p], static_cast<std::uint32_t>(i)});
    }
    g.out_total[i] = static_cast<float>(tot);
    const std::size_t m = std::min(k, row.size());
    std::partial_sort(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(m), row.end(),
                      [](const auto& x, const auto& y) { return stronger(x.first, x.second, y.first, y.second); });
    for (std::size_t q = 0; q < m; ++q)
      e.push_back({static_cast<std::uint32_t>(i), row[q].second, static_cast<float>(row[q].first)});
  }
  for (std::size_t j = 0; j < P.n; ++j) {
    auto& c = in[j];
    const std::size_t m = std::min(k, c.size());
    std::partial_sort(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(m), c.end(),
                      [](const auto& x, const auto& y) { return stronger(x.first, x.second, y.first, y.second); });
    for (std::size_t q = 0; q < m; ++q) e.push_back({c[q].second, static_cast<std::uint32_t>(j), static_cast<float>(c[q].first)});
  }
  std::sort(e.begin(), e.end(), [](const auto& x, const auto& y) { return x.a != y.a ? x.a < y.a : x.b < y.b; });
  e.erase(std::unique(e.begin(), e.end(), [](const auto& x, const auto& y) { return x.a == y.a && x.b == y.b; }), e.end());
  e.shrink_to_fit();
  g.edges = std::move(e);
  return g;
}

FlowSubgraph flow_subgraph(const FlowNeighbours& g, std::span<const std::uint32_t> focus, std::size_t k) {
  std::set<std::uint32_t> nodes(focus.begin(), focus.end());
  for (const auto f : focus) {
    std::map<std::uint32_t, double> nb;  // neighbour -> strongest edge with f (either direction)
    for (const auto& x : g.edges) {
      if (x.a == f && x.b != f) nb[x.b] = std::max(nb[x.b], static_cast<double>(x.raw));
      else if (x.b == f && x.a != f) nb[x.a] = std::max(nb[x.a], static_cast<double>(x.raw));
    }
    std::vector<std::pair<double, std::uint32_t>> v;
    for (const auto& [j, w] : nb) v.push_back({w, j});
    std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return stronger(x.first, x.second, y.first, y.second); });
    for (std::size_t q = 0; q < std::min(k, v.size()); ++q) nodes.insert(v[q].second);
  }
  FlowSubgraph s;
  s.nodes.assign(nodes.begin(), nodes.end());
  for (const auto& x : g.edges)
    if (nodes.count(x.a) && nodes.count(x.b)) s.edges.push_back(x);
  return s;
}

}  // namespace mr
