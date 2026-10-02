#include "pipeline/graph_slice.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "core/time.hpp"
#include "graph/csr.hpp"
#include "graph/markov_solver.hpp"

namespace mr {

GraphSlice pick_slice(const FluxAccumulator& slow, const Frame& f, std::size_t n, double alpha) {
  if (n == 0) throw std::invalid_argument("pick_slice: n must be >= 1");
  const std::size_t N = slow.size();
  if (f.active.size() != N || f.pi.size() != N) throw std::invalid_argument("pick_slice: frame size mismatch");
  GraphSlice s;
  s.t = f.t;
  s.alpha = alpha;
  std::size_t top = N;
  for (std::size_t i = 0; i < N; ++i) {
    if (!f.active[i]) continue;
    ++s.n_active;
    if (top == N || f.pi[i] > f.pi[top]) top = i;
  }
  if (top == N) throw std::runtime_error("pick_slice: no active node");
  // Combined raw flux between the top node and each active partner, both directions.
  std::map<std::size_t, double> strength;
  for (const auto& e : slow.rows()[top])
    if (e.j != top && f.active[e.j] && e.w > 0) strength[e.j] += e.w;
  for (std::size_t i = 0; i < N; ++i) {
    if (i == top || !f.active[i]) continue;
    for (const auto& e : slow.rows()[i])
      if (e.j == top && e.w > 0) strength[i] += e.w;
  }
  std::vector<std::pair<std::size_t, double>> partners(strength.begin(), strength.end());
  std::stable_sort(partners.begin(), partners.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  s.nodes.push_back(top);
  for (std::size_t k = 0; k < partners.size() && s.nodes.size() < n; ++k) s.nodes.push_back(partners[k].first);
  std::map<std::size_t, std::size_t> pos;
  for (std::size_t k = 0; k < s.nodes.size(); ++k) pos[s.nodes[k]] = k;
  for (std::size_t a = 0; a < s.nodes.size(); ++a) {
    const std::size_t i = s.nodes[a];
    s.pi.push_back(f.pi[i]);
    s.mr.push_back(market_rank_score(f.pi[i], s.n_active));
    for (const auto& e : slow.rows()[i]) {
      auto it = pos.find(e.j);
      if (it == pos.end() || e.j == i || !(e.w > 0)) continue;
      s.edges.push_back({a, it->second, e.w});
    }
  }
  s.slice_pi = slice_market_rank(s.nodes.size(), s.edges, alpha);
  return s;
}

std::vector<double> slice_market_rank(std::size_t k, const std::vector<SliceEdge>& edges, double alpha) {
  std::vector<double> dense(k * k, 0.0), rowsum(k, 0.0);
  for (const auto& e : edges) {
    if (e.from >= k || e.to >= k) throw std::invalid_argument("slice_market_rank: edge out of range");
    if (e.from == e.to || !(e.dollars > 0)) continue;
    dense[e.from * k + e.to] += e.dollars;
    rowsum[e.from] += e.dollars;
  }
  Csr P;
  P.n = k;
  P.row_ptr.push_back(0);
  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < k; ++j)
      if (dense[i * k + j] > 0) {
        P.col.push_back(static_cast<std::uint32_t>(j));
        P.val.push_back(dense[i * k + j] / rowsum[i]);
        P.raw.push_back(dense[i * k + j]);
      }
    P.row_ptr.push_back(P.col.size());
  }
  const SolveResult r = stationary(P, alpha, {}, 1e-13, 10000);
  if (!r.converged) throw std::runtime_error("slice_market_rank: did not converge");
  return r.pi;
}

GraphSlice export_slice(const Panel& panel, const CoreParams& params, std::size_t n) {
  if (panel.T() < 2) throw std::runtime_error("export_slice: need at least two bars");
  CorePipeline pipe(panel.N(), params);
  Frame f;
  for (std::size_t t = 1; t < panel.T(); ++t) f = pipe.step(panel, t);
  return pick_slice(pipe.slow_flux(), f, n, params.alpha);
}

nlohmann::json slice_json(const GraphSlice& s, const std::vector<Security>& universe, const std::string& bar,
                          const std::string& preset) {
  nlohmann::json j;
  j["t"] = format_rfc3339(s.t);
  j["bar"] = bar;
  j["preset"] = preset;
  j["p"] = 1.0 - s.alpha;
  j["n_active"] = s.n_active;
  j["tickers"] = nlohmann::json::array();
  j["sectors"] = nlohmann::json::array();
  for (std::size_t i : s.nodes) {
    j["tickers"].push_back(universe.at(i).ticker);
    j["sectors"].push_back(universe.at(i).sector);
  }
  j["pi"] = s.pi;
  j["mr"] = s.mr;
  j["slice_pi"] = s.slice_pi;
  j["edges"] = nlohmann::json::array();
  for (const auto& e : s.edges)
    j["edges"].push_back({{"from", universe.at(s.nodes[e.from]).ticker},
                          {"to", universe.at(s.nodes[e.to]).ticker},
                          {"dollars", e.dollars}});
  return j;
}

}  // namespace mr
