#include "graph/sparse_flux.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace fx {
namespace {

double dot(const double* a, const double* b, std::size_t w) {
  double s = 0;
  for (std::size_t k = 0; k < w; ++k) s += a[k] * b[k];
  return s;
}

bool by_weight_desc(const WEdge& x, const WEdge& y) {
  return x.w > y.w || (x.w == y.w && x.j < y.j);
}

bool by_column(const WEdge& x, const WEdge& y) { return x.j < y.j; }

}  // namespace

BarFlux bar_flux_sparse(std::span<const double> pressure, std::span<const double> unit,
                        std::size_t w, const SparseFluxParams& params) {
  const std::size_t n = pressure.size();
  BarFlux bf;
  bf.rows.resize(n);
  bf.out.assign(n, 0.0);
  bf.in.assign(n, 0.0);
  const bool affinity = params.lambda > 0 && w > 0 && unit.size() == n * w;
  const double lambda = affinity ? params.lambda : 0.0;

  std::vector<std::uint32_t> sinks, sources;
  double P = 0;
  for (std::size_t k = 0; k < n; ++k) {
    if (pressure[k] > 0) {
      sinks.push_back(static_cast<std::uint32_t>(k));
      P += pressure[k];
    } else if (pressure[k] < 0) {
      sources.push_back(static_cast<std::uint32_t>(k));
    }
  }
  if (sinks.empty() || sources.empty()) return bf;

  std::vector<double> g(affinity ? w : 0, 0.0);  // g = sum_k p_k u_k
  for (std::size_t s = 0; s < g.size(); ++s)
    for (auto k : sinks) g[s] += pressure[k] * unit[k * w + s];

  std::vector<std::uint32_t> cand(sinks);
  std::sort(cand.begin(), cand.end(), [&](auto a, auto b) {
    return pressure[a] > pressure[b] || (pressure[a] == pressure[b] && a < b);
  });
  if (cand.size() > params.sink_candidates) cand.resize(params.sink_candidates);

  // Per source: a_i = |p_i| / D_i and its kept edges (parallel, disjoint writes).
  std::vector<double> a_of(n, 0.0);
#pragma omp parallel
  {
    std::vector<WEdge> scored;
#pragma omp for schedule(dynamic, 64)
    for (std::size_t si = 0; si < sources.size(); ++si) {
      const std::size_t i = sources[si];
      const double* ui = affinity ? &unit[i * w] : nullptr;
      const double denom = P + (affinity ? lambda * dot(ui, g.data(), w) : 0.0);
      if (!(denom > 0)) continue;
      const double outflow = -pressure[i];
      const double a = outflow / denom;
      a_of[i] = a;
      bf.out[i] = outflow;
      scored.clear();
      for (auto j : cand) {
        const double score =
            pressure[j] * (1.0 + (affinity ? lambda * dot(ui, &unit[j * w], w) : 0.0));
        if (score > 0) scored.push_back({j, score});
      }
      const std::size_t m = std::min(params.sinks_per_source, scored.size());
      std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(m),
                        scored.end(), by_weight_desc);
      auto& row = bf.rows[i];
      row.reserve(m);
      for (std::size_t e = 0; e < m; ++e) row.push_back({scored[e].j, a * scored[e].w});
      std::sort(row.begin(), row.end(), by_column);
    }
  }

  // Serial, index-ordered reductions (determinism): A = sum a_i, Au = sum a_i u_i.
  double A = 0;
  std::vector<double> Au(affinity ? w : 0, 0.0);
  for (auto i : sources) {
    if (a_of[i] == 0) continue;
    A += a_of[i];
    for (std::size_t s = 0; s < Au.size(); ++s) Au[s] += a_of[i] * unit[i * w + s];
  }

#pragma omp parallel for schedule(static)
  for (std::size_t sk = 0; sk < sinks.size(); ++sk) {
    const std::size_t j = sinks[sk];
    bf.in[j] = pressure[j] * (A + (affinity ? lambda * dot(&unit[j * w], Au.data(), w) : 0.0));
  }
  return bf;
}

FluxAccumulator::FluxAccumulator(std::size_t n, double halflife, std::size_t row_cap)
    : n_(n), decay_(std::exp2(-1.0 / halflife)), cap_(row_cap), rows_(n), out_(n, 0.0), in_(n, 0.0) {
  if (!(halflife > 0)) throw std::invalid_argument("FluxAccumulator: halflife must be > 0");
  if (row_cap == 0) throw std::invalid_argument("FluxAccumulator: row_cap must be >= 1");
}

void FluxAccumulator::add(const BarFlux& bar) {
#pragma omp parallel
  {
    std::vector<WEdge> merged;
#pragma omp for schedule(dynamic, 64)
    for (std::size_t i = 0; i < n_; ++i) {
      out_[i] = decay_ * out_[i] + bar.out[i];
      in_[i] = decay_ * in_[i] + bar.in[i];
      auto& row = rows_[i];
      for (auto& e : row) e.w *= decay_;
      const auto& add = bar.rows[i];
      if (add.empty()) continue;
      merged.clear();
      merged.reserve(row.size() + add.size());
      std::size_t a = 0, b = 0;
      while (a < row.size() || b < add.size()) {
        if (b == add.size() || (a < row.size() && row[a].j < add[b].j)) {
          merged.push_back(row[a++]);
        } else if (a == row.size() || add[b].j < row[a].j) {
          merged.push_back(add[b++]);
        } else {
          merged.push_back({row[a].j, row[a].w + add[b].w});
          ++a;
          ++b;
        }
      }
      if (merged.size() > cap_) {
        std::nth_element(merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(cap_),
                         merged.end(), by_weight_desc);
        merged.resize(cap_);
        std::sort(merged.begin(), merged.end(), by_column);
      }
      row.swap(merged);
    }
  }
}

double FluxAccumulator::total() const {
  double s = 0;
  for (double x : out_) s += x;
  return s;
}

std::size_t FluxAccumulator::edge_count() const {
  std::size_t c = 0;
  for (const auto& r : rows_) c += r.size();
  return c;
}

}  // namespace fx
