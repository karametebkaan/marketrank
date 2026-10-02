#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fx {

struct WEdge {
  std::uint32_t j;
  double w;
};

struct BarFlux {
  std::vector<std::vector<WEdge>> rows;  // per source: kept edges, exact shares, ascending j
  std::vector<double> out;               // exact outflow per node
  std::vector<double> in;                // exact inflow per node (all source-sink pairs)
};

struct SparseFluxParams {
  double lambda = 1.0;                // affinity a_ij = 1 + lambda * rho_ij, lambda in [0, 1]
  std::size_t sink_candidates = 256;  // C: candidate sinks with the largest pressure
  std::size_t sinks_per_source = 64;  // M: edges kept per source
};

// Spec 5. unit: n x w unit vectors (ReturnWindow::unit_vectors) or empty for no affinity.
BarFlux bar_flux_sparse(std::span<const double> pressure, std::span<const double> unit,
                        std::size_t w, const SparseFluxParams& params);

class FluxAccumulator {
 public:
  FluxAccumulator(std::size_t n, double halflife, std::size_t row_cap = 256);

  void add(const BarFlux& bar);
  const std::vector<std::vector<WEdge>>& rows() const { return rows_; }
  const std::vector<double>& out() const { return out_; }
  const std::vector<double>& in() const { return in_; }
  double total() const;
  std::size_t edge_count() const;
  std::size_t size() const { return n_; }

 private:
  std::size_t n_;
  double decay_;
  std::size_t cap_;
  std::vector<std::vector<WEdge>> rows_;
  std::vector<double> out_, in_;
};

}  // namespace fx
