#pragma once
#include <cstddef>
#include <filesystem>
#include <vector>

#include "market/panel.hpp"
#include "market/universe.hpp"
#include "pipeline/core_pipeline.hpp"

namespace mr {

// Offline measurement of how persistent the flux-community structure really is, free of the landscape's display
// machinery (re-cluster cadence, warm start, label matching, cell hysteresis). The core pipeline is replayed over the
// panel; at sampled bars Louvain runs from scratch (resolution 1, min size 8, as the landscape) on three graphs of the
// active stocks (ETF/Fund excluded):
//   cum  - symmetric_flux_graph(Frame::P): the landscape's graph (slow, accumulated flux),
//   fast - symmetric_flux_graph(Frame::P_fast): the fast accumulator (halflife_fast bars),
//   bar  - the single bar's own sparse flux (CorePipeline::last_bar), symmetrized.
// Agreement (NMI, ARI) of partition(t) with partition(t + lag) on the stocks active in both, with baselines: two
// Louvain seeds on the same bar (noise ceiling; "_xseed" is the lag agreement across the two seeds), a
// block-size-preserving label permutation (chance), the sectors.
// A node-level recursive spectral bisection of the fast graph into Louvain's block count is measured alongside.
struct ClusterPersistenceOptions {
  std::size_t stride = 5;            // anchor bars every `stride` bars
  std::size_t warmup = 60;           // first anchor bar index
  std::vector<std::size_t> lags{1, 2, 5, 10, 20, 60};
  std::size_t spectral_stride = 20;  // spectral bisection anchors (fast graph), lags 1 and 5
  std::filesystem::path out_dir = "data/analysis";
};

// Writes <out_dir>/cluster_persistence.csv (lag, graph, metric, mean, median, q25, q75, n),
// cluster_persistence_by_year.csv (year, lag, graph, metric, mean, n) and prints a summary.
void run_cluster_persistence(const Panel& panel, const std::vector<Security>& nodes, const CoreParams& params,
                             const ClusterPersistenceOptions& opt);

}  // namespace mr
