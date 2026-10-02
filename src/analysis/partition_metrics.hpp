#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "geom/community.hpp"
#include "graph/csr.hpp"
#include "graph/sparse_flux.hpp"

namespace mr {

// Agreement between two partitions of the same node list (a[i], b[i] are the labels of node i; any int64 values).
// NMI with the arithmetic-mean normalization 2 I(A;B) / (H(A) + H(B)); 1 when both partitions are a single block.
// NaN for empty input. Throws std::invalid_argument on a size mismatch.
double partition_nmi(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b);
// Adjusted Rand index (Hubert-Arabie); 1 when the expected and maximum index coincide (e.g. both trivial).
double partition_ari(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b);
// The labels shuffled across positions (a uniformly random permutation from `seed`): block sizes are preserved, the
// node-to-block assignment is random. Agreement of x with permuted_labels(y) is the chance level.
std::vector<std::int64_t> permuted_labels(std::vector<std::int64_t> a, std::uint64_t seed);

// From-scratch Louvain (no warm start) on W with the nodes visited in a seeded random order: seed 0 is louvain()
// itself (index order); any other seed relabels the nodes by a random permutation, clusters, and maps the ids back.
// Louvain's result depends on the visiting order, so two seeds measure its own run-to-run noise.
CommunityResult louvain_seeded(const Csr& W, const std::vector<bool>& active, std::uint64_t seed, double resolution = 1.0,
                               int min_size = 8, int max_communities = 256);

// Node-level recursive spectral bisection of the active nodes of the symmetric graph W into (at most) k blocks.
// Connected components with fewer than min_size nodes (isolated nodes included) are pooled into one loose block;
// the other components are blocks. Then the largest block (if it has >= 2 min_size nodes) is split at its node-count
// median of the Fiedler vector of its induced subgraph's normalized Laplacian (power iteration on
// I + D^-1/2 W D^-1/2 with the trivial vector deflated, 500 iterations, the same scheme as spectral_order) until
// there are k non-loose blocks. Returns per node: -1 inactive, -2 loose, else block id 0..k-1.
std::vector<std::int32_t> spectral_bisection(const Csr& W, const std::vector<bool>& active, int k, int min_size = 8);

// A single bar's sparse flux as a transition-shaped Csr (val = raw = the bar's flux), for symmetric_flux_graph().
Csr bar_flux_csr(const BarFlux& bar);

}  // namespace mr
