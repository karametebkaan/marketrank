#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "graph/csr.hpp"

namespace fx {

struct CommunityResult {
  std::vector<std::int32_t> id;  // per node, -1 for inactive
  int count = 0;                 // ids are 0..count-1, ordered by smallest member index; the loose pool (if any) is last
  double modularity = 0;         // of the final partition on the input graph, the loose pool counted as one community
  int loose_id = -1;             // id of the pooled "loose" community, -1 if none
};

// Undirected weighted flux graph of the active nodes: W = R + R^T where R holds the off-diagonal kept edges of P with
// finite raw > 0. Rows are sorted by column, duplicates merged.
Csr symmetric_flux_graph(const Csr& P, const std::vector<bool>& active);

// Deterministic two-phase Louvain on the symmetric graph W (weights in W.val). Communities smaller than min_size merge
// into their most strongly connected neighbour (tie: lower id); those with no neighbour go to one pooled "loose"
// community. At most max_communities (excluding the loose pool) are kept by merging the smallest.
CommunityResult louvain(const Csr& W, const std::vector<bool>& active, double resolution = 1.0, int min_size = 8,
                        int max_communities = 256);

// Dense K x K matrix of summed edge weights between communities (diagonal 0).
std::vector<double> community_graph(const Csr& W, const CommunityResult& r);

// Layout order of communities: recursive spectral bisection (Fiedler vector of the normalized Laplacian), split at the
// stock-count median; disconnected sets fall back to their components (largest stock count first). The loose
// community, if any, always goes last. `cw` is the dense K x K matrix, `count[c]` the stock count of community c.
std::vector<int> spectral_order(const std::vector<double>& cw, const std::vector<std::size_t>& count, int loose_id);

// Flux-community assignment that is stable across frames. Re-clusters on the first update and then every
// `recluster_bars` updates, matching new communities to the previous ones so labels and layout order stay put.
// Between re-clusters a newly active node joins its strongest active neighbour's community (else loose) and an
// inactive node leaves.
class CommunityTracker {
 public:
  CommunityTracker(std::size_t n, int recluster_bars = 5, int min_size = 8);
  // Per-node group = position of the node's community in the layout order (the loose pool is last). 0 for inactive.
  const std::vector<std::uint32_t>& update(const Csr& P, const std::vector<bool>& active);
  // Per node: group position, or -1 for loose / inactive.
  const std::vector<std::int32_t>& node_group() const { return node_group_; }
  int communities() const { return communities_; }          // non-loose communities with members
  std::size_t loose_nodes() const { return loose_nodes_; }  // nodes in the loose pool
  double modularity() const { return modularity_; }         // from the last re-cluster
  double cluster_ms() const { return cluster_ms_; }         // duration of the last re-cluster
  bool reclustered() const { return reclustered_; }         // whether the last update re-clustered

 private:
  std::size_t n_;
  int bars_, min_size_;
  bool have_ = false;
  int since_ = 0;
  std::int64_t next_label_ = 0;
  std::vector<std::int64_t> label_;   // -1 inactive, -2 loose, >= 0 community label
  std::vector<std::int64_t> order_;   // labels in layout order (loose not included)
  std::vector<std::uint32_t> group_;
  std::vector<std::int32_t> node_group_;
  int communities_ = 0;
  std::size_t loose_nodes_ = 0;
  double modularity_ = 0, cluster_ms_ = 0;
  bool reclustered_ = false;
};

}  // namespace fx
