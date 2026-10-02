#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "core/types.hpp"
#include "geom/idw.hpp"
#include "geom/lattice.hpp"
#include "geom/community.hpp"
#include "geom/territory.hpp"
#include "pipeline/core_pipeline.hpp"

namespace mr {

enum class HeightMode { SignedLog, Linear };
HeightMode parse_height_mode(std::string_view s);  // "signed-log" | "linear"
std::string_view to_string(HeightMode m);
double display_height(double h, HeightMode m);

// What a territory is: a flux community (Louvain on the model's own flux graph) or a market sector.
enum class TerritoryMode { Flux, Sector };
TerritoryMode parse_territory_mode(std::string_view s);  // "flux" | "sector"
std::string_view to_string(TerritoryMode m);

struct LandscapeParams {
  IdwParams idw{1, 2.0, 3};  // subdivision 1: raster = lattice mesh vertices
  HeightMode height = HeightMode::SignedLog;
  double smooth = 1.0;            // display smoothing: Gaussian sigma in lattice cells after IDW (0 = off)
  double order_smoothing = 0.5;  // weight on the previous frame's hotness when ranking nodes inside a territory
  double rank_tolerance = 0.15;  // cell hysteresis: keep the cell while the spiral slot moves <= this x territory
                                 // size (at least 2 slots); see territory_layout
  std::size_t max_arcs = 2000;
  TerritoryMode territory = TerritoryMode::Flux;
  int recluster_bars = 5;   // flux mode: re-cluster every this many frames
  double resolution = 1.0;  // flux mode: Louvain resolution
  int warmup_bars = 5;      // serve mode: the first this many bars only feed the model (no landscape, no clustering)
};

struct LandscapeNode {
  std::uint32_t i;
  std::int32_t cell;
  float fx, fy;  // cell centre normalized to [0, 1]
  double h, hdisp, pi, score;
  std::int32_t group = -1;  // flux: persistent community label (stable across re-clusters); sector: sector id; -1 = loose
};

struct LandscapeArc {
  std::uint32_t a, b;
  double w;
};

struct LandscapeFrame {
  TimePoint t = 0;
  std::size_t n = 0;  // universe size the frame was built for
  LatticeSize size;
  std::vector<LandscapeNode> nodes;  // active nodes, ascending i
  std::vector<LandscapeArc> arcs;
  Raster raster;
  double compute_ms = 0;
  int communities = 0;          // territories with members, excluding the loose pool
  double modularity = 0;        // flux mode: of the last re-cluster
  std::size_t loose = 0;        // flux mode: stocks in the loose pool
  double cluster_ms = 0;        // flux mode: duration of the last re-cluster
  bool reclustered = false;     // flux mode: this frame re-clustered
};

// Whether two parameter sets place stocks identically. The display parameters (smooth, height and idw: power,
// radius, subdivision) only change how a frame is drawn; everything else (territory, ranking, clustering, arcs,
// warm-up) is placement.
bool same_placement(const LandscapeParams& a, const LandscapeParams& b);
// The frame redrawn with display parameters p: hdisp recomputed from h, the raster rebuilt from (cell, hdisp).
LandscapeFrame restyle(const LandscapeFrame& f, const LandscapeParams& p);

std::vector<LandscapeArc> top_arcs(const Csr& P, const std::vector<bool>& active, std::size_t max_arcs);
Raster delta_raster(const LandscapeFrame& base, const std::vector<double>& delta, const LandscapeParams& p);

// Territory placement (spec 6.1). Stateful across frames through the smoothed ranking hotness and, in flux mode,
// the community tracker.
class LandscapeBuilder {
 public:
  // group[i] is node i's sector id (used in sector mode); empty means one group for all nodes.
  LandscapeBuilder(std::size_t n, LandscapeParams params, std::vector<std::uint32_t> group = {});
  LandscapeFrame build(const Frame& f);

 private:
  std::size_t n_;
  LandscapeParams p_;
  std::vector<std::uint32_t> group_;
  std::vector<double> s_prev_;   // smoothed ranking hotness per node
  std::vector<char> has_prev_;   // node was active in the previous frame
  // Cell hysteresis: the previous frame's placement and each node's group identity in it.
  PlacementMemory mem_;
  std::vector<std::int64_t> prev_key_;
  CommunityTracker tracker_;
};

}  // namespace mr
