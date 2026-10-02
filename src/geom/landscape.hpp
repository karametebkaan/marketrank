#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
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

// What the landscape height (and the territory ordering, and the mountain/crater rule) is made of:
// - PiRelSize: log(pi_i / s_i), s_i the size share (trailing median dollar volume / sum over active stocks,
//   size_shares in graph/hotness.hpp): how much more money a stock attracts than its size predicts; 0 = as
//   predicted. The default under the marketrank preset.
// - Pi: the MarketRank score as log(pi * N_active), signed around 0 = average.
// - Hotness: hotness h through HeightMode.
// The truth (pi, the rank tables, /api/top) is the same under every value.
enum class LandscapeValue { Pi, Hotness, PiRelSize };
LandscapeValue parse_landscape_value(std::string_view s);  // "pi" | "hotness" | "pi_rel_size"
std::string_view to_string(LandscapeValue v);
// The landscape value a preset brings: PiRelSize under "marketrank", Hotness otherwise.
LandscapeValue default_landscape_value(std::string_view preset);
// Display value of a node: log(pi * n_active) under Pi, log(pi / size_share) under PiRelSize (NaN for pi <= 0 or
// a zero / non-finite size share; HeightMode is not applied), else display_height(h, height).
double landscape_value(double h, double pi, std::size_t n_active, double size_share, LandscapeValue v,
                       HeightMode height);

// Display smoother applied to the IDW raster (and the shock delta raster): CVT-weighted relaxation (default),
// Gaussian (LandscapeParams::smooth sigma) or none. Display-only: the cached frames are redrawn, not recomputed.
enum class Smoother { Gaussian, Cvt, None };
Smoother parse_smoother(std::string_view s);  // "gaussian" | "cvt" | "none"
std::string_view to_string(Smoother s);

struct LandscapeParams {
  LandscapeValue value = LandscapeValue::Hotness;  // PiRelSize under the marketrank preset (main, server)
  IdwParams idw{1, 2.0, 3};  // subdivision 1: raster = lattice mesh vertices
  HeightMode height = HeightMode::SignedLog;
  Smoother smoother = Smoother::Cvt;
  CvtParams cvt;                  // used when smoother == Cvt
  // Display pins: node indices (e.g. the portfolio holdings) the CVT surface must pass through exactly, at their
  // cell-centre pixel (any subdivision). Indices that are inactive or outside the universe are ignored. CVT only:
  // the Gaussian and none smoothers ignore pins (none is exact at subdivision 1 anyway). Display-only.
  std::vector<std::uint32_t> pinned;
  double smooth = 1.0;            // Gaussian sigma in lattice cells after IDW; used only when smoother == Gaussian (0 = off)
  double order_smoothing = 0.5;  // weight on the previous frame's hotness when ranking nodes inside a territory
  double rank_tolerance = 0.15;  // cell hysteresis: keep the cell while the spiral slot moves <= this x territory
                                 // size (at least 2 slots); see territory_layout
  std::size_t max_arcs = 2000;
  TerritoryMode territory = TerritoryMode::Flux;
  int recluster_bars = 5;   // flux mode: re-cluster every this many frames
  double resolution = 1.0;  // flux mode: Louvain resolution
  int warmup_bars = 5;      // serve mode: the first this many bars only feed the model (no landscape, no clustering)
  // Leave the ETF/Fund sector (kSectorEtfFund) out of the landscape surface: such nodes stay in the frame (exact pi,
  // tables, flows) with cell -1 but are not placed, not on the lattice and not in the IDW / smoothing. Placement.
  bool exclude_etf = true;
  bool operator==(const LandscapeParams&) const = default;
};

struct LandscapeNode {
  std::uint32_t i;
  std::int32_t cell;
  float fx, fy;  // cell centre normalized to [0, 1]
  double h, hdisp, pi, score;
  std::int32_t group = -1;  // flux: persistent community label (stable across re-clusters); sector: sector id; -1 = loose
  double pulse = std::numeric_limits<double>::quiet_NaN();       // heartbeat: Frame::pulse (delta log pi*N)
  double size_share = std::numeric_limits<double>::quiet_NaN();  // s_i (PiRelSize); NaN = no size
  bool floor = false;  // pi at the teleport floor (Frame::pi_floor); excluded from the territory medians under Pi
};

struct LandscapeArc {
  std::uint32_t a, b;
  double w;
};

struct LandscapeFrame {
  TimePoint t = 0;
  std::size_t n = 0;  // universe size the frame was built for
  LatticeSize size;
  std::vector<LandscapeNode> nodes;  // active nodes, ascending i; cell -1 = left out of the landscape (exclude_etf)
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
// radius, subdivision) only change how a frame is drawn; everything else (value, territory, ranking, clustering,
// arcs, warm-up) is placement. The value is placement because it orders the stocks inside a territory.
bool same_placement(const LandscapeParams& a, const LandscapeParams& b);
// IDW raster -> display raster under p.smoother (pins apply to CVT only).
Raster apply_smoother(Raster r, const LandscapeParams& p, const std::vector<CvtPin>& pins = {});
// CVT pins for p.pinned: each pinned node's cell-centre pixel (col·s + s/2, row·s + s/2) and its value.
// cell[i] < 0 or a non-finite value[i] skips node i.
std::vector<CvtPin> landscape_pins(const std::vector<std::int32_t>& cell, const std::vector<double>& value, LatticeSize size,
                                   const LandscapeParams& p);
// The frame redrawn with display parameters p: hdisp recomputed (landscape_value with p.value), the raster
// rebuilt from (cell, hdisp).
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
  // Nodes (ex[i] != 0) left out of the landscape surface; empty = none. They stay in the frame with cell -1.
  void set_excluded(std::vector<char> ex) { excluded_ = std::move(ex); }

 private:
  std::size_t n_;
  LandscapeParams p_;
  std::vector<std::uint32_t> group_;
  std::vector<char> excluded_;
  std::vector<double> s_prev_;   // smoothed ranking hotness per node
  std::vector<char> has_prev_;   // node was active in the previous frame
  // Cell hysteresis: the previous frame's placement and each node's group identity in it.
  PlacementMemory mem_;
  std::vector<std::int64_t> prev_key_;
  CommunityTracker tracker_;
};

}  // namespace mr
