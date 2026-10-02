#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "core/types.hpp"
#include "geom/idw.hpp"
#include "geom/lattice.hpp"
#include "geom/embedding.hpp"
#include "pipeline/core_pipeline.hpp"

namespace fx {

enum class HeightMode { SignedLog, Linear };
HeightMode parse_height_mode(std::string_view s);  // "signed-log" | "linear"
std::string_view to_string(HeightMode m);
double display_height(double h, HeightMode m);

struct LandscapeParams {
  EmbeddingParams embed;
  IdwParams idw{1, 2.0, 3};  // subdivision 1: raster = lattice mesh vertices
  HeightMode height = HeightMode::SignedLog;
  int max_shift = 2;
  std::size_t max_arcs = 2000;
};

struct LandscapeNode {
  std::uint32_t i;
  std::int32_t cell;
  float fx, fy;  // solve-embedding position normalized to [0, 1]
  double h, hdisp, pi, score;
};

struct LandscapeArc {
  std::uint32_t a, b;
  double w;
};

struct LandscapeFrame {
  TimePoint t = 0;
  LatticeSize size;
  std::vector<LandscapeNode> nodes;  // active nodes, ascending i
  std::vector<LandscapeArc> arcs;
  Raster raster;
  double compute_ms = 0;
};

std::vector<LandscapeArc> top_arcs(const Csr& P, const std::vector<bool>& active, std::size_t max_arcs);
Raster delta_raster(const LandscapeFrame& base, const std::vector<double>& delta, const LandscapeParams& p);

// Stateful across frames: Procrustes-aligned, smoothed solve-embedding positions and hysteresis on the lattice cells.
class LandscapeBuilder {
 public:
  LandscapeBuilder(std::size_t n, LandscapeParams params);
  LandscapeFrame build(const Frame& f);

 private:
  std::size_t n_;
  LandscapeParams p_;
  SolveEmbedding embed_;
  std::vector<double> xy_;
  std::vector<std::int32_t> cells_;
  LatticeSize size_;
  bool first_ = true;
};

}  // namespace fx
