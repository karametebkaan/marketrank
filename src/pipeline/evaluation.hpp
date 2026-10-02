#pragma once
#include <span>
#include <string>
#include <vector>

#include "market/panel.hpp"
#include "market/universe.hpp"
#include "pipeline/core_pipeline.hpp"

namespace fx {

double gini(std::span<const double> x);
double spearman(std::span<const double> a, std::span<const double> b);
double floor_share(const Frame& f, double alpha);
double structure_gain(const Frame& f);  // 1 - Spearman(pi, inflow share) over active nodes
double sector_coherence(const Frame& f, const std::vector<Security>& nodes);

struct EvalMetrics {
  double floor_share = 0, gini = 0, sector_coherence = 0;
  double structure_gain = 0;
  double ic_mean = 0, ic_t = 0;      // +1 forecast score vs next-bar return
  double ic_h_mean = 0, ic_h_t = 0;  // hotness h vs next-bar return
  std::size_t ic_samples = 0;
  double mean_frame_ms = 0;
};

// Spec 5.2.
EvalMetrics evaluate(const Panel& panel, const std::vector<Security>& nodes,
                     const CoreParams& params, std::size_t eval_bars);

struct EvalConfig {
  std::string name;
  CoreParams params;
};

std::vector<EvalConfig> evaluation_grid();

}  // namespace fx
