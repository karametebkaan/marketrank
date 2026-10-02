#include "market/synthetic_market.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace fx {

std::vector<Security> generate_synthetic(const SyntheticConfig& cfg, BarStore& store) {
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<double> factor_dist(0.0, 0.01), idio_dist(0.0, 0.008);
  const int n = cfg.sectors * cfg.per_sector;

  std::vector<Security> secs;
  for (int s = 0; s < cfg.sectors; ++s) {
    for (int k = 0; k < cfg.per_sector; ++k) {
      char ticker[16];
      std::snprintf(ticker, sizeof ticker, "S%d_%02d", s, k);
      secs.push_back({ticker, ticker, "Sector" + std::to_string(s)});
    }
  }

  std::vector<std::vector<Bar>> series(static_cast<std::size_t>(n));
  std::vector<double> price(static_cast<std::size_t>(n), 100.0);
  const TimePoint step = timeframe_seconds(cfg.tf);
  for (int b = 0; b < cfg.bars; ++b) {
    std::vector<double> factor(static_cast<std::size_t>(cfg.sectors));
    for (auto& f : factor) f = factor_dist(rng);
    const bool rotating = b >= cfg.rotation_start;
    for (int i = 0; i < n; ++i) {
      const int s = i / cfg.per_sector;
      double drift = 0.0, mult = 1.0;
      if (rotating && s == cfg.rotation_from) drift = -cfg.rotation_strength, mult = 1.5;
      if (rotating && s == cfg.rotation_to) drift = cfg.rotation_strength, mult = 1.5;
      const double r = 0.6 * factor[static_cast<std::size_t>(s)] + idio_dist(rng) + drift;
      const auto ui = static_cast<std::size_t>(i);
      const double o = price[ui];
      const double c = o * (1.0 + r);
      const double h = std::max(o, c) * 1.002;
      const double l = std::min(o, c) * 0.998;
      const double base = 1e6 * (1 + i % 5);
      const double v = base * (1.0 + 30.0 * std::abs(r)) * mult;
      series[ui].push_back({cfg.start + b * step, o, h, l, c, v, (o + h + l + c) / 4.0});
      price[ui] = c;
    }
  }
  for (int i = 0; i < n; ++i)
    store.merge(secs[static_cast<std::size_t>(i)].ticker, cfg.tf,
                series[static_cast<std::size_t>(i)]);
  return secs;
}

}  // namespace fx
