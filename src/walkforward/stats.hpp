#pragma once
#include <cstddef>
#include <cstdint>
#include <span>

#include "pipeline/evaluation.hpp"  // mr::spearman (average ranks, NaN pairs skipped, NaN if <3 pairs or zero variance)
namespace mr {
struct MeanT { double mean = 0, t = 0; std::size_t n = 0; };
MeanT mean_t(std::span<const double> x);
double norm_cdf(double x);
double norm_inv(double p);
struct CI { double lo = 0, hi = 0; };
CI block_bootstrap_mean_ci(std::span<const double> x, std::size_t block, std::size_t reps, std::uint64_t seed,
                           double level);
}  // namespace mr
