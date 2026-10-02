#include "walkforward/signals.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "graph/hotness.hpp"

namespace mr {
namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr const char* kNames[kSignals] = {"score",       "pulse1",      "pulse5",      "pulse20",
                                          "pi_rel_size", "neg_hotness", "inflow_mom5", "forecast"};
}  // namespace

std::string_view to_string(Signal s) { return kNames[static_cast<std::size_t>(s)]; }

Signal parse_signal(std::string_view name) {
  for (std::size_t i = 0; i < kSignals; ++i)
    if (name == kNames[i]) return static_cast<Signal>(i);
  throw std::invalid_argument("unknown signal: " + std::string(name));
}

SignalTracker::SignalTracker(std::size_t n) : n_(n) {
  for (auto& v : logs_) v.assign(n, kNaN);
  for (auto& v : inflow_) v.assign(n, kNaN);
}

std::array<std::vector<double>, kSignals> SignalTracker::update(const Frame& f) {
  if (f.pi.size() != n_ || f.active.size() != n_) throw std::invalid_argument("SignalTracker: frame size mismatch");
  head_ = (head_ + 1) % kRing;
  std::size_t n_active = 0;
  for (std::size_t i = 0; i < n_; ++i) n_active += f.active[i] ? 1 : 0;

  auto& cur = logs_[head_];
  auto& cur_in = inflow_[head_];
  for (std::size_t i = 0; i < n_; ++i) {
    const double sc = f.active[i] ? market_rank_score(f.pi[i], n_active) : kNaN;
    cur[i] = (f.active[i] && sc > 0) ? std::log(sc) : kNaN;
    cur_in[i] = i < f.inflow.size() ? f.inflow[i] : kNaN;
  }
  auto back = [&](std::size_t k) -> const std::vector<double>& { return logs_[(head_ + kRing - k) % kRing]; };

  std::array<std::vector<double>, kSignals> out;
  for (auto& v : out) v.assign(n_, kNaN);
  auto at = [&](Signal s) -> std::vector<double>& { return out[static_cast<std::size_t>(s)]; };

  std::vector<double> shares;
  if (f.size_ref.size() == n_) shares = size_shares(f.size_ref);
  const bool have_fc = !f.forecasts.empty() && f.forecasts[0].score.size() == n_;
  const auto& in5 = inflow_[(head_ + kRing - 5) % kRing];

  for (std::size_t i = 0; i < n_; ++i) {
    if (!f.active[i]) {
      // Only NegHotness, InflowMom5 and Forecast could be defined; keep inactive nodes NaN throughout.
      continue;
    }
    at(Signal::Score)[i] = cur[i];
    const std::pair<Signal, std::size_t> pk[] = {{Signal::Pulse1, 1}, {Signal::Pulse5, 5}, {Signal::Pulse20, 20}};
    for (auto [sig, k] : pk) at(sig)[i] = cur[i] - back(k)[i];  // NaN propagates
    if (!shares.empty() && std::isfinite(shares[i]) && shares[i] > 0 && f.pi[i] > 0)
      at(Signal::PiRelSize)[i] = std::log(f.pi[i] / shares[i]);
    at(Signal::NegHotness)[i] = -f.h[i];
    if (cur_in[i] > 0 && in5[i] > 0) at(Signal::InflowMom5)[i] = std::log(cur_in[i] / in5[i]);
    if (have_fc) at(Signal::Forecast)[i] = f.forecasts[0].score[i];
  }
  return out;
}

std::vector<double> zscore(std::span<const double> x, const std::vector<bool>& mask) {
  const std::size_t n = x.size();
  if (mask.size() != n) throw std::invalid_argument("zscore: mask size != x size");
  std::vector<double> z(n, kNaN);
  double sum = 0;
  std::size_t m = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (mask[i] && std::isfinite(x[i])) {
      sum += x[i];
      ++m;
    }
  if (m == 0) return z;
  const double mean = sum / static_cast<double>(m);
  double ss = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (mask[i] && std::isfinite(x[i])) ss += (x[i] - mean) * (x[i] - mean);
  const double sd = std::sqrt(ss / static_cast<double>(m));
  for (std::size_t i = 0; i < n; ++i)
    if (mask[i] && std::isfinite(x[i])) z[i] = sd > 0 ? std::clamp((x[i] - mean) / sd, -3.0, 3.0) : 0.0;
  return z;
}

}  // namespace mr
