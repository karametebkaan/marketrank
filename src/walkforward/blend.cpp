#include "walkforward/blend.hpp"

#include <algorithm>
#include <limits>
#include <cmath>
#include <span>
#include <stdexcept>

#include "walkforward/stats.hpp"

namespace mr {
namespace {
constexpr std::size_t kMinTrain = 6;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}

std::vector<BlendStep> run_blend(const std::vector<MonthRecord>& months, const BlendParams& p) {
  if (p.embargo < 1)
    throw std::invalid_argument("run_blend: embargo must be >= 1 (label(m) ends at the next execution)");
  const std::size_t M = months.size();
  std::vector<BlendStep> out(M);
  // ic[s][j] = spearman(z_s(j), label(j)); depends on label(j), only read for j <= m - embargo - 1.
  std::array<std::vector<double>, kSignals> ic;
  for (auto& v : ic) v.assign(M, kNaN);
  for (std::size_t j = 0; j < M; ++j)
    for (std::size_t s = 0; s < kSignals; ++s)
      if (months[j].z[s].size() == months[j].label.size()) ic[s][j] = spearman(months[j].z[s], months[j].label);

  for (std::size_t m = 0; m < M; ++m) {
    BlendStep& st = out[m];
    // Training window [m - embargo - train, m - embargo - 1], clipped at 0.
    if (m >= p.embargo + 1) {
      const std::size_t hi = m - p.embargo;  // exclusive
      const std::size_t lo = hi > p.train_months ? hi - p.train_months : 0;
      if (hi - lo >= kMinTrain) {
        double sum = 0;
        for (std::size_t s = 0; s < kSignals; ++s) {
          const MeanT mt = mean_t(std::span<const double>(ic[s].data() + lo, hi - lo));
          st.w[s] = (mt.n >= kMinTrain && mt.mean > 0) ? mt.mean : 0.0;
          sum += st.w[s];
        }
        if (sum > 0) for (double& w : st.w) w /= sum;
        else st.w.fill(0.0);
      }
    }
    const bool have_w = std::any_of(st.w.begin(), st.w.end(), [](double w) { return w > 0; });
    std::vector<double> score;
    if (have_w) {
      const std::size_t N = months[m].label.size();
      score.assign(N, kNaN);
      for (std::size_t i = 0; i < N; ++i) {
        double acc = 0;
        bool any = false;
        for (std::size_t s = 0; s < kSignals; ++s) {
          if (st.w[s] <= 0 || i >= months[m].z[s].size()) continue;
          const double z = months[m].z[s][i];
          if (!std::isfinite(z)) continue;
          acc += st.w[s] * z;
          any = true;
        }
        if (any) score[i] = acc;
      }
      st.oos_ic = spearman(score, months[m].label);  // recorded even if the gate is closed; used only by later months
    }
    // Gate: oos_ic over [m - embargo - gate, m - embargo - 1].
    if (have_w) {  // have_w implies m >= embargo + 1
      const std::size_t hi = m - p.embargo;
      const std::size_t lo = hi > p.gate_months ? hi - p.gate_months : 0;
      std::vector<double> o(hi - lo);
      for (std::size_t j = lo; j < hi; ++j) o[j - lo] = out[j].oos_ic;
      const MeanT mt = mean_t(o);
      st.gate_open = mt.n >= p.gate_min && mt.t > p.gate_t;  // NaN t compares false
    }
    if (st.gate_open) st.score = std::move(score);
  }
  return out;
}
}  // namespace mr
