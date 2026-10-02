#include "graph/transition.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mr {

LiftMode parse_lift_mode(std::string_view s) {
  if (s == "off") return LiftMode::Off;
  if (s == "excess") return LiftMode::Excess;
  if (s == "ratio") return LiftMode::Ratio;
  throw std::invalid_argument("unknown lift mode: " + std::string(s));
}

std::string_view to_string(LiftMode m) {
  switch (m) {
    case LiftMode::Off: return "off";
    case LiftMode::Excess: return "excess";
    case LiftMode::Ratio: return "ratio";
  }
  return "?";
}

namespace {
struct Cand {
  std::uint32_t i, j;
  double w, raw;
};
}  // namespace

Csr build_transition(const FluxAccumulator& acc, const TransitionParams& params,
                     const std::vector<bool>& active) {
  const std::size_t n = acc.size();
  if (!active.empty() && active.size() != n)
    throw std::invalid_argument("build_transition: active mask size != accumulator size");
  auto is_active = [&](std::size_t i) { return active.empty() || active[i]; };
  const auto& out = acc.out();
  const auto& in = acc.in();
  const double total = acc.total();

  // 1. Lifted candidates per row (parallel; rows are disjoint).
  std::vector<std::vector<Cand>> per_row(n);
#pragma omp parallel for schedule(dynamic, 64)
  for (std::size_t i = 0; i < n; ++i) {
    if (!is_active(i)) continue;
    for (const auto& e : acc.rows()[i]) {
      if (e.j == i || !is_active(e.j) || !(e.w > 0)) continue;
      double w = e.w;
      if (params.lift != LiftMode::Off) {
        const double expected = total > 0 ? out[i] * in[e.j] / total : 0.0;
        if (params.lift == LiftMode::Excess) {
          w = e.w - expected;
        } else {
          w = expected > 0 ? e.w / expected - 1.0 : 0.0;
        }
      }
      if (w > 0) per_row[i].push_back({static_cast<std::uint32_t>(i), e.j, w, e.w});
    }
  }
  std::vector<std::size_t> row_start(n + 1, 0);
  for (std::size_t i = 0; i < n; ++i) row_start[i + 1] = row_start[i] + per_row[i].size();
  std::vector<Cand> cands;
  cands.reserve(row_start[n]);
  for (auto& r : per_row) cands.insert(cands.end(), r.begin(), r.end());
  per_row.clear();

  auto by_weight = [&](std::size_t a, std::size_t b) {
    const Cand &x = cands[a], &y = cands[b];
    if (x.w != y.w) return x.w > y.w;
    if (x.i != y.i) return x.i < y.i;
    return x.j < y.j;
  };
  std::vector<char> keep(cands.size(), 0);

  // 2a. Top k_out per row.
#pragma omp parallel
  {
    std::vector<std::size_t> idx;
#pragma omp for schedule(dynamic, 64)
    for (std::size_t i = 0; i < n; ++i) {
      idx.clear();
      for (std::size_t c = row_start[i]; c < row_start[i + 1]; ++c) idx.push_back(c);
      const std::size_t m = std::min(params.k_out, idx.size());
      std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(m), idx.end(),
                        by_weight);
      for (std::size_t e = 0; e < m; ++e) keep[idx[e]] = 1;
    }
  }
  // 2b. Top k_in per column (each candidate belongs to exactly one column).
  if (params.k_in > 0) {
    std::vector<std::vector<std::size_t>> by_col(n);
    for (std::size_t c = 0; c < cands.size(); ++c) by_col[cands[c].j].push_back(c);
#pragma omp parallel for schedule(dynamic, 64)
    for (std::size_t j = 0; j < n; ++j) {
      auto& col = by_col[j];
      const std::size_t m = std::min(params.k_in, col.size());
      std::partial_sort(col.begin(), col.begin() + static_cast<std::ptrdiff_t>(m), col.end(),
                        by_weight);
      for (std::size_t e = 0; e < m; ++e) keep[col[e]] = 1;
    }
  }

  // 3. Assemble rows (serial).
  Csr P;
  P.n = n;
  P.row_ptr.reserve(n + 1);
  P.row_ptr.push_back(0);
  for (std::size_t i = 0; i < n; ++i) {
    double offsum = 0;
    for (std::size_t c = row_start[i]; c < row_start[i + 1]; ++c)
      if (keep[c]) offsum += cands[c].w;
    const double retained = is_active(i) ? params.retention * in[i] : 0.0;
    auto push_self = [&](double val) {
      P.col.push_back(static_cast<std::uint32_t>(i));
      P.val.push_back(val);
      P.raw.push_back(retained);
    };
    if (!(offsum > 0)) {
      // Teleport leaves an active row with nothing to give (no kept edge, nothing retained) empty.
      if (!is_active(i) || params.dangling == DanglingMode::SelfLoop || retained > 0) push_self(1.0);
    } else {
      const double denom = out[i] + retained;
      const double self_mass = denom > 0 ? retained / denom : 0.0;
      const double off_mass = 1.0 - self_mass;
      bool self_done = !(self_mass > 0);
      for (std::size_t c = row_start[i]; c < row_start[i + 1]; ++c) {
        if (!keep[c]) continue;
        if (!self_done && cands[c].j > i) {
          push_self(self_mass);
          self_done = true;
        }
        P.col.push_back(cands[c].j);
        P.val.push_back(off_mass * cands[c].w / offsum);
        P.raw.push_back(cands[c].raw);
      }
      if (!self_done) push_self(self_mass);
    }
    P.row_ptr.push_back(P.col.size());
  }
  return P;
}

}  // namespace mr
