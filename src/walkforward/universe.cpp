#include "walkforward/universe.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "core/time.hpp"

namespace mr {

Rebalance parse_rebalance(std::string_view name) {
  if (name == "monthly") return Rebalance::Monthly;
  if (name == "weekly") return Rebalance::Weekly;
  throw std::invalid_argument("unknown rebalance: " + std::string(name));
}

namespace {
// Group key of a bar: calendar month, or Monday-based week number (1970-01-01 was a Thursday, so
// (days+3)/7 is constant within a Mon..Sun week; this is equivalent to grouping by (ISO year, ISO week)).
std::int64_t group_key(TimePoint ts, Rebalance mode) {
  const std::int64_t days = floor_div(ts, 86400);
  if (mode == Rebalance::Weekly) return floor_div(days + 3, 7);
  const Civil c = civil_from_days(days);
  return static_cast<std::int64_t>(c.y) * 12 + static_cast<std::int64_t>(c.m) - 1;
}
}  // namespace

std::vector<std::size_t> rebalance_dates(const Panel& p, Rebalance mode, std::size_t warmup_bars) {
  std::vector<std::size_t> out;
  const std::size_t T = p.T();
  for (std::size_t t = 0; t < T; ++t) {
    const bool last = t + 1 == T || group_key(p.times[t + 1], mode) != group_key(p.times[t], mode);
    if (last && t >= warmup_bars && t + 2 <= T) out.push_back(t);
  }
  return out;
}

std::vector<bool> eligible_at(const Panel& p, std::size_t t, std::size_t window, double min_dollar_volume,
                              std::size_t top_n) {
  const std::size_t N = p.N();
  std::vector<bool> ok(N, false);
  if (t >= p.T() || window == 0) return ok;
  const std::size_t lo = t + 1 >= window ? t + 1 - window : 0;
  const std::size_t need = std::max<std::size_t>(window / 2, 1);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> med(N, nan), buf;
  buf.reserve(window);
  for (std::size_t i = 0; i < N; ++i) {
    buf.clear();
    for (std::size_t u = lo; u <= t; ++u) {
      const double dv = p.vwap[p.idx(u, i)] * p.volume[p.idx(u, i)];
      if (std::isfinite(dv)) buf.push_back(dv);
    }
    if (buf.size() < need) continue;
    const std::size_t mid = buf.size() / 2;
    std::nth_element(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(mid), buf.end());
    double m = buf[mid];
    if (buf.size() % 2 == 0) {
      const double lower = *std::max_element(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(mid));
      m = 0.5 * (m + lower);
    }
    med[i] = m;
  }
  std::vector<std::size_t> cand;
  for (std::size_t i = 0; i < N; ++i)
    if (std::isfinite(med[i]) && med[i] >= min_dollar_volume) cand.push_back(i);
  if (top_n > 0 && cand.size() > top_n) {
    std::stable_sort(cand.begin(), cand.end(), [&](std::size_t a, std::size_t b) { return med[a] > med[b]; });
    cand.resize(top_n);
  }
  for (std::size_t i : cand) ok[i] = true;
  return ok;
}

std::vector<double> forward_oo_return(const Panel& p, std::size_t t, std::size_t h) {
  const std::size_t N = p.N();
  std::vector<double> r(N, std::numeric_limits<double>::quiet_NaN());
  if (t + 1 + h >= p.T()) return r;
  for (std::size_t i = 0; i < N; ++i) {
    const double a = p.open[p.idx(t + 1, i)], b = p.open[p.idx(t + 1 + h, i)];
    if (std::isfinite(a) && std::isfinite(b)) r[i] = b / a - 1.0;
  }
  return r;
}

}  // namespace mr
