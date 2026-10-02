#include "walkforward/backtest.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mr {

namespace {
std::size_t ticker_index(const Panel& p, const std::string& ticker) {
  for (std::size_t i = 0; i < p.N(); ++i)
    if (p.tickers[i] == ticker) return i;
  throw std::invalid_argument("backtest: unknown ticker: " + ticker);
}

std::vector<double> base_weights(const Panel& p, const BacktestParams& bp) {
  std::vector<double> w(p.N(), 0.0);
  for (const BaseWeight& b : bp.base) w[ticker_index(p, b.ticker)] += b.weight;
  return w;
}

std::vector<double> equal_weight(std::size_t N, const std::vector<bool>& eligible) {
  std::vector<double> w(N, 0.0);
  std::size_t n = 0;
  for (std::size_t i = 0; i < N; ++i) n += eligible.empty() || eligible[i] ? 1 : 0;
  if (n == 0) return w;
  for (std::size_t i = 0; i < N; ++i)
    if (eligible.empty() || eligible[i]) w[i] = 1.0 / static_cast<double>(n);
  return w;
}

std::string trade_row(TimePoint t, const std::string& ticker, double dw, double px) {
  char buf[96];
  std::snprintf(buf, sizeof buf, ",%.12g,%.12g", dw, px);
  return std::to_string(t) + "," + ticker + buf;
}

struct Book {
  std::vector<double> shares;
  double cash = 1.0;
  std::vector<double> last_px;  // last finite price seen (open or close), NaN before the first
};

// Executes a trade to weights w at the open of bar t. Returns {cost, turnover}.
std::pair<double, double> execute(const Panel& p, const BacktestParams& bp, std::size_t t, const std::vector<double>& w,
                                  bool initial, Book& bk, EquityCurve& out) {
  const std::size_t N = p.N();
  double V = bk.cash;
  for (std::size_t i = 0; i < N; ++i)
    if (bk.shares[i] != 0.0) V += bk.shares[i] * bk.last_px[i];
  if (!(V > 0.0)) return {0.0, 0.0};

  std::vector<double> delta(N, 0.0);
  double gross = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    const double o = p.open[p.idx(t, i)];
    if (!std::isfinite(o)) continue;  // no open: keep shares
    delta[i] = w[i] * V - bk.shares[i] * o;
    gross += std::abs(delta[i]);
  }
  // Turnover cap (the initial funding is exempt).
  const double raw_turnover = gross / (2.0 * V);
  if (!initial && raw_turnover > bp.max_turnover) {
    const double s = std::max(0.0, bp.max_turnover) / raw_turnover;
    for (double& d : delta) d *= s;
  }
  // Funding: buys may not exceed cash plus sells.
  double buys = 0.0, sells = 0.0;
  for (double d : delta) (d > 0.0 ? buys : sells) += std::abs(d);
  const double avail = bk.cash + sells;
  if (buys > avail) {
    const double s = avail > 0.0 ? avail / buys : 0.0;
    for (double& d : delta)
      if (d > 0.0) d *= s;
  }
  gross = 0.0;
  for (double d : delta) gross += std::abs(d);
  const double cost = bp.cost_bps * 1e-4 * gross;

  // Post-trade notionals of the tradable names, then fund the cost by scaling them and the cash.
  std::vector<double> notional(N, 0.0);
  double cash = bk.cash, W = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    const double o = p.open[p.idx(t, i)];
    if (!std::isfinite(o)) continue;
    notional[i] = bk.shares[i] * o + delta[i];
    cash -= delta[i];
    W += notional[i];
  }
  W += cash;
  if (cost > 0.0) {
    if (W > cost) {
      const double f = (W - cost) / W;
      for (double& n : notional) n *= f;
      cash *= f;
    } else {
      cash -= cost;
    }
  }

  double post = cash;
  for (std::size_t i = 0; i < N; ++i) {
    const double o = p.open[p.idx(t, i)];
    if (std::isfinite(o)) {
      const double ns = notional[i] / o;
      if (ns != bk.shares[i]) out.trades_csv.push_back(trade_row(p.times[t], p.tickers[i], (ns - bk.shares[i]) * o / V, o));
      bk.shares[i] = ns;
      post += ns * o;
    } else if (bk.shares[i] != 0.0) {
      post += bk.shares[i] * bk.last_px[i];
    }
  }
  bk.cash = cash;
  if (std::abs(post - (V - cost)) > 1e-12 * V)
    throw std::logic_error("backtest: value not conserved across a trade");
  return {cost, gross / (2.0 * V)};
}
}  // namespace

std::vector<double> target_weights(const Panel& p, const BacktestParams& bp, const Decision& d) {
  const std::size_t N = p.N();
  if (!d.score.empty() && d.score.size() != N) throw std::invalid_argument("target_weights: score size != N");
  if (!d.eligible.empty() && d.eligible.size() != N) throw std::invalid_argument("target_weights: eligible size != N");
  if (d.score.empty() && bp.flat_holds_equal_weight) return equal_weight(N, d.eligible);
  std::vector<double> w = base_weights(p, bp);
  if (d.score.empty()) return w;
  for (double& x : w) x *= 1.0 - bp.tilt;
  if (bp.k == 0 || !(bp.tilt > 0.0)) return w;

  std::vector<std::size_t> cand;
  for (std::size_t i = 0; i < N; ++i)
    if (std::isfinite(d.score[i]) && (d.eligible.empty() || d.eligible[i])) cand.push_back(i);
  const std::size_t m = std::min(bp.k, cand.size());
  std::partial_sort(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(m), cand.end(),
                    [&](std::size_t a, std::size_t b) {
                      if (d.score[a] != d.score[b]) return d.score[a] > d.score[b];
                      return a < b;
                    });
  const double per = std::min(1.0 / static_cast<double>(bp.k), bp.max_name_tilt / bp.tilt);
  for (std::size_t j = 0; j < m; ++j) w[cand[j]] += bp.tilt * per;
  return w;
}

EquityCurve simulate(const Panel& p, const BacktestParams& bp, const std::vector<Decision>& decisions, BenchKind bench,
                     const std::string& single_ticker) {
  EquityCurve out;
  if (decisions.empty()) return out;
  const std::size_t T = p.T(), N = p.N();
  for (std::size_t j = 0; j < decisions.size(); ++j) {
    if (decisions[j].date + 1 >= T) throw std::invalid_argument("simulate: decision date has no next open");
    if (j > 0 && decisions[j].date <= decisions[j - 1].date)
      throw std::invalid_argument("simulate: decision dates must be strictly increasing");
    if (bench == BenchKind::EqualWeightEligible && !decisions[j].eligible.empty() && decisions[j].eligible.size() != N)
      throw std::invalid_argument("simulate: eligible size != N");
  }
  std::vector<double> fixed_w;
  if (bench == BenchKind::BuyHoldBase || bench == BenchKind::RebalancedBase) {
    fixed_w = base_weights(p, bp);
  } else if (bench == BenchKind::Single) {
    fixed_w.assign(N, 0.0);
    fixed_w[ticker_index(p, single_ticker)] = 1.0;
  }

  Book bk;
  bk.shares.assign(N, 0.0);
  bk.last_px.assign(N, std::numeric_limits<double>::quiet_NaN());
  const std::size_t first_exec = decisions.front().date + 1;
  std::size_t next = 0;
  for (std::size_t t = 0; t < T; ++t) {
    for (std::size_t i = 0; i < N; ++i) {
      const double o = p.open[p.idx(t, i)];
      if (std::isfinite(o)) bk.last_px[i] = o;
    }
    if (next < decisions.size() && decisions[next].date + 1 == t) {
      const bool initial = next == 0;
      const bool trade = bench == BenchKind::None || bench == BenchKind::RebalancedBase ||
                         bench == BenchKind::EqualWeightEligible || initial;
      if (trade) {
        const auto [cost, turnover] =
            execute(p, bp, t,
                    bench == BenchKind::None                  ? target_weights(p, bp, decisions[next])
                    : bench == BenchKind::EqualWeightEligible ? equal_weight(N, decisions[next].eligible)
                                                              : fixed_w,
                    initial, bk, out);
        out.costs += cost;
        if (!initial) out.turnover += turnover;
      }
      ++next;
    }
    for (std::size_t i = 0; i < N; ++i) {
      const double c = p.close[p.idx(t, i)];
      if (std::isfinite(c)) bk.last_px[i] = c;
    }
    if (t >= first_exec) {
      double v = bk.cash;
      for (std::size_t i = 0; i < N; ++i)
        if (bk.shares[i] != 0.0) v += bk.shares[i] * bk.last_px[i];
      out.t.push_back(p.times[t]);
      out.value.push_back(v);
    }
  }
  return out;
}

}  // namespace mr
