#include "flows13f/observed.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>

#include "core/csv.hpp"

namespace mr {

namespace {
constexpr double kMaxPairs = 1e6;
constexpr std::size_t kMaxSinks = 1000;

int col(const std::vector<std::string>& hdr, const std::string& name) {
  for (std::size_t i = 0; i < hdr.size(); ++i) if (hdr[i] == name) return static_cast<int>(i);
  return -1;
}
double num(const std::string& s) {
  try { return s.empty() ? 0.0 : std::stod(s); } catch (...) { return 0.0; }
}

struct Pos { double prev_shares = 0, cur_shares = 0, prev_value = 0, cur_value = 0; };
}  // namespace

QuarterHoldings load_quarter(const std::filesystem::path& dir, const std::string& quarter) {
  QuarterHoldings out;
  out.quarter = quarter;
  const auto map_rows = read_csv_file(dir / "13f" / "cusip_map.csv");
  std::unordered_map<std::string, std::string> tick;
  if (!map_rows.empty()) {
    const int c = col(map_rows[0], "cusip"), t = col(map_rows[0], "ticker");
    for (std::size_t i = 1; c >= 0 && t >= 0 && i < map_rows.size(); ++i) {
      const auto& r = map_rows[i];
      if (r.size() > static_cast<std::size_t>(std::max(c, t)) && !r[t].empty()) tick[r[c]] = r[t];
    }
  }
  const auto rows = read_csv_file(dir / "13f" / ("holdings_" + quarter + ".csv"));
  if (rows.empty()) return out;
  const int ck = col(rows[0], "cik"), cc = col(rows[0], "cusip"), cs = col(rows[0], "shares"), cv = col(rows[0], "value_usd");
  if (ck < 0 || cc < 0 || cs < 0 || cv < 0) return out;
  const std::size_t need = static_cast<std::size_t>(std::max({ck, cc, cs, cv})) + 1;
  for (std::size_t i = 1; i < rows.size(); ++i) {
    const auto& r = rows[i];
    if (r.size() < need) continue;
    const double v = num(r[cv]);
    out.total_value += v;
    auto it = tick.find(r[cc]);
    if (it == tick.end()) { out.dropped_value += v; continue; }
    out.rows.push_back({static_cast<std::uint64_t>(std::stoull(r[ck])), it->second, num(r[cs]), v});
  }
  return out;
}

ObservedFlows observed_flows(const QuarterHoldings& prev, const QuarterHoldings& cur, const std::vector<std::string>& tickers,
                             const std::vector<double>& prices, const std::function<double(const std::string&)>& split_ratio) {
  ObservedFlows res;
  res.quarter = cur.quarter;
  std::unordered_map<std::string, std::uint32_t> idx;
  for (std::size_t i = 0; i < tickers.size(); ++i) idx.emplace(tickers[i], static_cast<std::uint32_t>(i));

  // cik -> node -> position (ordered maps give deterministic summation order)
  std::map<std::uint64_t, std::map<std::uint32_t, Pos>> mgr;
  for (const auto& h : prev.rows) {
    auto it = idx.find(h.ticker);
    if (it == idx.end()) continue;
    auto& p = mgr[h.cik][it->second];
    p.prev_shares += h.shares; p.prev_value += h.value_usd;
  }
  for (const auto& h : cur.rows) {
    auto it = idx.find(h.ticker);
    if (it == idx.end()) continue;
    auto& p = mgr[h.cik][it->second];
    p.cur_shares += h.shares; p.cur_value += h.value_usd;
  }
  res.managers = mgr.size();

  std::unordered_map<std::uint64_t, double> acc;
  for (const auto& [cik, poss] : mgr) {
    std::vector<std::pair<std::uint32_t, double>> src, snk;  // (node, |d|)
    double sum_out = 0, sum_in = 0;
    for (const auto& [node, p] : poss) {
      const double ratio = split_ratio(tickers[node]);
      double price = prices.size() > node ? prices[node] : std::nan("");
      if (!std::isfinite(price)) {
        if (p.cur_shares > 0 && std::isfinite(p.cur_value / p.cur_shares)) price = p.cur_value / p.cur_shares;
        else if (p.prev_shares > 0 && std::isfinite(p.prev_value / p.prev_shares)) price = p.prev_value / p.prev_shares;
        else continue;
      }
      const double d = (p.cur_shares - ratio * p.prev_shares) * price;
      if (!std::isfinite(d)) continue;
      if (d < 0) { src.emplace_back(node, -d); sum_out -= d; }
      else if (d > 0) { snk.emplace_back(node, d); sum_in += d; }
    }
    if (sum_out <= 0 && sum_in <= 0) continue;
    if (src.empty() || snk.empty()) { res.unpaired_out += sum_out; res.unpaired_in += sum_in; continue; }

    double in_scale = 1.0;
    if (static_cast<double>(src.size()) * static_cast<double>(snk.size()) > kMaxPairs && snk.size() > kMaxSinks) {
      std::stable_sort(snk.begin(), snk.end(), [](auto& a, auto& b) { return a.second > b.second; });
      snk.resize(kMaxSinks);
      double kept = 0;
      for (auto& s : snk) kept += s.second;
      in_scale = sum_in / kept;  // spread the dropped remainder proportionally over kept sinks
    }
    const double paired = std::min(sum_in, sum_out);
    res.paired += paired;
    res.unpaired_in += sum_in - paired;
    res.unpaired_out += sum_out - paired;
    const double k = std::min(1.0, sum_in / sum_out) / sum_in;
    for (const auto& [a, out] : src)
      for (const auto& [b, in] : snk)
        acc[(static_cast<std::uint64_t>(a) << 32) | b] += out * in * in_scale * k;
  }
  res.edges.reserve(acc.size());
  for (const auto& [key, v] : acc)
    res.edges.push_back({static_cast<std::uint32_t>(key >> 32), static_cast<std::uint32_t>(key & 0xffffffffu), v});
  std::sort(res.edges.begin(), res.edges.end(),
            [](const FlowEdge& x, const FlowEdge& y) { return x.from != y.from ? x.from < y.from : x.to < y.to; });
  return res;
}

}  // namespace mr
