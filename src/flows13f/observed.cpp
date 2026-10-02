#include "flows13f/observed.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <numeric>
#include <tuple>
#include <stdexcept>
#include <map>
#include <unordered_map>

#include "core/csv.hpp"

namespace mr {

namespace {

int col(const std::vector<std::string>& hdr, const std::string& name) {
  for (std::size_t i = 0; i < hdr.size(); ++i) if (hdr[i] == name) return static_cast<int>(i);
  return -1;
}
constexpr double kZeroValueMaxUsd = 1e6;  // shares filed at value 0 worth more than this are a filing error
constexpr double kMaxPriceOff = 100.0;  // implied 13F price vs known price: beyond this the row is a filing error
struct Pos { double prev_shares = 0, cur_shares = 0, prev_value = 0, cur_value = 0; };

bool parse_num(const std::string& s, double& v) {
  if (s.empty()) { v = 0.0; return true; }
  const char* b = s.data(); const char* e = b + s.size();
  auto r = std::from_chars(b, e, v);
  return r.ec == std::errc() && r.ptr == e && std::isfinite(v);
}
bool parse_cik(const std::string& s, std::uint64_t& v) {
  const char* b = s.data(); const char* e = b + s.size();
  auto r = std::from_chars(b, e, v);
  return !s.empty() && r.ec == std::errc() && r.ptr == e;
}

}  // namespace

QuarterHoldings load_quarter(const std::filesystem::path& dir, const std::string& quarter) {
  QuarterHoldings out;
  out.quarter = quarter;
  const auto map_path = dir / "13f" / "cusip_map.csv";
  if (!std::filesystem::exists(map_path)) throw std::runtime_error("13f: missing " + map_path.string());
  const auto map_rows = read_csv_file(map_path);
  std::unordered_map<std::string, std::string> tick;
  if (!map_rows.empty()) {
    const int c = col(map_rows[0], "cusip"), t = col(map_rows[0], "ticker");
    for (std::size_t i = 1; c >= 0 && t >= 0 && i < map_rows.size(); ++i) {
      const auto& r = map_rows[i];
      if (r.size() > static_cast<std::size_t>(std::max(c, t)) && !r[t].empty()) tick[r[c]] = r[t];
    }
  }
  const auto hold_path = dir / "13f" / ("holdings_" + quarter + ".csv");
  if (!std::filesystem::exists(hold_path)) throw std::runtime_error("13f: missing " + hold_path.string());
  const auto rows = read_csv_file(hold_path);
  if (rows.empty()) return out;
  const int ck = col(rows[0], "cik"), cc = col(rows[0], "cusip"), cs = col(rows[0], "shares"), cv = col(rows[0], "value_usd");
  if (ck < 0 || cc < 0 || cs < 0 || cv < 0) throw std::runtime_error("13f: bad header in " + hold_path.string());
  const std::size_t need = static_cast<std::size_t>(std::max({ck, cc, cs, cv})) + 1;
  for (std::size_t i = 1; i < rows.size(); ++i) {
    const auto& r = rows[i];
    std::uint64_t cik = 0;
    double sh = 0, v = 0;
    if (r.size() < need || !parse_cik(r[ck], cik) || !parse_num(r[cs], sh) || !parse_num(r[cv], v)) { ++out.bad_rows; continue; }
    out.total_value += v;
    auto it = tick.find(r[cc]);
    if (it == tick.end()) { out.dropped_value += v; continue; }
    out.rows.push_back({cik, it->second, sh, v});
  }
  return out;
}

ObservedFlows observed_flows(const QuarterHoldings& prev, const QuarterHoldings& cur, const std::vector<std::string>& tickers,
                             const std::vector<double>& prices, const std::function<double(const std::string&)>& split_ratio,
                             const ObservedParams& params) {
  ObservedFlows res;
  res.quarter = cur.quarter;
  std::unordered_map<std::string, std::uint32_t> idx;
  for (std::size_t i = 0; i < tickers.size(); ++i) idx.emplace(tickers[i], static_cast<std::uint32_t>(i));

  // Node set: top_n tickers by total value over prev u cur (ties -> lower index).
  std::vector<double> tv(tickers.size(), 0.0);
  for (const auto* q : {&prev, &cur})
    for (const auto& h : q->rows) {
      auto it = idx.find(h.ticker);
      if (it != idx.end()) tv[it->second] += h.value_usd;
    }
  std::vector<std::uint32_t> order(tickers.size());
  std::iota(order.begin(), order.end(), 0u);
  const std::size_t n = std::min(params.top_n, order.size());
  std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) { return tv[a] > tv[b]; });
  order.resize(n);
  std::sort(order.begin(), order.end());
  res.nodes = order;
  std::vector<std::int32_t> local(tickers.size(), -1);
  for (std::size_t i = 0; i < n; ++i) local[order[i]] = static_cast<std::int32_t>(i);

  struct R { std::uint64_t cik; std::uint32_t node; bool is_cur; double shares, value; };
  std::vector<R> all;
  all.reserve(prev.rows.size() + cur.rows.size());
  for (int w = 0; w < 2; ++w)
    for (const auto& h : (w ? cur : prev).rows) {
      auto it = idx.find(h.ticker);
      if (it == idx.end()) { res.skipped_value += std::abs(h.value_usd); continue; }
      all.push_back({h.cik, it->second, w == 1, h.shares, h.value_usd});
    }
  std::sort(all.begin(), all.end(), [](const R& a, const R& b) {
    return std::tie(a.cik, a.node, a.is_cur, a.shares, a.value) < std::tie(b.cik, b.node, b.is_cur, b.shares, b.value);
  });

  std::vector<double> dense(n * n, 0.0);
  std::vector<std::pair<std::int32_t, double>> src, snk;
  std::vector<double> ratio_cache(tickers.size(), std::nan(""));
  std::size_t i = 0;
  while (i < all.size()) {
    const std::uint64_t cik = all[i].cik;
    ++res.managers;
    src.clear(); snk.clear();
    double sum_out = 0, sum_in = 0;
    while (i < all.size() && all[i].cik == cik) {
      const std::uint32_t node = all[i].node;
      Pos p;
      while (i < all.size() && all[i].cik == cik && all[i].node == node) {
        if (all[i].is_cur) { p.cur_shares += all[i].shares; p.cur_value += all[i].value; }
        else { p.prev_shares += all[i].shares; p.prev_value += all[i].value; }
        ++i;
      }
      if (std::isnan(ratio_cache[node])) ratio_cache[node] = split_ratio(tickers[node]);
      const double ratio = ratio_cache[node];
      double price = prices.size() > node ? prices[node] : std::nan("");
      if (std::isfinite(price) && price > 0) {
        // Each side's implied price must be within 100x of the known price (q's raw basis; q-1's is price * ratio).
        // A side with shares but no value is off too when those shares are worth more than $1M at that price
        // (a giant SHARES error filed at value 0).
        auto off = [&](double sh, double val, double expect) {
          if (!(sh > 0)) return false;
          if (!(val > 0)) return sh * expect > kZeroValueMaxUsd;
          return std::abs(std::log(val / sh / expect)) > std::log(kMaxPriceOff);
        };
        if (off(p.cur_shares, p.cur_value, price) || (ratio > 0 && off(p.prev_shares, p.prev_value, price * ratio))) {
          ++res.inconsistent_positions;
          res.inconsistent_value += std::abs(p.cur_value) + std::abs(p.prev_value);
          continue;
        }
      } else {
        price = std::nan("");
        if (p.cur_shares > 0 && p.cur_value > 0) price = p.cur_value / p.cur_shares;
        else if (p.prev_shares > 0 && ratio > 0 && p.prev_value > 0) price = p.prev_value / (ratio * p.prev_shares);
        if (!(std::isfinite(price) && price > 0)) { res.skipped_value += std::abs(p.cur_value) + std::abs(p.prev_value); continue; }
      }
      const double d = (p.cur_shares - ratio * p.prev_shares) * price;
      if (!std::isfinite(d) || d == 0) continue;
      const std::int32_t l = local[node];
      if (l < 0) { (d < 0 ? res.outside_out : res.outside_in) += std::abs(d); continue; }
      if (d < 0) { src.emplace_back(l, -d); sum_out -= d; }
      else { snk.emplace_back(l, d); sum_in += d; }
    }
    if (sum_out <= 0 && sum_in <= 0) continue;
    const double paired = std::min(sum_in, sum_out);
    res.paired += paired;
    res.unpaired_in += sum_in - paired;
    res.unpaired_out += sum_out - paired;
    if (src.empty() || snk.empty()) continue;
    const double k = paired / (sum_in * sum_out);  // out_i*in_j/sum_in*min(1,sum_in/sum_out)
    for (const auto& [a, out] : src) {
      double* row = &dense[static_cast<std::size_t>(a) * n];
      const double o = out * k;
      for (const auto& [b, in] : snk) row[b] += o * in;
    }
  }
  const double thresh = params.prune_rel * res.paired;
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b) {
      const double v = dense[a * n + b];
      if (v > 0 && v > thresh) res.edges.push_back({order[a], order[b], v});
    }
  return res;
}

}  // namespace mr
