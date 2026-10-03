#include "walkforward/external.hpp"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "walkforward/signals.hpp"

namespace mr {

void check_external_name(const std::string& name) {
  const bool chars = !name.empty() && name.size() <= 64 && std::all_of(name.begin(), name.end(), [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '-';
  });
  if (!chars)
    throw std::invalid_argument("external signal name must be 1-64 characters from A-Z a-z 0-9 _ . -, got '" + name +
                                "'");
  bool builtin = name == "blend";
  for (std::size_t s = 0; s < kSignals; ++s) builtin = builtin || name == to_string(static_cast<Signal>(s));
  if (builtin) throw std::invalid_argument("external signal name '" + name + "' is taken by a built-in curve");
}

ExternalSignal parse_external_signal(const std::string& name, const std::string& text, const Panel& panel) {
  check_external_name(name);
  ExternalSignal e;
  e.name = name;
  {
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : text) {
      h ^= c;
      h *= 1099511628211ULL;
    }
    char buf[24];
    std::snprintf(buf, sizeof buf, "%016" PRIx64, h);
    e.digest = buf;
  }
  const std::size_t N = panel.N();
  std::unordered_map<TimePoint, std::size_t> bar_of;
  for (std::size_t t = 0; t < panel.T(); ++t) bar_of.emplace(panel.times[t], t);
  std::unordered_map<std::string, std::size_t> node_of;
  for (std::size_t i = 0; i < N; ++i) node_of.emplace(panel.tickers[i], i);

  auto fail = [&](std::size_t line_no, const std::string& why) {
    throw std::runtime_error("--wf-external " + name + ": line " + std::to_string(line_no) + ": " + why);
  };
  std::istringstream in(text);
  std::string line;
  std::size_t line_no = 0;
  auto next = [&]() {
    if (!std::getline(in, line)) return false;
    ++line_no;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
  };
  if (!next() || line != "t,ticker,score") fail(1, "header must be 't,ticker,score'");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::set<std::pair<std::size_t, std::size_t>> seen;  // (bar, node) pairs given so far, NaN scores included
  while (next()) {
    if (line.empty()) continue;
    const auto c1 = line.find(',');
    const auto c2 = c1 == std::string::npos ? c1 : line.find(',', c1 + 1);
    if (c2 == std::string::npos || line.find(',', c2 + 1) != std::string::npos) fail(line_no, "expected 3 fields");
    const std::string ts = line.substr(0, c1), ticker = line.substr(c1 + 1, c2 - c1 - 1), sc = line.substr(c2 + 1);
    TimePoint t = 0;
    {
      std::size_t pos = 0;
      try {
        t = std::stoll(ts, &pos);
      } catch (const std::exception&) {
        pos = 0;
      }
      if (ts.empty() || pos != ts.size()) fail(line_no, "bad t '" + ts + "'");
    }
    double score = nan;
    if (!sc.empty() && sc != "nan" && sc != "NaN") {
      std::size_t pos = 0;
      try {
        score = std::stod(sc, &pos);
      } catch (const std::exception&) {
        pos = 0;
      }
      if (pos != sc.size() || !std::isfinite(score)) fail(line_no, "bad score '" + sc + "'");
    }
    const auto bt = bar_of.find(t);
    if (bt == bar_of.end()) fail(line_no, "t " + ts + " is not a bar time of the panel");
    const auto nd = node_of.find(ticker);
    if (nd == node_of.end()) {
      ++e.unknown_tickers;
      continue;
    }
    auto [it, fresh] = e.scores.try_emplace(bt->second);
    if (fresh) it->second.assign(N, nan);
    if (!seen.insert({bt->second, nd->second}).second) fail(line_no, "duplicate (t, ticker) " + ts + "," + ticker);
    it->second[nd->second] = score;
  }
  return e;
}

ExternalSignal load_external_signal(const std::string& name, const std::filesystem::path& csv, const Panel& panel) {
  std::ifstream in(csv, std::ios::binary);
  if (!in) throw std::runtime_error("--wf-external " + name + ": cannot read " + csv.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return parse_external_signal(name, ss.str(), panel);
}

}  // namespace mr
