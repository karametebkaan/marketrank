#include "market/intraday.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "core/csv.hpp"
#include "core/time.hpp"
#include "market/session_calendar.hpp"

namespace mr {
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------------------------------------------
// Universe
// ---------------------------------------------------------------------------------------------------------------

std::vector<LiquidName> rank_intraday_universe(const std::map<std::string, std::vector<Bar>>& daily, TimePoint from,
                                               TimePoint to, std::size_t n, double min_coverage) {
  std::vector<LiquidName> all;
  std::size_t max_days = 0;
  std::vector<double> dv;
  for (const auto& [ticker, bars] : daily) {
    dv.clear();
    for (const Bar& b : bars) {
      if (b.t < from || b.t >= to) continue;
      const double x = b.v * (b.vw > 0 ? b.vw : b.c);
      if (std::isfinite(x)) dv.push_back(x);
    }
    if (dv.empty()) continue;
    std::sort(dv.begin(), dv.end());
    const std::size_t m = dv.size();
    const double median = m % 2 ? dv[m / 2] : 0.5 * (dv[m / 2 - 1] + dv[m / 2]);
    all.push_back({ticker, median, m});
    max_days = std::max(max_days, m);
  }
  const double floor_days = min_coverage * static_cast<double>(max_days);
  std::erase_if(all, [&](const LiquidName& x) { return static_cast<double>(x.days) < floor_days; });
  std::sort(all.begin(), all.end(), [](const LiquidName& a, const LiquidName& b) {
    return a.median_dollar_volume > b.median_dollar_volume ||
           (a.median_dollar_volume == b.median_dollar_volume && a.ticker < b.ticker);
  });
  if (all.size() > n) all.resize(n);
  return all;
}

void write_intraday_universe(const fs::path& path, const std::vector<LiquidName>& names) {
  if (path.has_parent_path()) fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write " + path.string());
  out << "rank,ticker,median_dollar_volume,days\n";
  char buf[64];
  for (std::size_t k = 0; k < names.size(); ++k) {
    std::snprintf(buf, sizeof buf, "%.2f", names[k].median_dollar_volume);
    out << k + 1 << ',' << names[k].ticker << ',' << buf << ',' << names[k].days << '\n';
  }
  if (!out.flush()) throw std::runtime_error("cannot write " + path.string());
}

std::vector<std::string> read_tickers_file(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read tickers file " + path.string());
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  std::vector<std::string> out;
  std::set<std::string> seen;
  auto add = [&](std::string t) {
    const auto a = t.find_first_not_of(" \t\r"), b = t.find_last_not_of(" \t\r");
    if (a == std::string::npos) return;
    t = t.substr(a, b - a + 1);
    if (seen.insert(t).second) out.push_back(t);
  };
  const CsvRows rows = parse_csv(text);
  if (!rows.empty()) {
    const auto& head = rows.front();
    const auto col = std::find(head.begin(), head.end(), "ticker");
    if (col != head.end()) {
      const auto k = static_cast<std::size_t>(col - head.begin());
      for (std::size_t r = 1; r < rows.size(); ++r)
        if (k < rows[r].size()) add(rows[r][k]);
      return out;
    }
  }
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    add(line);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Sync
// ---------------------------------------------------------------------------------------------------------------

namespace {

using Intervals = std::vector<std::pair<TimePoint, TimePoint>>;

std::unordered_map<std::string, Intervals> load_progress(const fs::path& file) {
  std::unordered_map<std::string, Intervals> out;
  std::ifstream in(file);
  std::string line;
  while (std::getline(in, line)) {
    const auto c1 = line.find(','), c2 = line.find(',', c1 == std::string::npos ? c1 : c1 + 1);
    if (c1 == std::string::npos || c2 == std::string::npos) continue;  // a torn last line is ignored
    try {
      std::size_t p1 = 0, p2 = 0;
      const std::string s1 = line.substr(c1 + 1, c2 - c1 - 1), s2 = line.substr(c2 + 1);
      const TimePoint a = std::stoll(s1, &p1), b = std::stoll(s2, &p2);
      if (p1 != s1.size() || p2 != s2.size() || a >= b) continue;
      out[line.substr(0, c1)].emplace_back(a, b);
    } catch (const std::exception&) {
    }
  }
  return out;
}

// Whether [a, b) lies inside the union of the intervals.
bool covered(Intervals iv, TimePoint a, TimePoint b) {
  std::sort(iv.begin(), iv.end());
  TimePoint reach = a;
  for (const auto& [x, y] : iv) {
    if (x > reach) break;
    reach = std::max(reach, y);
    if (reach >= b) return true;
  }
  return reach >= b;
}

TimePoint next_month(TimePoint t) {
  const Civil c = civil_from_days(floor_div(t, 86400));
  return c.m == 12 ? utc_seconds(c.y + 1, 1, 1) : utc_seconds(c.y, c.m + 1, 1);
}

}  // namespace

IntradaySyncStats sync_intraday(AlpacaClient& client, Lake& lake, const std::vector<std::string>& tickers_in,
                                TimePoint from, TimePoint to, const IntradaySyncOptions& opt) {
  IntradaySyncStats st;
  if (from >= to) return st;
  std::vector<std::string> tickers;
  {
    std::set<std::string> seen;
    for (const auto& t : tickers_in)
      if (seen.insert(t).second) tickers.push_back(t);
  }
  const fs::path progress_file = opt.progress_file.empty() ? lake.root() / "progress_15m.csv" : opt.progress_file;
  auto progress = load_progress(progress_file);
  const TimePoint cutoff = opt.now == std::numeric_limits<TimePoint>::max() ? opt.now : opt.now - 16 * 60;
  std::set<std::string> failed;
  const std::size_t requests_before = client.requests();
  for (TimePoint a = from; a < to;) {
    const TimePoint b = std::min(to, next_month(a));
    std::vector<std::string> group;
    for (const auto& t : tickers) {
      auto it = progress.find(t);
      if (it != progress.end() && covered(it->second, a, b)) ++st.chunks_skipped;
      else group.push_back(t);
    }
    const TimePoint req_end = std::min(b, cutoff);
    if (!group.empty() && req_end > a) {
      const bool complete = b <= cutoff;
      const FetchResult r = client.fetch_bars(
          group, alpaca_timeframe(Timeframe::Min15), a, req_end,
          [&](const std::vector<std::string>& batch_symbols, const std::map<std::string, std::vector<Bar>>& batch_bars) {
            std::vector<LakeRow> rows;
            for (const auto& [ticker, bars] : batch_bars) {
              st.raw_bars += bars.size();
              for (const Bar& bar : bars)
                if (bar.t >= a && bar.t < req_end && is_regular_session_bar(bar.t)) rows.push_back({ticker, bar});
            }
            st.kept_bars += rows.size();
            lake.write(Timeframe::Min15, rows, {});
            st.chunks_fetched += batch_symbols.size();
            if (!complete) return;
            {
              std::ofstream out(progress_file, std::ios::app);
              for (const auto& t : batch_symbols) {
                out << t << ',' << a << ',' << b << '\n';
                progress[t].emplace_back(a, b);
              }
              out.flush();
              if (!out) throw std::runtime_error("cannot append to " + progress_file.string());
            }
            fsync_path(progress_file);
          });
      failed.insert(r.stale.begin(), r.stale.end());
      if (opt.log)
        std::cerr << "sync-intraday: " << format_rfc3339(a).substr(0, 10) << " .. " << format_rfc3339(b).substr(0, 10)
                  << ": " << group.size() - r.stale.size() << " tickers fetched, " << r.stale.size() << " failed, "
                  << st.kept_bars << " bars kept so far, " << client.requests() - requests_before << " requests\n";
    }
    a = b;
  }
  st.requests = client.requests() - requests_before;
  st.failed.assign(failed.begin(), failed.end());
  return st;
}

// ---------------------------------------------------------------------------------------------------------------
// Session index
// ---------------------------------------------------------------------------------------------------------------

SessionIndex session_index(const std::vector<TimePoint>& times, TimePoint bar_seconds) {
  SessionIndex si;
  const std::size_t T = times.size();
  si.session.resize(T);
  si.slot.resize(T);
  si.first.assign(T, 0);
  si.last.assign(T, 0);
  std::int64_t prev_day = std::numeric_limits<std::int64_t>::min();
  for (std::size_t t = 0; t < T; ++t) {
    const EtTime et = to_eastern(times[t]);
    const std::int64_t day = days_from_civil(et.date.y, et.date.m, et.date.d);
    if (t == 0 || day != prev_day) {
      ++si.sessions;
      char buf[16];
      std::snprintf(buf, sizeof buf, "%04d-%02u-%02u", et.date.y, et.date.m, et.date.d);
      si.dates.push_back(buf);
      si.first[t] = 1;
      if (t > 0) si.last[t - 1] = 1;
    }
    prev_day = day;
    si.session[t] = static_cast<std::uint32_t>(si.sessions - 1);
    const int minute = et.hour * 60 + et.minute - (9 * 60 + 30);
    si.slot[t] = static_cast<std::uint32_t>(std::max(0, minute) / static_cast<int>(bar_seconds / 60));
  }
  if (T > 0) si.last[T - 1] = 1;
  return si;
}

}  // namespace mr
