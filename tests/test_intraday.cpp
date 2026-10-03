// M5 intraday: 15-minute timeframe, regular-session filter, intraday universe, --sync-intraday (fake HTTP),
// session index, the intraday pipeline preset and the 15m --export-panel (label mask, prior edges, causality).
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cli/export_panel.hpp"
#include "core/time.hpp"
#include "market/alpaca_client.hpp"
#include "market/intraday.hpp"
#include "market/panel.hpp"
#include "market/session_calendar.hpp"
#include "pipeline/core_pipeline.hpp"
#include "storage/lake.hpp"
#include "test_util.hpp"

using namespace mr;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------------------------------------------
// Timeframe
// ---------------------------------------------------------------------------------------------------------------

TEST_CASE("Timeframe::Min15 is 15m, 900 s, Alpaca 15Min; 1h/1d/1w unchanged") {
  CHECK(to_string(Timeframe::Min15) == "15m");
  CHECK(parse_timeframe("15m") == Timeframe::Min15);
  CHECK(timeframe_seconds(Timeframe::Min15) == 900);
  CHECK(alpaca_timeframe(Timeframe::Min15) == "15Min");
  CHECK(alpaca_timeframe(Timeframe::Hour) == "30Min");
  CHECK(alpaca_timeframe(Timeframe::Day) == "1Day");
  CHECK(alpaca_timeframe(Timeframe::Week) == "1Week");
  CHECK(to_string(Timeframe::Day) == "1d");
  CHECK(timeframe_seconds(Timeframe::Day) == 86400);
}

TEST_CASE("the lake stores 15m bars in their own tf=15m partition") {
  auto dir = test::temp_dir("lake15");
  Lake lake(dir);
  const TimePoint t = utc_seconds(2026, 9, 30, 13, 30);
  lake.write(Timeframe::Min15, {{"AAPL", {t, 1, 2, 0.5, 1.5, 100, 1.2}}}, {});
  CHECK(fs::is_directory(dir / "bars" / "tf=15m"));
  CHECK_FALSE(fs::exists(dir / "bars" / "tf=1d"));
  const auto r = lake.read(Timeframe::Min15, {"AAPL"}, 0, 1LL << 40);
  REQUIRE(r.at("AAPL").size() == 1);
  CHECK(r.at("AAPL")[0].c == 1.5);
  CHECK(lake.read(Timeframe::Day, {"AAPL"}, 0, 1LL << 40).empty());
}

// ---------------------------------------------------------------------------------------------------------------
// Session filter: DST and half days
// ---------------------------------------------------------------------------------------------------------------

TEST_CASE("early closes: day after Thanksgiving, July 3 and Dec 24 on Mon-Thu") {
  CHECK(is_early_close({2024, 11, 29}));
  CHECK(is_early_close({2024, 12, 24}));
  CHECK(is_early_close({2025, 7, 3}));
  CHECK(is_early_close({2025, 11, 28}));
  CHECK(is_early_close({2025, 12, 24}));
  CHECK(is_early_close({2026, 11, 27}));
  CHECK(is_early_close({2026, 12, 24}));
  CHECK(is_early_close({2024, 7, 3}));
  CHECK_FALSE(is_early_close({2026, 7, 3}));   // Friday: the observed Independence Day holiday
  CHECK_FALSE(is_early_close({2026, 7, 2}));
  CHECK_FALSE(is_early_close({2021, 12, 24}));  // Friday: observed Christmas
  CHECK_FALSE(is_early_close({2025, 11, 27}));  // Thanksgiving itself is a holiday, not an early close
  CHECK_FALSE(is_early_close({2025, 10, 1}));
  CHECK(session_close_minute({2025, 11, 28}) == 13 * 60);
  CHECK(session_close_minute({2025, 11, 26}) == 16 * 60);
}

TEST_CASE("regular-session 15m bars: 09:30 <= start < close ET, across DST, half days and weekends") {
  // 2025-03-07 (Fri) is EST (UTC-5): 09:30 ET = 14:30Z.
  CHECK_FALSE(is_regular_session_bar(utc_seconds(2025, 3, 7, 14, 15)));
  CHECK(is_regular_session_bar(utc_seconds(2025, 3, 7, 14, 30)));
  CHECK(is_regular_session_bar(utc_seconds(2025, 3, 7, 20, 45)));  // 15:45 EST, the last bar
  CHECK_FALSE(is_regular_session_bar(utc_seconds(2025, 3, 7, 21, 0)));
  // 2025-03-10 (Mon) is EDT (UTC-4): 09:30 ET = 13:30Z.
  CHECK_FALSE(is_regular_session_bar(utc_seconds(2025, 3, 10, 13, 15)));
  CHECK(is_regular_session_bar(utc_seconds(2025, 3, 10, 13, 30)));
  CHECK(is_regular_session_bar(utc_seconds(2025, 3, 10, 19, 45)));
  CHECK_FALSE(is_regular_session_bar(utc_seconds(2025, 3, 10, 20, 0)));
  // 2024-11-29 half day (EST): the session ends at 13:00 ET = 18:00Z.
  CHECK(is_regular_session_bar(utc_seconds(2024, 11, 29, 17, 45)));
  CHECK_FALSE(is_regular_session_bar(utc_seconds(2024, 11, 29, 18, 0)));
  CHECK_FALSE(is_regular_session_bar(utc_seconds(2024, 11, 29, 20, 45)));
  // Saturday.
  CHECK_FALSE(is_regular_session_bar(utc_seconds(2025, 3, 8, 15, 0)));
  // Off-grid starts are not 15m bars.
  CHECK_FALSE(is_regular_session_bar(utc_seconds(2025, 3, 10, 13, 40)));

  std::vector<Bar> in;
  for (TimePoint t = utc_seconds(2024, 11, 29, 9); t < utc_seconds(2024, 11, 30, 1); t += 900) in.push_back({t, 1, 1, 1, 1, 1, 1});
  const auto kept = filter_regular_session(in);
  REQUIRE(kept.size() == 14);  // 09:30 .. 12:45 EST
  CHECK(kept.front().t == utc_seconds(2024, 11, 29, 14, 30));
  CHECK(kept.back().t == utc_seconds(2024, 11, 29, 17, 45));
}

// ---------------------------------------------------------------------------------------------------------------
// Intraday universe
// ---------------------------------------------------------------------------------------------------------------

TEST_CASE("intraday universe: top N by median daily dollar volume in the window, ties by ticker, coverage floor") {
  std::map<std::string, std::vector<Bar>> daily;
  const TimePoint from = utc_seconds(2025, 1, 6), to = utc_seconds(2025, 1, 16);  // 10 days, [from, to)
  auto series = [&](double dv, int days, TimePoint start) {
    std::vector<Bar> s;
    for (int d = 0; d < days; ++d) s.push_back({start + d * 86400 + 75600, 10, 10, 10, 10, dv / 20.0, 20});  // vw*v = dv
    return s;
  };
  daily["BIG"] = series(5e9, 10, from);
  daily["TIEB"] = series(1e9, 10, from);
  daily["TIEA"] = series(1e9, 10, from);
  daily["SMALL"] = series(1e6, 10, from);
  daily["NEWIPO"] = series(9e9, 3, from + 7 * 86400);  // too few days in the window
  auto out_of_window = series(1e12, 10, from - 20 * 86400);
  daily["OLD"] = out_of_window;  // only bars before the window
  // A bar outside the window must not count: BIG's pre-window bar would raise its median.
  daily["BIG"].insert(daily["BIG"].begin(), Bar{from - 86400, 1, 1, 1, 1, 1e12, 1});
  const auto top = rank_intraday_universe(daily, from, to, 3);
  REQUIRE(top.size() == 3);
  CHECK(top[0].ticker == "BIG");
  CHECK(top[0].median_dollar_volume == doctest::Approx(5e9));
  CHECK(top[0].days == 10);
  CHECK(top[1].ticker == "TIEA");
  CHECK(top[2].ticker == "TIEB");

  auto dir = test::temp_dir("iuniv");
  write_intraday_universe(dir / "u.csv", top);
  CHECK(read_tickers_file(dir / "u.csv") == std::vector<std::string>{"BIG", "TIEA", "TIEB"});
  test::write_file(dir / "plain.txt", "# comment\nAAPL\n\nMSFT\n");
  CHECK(read_tickers_file(dir / "plain.txt") == std::vector<std::string>{"AAPL", "MSFT"});
}

// ---------------------------------------------------------------------------------------------------------------
// --sync-intraday with a fake Alpaca
// ---------------------------------------------------------------------------------------------------------------

namespace {

AlpacaConfig fake_config() {
  AlpacaConfig c;
  c.key_id = "k";
  c.secret = "s";
  c.backoff_initial_ms = 0;
  c.backoff_max_ms = 0;
  c.max_retries = 0;
  c.min_request_interval_ms = 0;
  return c;
}

std::string query_value(const std::string& path, const std::string& key) {
  const auto p = path.find("&" + key + "=") != std::string::npos ? path.find("&" + key + "=") + key.size() + 2
                                                                 : path.find("?" + key + "=") + key.size() + 2;
  const auto e = path.find('&', p);
  return path.substr(p, e == std::string::npos ? std::string::npos : e - p);
}

// A fake /v2/stocks/bars: 15Min bars 04:00-20:00 ET (extended hours too) on weekdays in [start, end) for every
// requested symbol, paged `page_size` bars per response with a numeric page token.
struct FakeAlpaca {
  std::size_t page_size = 50;
  std::vector<std::string> paths;
  std::set<std::string> fail_starts;  // requests whose start= is listed answer 403

  HttpResponse operator()(const std::string& path) {
    paths.push_back(path);
    const std::string start_s = query_value(path, "start");
    if (fail_starts.count(start_s)) return {403, "forbidden"};
    const TimePoint start = parse_rfc3339(start_s), end = parse_rfc3339(query_value(path, "end"));
    std::vector<std::string> syms;
    {
      std::string s = query_value(path, "symbols");
      std::size_t a = 0;
      for (;;) {
        const auto c = s.find(',', a);
        syms.push_back(s.substr(a, c == std::string::npos ? std::string::npos : c - a));
        if (c == std::string::npos) break;
        a = c + 1;
      }
    }
    std::vector<std::pair<std::string, TimePoint>> all;
    for (const auto& sym : syms)
      for (TimePoint t = start - start % 900; t < end; t += 900) {
        if (t < start) continue;
        const EtTime et = to_eastern(t);
        const int m = et.hour * 60 + et.minute;
        if (et.weekday == 0 || et.weekday == 6 || m < 4 * 60 || m >= 20 * 60) continue;
        all.emplace_back(sym, t);
      }
    std::size_t off = 0;
    if (path.find("page_token=") != std::string::npos) off = std::stoul(query_value(path, "page_token"));
    nlohmann::json j;
    j["bars"] = nlohmann::json::object();
    for (std::size_t k = off; k < std::min(all.size(), off + page_size); ++k) {
      const double px = 100.0 + static_cast<double>(all[k].second % 7919) / 100.0;
      j["bars"][all[k].first].push_back({{"t", format_rfc3339(all[k].second)}, {"o", px}, {"h", px + 1},
                                          {"l", px - 1}, {"c", px + 0.5}, {"v", 1000}, {"vw", px}});
    }
    if (off + page_size < all.size()) j["next_page_token"] = std::to_string(off + page_size);
    else j["next_page_token"] = nullptr;
    return {200, j.dump()};
  }
};

std::size_t regular_bars(TimePoint from, TimePoint to) {
  std::size_t n = 0;
  for (TimePoint t = from; t < to; t += 900) n += is_regular_session_bar(t) ? 1 : 0;
  return n;
}

}  // namespace

TEST_CASE("sync-intraday: monthly chunks, pagination, session filter, tf=15m; a rerun requests nothing") {
  auto dir = test::temp_dir("sync15");
  FakeAlpaca fake;
  const TimePoint from = utc_seconds(2024, 11, 25), to = utc_seconds(2024, 12, 4);  // spans a month end and a half day
  const std::vector<std::string> tickers{"AAA", "BBB"};
  {
    Lake lake(dir);
    AlpacaClient client(fake_config(), [&](const std::string& p) { return fake(p); });
    IntradaySyncStats st = sync_intraday(client, lake, tickers, from, to);
    CHECK(st.failed.empty());
    CHECK(st.chunks_fetched == 4);  // 2 tickers x {Nov, Dec}
    CHECK(st.chunks_skipped == 0);
    CHECK(st.kept_bars == 2 * regular_bars(from, to));
    CHECK(st.raw_bars > st.kept_bars);
    // Pagination was followed (page_size 50 is far below one month of bars).
    bool paged = false;
    for (const auto& p : fake.paths) paged = paged || p.find("page_token=") != std::string::npos;
    CHECK(paged);
    CHECK(fake.paths.front().find("timeframe=15Min") != std::string::npos);
    CHECK(fake.paths.front().find("start=2024-11-25T00:00:00Z") != std::string::npos);
    CHECK(fake.paths.front().find("end=2024-12-01T00:00:00Z") != std::string::npos);
    const auto r = lake.read(Timeframe::Min15, tickers, 0, 1LL << 40);
    REQUIRE(r.at("AAA").size() == regular_bars(from, to));
    for (const Bar& b : r.at("AAA")) CHECK(is_regular_session_bar(b.t));
    // Thanksgiving Friday 2024-11-29 has 14 bars (09:30..12:45), a normal day 26.
    std::map<unsigned, std::size_t> per_day;
    for (const Bar& b : r.at("AAA")) ++per_day[to_eastern(b.t).date.d];
    CHECK(per_day[29] == 14);
    CHECK(per_day[27] == 26);
    CHECK(per_day[28] == 26);  // the fake has no holidays; the real feed has no regular bars on Thanksgiving
    CHECK(lake.read(Timeframe::Day, tickers, 0, 1LL << 40).empty());
  }
  {
    fake.paths.clear();
    Lake lake(dir);
    AlpacaClient client(fake_config(), [&](const std::string& p) { return fake(p); });
    IntradaySyncStats st = sync_intraday(client, lake, tickers, from, to);
    CHECK(fake.paths.empty());
    CHECK(st.chunks_fetched == 0);
    CHECK(st.chunks_skipped == 4);
  }
}

TEST_CASE("sync-intraday resumes: only the failed chunk and new tickers are fetched again") {
  auto dir = test::temp_dir("sync15r");
  FakeAlpaca fake;
  fake.page_size = 1000;
  const TimePoint from = utc_seconds(2025, 1, 20), to = utc_seconds(2025, 2, 5);
  fake.fail_starts.insert("2025-02-01T00:00:00Z");
  {
    Lake lake(dir);
    AlpacaClient client(fake_config(), [&](const std::string& p) { return fake(p); });
    const auto st = sync_intraday(client, lake, {"AAA", "BBB"}, from, to);
    CHECK(st.failed == std::vector<std::string>{"AAA", "BBB"});
    CHECK(st.chunks_fetched == 2);
  }
  fake.fail_starts.clear();
  fake.paths.clear();
  {
    Lake lake(dir);
    AlpacaClient client(fake_config(), [&](const std::string& p) { return fake(p); });
    const auto st = sync_intraday(client, lake, {"AAA", "BBB", "CCC"}, from, to);
    CHECK(st.failed.empty());
    CHECK(st.chunks_skipped == 2);  // AAA, BBB January
    CHECK(st.chunks_fetched == 4);  // AAA, BBB February; CCC January and February
    REQUIRE(fake.paths.size() == 2);
    CHECK(fake.paths[0].find("symbols=CCC&") != std::string::npos);  // January: only the new ticker
    CHECK(fake.paths[0].find("start=2025-01-20T00:00:00Z") != std::string::npos);
    CHECK(fake.paths[1].find("symbols=AAA,BBB,CCC&") != std::string::npos);
    CHECK(fake.paths[1].find("start=2025-02-01T00:00:00Z") != std::string::npos);
    const auto r = lake.read(Timeframe::Min15, {"AAA", "CCC"}, 0, 1LL << 40);
    CHECK(r.at("AAA").size() == regular_bars(from, to));
    CHECK(r.at("CCC").size() == regular_bars(from, to));
  }
}

TEST_CASE("sync-intraday does not mark a chunk done when it reaches past `now`") {
  auto dir = test::temp_dir("sync15n");
  FakeAlpaca fake;
  fake.page_size = 1000;
  const TimePoint from = utc_seconds(2025, 1, 27), to = utc_seconds(2025, 2, 5);
  IntradaySyncOptions opt;
  opt.now = utc_seconds(2025, 2, 3, 15);
  for (int run = 0; run < 2; ++run) {
    Lake lake(dir);
    AlpacaClient client(fake_config(), [&](const std::string& p) { return fake(p); });
    const auto st = sync_intraday(client, lake, {"AAA"}, from, to, opt);
    CHECK(st.chunks_fetched == (run == 0 ? 2u : 1u));  // February is refetched: it was not complete
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Session index of a 15m panel
// ---------------------------------------------------------------------------------------------------------------

namespace {

// Session times: the regular 15m bars of the given ET dates (weekday, early closes honoured).
std::vector<TimePoint> session_times(const std::vector<Civil>& days) {
  std::vector<TimePoint> out;
  for (const Civil& d : days)
    for (TimePoint t = utc_seconds(d.y, d.m, d.d, 12); t < utc_seconds(d.y, d.m, d.d, 23); t += 900)
      if (is_regular_session_bar(t)) out.push_back(t);
  return out;
}

// A hand-built 15m panel: N names, random-walk prices with a common factor and lead-lag, U-shaped volume.
Panel intraday_panel(std::size_t N, const std::vector<Civil>& days, unsigned seed = 7) {
  Panel p;
  p.times = session_times(days);
  for (std::size_t i = 0; i < N; ++i) p.tickers.push_back("T" + std::to_string(i));
  const std::size_t T = p.T();
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> z(0.0, 1.0);
  std::vector<double> px(N), prev_r(N, 0.0);
  for (std::size_t i = 0; i < N; ++i) px[i] = 20.0 + 5.0 * static_cast<double>(i);
  const double nan = std::nan("");
  for (auto* v : {&p.open, &p.high, &p.low, &p.close, &p.volume, &p.vwap}) v->assign(T * N, nan);
  const SessionIndex si = session_index(p.times);
  for (std::size_t t = 0; t < T; ++t) {
    const double f = 0.002 * z(rng);
    const double u = std::abs(static_cast<double>(si.slot[t]) - 12.5) / 12.5;  // U-shape over the day
    std::vector<double> r(N);
    for (std::size_t i = 0; i < N; ++i) {
      r[i] = f * (0.5 + 0.1 * static_cast<double>(i % 5)) + 0.003 * z(rng) + (i > 0 ? 0.3 * prev_r[i - 1] : 0.0);
      if (si.first[t]) r[i] += 0.01 * z(rng);  // overnight gap
      const double o = px[i];
      px[i] *= std::exp(r[i]);
      const std::size_t k = p.idx(t, i);
      if ((t * 31 + i * 17) % 97 == 0) continue;  // an occasional missing bar
      p.open[k] = o;
      p.close[k] = px[i];
      p.high[k] = std::max(o, px[i]) * 1.001;
      p.low[k] = std::min(o, px[i]) * 0.999;
      p.vwap[k] = 0.5 * (o + px[i]);
      p.volume[k] = (1000.0 + 300.0 * static_cast<double>(i)) * (1.0 + 2.0 * u * u) * std::exp(0.3 * z(rng));
    }
    prev_r = r;
  }
  return p;
}

std::vector<Civil> trading_days() {
  // Mon 2024-11-25 .. Fri 2024-12-06 without Thanksgiving (11-28); 11-29 is a half day.
  return {{2024, 11, 25}, {2024, 11, 26}, {2024, 11, 27}, {2024, 11, 29}, {2024, 12, 2},
          {2024, 12, 3},  {2024, 12, 4},  {2024, 12, 5},  {2024, 12, 6}};
}

IntradayExportParams small_export_params() {
  IntradayExportParams ep;
  ep.core.min_dollar_volume = 0;
  ep.core.adv_window = 26;
  ep.core.corr_window = 52;
  ep.vol_window = 52;
  ep.vol_need = 26;
  ep.dv_sessions = 3;
  ep.dv_need = 2;
  ep.elig_window = 26;
  ep.elig_min_dollar_volume = 0;
  ep.elig_top_n = 8;
  return ep;
}

bool same_prefix(const std::vector<float>& a, const std::vector<float>& b, std::size_t N, std::size_t upto) {
  for (std::size_t k = 0; k < (upto + 1) * N; ++k)
    if (!(a[k] == b[k] || (std::isnan(a[k]) && std::isnan(b[k])))) return false;
  return true;
}

std::vector<char> read_bytes(const fs::path& f) {
  std::ifstream in(f, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(in), {});
}

}  // namespace

TEST_CASE("session index: session per bar, slot, first/last-of-session flags (half day included)") {
  const auto times = session_times({{2024, 11, 27}, {2024, 11, 29}, {2024, 12, 2}});
  REQUIRE(times.size() == 26 + 14 + 26);
  const SessionIndex si = session_index(times);
  REQUIRE(si.session.size() == times.size());
  CHECK(si.sessions == 3);
  CHECK(si.session[0] == 0);
  CHECK(si.session[25] == 0);
  CHECK(si.session[26] == 1);
  CHECK(si.session[39] == 1);
  CHECK(si.session[40] == 2);
  CHECK(si.slot[0] == 0);
  CHECK(si.slot[25] == 25);
  CHECK(si.slot[39] == 13);
  CHECK(si.first[0] == 1);
  CHECK(si.first[1] == 0);
  CHECK(si.last[25] == 1);
  CHECK(si.first[26] == 1);
  CHECK(si.last[39] == 1);  // half day: 12:45 is the last bar
  CHECK(si.first[40] == 1);
  CHECK(si.last[65] == 1);
  std::size_t firsts = 0, lasts = 0;
  for (std::size_t t = 0; t < times.size(); ++t) firsts += si.first[t], lasts += si.last[t];
  CHECK(firsts == 3);
  CHECK(lasts == 3);
  CHECK(si.dates[1] == "2024-11-29");
}

TEST_CASE("intraday preset: wall-clock-equivalent windows in bars") {
  const CoreParams p = CoreParams::market_rank_intraday();
  const CoreParams d = CoreParams::market_rank();
  CHECK(p.adv_window == 520);
  CHECK(p.corr_window == 520);
  CHECK(p.stale_bars == 26);
  CHECK(p.halflife_slow == 1e9);
  CHECK(p.halflife_fast == 26);
  CHECK(p.min_dollar_volume == doctest::Approx(1e6 / 26));
  CHECK(p.pressure == d.pressure);
  CHECK(p.transition == d.transition);
  CHECK(p.alpha == d.alpha);
  CHECK(p.h_ref == d.h_ref);
  p.validate();
}

TEST_CASE("label_6: open[t+7]/open[t+1]-1, NaN unless bars t+1..t+7 are one session") {
  const Panel p = intraday_panel(5, {{2024, 11, 26}, {2024, 11, 27}, {2024, 11, 29}, {2024, 12, 2}});
  const SessionIndex si = session_index(p.times);
  const auto lab = export_session_label(p, si, 6);
  std::size_t finite = 0;
  for (std::size_t t = 0; t < p.T(); ++t)
    for (std::size_t i = 0; i < p.N(); ++i) {
      const float v = lab[p.idx(t, i)];
      const bool same = t + 7 < p.T() && si.session[t + 1] == si.session[t + 7];
      const double a = p.open[p.idx(std::min(t + 1, p.T() - 1), i)];
      const double b = p.open[p.idx(std::min(t + 7, p.T() - 1), i)];
      if (!same || !std::isfinite(a) || !std::isfinite(b)) {
        CHECK(std::isnan(v));
      } else {
        CHECK(v == static_cast<float>(b / a - 1.0));
        ++finite;
      }
    }
  CHECK(finite > 0);
  // Bar 19 of a full session: t+7 = 26 is the next session's first bar -> masked. Bar 18: t+7 = 25 -> finite.
  CHECK(std::isnan(lab[p.idx(19, 0)]));
  CHECK(std::isfinite(lab[p.idx(18, 0)]));
  // The last bar of a session decides overnight, but its window t+1..t+7 lies inside the next session: open to open
  // of the next morning, no overnight return in the label.
  CHECK(std::isfinite(lab[p.idx(25, 0)]));
}

TEST_CASE("15m export: arrays, session meta, prior edges equal the pipeline's Frame::P kept edges") {
  const Panel p = intraday_panel(12, trading_days());
  const std::size_t T = p.T(), N = p.N();
  const IntradayExportParams ep = small_export_params();
  std::vector<std::string> sectors(N, "S");
  auto dir = test::temp_dir("export15");
  export_panel_intraday(p, sectors, ep, dir);

  std::ifstream mf(dir / "meta.json");
  const auto meta = nlohmann::json::parse(mf);
  CHECK(meta["timeframe"] == "15m");
  CHECK(meta["T"] == T);
  CHECK(meta["N"] == N);
  CHECK(meta["session"]["index"].size() == T);
  CHECK(meta["session"]["first_of_session"].size() == T);
  CHECK(meta["session"]["last_of_session"].size() == T);
  CHECK(meta["label_horizons"]["label_6"] == 6);
  CHECK(meta["files"]["label_6"] == "label_6.f32");
  CHECK(meta["files"]["vol"] == "vol20.f32");
  CHECK(meta["rebalance"]["mode"] == "every_bar");
  for (const auto& name : intraday_array_names()) {
    CAPTURE(name);
    CHECK(fs::file_size(dir / (name + ".f32")) == T * N * 4);
  }
  const auto lab = read_f32(dir / "label_6.f32");
  const auto want = export_session_label(p, session_index(p.times), 6);
  CHECK(std::memcmp(lab.data(), want.data(), lab.size() * 4) == 0);

  const PriorEdges prior = read_prior(dir / "prior");
  REQUIRE(prior.offsets.size() == T + 1);
  CHECK(prior.offsets[0] == 0);
  CHECK(prior.offsets[1] == 0);  // bar 0: no step
  CHECK(prior.offsets[T] == prior.edges.size());
  CHECK(prior.edges.size() > 0);
  CHECK(fs::file_size(dir / "prior" / "edges.bin") == prior.edges.size() * 20);

  CorePipeline pipe(N, ep.core);
  const auto pa_active = read_f32(dir / "active.f32");
  std::size_t compared = 0;
  for (std::size_t t = 1; t < T; ++t) {
    const Frame f = pipe.step(p, t);
    std::vector<PriorEdge> want_t;
    for (std::size_t i = 0; i < N; ++i) {
      CHECK(pa_active[p.idx(t, i)] == (f.active[i] ? 1.0f : 0.0f));
      if (!f.active[i]) continue;
      for (auto e = f.P.row_ptr[i]; e < f.P.row_ptr[i + 1]; ++e)
        want_t.push_back({static_cast<std::uint32_t>(t), static_cast<std::uint32_t>(i), f.P.col[e],
                          static_cast<float>(f.P.val[e]), static_cast<float>(f.P.raw[e])});
    }
    REQUIRE(prior.offsets[t + 1] - prior.offsets[t] == want_t.size());
    for (std::size_t k = 0; k < want_t.size(); ++k) {
      const PriorEdge& g = prior.edges[prior.offsets[t] + k];
      CHECK(g.t == want_t[k].t);
      CHECK(g.src == want_t[k].src);
      CHECK(g.dst == want_t[k].dst);
      CHECK(g.p == want_t[k].p);
      CHECK(g.raw == want_t[k].raw);
      ++compared;
    }
  }
  CHECK(compared == prior.edges.size());
  // Sorted by (t, src, dst).
  for (std::size_t k = 1; k < prior.edges.size(); ++k) {
    const auto& a = prior.edges[k - 1];
    const auto& b = prior.edges[k];
    CHECK((a.t < b.t || (a.t == b.t && (a.src < b.src || (a.src == b.src && a.dst < b.dst)))));
  }
}

TEST_CASE("15m export is causal: perturbing bars after t leaves features and prior at bars <= t unchanged") {
  const Panel base = intraday_panel(10, trading_days(), 11);
  const std::size_t N = base.N();
  const IntradayExportParams ep = small_export_params();
  std::vector<std::string> sectors(N, "S");
  auto d0 = test::temp_dir("causal15_base");
  export_panel_intraday(base, sectors, ep, d0);
  const PriorEdges p0 = read_prior(d0 / "prior");
  for (std::size_t cut : {std::size_t{60}, std::size_t{140}}) {
    CAPTURE(cut);
    Panel q = base;
    for (std::size_t t = cut + 1; t < q.T(); ++t)
      for (std::size_t i = 0; i < N; ++i) {
        const double f = 1.0 + 0.05 * std::sin(static_cast<double>(t * 13 + i * 5));
        for (auto* v : {&q.open, &q.close, &q.vwap, &q.high, &q.low})
          if (std::isfinite((*v)[q.idx(t, i)])) (*v)[q.idx(t, i)] *= f;
        if (std::isfinite(q.volume[q.idx(t, i)])) q.volume[q.idx(t, i)] *= (i + t) % 2 ? 40.0 : 0.03;
      }
    auto d1 = test::temp_dir("causal15_q");
    export_panel_intraday(q, sectors, ep, d1);
    for (const auto& name : intraday_array_names()) {
      if (name == "label_6") continue;  // forward by construction
      CAPTURE(name);
      const auto a = read_f32(d0 / (name + ".f32")), b = read_f32(d1 / (name + ".f32"));
      CHECK(same_prefix(a, b, N, cut));
      if (name == "pressure" || name == "dvshock" || name == "vol20" || name == "ret1")
        CHECK_FALSE(same_prefix(a, b, N, cut + 1));  // not vacuous
    }
    const PriorEdges p1 = read_prior(d1 / "prior");
    REQUIRE(p1.offsets.size() == p0.offsets.size());
    for (std::size_t t = 0; t <= cut + 1; ++t) CHECK(p1.offsets[t] == p0.offsets[t]);
    const std::size_t n_e = p0.offsets[cut + 1];
    bool same = true;
    for (std::size_t k = 0; k < n_e; ++k) {
      const auto &a = p0.edges[k], &b = p1.edges[k];
      same = same && a.t == b.t && a.src == b.src && a.dst == b.dst && a.p == b.p && a.raw == b.raw;
    }
    CHECK(same);
    // The prior after the cut does move.
    bool moved = p1.offsets.back() != p0.offsets.back();
    for (std::size_t k = n_e; !moved && k < std::min(p0.edges.size(), p1.edges.size()); ++k)
      moved = p0.edges[k].p != p1.edges[k].p || p0.edges[k].dst != p1.edges[k].dst;
    CHECK(moved);
  }
  (void)read_bytes;
}

TEST_CASE("15m features: dvshock is slot-matched over previous sessions, vol over a bar window") {
  const Panel p = intraday_panel(4, trading_days(), 3);
  const SessionIndex si = session_index(p.times);
  const auto dvs = export_dvshock_slot(p, si, 3, 2);
  // Bar t in session s, slot k: dv[t] / median dv at slot k of the previous 3 sessions (>= 2 values).
  for (std::size_t t = 0; t < p.T(); ++t)
    for (std::size_t i = 0; i < p.N(); ++i) {
      std::vector<double> prev;
      for (std::size_t u = 0; u < t; ++u)
        if (si.slot[u] == si.slot[t] && si.session[u] + 3 >= si.session[t] && si.session[u] < si.session[t]) {
          const double x = p.close[p.idx(u, i)] * p.volume[p.idx(u, i)];
          if (std::isfinite(x)) prev.push_back(x);
        }
      const double dv = p.close[p.idx(t, i)] * p.volume[p.idx(t, i)];
      const float got = dvs[p.idx(t, i)];
      if (prev.size() < 2 || !std::isfinite(dv)) {
        CHECK(std::isnan(got));
        continue;
      }
      std::sort(prev.begin(), prev.end());
      const std::size_t m = prev.size();
      const double med = m % 2 ? prev[m / 2] : 0.5 * (prev[m / 2 - 1] + prev[m / 2]);
      CHECK(got == static_cast<float>(dv / med));
    }
  // The generic vol at window 20 / need 10 is exactly the M4 vol20.
  const auto v20 = export_vol(p, 20, 10), m4 = export_vol20(p);
  CHECK(std::memcmp(v20.data(), m4.data(), v20.size() * 4) == 0);
}

// ---------------------------------------------------------------------------------------------------------------
// CLI and lake helpers
// ---------------------------------------------------------------------------------------------------------------

TEST_CASE("Lake::tickers lists the stored tickers of one timeframe") {
  auto dir = test::temp_dir("lake_tickers");
  Lake lake(dir);
  CHECK(lake.tickers(Timeframe::Day).empty());
  lake.write(Timeframe::Day, {{"MSFT", {utc_seconds(2026, 9, 1), 1, 1, 1, 1, 1, 1}},
                              {"AAPL", {utc_seconds(2026, 9, 1), 1, 1, 1, 1, 1, 1}},
                              {"AAPL", {utc_seconds(2026, 9, 2), 1, 1, 1, 1, 1, 1}}}, {});
  lake.write(Timeframe::Min15, {{"ZZZ", {utc_seconds(2026, 9, 1, 13, 30), 1, 1, 1, 1, 1, 1}}}, {});
  CHECK(lake.tickers(Timeframe::Day) == std::vector<std::string>{"AAPL", "MSFT"});
  CHECK(lake.tickers(Timeframe::Min15) == std::vector<std::string>{"ZZZ"});
}

#include "cli/args.hpp"

TEST_CASE("cli: --intraday-universe, --sync-intraday, --tickers-file, --from/--to, 15m export") {
  const CliArgs d = parse_cli({});
  CHECK(d.intraday_universe == 0);
  CHECK_FALSE(d.sync_intraday);
  CHECK(d.from_day == "2024-10-01");
  CHECK(d.to_day == "2026-10-02");
  CHECK(d.tickers_file == fs::path("data/universe/intraday_top1000.csv"));

  const CliArgs u = parse_cli({"--intraday-universe", "1000"});
  CHECK(u.intraday_universe == 1000);
  CHECK_THROWS_AS(parse_cli({"--intraday-universe", "0"}), std::invalid_argument);

  const CliArgs s = parse_cli({"--mode", "alpaca", "--sync-intraday", "--tickers-file", "t.csv", "--from",
                               "2024-10-01", "--to", "2026-10-02"});
  CHECK(s.sync_intraday);
  CHECK(s.tickers_file == fs::path("t.csv"));
  CHECK(parse_day_start(s.from_day) == utc_seconds(2024, 10, 1));
  CHECK(parse_day_start(s.to_day) == utc_seconds(2026, 10, 2));
  CHECK_THROWS_AS(parse_cli({"--sync-intraday"}), std::invalid_argument);  // needs --mode alpaca
  CHECK_THROWS_AS(parse_cli({"--mode", "alpaca", "--sync-intraday", "--from", "2024-13-01"}), std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--mode", "alpaca", "--sync-intraday", "--from", "2026-10-02", "--to", "2026-10-01"}),
                  std::invalid_argument);
  CHECK_THROWS_AS(parse_cli({"--mode", "alpaca", "--sync-intraday", "--refetch-full"}), std::invalid_argument);

  const CliArgs e = parse_cli({"--mode", "replay", "--export-panel", "out", "--timeframe", "15m"});
  CHECK(e.tf == Timeframe::Min15);
  CHECK(e.export_panel == fs::path("out"));
  CHECK(e.params == CoreParams::market_rank_intraday());  // 15m uses the intraday preset
  CHECK(cli_usage().find("--sync-intraday") != std::string::npos);
  CHECK(cli_usage().find("--intraday-universe") != std::string::npos);
}
