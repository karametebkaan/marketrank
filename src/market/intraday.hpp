#pragma once
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "market/alpaca_client.hpp"
#include "storage/lake.hpp"

namespace mr {

// --- M5 intraday universe -----------------------------------------------------------------------------------------
// The M5 universe is static: the N names with the highest median DAILY dollar volume (v * vwap, close when vwap is
// missing) over [from, to) in the daily lake, ties by ticker. A name needs bars on at least min_coverage x the
// largest per-name bar count in the window (listed for most of the window). Survivorship: the list is chosen with
// knowledge of the whole window (names that delisted early fail the coverage floor; names that became liquid late
// are included), so it is not point-in-time. The M5 evaluation acknowledges this bias.
struct LiquidName {
  std::string ticker;
  double median_dollar_volume = 0;
  std::size_t days = 0;
};
std::vector<LiquidName> rank_intraday_universe(const std::map<std::string, std::vector<Bar>>& daily, TimePoint from,
                                               TimePoint to, std::size_t n, double min_coverage = 0.5);
// CSV: rank,ticker,median_dollar_volume,days.
void write_intraday_universe(const std::filesystem::path& path, const std::vector<LiquidName>& names);
// Tickers in file order: a CSV with a "ticker" header column, or a plain list (one per line, # comments).
std::vector<std::string> read_tickers_file(const std::filesystem::path& path);

// --- --sync-intraday ------------------------------------------------------------------------------------------------
// Fetches 15Min bars for [from, to) in calendar-month chunks (UTC; a month boundary never splits a US session).
// Each chunk is requested for every ticker that has not completed it, 100 symbols per request with pagination and
// the client's rate limit and retries; only regular-session bars are kept (is_regular_session_bar) and written to
// the lake as tf=15m, one lake write per 100-symbol batch. After the write, the (ticker, chunk) pairs are appended
// to the progress file (fsynced): a rerun skips them, so an interrupted sync resumes where it stopped and a failed
// batch is retried next time. A chunk that reaches past now - 16 min is fetched (up to then) but never marked done.
struct IntradaySyncOptions {
  TimePoint now = std::numeric_limits<TimePoint>::max();
  std::filesystem::path progress_file;  // empty = <lake root>/progress_15m.csv
  bool log = false;                     // per-chunk progress on stderr
};
struct IntradaySyncStats {
  std::size_t chunks_fetched = 0;  // (ticker, chunk) pairs fetched successfully
  std::size_t chunks_skipped = 0;  // (ticker, chunk) pairs already done
  std::size_t requests = 0;        // HTTP requests (pages), as counted by the batches
  std::size_t raw_bars = 0;        // bars received (extended hours included)
  std::size_t kept_bars = 0;       // regular-session bars written
  std::vector<std::string> failed;  // tickers with at least one failed chunk (sorted, unique)
};
IntradaySyncStats sync_intraday(AlpacaClient& client, Lake& lake, const std::vector<std::string>& tickers,
                                TimePoint from, TimePoint to, const IntradaySyncOptions& opt = {});

// --- Session index of a 15m panel ------------------------------------------------------------------------------------
// A session is one ET trading date. session[t] counts sessions from 0 in time order; slot[t] is the 15-minute slot
// within the session (0 = the 09:30 bar); first[t] / last[t] flag the first / last bar of its session in the panel
// (the panel's final bar counts as last). dates[s] is session s's ET date (YYYY-MM-DD).
struct SessionIndex {
  std::vector<std::uint32_t> session;
  std::vector<std::uint32_t> slot;
  std::vector<std::uint8_t> first, last;
  std::vector<std::string> dates;
  std::size_t sessions = 0;
};
SessionIndex session_index(const std::vector<TimePoint>& times, TimePoint bar_seconds = 900);

}  // namespace mr
