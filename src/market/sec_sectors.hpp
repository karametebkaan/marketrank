#pragma once
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "market/alpaca_client.hpp"
#include "market/universe.hpp"

namespace fx {

inline constexpr const char* kSectorEtfFund = "ETF/Fund";
inline constexpr const char* kSectorUnclassified = "Unclassified";

struct SecSubmission {
  std::string sic;          // may be empty
  std::string description;  // sicDescription
};

// company_tickers.json -> ticker -> CIK digits. Tickers are uppercased; a class-share ticker is
// reachable in both the SEC form (BRK-B) and Alpaca's form (BRK.B).
std::map<std::string, std::string> parse_company_tickers(const std::string& json);
SecSubmission parse_submission(const std::string& json);  // throws std::runtime_error on bad JSON

// SIC code -> GICS-style sector name (the 11 names data/universe/sp500.csv uses); "" if unknown.
// Range table in sec_sectors.cpp; specific codes take priority over division ranges.
std::string sic_to_sector(int sic);
std::string sic_to_sector(const std::string& sic);  // "" or non-numeric -> ""

// Fund/ETF name heuristic (no network).
bool looks_like_fund(const std::string& name);

struct SecRow {
  std::string ticker, cik, sic, sic_description, sector;
  std::int64_t fetched_at = 0;  // epoch seconds; stored as RFC 3339
};
using SecCache = std::map<std::string, SecRow>;

SecCache load_sec_cache(const std::filesystem::path& path);  // missing file -> empty
void save_sec_cache(const std::filesystem::path& path, const SecCache& cache);  // temp + rename

struct SecConfig {
  std::string user_agent;  // never printed or logged
  int min_request_interval_ms = 150;  // under SEC's 10 requests/second limit
  int backoff_initial_ms = 500;
  int backoff_max_ms = 30000;
  int max_attempts = 3;
  int max_retry_after_s = 60;
  // Injectable for tests; empty means the real steady clock / this_thread::sleep_for.
  std::function<std::chrono::steady_clock::time_point()> now;
  std::function<void(std::chrono::milliseconds)> sleep;
};

// Fatal for the whole sync (HTTP 403, or too many consecutive failures); progress is saved first.
struct SecAbort : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The User-Agent is passed on every call so the transport must send it (and tests can see it).
using SecHttpGet = std::function<HttpResponse(const std::string& host, const std::string& path,
                                              const std::string& user_agent)>;

class SecClient {
 public:
  explicit SecClient(SecConfig config, SecHttpGet get = {});
  std::string get(const std::string& host, const std::string& path);  // retries 429/5xx/transport

 private:
  std::optional<std::chrono::steady_clock::time_point> last_request_;
  SecConfig config_;
  SecHttpGet get_;
};

// Requires SEC_USER_AGENT (load_dotenv first); throws std::runtime_error with the setup hint if unset.
SecConfig sec_config_from_env();
SecClient make_sec_client();

using SecClientFactory = std::function<SecClient()>;  // called only when there is something to fetch

struct SecSyncOptions {
  std::int64_t now = 0;
  int stale_days = 90;
  std::size_t commit_every = 500;
  std::size_t max_consecutive_failures = 20;
};
struct SecSyncStats {
  std::size_t fresh = 0, fetched = 0, no_cik = 0, failed = 0;
};

// Fetches SIC for tickers missing from the cache or older than stale_days. Progress is written
// atomically every commit_every fetched tickers. No-CIK tickers are cached with an empty SIC;
// tickers whose fetch fails are left out so the next run retries them.
SecSyncStats sync_sec_sectors(const std::vector<std::string>& tickers,
                              const std::filesystem::path& cache_path,
                              const SecClientFactory& client_factory, const SecSyncOptions& options);

// Fill order: S&P GICS sector (anything but ""/Unclassified), SEC sector, ETF/Fund, Unclassified.
std::string resolve_sector(const Security& s, bool known_fund, const SecCache& cache);
// Applies resolve_sector to every node still unclassified. Returns how many changed.
std::size_t apply_sector_fill(Universe& universe, const SecCache& cache);

}  // namespace fx
