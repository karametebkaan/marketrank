#include "market/sec_sectors.hpp"

#include <fcntl.h>
#include <httplib.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <thread>

#include "core/csv.hpp"
#include "core/time.hpp"

namespace mr {
namespace {

struct SicRange {
  int lo, hi;
  const char* sector;
};

constexpr const char* kIT = "Information Technology";
constexpr const char* kHC = "Health Care";
constexpr const char* kFin = "Financials";
constexpr const char* kCD = "Consumer Discretionary";
constexpr const char* kCS = "Consumer Staples";
constexpr const char* kInd = "Industrials";
constexpr const char* kEn = "Energy";
constexpr const char* kMat = "Materials";
constexpr const char* kUt = "Utilities";
constexpr const char* kRE = "Real Estate";
constexpr const char* kCom = "Communication Services";

// First match wins, so specific codes precede the division ranges they carve out of.
// Based on SEC's SIC list; mapped to GICS-style sectors by industry, not an official crosswalk.
constexpr SicRange kSicTable[] = {
    // --- specific carve-outs ---
    {6798, 6798, kRE},      // REIT
    {6324, 6324, kHC},      // hospital and medical service plans (6411 stays Financials)
    {1220, 1241, kEn},      // coal mining
    {4922, 4924, kEn},      // natural gas transmission and distribution (pipelines)
    {3020, 3029, kCD},      // footwear (before rubber and plastics 3000-3099)
    {3826, 3826, kHC},      // laboratory analytical instruments
    {3829, 3829, kHC},      // measuring and controlling devices
    {7311, 7311, kCom},     // advertising agencies
    {7812, 7841, kCom},     // film, video production, distribution, theaters
    {5331, 5331, kCS},      // variety stores
    {5399, 5399, kCS},      // misc general merchandise stores
    {5400, 5499, kCS},      // food stores
    {7389, 7389, kInd},     // business services NEC: kept in Industrials on purpose (catch-all)
    {6770, 6770, kFin},     // blank check / SPAC
    {6500, 6553, kRE},      // real estate operators, developers, lessors
    {2833, 2836, kHC},      // pharma, biologics
    {2840, 2844, kCS},      // soaps, cosmetics, personal care
    {3841, 3851, kHC},      // medical instruments, supplies, dental, ophthalmic
    {8731, 8731, kHC},      // commercial biological research
    {5122, 5122, kHC},      // drug wholesale
    {5047, 5047, kHC},      // medical equipment wholesale
    {5171, 5172, kEn},      // petroleum wholesale
    {5912, 5912, kCS},      // drug stores
    {5140, 5149, kCS},      // grocery wholesale
    {5150, 5159, kCS},      // farm products wholesale
    {3570, 3579, kIT},      // computers and office equipment
    {3661, 3669, kIT},      // telephone, communications equipment
    {3670, 3679, kIT},      // electronic components, semiconductors
    {7370, 7379, kIT},      // computer services and software
    {3711, 3716, kCD},      // motor vehicles, bodies, motor homes
    {3751, 3751, kCD},      // motorcycles, bicycles
    {3790, 3799, kCD},
    {3011, 3011, kCD},      // tires
    {3630, 3639, kCD},      // household appliances
    {1520, 1531, kCD},      // homebuilders
    // --- division ranges ---
    {100, 999, kCS},        // agriculture, fishing
    {1000, 1299, kMat},     // metal and coal mining
    {1300, 1399, kEn},      // oil and gas extraction and services
    {1400, 1499, kMat},     // nonmetallic minerals mining
    {1500, 1799, kInd},     // construction
    {2000, 2199, kCS},      // food, beverage, tobacco
    {2200, 2399, kCD},      // textiles, apparel
    {2400, 2499, kMat},     // lumber and wood
    {2500, 2599, kCD},      // furniture
    {2600, 2699, kMat},     // paper
    {2700, 2799, kCom},     // publishing, printing
    {2800, 2829, kMat},     // chemicals
    {2845, 2899, kMat},     // other chemicals
    {2900, 2999, kEn},      // petroleum refining
    {3000, 3099, kMat},     // rubber and plastics
    {3100, 3199, kCD},      // leather, footwear
    {3200, 3299, kMat},     // stone, clay, glass, cement
    {3300, 3399, kMat},     // primary metals
    {3400, 3599, kInd},     // fabricated metals, machinery
    {3600, 3699, kInd},     // electrical equipment
    {3700, 3799, kInd},     // transportation equipment (aerospace, rail, ships)
    {3800, 3899, kInd},     // instruments, defense electronics, photographic
    {3900, 3999, kCD},      // misc manufacturing (toys, jewelry, sporting goods)
    {4000, 4599, kInd},     // rail, trucking, water, air transport
    {4600, 4699, kEn},      // pipelines
    {4700, 4799, kInd},     // transportation services
    {4800, 4899, kCom},     // communications
    {4900, 4949, kUt},      // electric, gas, water utilities
    {4950, 4999, kInd},     // sanitary services
    {5000, 5199, kInd},     // wholesale trade
    {5200, 5999, kCD},      // retail trade (food stores below)
    {6000, 6499, kFin},     // banks, brokers, insurance
    {6700, 6799, kFin},     // holding companies, investment offices
    {7000, 7099, kCD},      // hotels and lodging
    {7200, 7299, kCD},      // personal services
    {7300, 7369, kInd},     // business services
    {7380, 7399, kInd},
    {7500, 7599, kCD},      // auto rental and repair
    {7600, 7699, kInd},     // misc repair
    {7830, 7999, kCD},      // movie theaters, amusement, recreation
    {8000, 8099, kHC},      // health services
    {8100, 8199, kInd},     // legal services
    {8200, 8299, kCD},      // education
    {8300, 8699, kInd},     // social, membership, and other services
    {8700, 8799, kInd},     // engineering, management, research services
};

std::string upper(std::string s) {
  for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
}
std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string csv_quote(const std::string& s) {
  if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
  std::string q = "\"";
  for (char ch : s) {
    if (ch == '"') q += '"';
    q += ch;
  }
  return q + "\"";
}

bool is_unclassified(const std::string& sector) { return sector.empty() || sector == kSectorUnclassified; }

bool stale(const SecRow& r, std::int64_t now, int stale_days) {
  return r.fetched_at <= 0 || now - r.fetched_at > static_cast<std::int64_t>(stale_days) * 86400;
}

}  // namespace

std::map<std::string, std::string> parse_company_tickers(const std::string& json) {
  const auto j = nlohmann::json::parse(json, nullptr, false);
  if (j.is_discarded() || !j.is_object()) throw std::runtime_error("company_tickers is not a JSON object");
  std::map<std::string, std::string> out;
  std::vector<std::pair<std::string, std::string>> real;
  for (const auto& [key, e] : j.items()) {
    if (!e.is_object() || !e.contains("ticker") || !e["ticker"].is_string() || !e.contains("cik_str"))
      continue;
    const std::string cik = e["cik_str"].is_string() ? e["cik_str"].get<std::string>()
                                                     : std::to_string(e["cik_str"].get<long long>());
    const std::string t = upper(e["ticker"].get<std::string>());
    if (t.empty()) continue;
    out.emplace(t, cik);
    real.emplace_back(t, cik);
  }
  if (out.empty()) throw std::runtime_error("company_tickers is empty");
  // Synthetic dash/dot spellings never displace a real ticker.
  for (const auto& [t, cik] : real) {
    std::string dotted = t, dashed = t;
    std::replace(dotted.begin(), dotted.end(), '-', '.');
    std::replace(dashed.begin(), dashed.end(), '.', '-');
    out.emplace(dotted, cik);
    out.emplace(dashed, cik);
  }
  return out;
}

SecSubmission parse_submission(const std::string& json) {
  const auto j = nlohmann::json::parse(json, nullptr, false);
  if (j.is_discarded() || !j.is_object()) throw std::runtime_error("submission is not a JSON object");
  SecSubmission s;
  if (j.contains("sic")) {
    if (j["sic"].is_string()) s.sic = j["sic"].get<std::string>();
    else if (j["sic"].is_number_integer()) s.sic = std::to_string(j["sic"].get<long long>());
  }
  if (j.contains("sicDescription") && j["sicDescription"].is_string())
    s.description = j["sicDescription"].get<std::string>();
  return s;
}

std::string sic_to_sector(int sic) {
  for (const auto& r : kSicTable)
    if (sic >= r.lo && sic <= r.hi) return r.sector;
  return "";
}

std::string sic_to_sector(const std::string& sic) {
  int v = 0;
  const auto [end, ec] = std::from_chars(sic.data(), sic.data() + sic.size(), v);
  if (sic.empty() || ec != std::errc{} || end != sic.data() + sic.size()) return "";
  return sic_to_sector(v);
}

bool looks_like_fund(const std::string& name) {
  const std::string n = " " + lower(name) + " ";
  auto has = [&](const char* s) { return n.find(s) != std::string::npos; };
  // Word-bounded "fund"/"funds" so "Fundamental" does not match.
  auto has_word = [&](const std::string& w) {
    for (std::size_t p = n.find(w); p != std::string::npos; p = n.find(w, p + 1)) {
      const bool left = !std::isalpha(static_cast<unsigned char>(n[p - 1]));
      const std::size_t e = p + w.size();
      const bool right = !std::isalpha(static_cast<unsigned char>(n[e])) ||
                         (n[e] == 's' && !std::isalpha(static_cast<unsigned char>(n[e + 1])));
      if (left && right) return true;
    }
    return false;
  };
  if (has_word("etf") || has_word("etn") || has("ishares") || has("spdr") || has("proshares") ||
      has("invesco qqq"))
    return true;
  if (has_word("fund")) return true;
  if (has_word("trust") && (has_word("etf") || has_word("index") || has_word("shares"))) return true;
  return false;
}

SecCache load_sec_cache(const std::filesystem::path& path) {
  SecCache cache;
  if (!std::filesystem::exists(path)) return cache;
  const CsvRows rows = read_csv_file(path);
  for (std::size_t r = 1; r < rows.size(); ++r) {
    const auto& f = rows[r];
    if (f.size() < 6 || f[0].empty()) continue;
    SecRow row{f[0], f[1], f[2], f[3], f[4], 0};
    try {
      row.fetched_at = parse_rfc3339(f[5]);
    } catch (const std::exception&) {
      row.fetched_at = 0;  // treated as stale
    }
    cache[row.ticker] = std::move(row);
  }
  return cache;
}

void save_sec_cache(const std::filesystem::path& path, const SecCache& cache) {
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  std::string text = "ticker,cik,sic,sic_description,sector,fetched_at\n";
  for (const auto& [t, r] : cache)
    text += csv_quote(r.ticker) + ',' + csv_quote(r.cik) + ',' + csv_quote(r.sic) + ',' +
            csv_quote(r.sic_description) + ',' + csv_quote(r.sector) + ',' + format_rfc3339(r.fetched_at) + '\n';
  // Unique temp name per call, fsync before the rename so a crash never leaves a torn cache.
  static std::atomic<unsigned> counter{0};
  const std::string tmp = path.string() + "." + std::to_string(::getpid()) + "." +
                          std::to_string(counter++) + ".tmp";
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0) throw std::runtime_error("cannot write " + tmp);
  bool ok = true;
  for (std::size_t off = 0; ok && off < text.size();) {
    const ssize_t n = ::write(fd, text.data() + off, text.size() - off);
    if (n <= 0) ok = false;
    else off += static_cast<std::size_t>(n);
  }
  ok = ok && ::fsync(fd) == 0;
  ::close(fd);
  if (!ok) {
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    throw std::runtime_error("write failed for " + tmp);
  }
  std::filesystem::rename(tmp, path);
}

SecClient::SecClient(SecConfig config, SecHttpGet get) : config_(std::move(config)), get_(std::move(get)) {
  if (!get_) {
    // One keep-alive connection per host. The User-Agent lives only in this closure.
    auto clients = std::make_shared<std::map<std::string, std::unique_ptr<httplib::SSLClient>>>();
    get_ = [clients](const std::string& host, const std::string& path,
                     const std::string& ua) -> HttpResponse {
      auto& cli = (*clients)[host];
      if (!cli) {
        cli = std::make_unique<httplib::SSLClient>(host);
        cli->set_connection_timeout(10);
        cli->set_read_timeout(60);
        cli->set_follow_location(true);
      }
      auto res = cli->Get(path, httplib::Headers{{"User-Agent", ua}, {"Accept", "application/json"}});
      if (!res) return {0, httplib::to_string(res.error())};
      int retry_after = 0;
      if (res->has_header("Retry-After")) {
        const std::string v = res->get_header_value("Retry-After");
        std::from_chars(v.data(), v.data() + v.size(), retry_after);
      }
      return {res->status, res->body, retry_after};
    };
  }
}

std::string SecClient::get(const std::string& host, const std::string& path) {
  const auto now = [&] { return config_.now ? config_.now() : std::chrono::steady_clock::now(); };
  const auto sleep = [&](std::chrono::milliseconds d) {
    if (config_.sleep) config_.sleep(d);
    else std::this_thread::sleep_for(d);
  };
  int delay_ms = config_.backoff_initial_ms;
  for (int attempt = 1;; ++attempt) {
    if (config_.min_request_interval_ms > 0) {
      const auto gap = std::chrono::milliseconds(config_.min_request_interval_ms);
      if (last_request_ && now() - *last_request_ < gap)
        sleep(std::chrono::duration_cast<std::chrono::milliseconds>(gap - (now() - *last_request_)));
      last_request_ = now();
    }
    HttpResponse res = get_(host, path, config_.user_agent);
    if (res.status == 200) return res.body;
    if (res.status == 403)
      throw SecAbort("SEC returned HTTP 403 for " + path +
                     ": blocked or User-Agent rejected; check SEC_USER_AGENT and wait before retrying");
    const bool retryable = res.status == 0 || res.status == 429 || res.status >= 500;
    if (!retryable || attempt >= config_.max_attempts)
      throw std::runtime_error("SEC GET " + path + " failed (HTTP " + std::to_string(res.status) + ")");
    if (res.status == 429 && res.retry_after_s > 0) {
      sleep(std::chrono::milliseconds(std::min(res.retry_after_s, config_.max_retry_after_s) * 1000));
    } else if (delay_ms > 0) {
      sleep(std::chrono::milliseconds(delay_ms));
    }
    delay_ms = std::min(delay_ms * 2, config_.backoff_max_ms);
  }
}

SecConfig sec_config_from_env() {
  const char* ua = std::getenv("SEC_USER_AGENT");
  if (!ua || !*ua)
    throw std::runtime_error(
        "SEC_USER_AGENT is not set: add SEC_USER_AGENT=\"Your Name your@email\" to .env (SEC requires a contact)");
  SecConfig c;
  c.user_agent = ua;
  return c;
}

SecClient make_sec_client() { return SecClient(sec_config_from_env()); }

SecSyncStats sync_sec_sectors(const std::vector<std::string>& tickers,
                              const std::filesystem::path& cache_path,
                              const SecClientFactory& client_factory, const SecSyncOptions& options) {
  SecCache cache = load_sec_cache(cache_path);
  SecSyncStats stats;
  std::vector<std::string> todo;
  for (const auto& t : tickers) {
    const auto it = cache.find(t);
    if (it != cache.end() && !stale(it->second, options.now, options.stale_days)) ++stats.fresh;
    else todo.push_back(t);
  }
  if (todo.empty()) return stats;

  SecClient client = client_factory();
  const auto cik_map = parse_company_tickers(client.get("www.sec.gov", "/files/company_tickers.json"));
  std::size_t since_commit = 0, consecutive_failures = 0;
  for (const auto& t : todo) {
    SecRow row{t, "", "", "", "", options.now};
    const auto c = cik_map.find(upper(t));
    if (c == cik_map.end()) {
      ++stats.no_cik;
    } else {
      try {
        std::string padded = c->second;
        if (padded.size() < 10) padded.insert(0, 10 - padded.size(), '0');
        const SecSubmission s = parse_submission(client.get("data.sec.gov", "/submissions/CIK" + padded + ".json"));
        row.cik = c->second;
        row.sic = s.sic;
        row.sic_description = s.description;
        row.sector = sic_to_sector(s.sic);
        ++stats.fetched;
        consecutive_failures = 0;
      } catch (const SecAbort&) {
        save_sec_cache(cache_path, cache);
        throw;
      } catch (const std::runtime_error&) {
        ++stats.failed;  // not cached: retried next run
        if (++consecutive_failures >= options.max_consecutive_failures) {
          save_sec_cache(cache_path, cache);
          throw SecAbort(std::to_string(consecutive_failures) +
                         " consecutive SEC fetches failed; aborting (progress saved, rerun to resume)");
        }
        continue;
      }
    }
    cache[t] = std::move(row);
    if (++since_commit >= options.commit_every) {
      save_sec_cache(cache_path, cache);
      since_commit = 0;
    }
  }
  save_sec_cache(cache_path, cache);
  return stats;
}

std::string resolve_sector(const Security& s, bool known_fund, const SecCache& cache) {
  if (!is_unclassified(s.sector)) return s.sector;
  if (const auto it = cache.find(s.ticker); it != cache.end() && !it->second.sector.empty())
    return it->second.sector;
  if (known_fund || looks_like_fund(s.name)) return kSectorEtfFund;
  return kSectorUnclassified;
}

std::size_t apply_sector_fill(Universe& universe, const SecCache& cache) {
  std::size_t changed = 0;
  for (std::size_t i = 0; i < universe.nodes().size(); ++i) {
    const Security& s = universe.nodes()[i];
    if (!is_unclassified(s.sector)) continue;
    std::string sector = resolve_sector(s, universe.is_fund(s.ticker), cache);
    if (sector != s.sector) {
      universe.set_sector(i, std::move(sector));
      ++changed;
    }
  }
  return changed;
}

}  // namespace mr
