#pragma once
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.hpp"

namespace fx {

struct HttpResponse {
  int status = 0;
  std::string body;
};

using HttpGet = std::function<HttpResponse(const std::string& path_and_query)>;

struct BarsPage {
  std::map<std::string, std::vector<Bar>> bars;
  std::optional<std::string> next_page_token;
};

BarsPage parse_bars_page(const std::string& json);

struct AlpacaConfig {
  std::string key_id, secret;
  std::string feed = "sip";
  std::string host = "data.alpaca.markets";
  int backoff_initial_ms = 500;
  int backoff_max_ms = 30000;
  int max_retries = 6;
  int min_request_interval_ms = 334;  // <= 180 requests/minute
  std::string trading_host = "paper-api.alpaca.markets";
};

void load_dotenv(const std::filesystem::path& path);
std::optional<AlpacaConfig> alpaca_config_from_env();
std::string url_encode(std::string_view s);
std::string_view alpaca_timeframe(Timeframe tf);

struct FetchResult {
  std::map<std::string, std::vector<Bar>> bars;
  std::vector<std::string> stale;  // symbols from batches that failed
};

class AlpacaClient {
 public:
  explicit AlpacaClient(AlpacaConfig config, HttpGet get = {});

  FetchResult fetch_bars(const std::vector<std::string>& symbols, std::string_view timeframe,
                         TimePoint start, TimePoint end);
  std::string get(const std::string& path);

 private:
  void throttle();
  std::optional<std::chrono::steady_clock::time_point> last_request_;
  HttpResponse get_with_retry(const std::string& path);
  AlpacaConfig config_;
  HttpGet get_;
};

}  // namespace fx
