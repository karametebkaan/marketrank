#pragma once
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
};

void load_dotenv(const std::filesystem::path& path);
std::optional<AlpacaConfig> alpaca_config_from_env();
std::string url_encode(std::string_view s);
std::string_view alpaca_timeframe(Timeframe tf);

class AlpacaClient {
 public:
  explicit AlpacaClient(AlpacaConfig config, HttpGet get = {});

  std::map<std::string, std::vector<Bar>> fetch_bars(const std::vector<std::string>& symbols,
                                                     std::string_view timeframe, TimePoint start,
                                                     TimePoint end);

 private:
  HttpResponse get_with_retry(const std::string& path);
  AlpacaConfig config_;
  HttpGet get_;
};

}  // namespace fx
