#include "market/alpaca_client.hpp"

#include <httplib.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <thread>

#include "core/time.hpp"

namespace fx {

BarsPage parse_bars_page(const std::string& json) {
  const auto j = nlohmann::json::parse(json);
  BarsPage page;
  if (j.contains("bars") && j["bars"].is_object()) {
    for (const auto& [symbol, arr] : j["bars"].items()) {
      auto& out = page.bars[symbol];
      for (const auto& b : arr) {
        out.push_back({parse_rfc3339(b.at("t").get<std::string>()), b.at("o").get<double>(),
                       b.at("h").get<double>(), b.at("l").get<double>(), b.at("c").get<double>(),
                       b.at("v").get<double>(), b.value("vw", 0.0)});
      }
    }
  }
  if (j.contains("next_page_token") && j["next_page_token"].is_string())
    page.next_page_token = j["next_page_token"].get<std::string>();
  return page;
}

void load_dotenv(const std::filesystem::path& path) {
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = line.substr(0, eq), value = line.substr(eq + 1);
    while (!value.empty() && (value.back() == '\r' || value.back() == ' ' || value.back() == '\t'))
      value.pop_back();
    if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') &&
        value.back() == value.front())
      value = value.substr(1, value.size() - 2);
    ::setenv(key.c_str(), value.c_str(), 0);
  }
}

std::optional<AlpacaConfig> alpaca_config_from_env() {
  const char* key = std::getenv("APCA_API_KEY_ID");
  const char* secret = std::getenv("APCA_API_SECRET_KEY");
  if (!key || !secret || !*key || !*secret) return std::nullopt;
  AlpacaConfig c;
  c.key_id = key;
  c.secret = secret;
  if (const char* feed = std::getenv("APCA_DATA_FEED"); feed && *feed) c.feed = feed;
  if (const char* th = std::getenv("APCA_TRADING_HOST"); th && *th) c.trading_host = th;
  return c;
}

std::string url_encode(std::string_view s) {
  static const char* kHex = "0123456789ABCDEF";
  std::string out;
  for (unsigned char ch : s) {
    if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == '~') {
      out += static_cast<char>(ch);
    } else {
      out += '%';
      out += kHex[ch >> 4];
      out += kHex[ch & 15];
    }
  }
  return out;
}

std::string_view alpaca_timeframe(Timeframe tf) {
  switch (tf) {
    case Timeframe::Hour: return "30Min";
    case Timeframe::Day: return "1Day";
    case Timeframe::Week: return "1Week";
  }
  return "1Day";
}

AlpacaClient::AlpacaClient(AlpacaConfig config, HttpGet get)
    : config_(std::move(config)), get_(std::move(get)) {
  if (!get_) {
    get_ = [cfg = config_](const std::string& path) -> HttpResponse {
      httplib::SSLClient cli(cfg.host);
      cli.set_connection_timeout(10);
      cli.set_read_timeout(60);
      httplib::Headers headers = {{"APCA-API-KEY-ID", cfg.key_id},
                                  {"APCA-API-SECRET-KEY", cfg.secret}};
      auto res = cli.Get(path, headers);
      if (!res) return {0, httplib::to_string(res.error())};
      return {res->status, res->body};
    };
  }
}

void AlpacaClient::throttle() {
  if (config_.min_request_interval_ms <= 0) return;
  const auto gap = std::chrono::milliseconds(config_.min_request_interval_ms);
  const auto now = std::chrono::steady_clock::now();
  if (last_request_ && now - *last_request_ < gap) std::this_thread::sleep_for(gap - (now - *last_request_));
  last_request_ = std::chrono::steady_clock::now();
}

std::string AlpacaClient::get(const std::string& path) { return get_with_retry(path).body; }

HttpResponse AlpacaClient::get_with_retry(const std::string& path) {
  std::mt19937 jitter_rng(std::random_device{}());
  int delay_ms = config_.backoff_initial_ms;
  for (int attempt = 0;; ++attempt) {
    throttle();
    HttpResponse res = get_(path);
    if (res.status == 200) return res;
    const bool retryable = res.status == 0 || res.status == 429 || res.status >= 500;
    if (!retryable || attempt >= config_.max_retries) {
      throw std::runtime_error("Alpaca GET failed (HTTP " + std::to_string(res.status) +
                               "): " + res.body.substr(0, 200));
    }
    if (delay_ms > 0) {
      std::uniform_int_distribution<int> jitter(0, delay_ms / 2);
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms + jitter(jitter_rng)));
    }
    delay_ms = std::min(delay_ms * 2, config_.backoff_max_ms);
  }
}

FetchResult AlpacaClient::fetch_bars(const std::vector<std::string>& symbols,
                                     std::string_view timeframe, TimePoint start, TimePoint end) {
  FetchResult result;
  constexpr std::size_t kBatch = 100;
  for (std::size_t b = 0; b < symbols.size(); b += kBatch) {
    const std::size_t batch_end = std::min(symbols.size(), b + kBatch);
    std::string joined;
    for (std::size_t i = b; i < batch_end; ++i) {
      if (!joined.empty()) joined += ',';
      joined += url_encode(symbols[i]);
    }
    const std::string base = "/v2/stocks/bars?symbols=" + joined +
                             "&timeframe=" + std::string(timeframe) +
                             "&start=" + format_rfc3339(start) + "&end=" + format_rfc3339(end) +
                             "&limit=10000&adjustment=all&feed=" + config_.feed;
    std::map<std::string, std::vector<Bar>> batch_bars;
    try {
      std::optional<std::string> token;
      do {
        std::string path = base;
        if (token) path += "&page_token=" + url_encode(*token);
        BarsPage page;
        try {
          page = parse_bars_page(get_with_retry(path).body);
        } catch (const std::runtime_error&) {
          throw;
        } catch (const std::exception&) {
          throw std::runtime_error("Alpaca response could not be parsed");
        }
        for (auto& [sym, bars] : page.bars)
          batch_bars[sym].insert(batch_bars[sym].end(), bars.begin(), bars.end());
        token = page.next_page_token;
      } while (token);
    } catch (const std::runtime_error&) {
      result.stale.insert(result.stale.end(), symbols.begin() + static_cast<std::ptrdiff_t>(b),
                          symbols.begin() + static_cast<std::ptrdiff_t>(batch_end));
      continue;
    }
    for (auto& [sym, bars] : batch_bars)
      result.bars[sym].insert(result.bars[sym].end(), bars.begin(), bars.end());
  }
  return result;
}

}  // namespace fx
