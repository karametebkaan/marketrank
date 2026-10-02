#pragma once
#include <httplib.h>

#include <atomic>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "market/universe.hpp"
#include "server/frame_store.hpp"

namespace fx {

// Whether `host` names the loopback interface (127.0.0.0/8, localhost or ::1).
bool is_loopback_host(const std::string& host);

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8765;
  std::filesystem::path web_root = "web";
};

class FluxServer {
 public:
  FluxServer(FrameStore& store, std::optional<PortfolioSpec> portfolio, std::string label);
  int bind(const ServerOptions& opts);
  bool listen();  // false if the socket failed (or stop() came first)
  void stop();

 private:
  void routes();
  FrameStore& store_;
  std::optional<PortfolioSpec> portfolio_;
  std::string label_;
  httplib::Server svr_;
  bool guard_post(const httplib::Request& req, httplib::Response& res) const;
  std::atomic<bool> stopping_{false};
  std::atomic<bool> listen_active_{false};
  std::vector<std::string> allowed_hosts_;    // Host header values accepted (set by bind)
  std::vector<std::string> allowed_origins_;  // "http://" + each allowed host
  std::mutex shock_m_;
  std::optional<Raster> last_shock_;
};

}  // namespace fx
