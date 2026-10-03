#pragma once
#include <httplib.h>

#include <atomic>
#include <filesystem>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "market/universe.hpp"
#include "server/frame_store.hpp"

namespace mr {

// Whether `host` names the loopback interface (127.0.0.0/8, localhost or ::1).
bool is_loopback_host(const std::string& host);

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8765;
  std::filesystem::path web_root = "web";
};

class FluxServer {
 public:
  // portfolio_path: where POST /api/portfolio saves the user's holdings (atomically); empty = not saved.
  FluxServer(FrameStore& store, std::optional<PortfolioSpec> portfolio, std::string label,
             std::filesystem::path portfolio_path = {});
  int bind(const ServerOptions& opts);
  bool listen();  // false if the socket failed (or stop() came first)
  void stop();

 private:
  void routes();
  FrameStore& store_;
  std::optional<PortfolioSpec> portfolio_;  // guarded by portfolio_m_
  std::filesystem::path portfolio_path_;
  mutable std::mutex portfolio_m_;
  std::optional<PortfolioSpec> portfolio() const;
  std::string label_;
  httplib::Server svr_;
  bool guard_post(const httplib::Request& req, httplib::Response& res) const;
  std::atomic<bool> stopping_{false};
  std::atomic<bool> listen_active_{false};
  std::vector<std::string> allowed_hosts_;    // Host header values accepted (set by bind)
  std::vector<std::string> allowed_origins_;  // "http://" + each allowed host
  std::atomic<int> sse_clients_{0};  // open /api/events streams (at most kMaxSse)
  std::mutex shock_m_;  // guards shocks_ and next_shock_id_
  std::list<std::pair<std::uint64_t, Raster>> shocks_;  // the last 8 shock grids, most recently used first
  std::uint64_t next_shock_id_ = 1;
};

}  // namespace mr
