#pragma once
#include <httplib.h>

#include <atomic>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>

#include "market/universe.hpp"
#include "server/frame_store.hpp"

namespace fx {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::filesystem::path web_root = "web";
};

class FluxServer {
 public:
  FluxServer(FrameStore& store, std::optional<PortfolioSpec> portfolio, std::string label);
  int bind(const ServerOptions& opts);
  void listen();
  void stop();

 private:
  void routes();
  FrameStore& store_;
  std::optional<PortfolioSpec> portfolio_;
  std::string label_;
  httplib::Server svr_;
  std::atomic<bool> stopping_{false};  // lets SSE loops exit so listen() can join its workers
  std::mutex shock_m_;
  std::optional<Raster> last_shock_;
};

}  // namespace fx
