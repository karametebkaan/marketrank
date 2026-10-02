#include "server/http_server.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <thread>

#include "cli/args.hpp"
#include "core/time.hpp"
#include "graph/hotness.hpp"
#include "graph/pressure.hpp"
#include "graph/transition.hpp"
#include "server/top_list.hpp"

namespace mr {
using nlohmann::json;

namespace {

json num(double v) { return std::isfinite(v) ? json(v) : json(nullptr); }

void send_json(httplib::Response& res, int status, const json& j) {
  res.status = status;
  res.set_content(j.dump(), "application/json");
}

void send_raster(httplib::Response& res, const Raster& r) {
  static_assert(std::endian::native == std::endian::little, "grid payload is little-endian float32");
  std::string body(reinterpret_cast<const char*>(r.z.data()), r.z.size() * sizeof(float));
  res.set_content(std::move(body), "application/octet-stream");
}

// Strict JSON field readers: wrong type, non-finite or out-of-range values throw invalid_argument.
double get_num(const json& b, const char* key, double lo, double hi, bool lo_open = false) {
  const json& v = b.at(key);
  if (!v.is_number()) throw std::invalid_argument(std::string(key) + " must be a number");
  const double x = v.get<double>();
  if (!std::isfinite(x) || x > hi || x < lo || (lo_open && x <= lo))
    throw std::invalid_argument(std::string(key) + " out of range");
  return x;
}

long long get_int(const json& b, const char* key, long long lo, long long hi) {
  const json& v = b.at(key);
  if (!v.is_number_integer()) throw std::invalid_argument(std::string(key) + " must be an integer");
  if (v.is_number_unsigned() && v.get<std::uint64_t>() > static_cast<std::uint64_t>(hi))
    throw std::invalid_argument(std::string(key) + " out of range");
  const long long x = v.get<long long>();
  if (x < lo || x > hi) throw std::invalid_argument(std::string(key) + " out of range");
  return x;
}

std::string get_str(const json& b, const char* key) {
  if (!b.at(key).is_string()) throw std::invalid_argument(std::string(key) + " must be a string");
  return b.at(key).get<std::string>();
}

json raster_meta(const Raster& r) { return {{"w", r.w}, {"h", r.h}, {"zmin", r.zmin}, {"zmax", r.zmax}}; }

// Strict whole-number parse: the entire string must be an integer (no sign prefix '+', spaces or suffix).
long long parse_int(const std::string& v) {
  long long x = 0;
  const auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), x);
  if (ec != std::errc{} || end != v.data() + v.size() || v.empty()) throw std::invalid_argument("not an integer: " + v);
  return x;
}

std::optional<TimePoint> query_t(const httplib::Request& req) {
  if (!req.has_param("t")) return std::nullopt;
  return parse_int(req.get_param_value("t"));
}

constexpr std::size_t kShockCache = 8;
constexpr int kMaxSse = 8;  // open /api/events streams; the worker pool is larger, so other requests still run

// Holds one SSE slot; released when the stream's content provider is destroyed.
struct SseSlot {
  std::atomic<int>& n;
  explicit SseSlot(std::atomic<int>& c) : n(c) {}
  ~SseSlot() { --n; }
};

// "marketrank", "money-flow" or "legacy" when the model parameters equal that preset, otherwise "custom". Only the
// model parameters count: a landscape change (value, smoother, ...) keeps the preset name.
std::string preset_name(const CoreParams& p) {
  if (p == CoreParams::market_rank()) return "marketrank";
  if (p == CoreParams::money_flow()) return "money-flow";
  if (p == CoreParams::legacy()) return "legacy";
  return "custom";
}

json status_json(const FrameStore& s, const std::string& label) {
  const auto st = s.status();
  const CoreParams core = s.core_params();
  const LandscapeParams lp = s.landscape_params();
  return {{"computed", st.computed}, {"total", st.total},     {"running", st.running},
          {"ready", st.ready},       {"error", st.error},     {"generation", st.generation},
          {"params", describe(core)}, {"preset", preset_name(core)},
          {"h_ref", std::string(to_string(core.h_ref))},
          {"height", std::string(to_string(lp.height))},
          {"value", std::string(to_string(lp.value))},
          {"label", label},          {"nodes", s.nodes().size()},
          {"smooth", lp.smooth},     {"smoother", std::string(to_string(lp.smoother))},
          {"cvt_iterations", lp.cvt.iterations}, {"cvt_lambda", lp.cvt.lambda}, {"cvt_eps", lp.cvt.eps_frac},
          {"idw_power", lp.idw.power},
          {"idw_radius", lp.idw.radius_cells}, {"subdivision", lp.idw.subdivision},
          {"territory", std::string(to_string(lp.territory))}, {"show_etf", !lp.exclude_etf}};
}

}  // namespace

bool is_loopback_host(const std::string& host) {
  return host == "localhost" || host == "::1" || host == "[::1]" || host.rfind("127.", 0) == 0;
}

FluxServer::FluxServer(FrameStore& store, std::optional<PortfolioSpec> portfolio, std::string label)
    : store_(store), portfolio_(std::move(portfolio)), label_(std::move(label)) {
  routes();
}

int FluxServer::bind(const ServerOptions& opts) {
  if (!opts.web_root.empty() && std::filesystem::is_directory(opts.web_root))
    svr_.set_mount_point("/", opts.web_root.string());
  const int port = opts.port == 0 ? svr_.bind_to_any_port(opts.host) : (svr_.bind_to_port(opts.host, opts.port) ? opts.port : -1);
  if (port < 0) throw std::runtime_error("cannot bind " + opts.host + ":" + std::to_string(opts.port));
  const std::string p = ":" + std::to_string(port);
  const std::string configured = opts.host.find(':') != std::string::npos && opts.host.front() != '[' ? "[" + opts.host + "]" : opts.host;
  allowed_hosts_ = {"127.0.0.1" + p, "localhost" + p, "[::1]" + p};
  if (std::find(allowed_hosts_.begin(), allowed_hosts_.end(), configured + p) == allowed_hosts_.end())
    allowed_hosts_.push_back(configured + p);
  allowed_origins_.clear();
  for (const auto& h : allowed_hosts_) allowed_origins_.push_back("http://" + h);
  return port;
}

bool FluxServer::listen() {
  listen_active_ = true;
  bool ok = false;
  if (!stopping_) ok = svr_.listen_after_bind();
  listen_active_ = false;
  return ok;
}

void FluxServer::stop() {
  stopping_ = true;
  // httplib's stop() is a no-op until the accept loop is running. If listen() has been entered but is not
  // running yet, keep nudging until it has stopped (listen() re-checks stopping_ before it starts).
  svr_.stop();
  while (listen_active_) {
    svr_.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

bool FluxServer::guard_post(const httplib::Request& req, httplib::Response& res) const {
  if (const auto origin = req.get_header_value("Origin"); !origin.empty()) {
    if (std::find(allowed_origins_.begin(), allowed_origins_.end(), origin) == allowed_origins_.end()) {
      send_json(res, 403, {{"error", "cross-origin request refused"}});
      return false;
    }
  }
  const auto ct = req.get_header_value("Content-Type");
  if (ct.rfind("application/json", 0) != 0) {
    send_json(res, 415, {{"error", "Content-Type must be application/json"}});
    return false;
  }
  return true;
}

void FluxServer::routes() {
  svr_.set_payload_max_length(64 * 1024);
  // Workers: room for kMaxSse long-lived event streams plus ordinary requests.
  svr_.new_task_queue = [] { return new httplib::ThreadPool(std::max(16u, std::thread::hardware_concurrency())); };
  // DNS rebinding: a page on evil.example rebound to 127.0.0.1 reaches us with Host evil.example:<port>. Every
  // request (API and static files) must name this server by a loopback name or the configured --host.
  svr_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
    const auto host = req.get_header_value("Host");
    if (std::find(allowed_hosts_.begin(), allowed_hosts_.end(), host) != allowed_hosts_.end())
      return httplib::Server::HandlerResponse::Unhandled;
    send_json(res, 403, {{"error", "unexpected Host header"}});
    return httplib::Server::HandlerResponse::Handled;
  });
  svr_.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    std::string what = "internal error";
    try {
      if (ep) std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      what = e.what();
    } catch (...) {
    }
    res.status = 500;
    res.set_content(json{{"error", what}}.dump(), "application/json");
  });

  svr_.Get("/api/status", [this](const httplib::Request&, httplib::Response& res) {
    send_json(res, 200, status_json(store_, label_));
  });

  svr_.Get("/api/times", [this](const httplib::Request&, httplib::Response& res) {
    send_json(res, 200, json(store_.times()));
  });

  svr_.Get("/api/frame", [this](const httplib::Request& req, httplib::Response& res) {
    std::shared_ptr<const LandscapeFrame> f;
    try {
      f = store_.landscape(query_t(req));
    } catch (const std::exception&) {
      return send_json(res, 400, {{"error", "bad t"}});
    }
    if (!f) return send_json(res, req.has_param("t") && !store_.times().empty() ? 404 : 503, {{"error", "no such frame"}});
    const auto& nodes = store_.nodes();
    json jn = json::array();
    for (const auto& n : f->nodes)
      jn.push_back({n.i, nodes[n.i].ticker, nodes[n.i].sector, n.cell, n.fx, n.fy, num(n.h), num(n.hdisp), num(n.pi),
                    num(n.score), n.group, num(market_rank_score(n.pi, f->nodes.size())), num(n.pulse), n.floor});
    json ja = json::array();
    for (const auto& a : f->arcs) ja.push_back({a.a, a.b, a.w});
    json jp = json::array();
    if (portfolio_) {
      std::map<std::string, std::size_t> idx;
      for (std::size_t i = 0; i < nodes.size(); ++i) idx[nodes[i].ticker] = i;
      for (const auto& h : portfolio_->holdings) {
        auto it = idx.find(h.ticker);
        jp.push_back({{"ticker", h.ticker}, {"weight", h.weight}, {"i", it == idx.end() ? json(nullptr) : json(it->second)}});
      }
    }
    send_json(res, 200,
              {{"t", f->t}, {"time", format_rfc3339(f->t)}, {"lattice", {{"cols", f->size.cols}, {"rows", f->size.rows}}},
               {"raster", raster_meta(f->raster)}, {"nodes", jn}, {"arcs", ja}, {"portfolio", jp},
               {"params", describe(store_.core_params())}, {"compute_ms", f->compute_ms},
               {"communities", {{"count", f->communities}, {"modularity", num(f->modularity)}, {"loose", f->loose},
                                {"cluster_ms", f->cluster_ms}, {"reclustered", f->reclustered}}}});
  });

  svr_.Get("/api/top", [this](const httplib::Request& req, httplib::Response& res) {
    std::optional<TimePoint> t;
    long long n = 10, bars = 30;
    TopBy by = TopBy::Pi;
    auto int_param = [&](const char* key, long long lo, long long hi, long long& out) {
      if (!req.has_param(key)) return;
      const long long x = parse_int(req.get_param_value(key));
      if (x < lo || x > hi) throw std::invalid_argument(std::string(key) + " out of range");
      out = x;
    };
    try {
      int_param("n", 1, 50, n);
      int_param("bars", 1, 300, bars);
      if (req.has_param("by")) {
        const std::string b = req.get_param_value("by");
        if (b == "pi") by = TopBy::Pi;
        else if (b == "h") by = TopBy::Hotness;
        else throw std::invalid_argument("by must be pi or h");
      }
      if (req.has_param("t")) {
        long long tv = 0;
        int_param("t", std::numeric_limits<long long>::min(), std::numeric_limits<long long>::max(), tv);
        t = tv;
      }
    } catch (const std::exception&) {
      return send_json(res, 400, {{"error", "bad n, bars, by or t (n in [1, 50], bars in [1, 300], by pi or h)"}});
    }
    if (!store_.status().ready) return send_json(res, 503, {{"error", "landscapes are still being computed"}});
    const auto frames = store_.recent(t, static_cast<std::size_t>(std::max<long long>(bars, 2)));
    if (frames.empty()) return send_json(res, 404, {{"error", "no such frame"}});
    const auto& nodes = store_.nodes();
    json rows = json::array();
    for (const auto& r : top_hot(frames, static_cast<std::size_t>(n), static_cast<std::size_t>(bars), by)) {
      json series = json::array();
      for (double v : r.series) series.push_back(num(v));
      rows.push_back({{"rank", r.rank}, {"i", r.i}, {"ticker", nodes[r.i].ticker}, {"sector", nodes[r.i].sector},
                      {"h", num(r.h)}, {"hdisp", num(r.hdisp)}, {"pi", num(r.pi)}, {"mr", num(r.mr)},
                      {"pulse", num(r.pulse)}, {"score", num(r.score)},
                      {"prev_rank", r.prev_rank ? json(*r.prev_rank) : json(nullptr)}, {"series", series}});
    }
    send_json(res, 200, {{"t", frames.back()->t}, {"by", by == TopBy::Pi ? "pi" : "h"}, {"rows", rows}});
  });

  svr_.Get("/api/frame/grid", [this](const httplib::Request& req, httplib::Response& res) {
    std::shared_ptr<const LandscapeFrame> f;
    try {
      f = store_.landscape(query_t(req));
    } catch (const std::exception&) {
      return send_json(res, 400, {{"error", "bad t"}});
    }
    if (!f) return send_json(res, req.has_param("t") && !store_.times().empty() ? 404 : 503, {{"error", "no such frame"}});
    send_raster(res, f->raster);
  });

  svr_.Post("/api/params", [this](const httplib::Request& req, httplib::Response& res) {
    if (!guard_post(req, res)) return;
    try {
      const json b = json::parse(req.body);
      if (!b.is_object()) throw std::invalid_argument("body must be a JSON object");
      // With a preset the model parameters start from it; without one they start from the current ones (which may
      // carry CLI flags such as --lambda), and only the given fields change.
      CoreParams p = store_.core_params();
      LandscapeParams lp = store_.landscape_params();
      if (b.contains("preset")) {
        const std::string preset = get_str(b, "preset");
        // The landscape value follows the preset unless given: pi relative to size under marketrank, else hotness.
        lp.value = default_landscape_value(preset);
        if (preset == "marketrank") p = CoreParams::market_rank();
        else if (preset == "money-flow") p = CoreParams::money_flow();
        else if (preset == "legacy") p = CoreParams::legacy();
        else if (preset == "defaults") p = CoreParams{};
        else throw std::invalid_argument("unknown preset: " + preset);
      }
      constexpr long long kMaxK = 1000000;
      if (b.contains("h_ref")) p.h_ref = parse_hot_ref(get_str(b, "h_ref"));
      if (b.contains("pressure")) p.pressure = parse_pressure_mode(get_str(b, "pressure"));
      if (b.contains("lift")) p.transition.lift = parse_lift_mode(get_str(b, "lift"));
      if (b.contains("k_out")) p.transition.k_out = static_cast<std::size_t>(get_int(b, "k_out", 0, kMaxK));
      if (b.contains("k_in")) p.transition.k_in = static_cast<std::size_t>(get_int(b, "k_in", 0, kMaxK));
      constexpr double kInf = std::numeric_limits<double>::max();
      if (b.contains("retention")) p.transition.retention = get_num(b, "retention", -kInf, kInf);
      if (b.contains("lambda")) p.flux.lambda = get_num(b, "lambda", -kInf, kInf);
      if (b.contains("vol_scale")) {
        if (!b["vol_scale"].is_boolean()) throw std::invalid_argument("vol_scale must be a boolean");
        p.vol_scale = b["vol_scale"].get<bool>();
      }
      if (b.contains("value")) lp.value = parse_landscape_value(get_str(b, "value"));
      if (b.contains("height")) lp.height = parse_height_mode(get_str(b, "height"));
      if (b.contains("idw_power")) lp.idw.power = get_num(b, "idw_power", 0.0, 8.0, true);
      if (b.contains("idw_radius")) lp.idw.radius_cells = static_cast<int>(get_int(b, "idw_radius", 0, 16));
      if (b.contains("subdivision")) lp.idw.subdivision = static_cast<int>(get_int(b, "subdivision", 1, 8));
      if (b.contains("smooth")) lp.smooth = get_num(b, "smooth", 0.0, 4.0);
      if (b.contains("smoother")) lp.smoother = parse_smoother(get_str(b, "smoother"));
      if (b.contains("cvt_iterations")) lp.cvt.iterations = static_cast<int>(get_int(b, "cvt_iterations", 0, 50));
      if (b.contains("cvt_lambda")) lp.cvt.lambda = get_num(b, "cvt_lambda", 0.0, 1.0, true);
      if (b.contains("cvt_eps")) lp.cvt.eps_frac = get_num(b, "cvt_eps", 0.0, 10.0, true);
      if (b.contains("show_etf")) {
        if (!b["show_etf"].is_boolean()) throw std::invalid_argument("show_etf must be a boolean");
        lp.exclude_etf = !b["show_etf"].get<bool>();
      }
      if (b.contains("territory")) lp.territory = parse_territory_mode(get_str(b, "territory"));
      p.validate();
      const std::uint64_t gen = store_.set_params(p, lp);
      {
        std::lock_guard<std::mutex> lk(shock_m_);
        shocks_.clear();  // shocks belong to the parameters they were computed under
      }
      send_json(res, 202, {{"generation", gen}});
    } catch (const std::exception& e) {
      send_json(res, 400, {{"error", e.what()}});
    }
  });

  svr_.Post("/api/shock", [this](const httplib::Request& req, httplib::Response& res) {
    if (!guard_post(req, res)) return;
    std::vector<Shock> shocks;
    std::vector<std::string> names;
    try {
      const json b = json::parse(req.body);
      const auto& nodes = store_.nodes();
      if (!b.at("shocks").is_array()) throw std::invalid_argument("shocks must be an array");
      for (const auto& s : b.at("shocks")) {
        const std::string t = get_str(s, "ticker");
        const double size = get_num(s, "size", -std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
        auto it = std::find_if(nodes.begin(), nodes.end(), [&](const Security& x) { return x.ticker == t; });
        if (it == nodes.end()) throw std::invalid_argument("unknown ticker: " + t);
        shocks.push_back({static_cast<std::size_t>(it - nodes.begin()), size});
        names.push_back(t);
      }
      if (shocks.empty()) throw std::invalid_argument("no shocks");
    } catch (const std::exception& e) {
      return send_json(res, 400, {{"error", e.what()}});
    }
    if (!store_.status().ready) return send_json(res, 503, {{"error", "landscapes are still being computed"}});
    FrameStore::ShockResult r;
    try {
      r = store_.shock(shocks);
    } catch (const std::invalid_argument& e) {
      return send_json(res, 400, {{"error", e.what()}});
    } catch (const std::exception& e) {
      // Only "not ready" is a 503: re-check in case parameters changed since the pre-check.
      return send_json(res, store_.status().ready ? 500 : 503, {{"error", e.what()}});
    }
    const auto& nodes = store_.nodes();
    std::vector<char> shocked(nodes.size(), 0);
    for (const auto& s : shocks) shocked[s.node] = 1;
    std::vector<std::size_t> act;  // receivers and losers: active, not shocked themselves
    for (const auto& n : r.base->nodes)
      if (std::isfinite(r.delta.dh[n.i]) && !shocked[n.i]) act.push_back(n.i);
    std::sort(act.begin(), act.end(), [&](auto a, auto b) {
      return r.delta.dh[a] > r.delta.dh[b] || (r.delta.dh[a] == r.delta.dh[b] && nodes[a].ticker < nodes[b].ticker);
    });
    auto row = [&](std::size_t i) {
      return json{{"ticker", nodes[i].ticker}, {"sector", nodes[i].sector}, {"dh", num(r.delta.dh[i])},
                  {"dpi", num(r.delta.dpi[i])}, {"dscore", num(r.delta.dscore[i])}};
    };
    json rec = json::array(), los = json::array(), sh = json::array();
    for (std::size_t k = 0; k < std::min<std::size_t>(15, act.size()); ++k) rec.push_back(row(act[k]));
    for (std::size_t k = 0; k < std::min<std::size_t>(15, act.size()); ++k) los.push_back(row(act[act.size() - 1 - k]));
    for (const auto& s : shocks) sh.push_back(row(s.node));
    std::uint64_t id = 0;
    {
      std::lock_guard<std::mutex> lk(shock_m_);
      id = next_shock_id_++;
      shocks_.emplace_front(id, r.raster);
      if (shocks_.size() > kShockCache) shocks_.pop_back();
    }
    send_json(res, 200, {{"shock_id", id}, {"t", r.t}, {"l1_dpi", r.delta.l1_dpi}, {"raster", raster_meta(r.raster)},
                         {"shocked", sh}, {"receivers", rec}, {"losers", los}});
  });

  svr_.Get("/api/shock/grid", [this](const httplib::Request& req, httplib::Response& res) {
    long long id = 0;
    try {
      if (!req.has_param("id")) throw std::invalid_argument("missing id");
      id = parse_int(req.get_param_value("id"));
    } catch (const std::exception&) {
      return send_json(res, 400, {{"error", "id must be the shock_id returned by POST /api/shock"}});
    }
    std::lock_guard<std::mutex> lk(shock_m_);
    auto it = std::find_if(shocks_.begin(), shocks_.end(), [&](const auto& e) { return static_cast<long long>(e.first) == id; });
    if (it == shocks_.end()) return send_json(res, 404, {{"error", "unknown or expired shock id"}});
    shocks_.splice(shocks_.begin(), shocks_, it);  // most recently used
    send_raster(res, shocks_.front().second);
  });

  svr_.Get("/api/events", [this](const httplib::Request&, httplib::Response& res) {
    if (sse_clients_.fetch_add(1) >= kMaxSse) {
      --sse_clients_;
      return send_json(res, 503, {{"error", "too many event streams"}});
    }
    auto slot = std::make_shared<SseSlot>(sse_clients_);
    res.set_header("Cache-Control", "no-cache");
    struct Sse {
      std::uint64_t seen = 0;
      std::chrono::steady_clock::time_point last_write = std::chrono::steady_clock::now();
    };
    auto st = std::make_shared<Sse>();
    res.set_chunked_content_provider("text/event-stream", [this, st, slot](size_t, httplib::DataSink& sink) {
      if (stopping_) return false;
      // Bounded wait so a stop request is noticed within ~250 ms.
      const auto v = store_.wait_for_change(st->seen, std::chrono::milliseconds(250));
      if (stopping_) return false;
      const auto now = std::chrono::steady_clock::now();
      std::string msg;
      if (v != st->seen) {
        st->seen = v;
        msg = "event: status\ndata: " + status_json(store_, label_).dump() + "\n\n";
      } else if (now - st->last_write >= std::chrono::seconds(15)) {
        msg = ": ping\n\n";
      } else {
        return true;
      }
      if (!sink.write(msg.data(), msg.size())) return false;
      st->last_write = now;
      // Throttle: at most one status event every 250 ms (sliced so stop stays responsive).
      for (int k = 0; k < 5 && !stopping_; ++k) std::this_thread::sleep_for(std::chrono::milliseconds(50));
      return true;
    });
  });
}

}  // namespace mr
