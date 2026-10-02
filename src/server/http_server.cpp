#include "server/http_server.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <thread>

#include "cli/args.hpp"
#include "core/time.hpp"
#include "graph/hotness.hpp"
#include "graph/pressure.hpp"
#include "graph/transition.hpp"

namespace fx {
using nlohmann::json;

namespace {

json num(double v) { return std::isfinite(v) ? json(v) : json(nullptr); }

void send_json(httplib::Response& res, int status, const json& j) {
  res.status = status;
  res.set_content(j.dump(), "application/json");
}

void send_raster(httplib::Response& res, const Raster& r) {
  std::string body(reinterpret_cast<const char*>(r.z.data()), r.z.size() * sizeof(float));
  res.set_content(std::move(body), "application/octet-stream");
}

json raster_meta(const Raster& r) { return {{"w", r.w}, {"h", r.h}, {"zmin", r.zmin}, {"zmax", r.zmax}}; }

std::optional<TimePoint> query_t(const httplib::Request& req) {
  if (!req.has_param("t")) return std::nullopt;
  return std::stoll(req.get_param_value("t"));
}

json status_json(const FrameStore& s, const std::string& label) {
  const auto st = s.status();
  return {{"computed", st.computed}, {"total", st.total},     {"running", st.running},
          {"ready", st.ready},       {"error", st.error},     {"generation", st.generation},
          {"params", describe(s.core_params())},
          {"height", std::string(to_string(s.landscape_params().height))},
          {"label", label},          {"nodes", s.nodes().size()}};
}

}  // namespace

FluxServer::FluxServer(FrameStore& store, std::optional<PortfolioSpec> portfolio, std::string label)
    : store_(store), portfolio_(std::move(portfolio)), label_(std::move(label)) {
  routes();
}

int FluxServer::bind(const ServerOptions& opts) {
  if (!opts.web_root.empty() && std::filesystem::is_directory(opts.web_root))
    svr_.set_mount_point("/", opts.web_root.string());
  const int port = opts.port == 0 ? svr_.bind_to_any_port(opts.host) : (svr_.bind_to_port(opts.host, opts.port) ? opts.port : -1);
  if (port < 0) throw std::runtime_error("cannot bind " + opts.host + ":" + std::to_string(opts.port));
  return port;
}

void FluxServer::listen() { svr_.listen_after_bind(); }
void FluxServer::stop() {
  stopping_ = true;
  svr_.stop();
}

void FluxServer::routes() {
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
    if (!f) return send_json(res, req.has_param("t") ? 404 : 503, {{"error", "no such frame"}});
    const auto& nodes = store_.nodes();
    json jn = json::array();
    for (const auto& n : f->nodes)
      jn.push_back({n.i, nodes[n.i].ticker, nodes[n.i].sector, n.cell, n.fx, n.fy, num(n.h), num(n.hdisp), num(n.pi), num(n.score)});
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
               {"params", describe(store_.core_params())}, {"compute_ms", f->compute_ms}});
  });

  svr_.Get("/api/frame/grid", [this](const httplib::Request& req, httplib::Response& res) {
    std::shared_ptr<const LandscapeFrame> f;
    try {
      f = store_.landscape(query_t(req));
    } catch (const std::exception&) {
      return send_json(res, 400, {{"error", "bad t"}});
    }
    if (!f) return send_json(res, req.has_param("t") ? 404 : 503, {{"error", "no such frame"}});
    send_raster(res, f->raster);
  });

  svr_.Post("/api/params", [this](const httplib::Request& req, httplib::Response& res) {
    try {
      const json b = json::parse(req.body);
      CoreParams p;
      const std::string preset = b.value("preset", std::string("money-flow"));
      if (preset == "money-flow") p = CoreParams::money_flow();
      else if (preset == "legacy") p = CoreParams::legacy();
      else if (preset == "defaults") p = CoreParams{};
      else throw std::invalid_argument("unknown preset: " + preset);
      if (b.contains("h_ref")) p.h_ref = parse_hot_ref(b["h_ref"].get<std::string>());
      if (b.contains("pressure")) p.pressure = parse_pressure_mode(b["pressure"].get<std::string>());
      if (b.contains("lift")) p.transition.lift = parse_lift_mode(b["lift"].get<std::string>());
      if (b.contains("k_out")) p.transition.k_out = b["k_out"].get<std::size_t>();
      if (b.contains("k_in")) p.transition.k_in = b["k_in"].get<std::size_t>();
      if (b.contains("retention")) p.transition.retention = b["retention"].get<double>();
      if (b.contains("lambda")) p.flux.lambda = b["lambda"].get<double>();
      if (b.contains("vol_scale")) p.vol_scale = b["vol_scale"].get<bool>();
      LandscapeParams lp = store_.landscape_params();
      if (b.contains("height")) lp.height = parse_height_mode(b["height"].get<std::string>());
      if (b.contains("idw_power")) lp.idw.power = b["idw_power"].get<double>();
      if (b.contains("idw_radius")) lp.idw.radius_cells = b["idw_radius"].get<int>();
      if (b.contains("subdivision")) lp.idw.subdivision = std::clamp(b["subdivision"].get<int>(), 1, 8);
      p.validate();
      store_.set_params(p, lp);
      send_json(res, 202, {{"generation", store_.status().generation}});
    } catch (const std::exception& e) {
      send_json(res, 400, {{"error", e.what()}});
    }
  });

  svr_.Post("/api/shock", [this](const httplib::Request& req, httplib::Response& res) {
    std::vector<Shock> shocks;
    std::vector<std::string> names;
    try {
      const json b = json::parse(req.body);
      const auto& nodes = store_.nodes();
      for (const auto& s : b.at("shocks")) {
        const std::string t = s.at("ticker").get<std::string>();
        const double size = s.at("size").get<double>();
        if (!std::isfinite(size)) throw std::invalid_argument("size must be finite");
        auto it = std::find_if(nodes.begin(), nodes.end(), [&](const Security& x) { return x.ticker == t; });
        if (it == nodes.end()) throw std::invalid_argument("unknown ticker: " + t);
        shocks.push_back({static_cast<std::size_t>(it - nodes.begin()), size});
        names.push_back(t);
      }
      if (shocks.empty()) throw std::invalid_argument("no shocks");
    } catch (const std::exception& e) {
      return send_json(res, 400, {{"error", e.what()}});
    }
    FrameStore::ShockResult r;
    try {
      r = store_.shock(shocks);
    } catch (const std::invalid_argument& e) {
      return send_json(res, 400, {{"error", e.what()}});
    } catch (const std::exception& e) {
      return send_json(res, 503, {{"error", e.what()}});
    }
    const auto& nodes = store_.nodes();
    std::vector<std::size_t> act;
    for (const auto& n : r.base->nodes)
      if (std::isfinite(r.delta.dh[n.i])) act.push_back(n.i);
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
    {
      std::lock_guard<std::mutex> lk(shock_m_);
      last_shock_ = r.raster;
    }
    send_json(res, 200, {{"t", r.t}, {"l1_dpi", r.delta.l1_dpi}, {"raster", raster_meta(r.raster)},
                         {"shocked", sh}, {"receivers", rec}, {"losers", los}});
  });

  svr_.Get("/api/shock/grid", [this](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(shock_m_);
    if (!last_shock_) return send_json(res, 404, {{"error", "no shock yet"}});
    send_raster(res, *last_shock_);
  });

  svr_.Get("/api/events", [this](const httplib::Request&, httplib::Response& res) {
    res.set_header("Cache-Control", "no-cache");
    struct Sse {
      std::uint64_t seen = 0;
      std::chrono::steady_clock::time_point last_write = std::chrono::steady_clock::now();
    };
    auto st = std::make_shared<Sse>();
    res.set_chunked_content_provider("text/event-stream", [this, st](size_t, httplib::DataSink& sink) {
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

}  // namespace fx
