#include "server/frame_store.hpp"

#include <stdexcept>

namespace fx {

FrameStore::FrameStore(Panel panel, std::vector<Security> nodes, CoreParams core, LandscapeParams land,
                       std::size_t max_frames)
    : panel_(std::move(panel)), nodes_(std::move(nodes)), max_frames_(max_frames), core_(std::move(core)),
      land_(land) {
  if (nodes_.size() != panel_.N()) throw std::invalid_argument("FrameStore: nodes/panel size mismatch");
  core_.validate();
}

FrameStore::~FrameStore() { stop_worker(); }

void FrameStore::bump() {
  ++version_;
  cv_.notify_all();
}

void FrameStore::stop_worker() {
  ++gen_;
  if (worker_.joinable()) worker_.join();
}

void FrameStore::start() {
  stop_worker();
  std::lock_guard<std::mutex> lk(m_);
  const std::uint64_t gen = ++gen_;
  frames_.clear();
  pre_last_.reset();
  last_core_.reset();
  status_ = Status{};
  status_.total = panel_.T() > 1 ? panel_.T() - 1 : 0;
  status_.running = true;
  status_.generation = gen;
  bump();
  worker_ = std::thread([this, gen, core = core_, land = land_] { run(gen, core, land); });
}

void FrameStore::set_params(CoreParams core, LandscapeParams land) {
  core.validate();
  stop_worker();
  {
    std::lock_guard<std::mutex> lk(m_);
    core_ = std::move(core);
    land_ = land;
  }
  start();
}

void FrameStore::run(std::uint64_t gen, CoreParams core, LandscapeParams land) {
  try {
    const std::size_t T = panel_.T();
    if (T < 3) throw std::runtime_error("need at least three bars to serve landscapes");
    CorePipeline pipe(panel_.N(), core);
    LandscapeBuilder builder(panel_.N(), land);
    const std::size_t first_landscape = T - 1 > max_frames_ ? T - max_frames_ : 1;
    for (std::size_t t = 1; t < T; ++t) {
      if (gen_.load() != gen) return;
      if (t == T - 1) {
        std::lock_guard<std::mutex> lk(m_);
        pre_last_ = pipe;  // state after bar T-2: shocks re-step bar T-1 from here
      }
      auto f = std::make_shared<Frame>(pipe.step(panel_, t));
      std::shared_ptr<const LandscapeFrame> lf;
      if (t >= first_landscape) lf = std::make_shared<LandscapeFrame>(builder.build(*f));
      std::lock_guard<std::mutex> lk(m_);
      if (gen_.load() != gen) return;
      if (lf) frames_[lf->t] = lf;
      if (t == T - 1) last_core_ = f;
      status_.computed = t;
      bump();
    }
    std::lock_guard<std::mutex> lk(m_);
    if (gen_.load() != gen) return;
    status_.running = false;
    status_.ready = true;
    bump();
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> lk(m_);
    if (gen_.load() != gen) return;
    status_.running = false;
    status_.error = e.what();
    bump();
  }
}

FrameStore::Status FrameStore::status() const {
  std::lock_guard<std::mutex> lk(m_);
  return status_;
}

std::vector<TimePoint> FrameStore::times() const {
  std::lock_guard<std::mutex> lk(m_);
  std::vector<TimePoint> out;
  out.reserve(frames_.size());
  for (const auto& [t, f] : frames_) out.push_back(t);
  return out;
}

std::shared_ptr<const LandscapeFrame> FrameStore::landscape(std::optional<TimePoint> t) const {
  std::lock_guard<std::mutex> lk(m_);
  if (frames_.empty()) return nullptr;
  if (!t) return frames_.rbegin()->second;
  auto it = frames_.find(*t);
  return it == frames_.end() ? nullptr : it->second;
}

FrameStore::ShockResult FrameStore::shock(const std::vector<Shock>& shocks) {
  std::optional<CorePipeline> pipe;
  std::shared_ptr<const Frame> base;
  std::shared_ptr<const LandscapeFrame> base_land;
  LandscapeParams land;
  {
    std::lock_guard<std::mutex> lk(m_);
    if (!status_.ready || !pre_last_ || !last_core_ || frames_.empty())
      throw std::runtime_error("landscapes are still being computed");
    pipe = pre_last_;  // copy: the cached state stays reusable
    base = last_core_;
    base_land = frames_.rbegin()->second;
    land = land_;
  }
  const Frame shocked = pipe->step(panel_, panel_.T() - 1, shocks);
  ShockResult r;
  r.t = shocked.t;
  r.delta = shock_response(*base, shocked);
  r.raster = delta_raster(*base_land, r.delta.dh, land);
  r.base = base_land;
  return r;
}

std::uint64_t FrameStore::wait_for_change(std::uint64_t seen, std::chrono::milliseconds timeout) const {
  std::unique_lock<std::mutex> lk(m_);
  cv_.wait_for(lk, timeout, [&] { return version_ != seen; });
  return version_;
}

CoreParams FrameStore::core_params() const {
  std::lock_guard<std::mutex> lk(m_);
  return core_;
}

LandscapeParams FrameStore::landscape_params() const {
  std::lock_guard<std::mutex> lk(m_);
  return land_;
}

}  // namespace fx
