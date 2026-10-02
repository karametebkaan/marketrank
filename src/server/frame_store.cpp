#include "server/frame_store.hpp"

#include <omp.h>

#include <map>
#include <stdexcept>
#include <string>

namespace fx {

namespace {
// Sector ids in ascending sector-string order, so they are stable for a given universe.
std::vector<std::uint32_t> sector_groups(const std::vector<Security>& nodes) {
  std::map<std::string, std::uint32_t> ids;
  for (const auto& s : nodes) ids.emplace(s.sector, 0);
  std::uint32_t k = 0;
  for (auto& kv : ids) kv.second = k++;
  std::vector<std::uint32_t> g;
  g.reserve(nodes.size());
  for (const auto& s : nodes) g.push_back(ids[s.sector]);
  return g;
}
}  // namespace

FrameStore::FrameStore(Panel panel, std::vector<Security> nodes, CoreParams core, LandscapeParams land,
                       std::size_t max_frames)
    : panel_(std::move(panel)), nodes_(std::move(nodes)), max_frames_(max_frames), core_(std::move(core)),
      land_(land) {
  if (max_frames_ == 0) throw std::invalid_argument("FrameStore: max_frames must be positive");
  if (nodes_.size() != panel_.N()) throw std::invalid_argument("FrameStore: nodes/panel size mismatch");
  core_.validate();
}

FrameStore::~FrameStore() {
  std::lock_guard<std::mutex> ck(control_m_);
  stop_worker();
}

void FrameStore::bump() {
  ++version_;
  cv_.notify_all();
}

void FrameStore::stop_worker() {
  ++gen_;
  if (worker_.joinable()) worker_.join();
}

std::uint64_t FrameStore::launch_locked(std::optional<CoreParams> core, std::optional<LandscapeParams> land,
                               int threads) {
  std::lock_guard<std::mutex> lk(m_);
  if (core) core_ = std::move(*core);
  if (land) land_ = *land;
  omp_threads_ = threads;
  const std::uint64_t gen = ++gen_;
  frames_.clear();
  pre_last_.reset();
  last_core_.reset();
  status_ = Status{};
  status_.total = panel_.T() > 1 ? panel_.T() - 1 : 0;
  status_.running = true;
  status_.generation = gen;
  bump();
  try {
    worker_ = std::thread([this, gen, c = core_, l = land_, threads] { run(gen, c, l, threads); });
  } catch (const std::exception& e) {
    status_.running = false;
    status_.error = std::string("thread launch failed: ") + e.what();
    bump();
  }
  return gen;
}

void FrameStore::start() {
  const int threads = omp_get_max_threads();
  std::lock_guard<std::mutex> ck(control_m_);
  stop_worker();
  launch_locked(std::nullopt, std::nullopt, threads);
}

std::uint64_t FrameStore::set_params(CoreParams core, LandscapeParams land) {
  core.validate();
  const int threads = omp_get_max_threads();
  std::lock_guard<std::mutex> ck(control_m_);
  stop_worker();
  return launch_locked(std::move(core), land, threads);
}

void FrameStore::run(std::uint64_t gen, CoreParams core, LandscapeParams land, int threads) {
  try {
    omp_set_num_threads(threads);  // the ICV is per-thread: follow the caller that started us
    {
      std::lock_guard<std::mutex> lk(m_);
      if (gen_.load() == gen) status_.threads = omp_get_max_threads();
    }
    const std::size_t T = panel_.T();
    if (T < 3) throw std::runtime_error("need at least three bars to serve landscapes");
    CorePipeline pipe(panel_.N(), core);
    LandscapeBuilder builder(panel_.N(), land, sector_groups(nodes_));
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
  } catch (...) {
    std::lock_guard<std::mutex> lk(m_);
    if (gen_.load() != gen) return;
    status_.running = false;
    status_.error = "unknown error";
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
  int threads = 1;
  {
    std::lock_guard<std::mutex> lk(m_);
    if (!status_.ready || !pre_last_ || !last_core_ || frames_.empty())
      throw std::runtime_error("landscapes are still being computed");
    pipe = pre_last_;  // copy: the cached state stays reusable
    base = last_core_;
    base_land = frames_.rbegin()->second;
    land = land_;
    threads = omp_threads_;
  }
  const int saved_threads = omp_get_max_threads();
  omp_set_num_threads(threads);
  struct Restore {
    int n;
    ~Restore() { omp_set_num_threads(n); }
  } restore{saved_threads};
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
