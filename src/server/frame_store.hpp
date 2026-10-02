#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "geom/landscape.hpp"
#include "market/panel.hpp"
#include "market/universe.hpp"
#include "pipeline/core_pipeline.hpp"
#include "pipeline/shock.hpp"

namespace mr {

class FrameStore {
 public:
  struct Status {
    std::size_t computed = 0, total = 0;
    bool running = false, ready = false;
    std::string error;
    std::uint64_t generation = 0;
    int threads = 0;  // OpenMP thread count the worker actually ran with (0 until it starts)
  };
  struct ShockResult {
    TimePoint t = 0;
    ShockDelta delta;
    Raster raster;
    std::shared_ptr<const LandscapeFrame> base;
  };

  FrameStore(Panel panel, std::vector<Security> nodes, CoreParams core, LandscapeParams land,
             std::size_t max_frames = 300);
  ~FrameStore();
  FrameStore(const FrameStore&) = delete;
  FrameStore& operator=(const FrameStore&) = delete;

  void start();
  // Returns the generation the new parameters run under. When only display parameters change (same CoreParams and
  // same_placement) and every frame has been computed, the cached frames are redrawn (restyle) in the worker
  // instead of re-running the pipeline; the generation still advances. Both ETF layouts (exclude_etf true and false)
  // are built in the same pass, so flipping only exclude_etf on a complete store swaps them in at once (ready
  // immediately, under a new generation).
  std::uint64_t set_params(CoreParams core, LandscapeParams land);
  Status status() const;
  std::vector<TimePoint> times() const;
  std::shared_ptr<const LandscapeFrame> landscape(std::optional<TimePoint> t) const;
  // Up to k cached frames ending at t (latest when empty), oldest first; empty if t is not cached.
  std::vector<std::shared_ptr<const LandscapeFrame>> recent(std::optional<TimePoint> t, std::size_t k) const;
  ShockResult shock(const std::vector<Shock>& shocks);
  // Callers blocked here must be released (by progress, a timeout, or stopping them) before the store
  // is destroyed: destruction does not wake waiters.
  std::uint64_t wait_for_change(std::uint64_t seen, std::chrono::milliseconds timeout) const;
  const Panel& panel() const { return panel_; }
  const std::vector<Security>& nodes() const { return nodes_; }
  CoreParams core_params() const;
  LandscapeParams landscape_params() const;
  std::size_t pipeline_steps() const { return steps_.load(); }  // core frames computed so far (all generations)

 private:
  void stop_worker();                  // call with control_m_ held and m_ NOT held (it joins)
  std::uint64_t launch_locked(std::optional<CoreParams> core, std::optional<LandscapeParams> land, int threads,
                              bool restyle_only = false);
  // (launch_locked: call with control_m_ held and m_ NOT held)
  void run(std::uint64_t gen, CoreParams core, LandscapeParams land, int threads);
  void run_restyle(std::uint64_t gen, LandscapeParams land, int threads);
  void bump();  // version++ and notify (call with m_ held)

  const Panel panel_;
  const std::vector<Security> nodes_;
  const std::size_t max_frames_;
  std::mutex control_m_;  // serializes start/set_params/stop; guards worker_. Always taken before m_.
  mutable std::mutex m_;
  mutable std::condition_variable cv_;
  std::thread worker_;
  std::atomic<std::uint64_t> gen_{0};
  std::atomic<std::size_t> steps_{0};
  bool complete_ = false;  // every frame of the current core/placement parameters has been computed (guarded by m_)
  std::uint64_t version_ = 0;
  CoreParams core_;
  LandscapeParams land_;
  int omp_threads_ = 1;  // caller's omp_get_max_threads() at start()/set_params(); applied to worker and shock()
  Status status_;
  std::map<TimePoint, std::shared_ptr<const LandscapeFrame>> frames_;
  std::map<TimePoint, std::shared_ptr<const LandscapeFrame>> alt_frames_;  // same bars, exclude_etf flipped
  std::optional<CorePipeline> pre_last_;
  std::shared_ptr<const Frame> last_core_;
};

}  // namespace mr
