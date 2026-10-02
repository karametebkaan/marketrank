#include "market/bar_store.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <set>

namespace fx {

BarStore::BarStore(std::filesystem::path lake_root) : root_(std::move(lake_root)) {}

BarStore::~BarStore() {
  try {
    flush();
  } catch (...) {
  }
}

Lake& BarStore::lake() {
  if (!lake_) lake_ = std::make_unique<Lake>(root_);
  return *lake_;
}

void BarStore::upsert(const Key& key, const std::vector<Bar>& incoming) {
  auto& series = series_[key];
  std::map<TimePoint, Bar> by_time;
  for (const Bar& b : series) by_time[b.t] = b;
  for (const Bar& b : incoming) by_time[b.t] = b;
  series.clear();
  series.reserve(by_time.size());
  for (const auto& [t, b] : by_time) series.push_back(b);
}

void BarStore::merge(const std::string& ticker, Timeframe tf, const std::vector<Bar>& incoming) {
  const Key key{ticker, tf};
  upsert(key, incoming);
  auto& q = queued_[key];
  q.insert(q.end(), incoming.begin(), incoming.end());
}

const std::vector<Bar>& BarStore::bars(const std::string& ticker, Timeframe tf) const {
  static const std::vector<Bar> kEmpty;
  auto it = series_.find({ticker, tf});
  return it == series_.end() ? kEmpty : it->second;
}

std::optional<TimePoint> BarStore::last_time(const std::string& ticker, Timeframe tf) const {
  const auto& b = bars(ticker, tf);
  if (b.empty()) return std::nullopt;
  return b.back().t;
}

std::optional<TimePoint> BarStore::first_time(const std::string& ticker, Timeframe tf) const {
  const auto& b = bars(ticker, tf);
  if (b.empty()) return std::nullopt;
  return b.front().t;
}

std::optional<TimePoint> BarStore::covered_from(const std::string& ticker, Timeframe tf) const {
  auto it = covered_.find({ticker, tf});
  if (it == covered_.end()) return std::nullopt;
  return it->second;
}

void BarStore::set_covered_from(const std::string& ticker, Timeframe tf, TimePoint t) {
  const Key key{ticker, tf};
  auto it = covered_.find(key);
  if (it != covered_.end() && it->second <= t) return;
  covered_[key] = t;
  auto [w, inserted] = cov_to_write_.emplace(key, t);
  if (!inserted) w->second = std::min(w->second, t);
}

void BarStore::save(const std::string& ticker, Timeframe tf) {
  const Key key{ticker, tf};
  auto it = queued_.find(key);
  if (it == queued_.end()) return;
  auto& w = to_write_[key];
  w.insert(w.end(), it->second.begin(), it->second.end());
  queued_.erase(it);
}

void BarStore::flush() {
  if (to_write_.empty() && cov_to_write_.empty()) return;
  for (Timeframe tf : {Timeframe::Hour, Timeframe::Day, Timeframe::Week}) {
    std::vector<LakeRow> rows;
    std::vector<std::pair<std::string, TimePoint>> cov;
    for (const auto& [key, bars] : to_write_)
      if (key.second == tf)
        for (const Bar& b : bars) rows.push_back({key.first, b});
    for (const auto& [key, t] : cov_to_write_)
      if (key.second == tf) cov.emplace_back(key.first, t);
    if (!rows.empty() || !cov.empty()) lake().write(tf, rows, cov);
  }
  to_write_.clear();
  cov_to_write_.clear();
}

void BarStore::load_range(const std::vector<std::string>& tickers, Timeframe tf, TimePoint start,
                          TimePoint end) {
  lake();  // an open failure (e.g. lake in use by another process) must surface
  try {
    for (auto& [ticker, loaded] : lake().read(tf, tickers, start, end)) {
      const Key key{ticker, tf};
      // Unsaved (queued_) or unflushed (to_write_) bars are newer than the lake: keep them.
      std::set<TimePoint> pending;
      for (const auto* m : {&queued_, &to_write_})
        if (auto it = m->find(key); it != m->end())
          for (const Bar& b : it->second) pending.insert(b.t);
      if (!pending.empty())
        std::erase_if(loaded, [&](const Bar& b) { return pending.count(b.t) > 0; });
      upsert(key, loaded);
    }
    for (const auto& [ticker, t] : lake().coverage(tf, tickers)) {
      auto it = covered_.find({ticker, tf});
      if (it == covered_.end() || t < it->second) covered_[{ticker, tf}] = t;
    }
  } catch (const std::exception& e) {
    // Contract 4: unreadable data never aborts a load; the caller sees missing series.
    std::cerr << "bar store: load of " << to_string(tf) << " failed: " << e.what() << "\n";
  }
}

void BarStore::load_all(const std::vector<std::string>& tickers, Timeframe tf) {
  load_range(tickers, tf, std::numeric_limits<TimePoint>::min(), std::numeric_limits<TimePoint>::max());
}

}  // namespace fx
