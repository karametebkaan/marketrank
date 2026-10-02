#include "geom/layout.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace fx {
namespace {

inline void add_force(double dx, double dy, double m, double& fx, double& fy) {
  const double d2 = dx * dx + dy * dy + 1e-9;
  fx += m * dx / d2;
  fy += m * dy / d2;
}

struct QNode {
  double cx = 0, cy = 0, mass = 0, x0 = 0, y0 = 0, size = 0;
  int child[4] = {-1, -1, -1, -1};
  int first = 0, count = 0;
  bool leaf() const { return child[0] < 0 && child[1] < 0 && child[2] < 0 && child[3] < 0; }
};

class QuadTree {
 public:
  QuadTree(const std::vector<double>& xy, std::vector<std::uint32_t> pts) : xy_(xy), idx_(std::move(pts)) {
    if (idx_.empty()) return;
    double minx = 1e300, miny = 1e300, maxx = -1e300, maxy = -1e300;
    for (auto i : idx_) {
      minx = std::min(minx, xy_[2 * i]);
      maxx = std::max(maxx, xy_[2 * i]);
      miny = std::min(miny, xy_[2 * i + 1]);
      maxy = std::max(maxy, xy_[2 * i + 1]);
    }
    const double size = std::max(maxx - minx, maxy - miny) * 1.0001 + 1e-9;
    build(0, static_cast<int>(idx_.size()), minx, miny, size, 0);
  }

  void force(std::uint32_t i, double theta, double& fx, double& fy) const {
    fx = fy = 0;
    if (nodes_.empty()) return;
    const double x = xy_[2 * i], y = xy_[2 * i + 1];
    int stack[256];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
      const QNode& q = nodes_[static_cast<std::size_t>(stack[--top])];
      if (q.leaf()) {
        for (int k = q.first; k < q.first + q.count; ++k) {
          const auto j = idx_[static_cast<std::size_t>(k)];
          if (j != i) add_force(x - xy_[2 * j], y - xy_[2 * j + 1], 1.0, fx, fy);
        }
        continue;
      }
      const bool inside = x >= q.x0 && x < q.x0 + q.size && y >= q.y0 && y < q.y0 + q.size;
      const double dx = x - q.cx, dy = y - q.cy;
      const double d = std::sqrt(dx * dx + dy * dy);
      if (!inside && q.size < theta * d) {
        add_force(dx, dy, q.mass, fx, fy);
        continue;
      }
      for (int c = 3; c >= 0; --c)
        if (q.child[c] >= 0) stack[top++] = q.child[c];
    }
  }

 private:
  static constexpr int kMaxDepth = 48;

  int build(int lo, int hi, double x0, double y0, double size, int depth) {
    QNode q;
    q.x0 = x0;
    q.y0 = y0;
    q.size = size;
    q.first = lo;
    q.count = hi - lo;
    double sx = 0, sy = 0;
    for (int k = lo; k < hi; ++k) {
      sx += xy_[2 * idx_[static_cast<std::size_t>(k)]];
      sy += xy_[2 * idx_[static_cast<std::size_t>(k)] + 1];
    }
    q.mass = hi - lo;
    q.cx = sx / q.mass;
    q.cy = sy / q.mass;
    const int me = static_cast<int>(nodes_.size());
    nodes_.push_back(q);
    if (hi - lo <= 1 || depth >= kMaxDepth) return me;
    const double h = size / 2, mx = x0 + h, my = y0 + h;
    auto b = idx_.begin();
    const int m = static_cast<int>(std::stable_partition(b + lo, b + hi, [&](auto i) { return xy_[2 * i + 1] < my; }) - b);
    const int a = static_cast<int>(std::stable_partition(b + lo, b + m, [&](auto i) { return xy_[2 * i] < mx; }) - b);
    const int c2 = static_cast<int>(std::stable_partition(b + m, b + hi, [&](auto i) { return xy_[2 * i] < mx; }) - b);
    const int ranges[5] = {lo, a, m, c2, hi};
    const double ox[4] = {x0, mx, x0, mx}, oy[4] = {y0, y0, my, my};
    for (int c = 0; c < 4; ++c) {
      if (ranges[c + 1] > ranges[c]) {
        const int ch = build(ranges[c], ranges[c + 1], ox[c], oy[c], h, depth + 1);
        nodes_[static_cast<std::size_t>(me)].child[c] = ch;
      }
    }
    return me;
  }

  const std::vector<double>& xy_;
  std::vector<std::uint32_t> idx_;
  std::vector<QNode> nodes_;
};

std::vector<std::uint32_t> active_ids(const std::vector<bool>& active) {
  std::vector<std::uint32_t> ids;
  for (std::size_t i = 0; i < active.size(); ++i)
    if (active[i]) ids.push_back(static_cast<std::uint32_t>(i));
  return ids;
}

}  // namespace

std::vector<double> initial_positions(std::size_t n, std::uint64_t seed) {
  std::vector<double> xy(2 * n);
  const std::size_t cols = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n)))));
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> jitter(-0.01, 0.01);
  for (std::size_t i = 0; i < n; ++i) {
    const double gx = (static_cast<double>(i % cols) + 0.5) / static_cast<double>(cols);
    const double gy = (static_cast<double>(i / cols) + 0.5) / static_cast<double>(cols);
    xy[2 * i] = std::clamp(gx * 2 - 1 + jitter(rng), -1.0, 1.0);
    xy[2 * i + 1] = std::clamp(gy * 2 - 1 + jitter(rng), -1.0, 1.0);
  }
  return xy;
}

std::vector<LayoutEdge> layout_edges(const Csr& P, const std::vector<bool>& active, std::size_t per_node) {
  std::vector<LayoutEdge> out;
  std::vector<std::pair<double, std::uint32_t>> row;
  for (std::size_t i = 0; i < P.n; ++i) {
    if (!active[i]) continue;
    row.clear();
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) {
      const auto j = P.col[e];
      if (j == i || !active[j] || !(P.raw[e] > 0)) continue;
      row.emplace_back(P.raw[e], j);
    }
    const std::size_t m = std::min(per_node, row.size());
    std::partial_sort(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(m), row.end(),
                      [](const auto& x, const auto& y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
    for (std::size_t k = 0; k < m; ++k) out.push_back({static_cast<std::uint32_t>(i), row[k].second, row[k].first});
  }
  return out;
}

std::vector<double> exact_repulsion(const std::vector<double>& xy, const std::vector<bool>& active) {
  const std::size_t n = active.size();
  std::vector<double> f(2 * n, 0.0);
  const auto ids = active_ids(active);
#pragma omp parallel for schedule(dynamic, 64)
  for (std::size_t k = 0; k < ids.size(); ++k) {
    const auto i = ids[k];
    double fx = 0, fy = 0;
    for (auto j : ids)
      if (j != i) add_force(xy[2 * i] - xy[2 * j], xy[2 * i + 1] - xy[2 * j + 1], 1.0, fx, fy);
    f[2 * i] = fx;
    f[2 * i + 1] = fy;
  }
  return f;
}

std::vector<double> barnes_hut_repulsion(const std::vector<double>& xy, const std::vector<bool>& active,
                                         double theta) {
  const std::size_t n = active.size();
  std::vector<double> f(2 * n, 0.0);
  const auto ids = active_ids(active);
  const QuadTree tree(xy, ids);
#pragma omp parallel for schedule(dynamic, 64)
  for (std::size_t k = 0; k < ids.size(); ++k) {
    double fx = 0, fy = 0;
    tree.force(ids[k], theta, fx, fy);
    f[2 * ids[k]] = fx;
    f[2 * ids[k] + 1] = fy;
  }
  return f;
}

void run_layout(std::vector<double>& xy, const std::vector<bool>& active,
                const std::vector<LayoutEdge>& edges, const LayoutParams& p, int iterations) {
  const std::size_t n = active.size();
  std::size_t n_active = 0;
  for (bool a : active) n_active += a ? 1 : 0;
  if (n_active == 0 || iterations <= 0) return;
  double wmax = 0;
  for (const auto& e : edges) wmax = std::max(wmax, e.w);
  const double rep = p.repulsion / static_cast<double>(n_active);
  std::vector<double> f(2 * n, 0.0);
  for (int it = 0; it < iterations; ++it) {
    const double temp = p.max_step * (1.0 - static_cast<double>(it) / iterations) + 1e-4;
    const std::vector<double> r = barnes_hut_repulsion(xy, active, p.theta);
    for (std::size_t i = 0; i < n; ++i) {
      f[2 * i] = active[i] ? rep * r[2 * i] - p.gravity * xy[2 * i] : 0.0;
      f[2 * i + 1] = active[i] ? rep * r[2 * i + 1] - p.gravity * xy[2 * i + 1] : 0.0;
    }
    for (const auto& e : edges) {  // serial: deterministic accumulation
      if (!active[e.a] || !active[e.b] || !(wmax > 0)) continue;
      const double w = p.attraction * e.w / wmax;
      const double dx = xy[2 * e.b] - xy[2 * e.a], dy = xy[2 * e.b + 1] - xy[2 * e.a + 1];
      f[2 * e.a] += w * dx;
      f[2 * e.a + 1] += w * dy;
      f[2 * e.b] -= w * dx;
      f[2 * e.b + 1] -= w * dy;
    }
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
      if (!active[i]) continue;
      const double fx = f[2 * i], fy = f[2 * i + 1];
      const double m = std::sqrt(fx * fx + fy * fy);
      const double s = m > temp ? temp / m : 1.0;
      xy[2 * i] += fx * s;
      xy[2 * i + 1] += fy * s;
    }
  }
}

}  // namespace fx
