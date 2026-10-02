# Fluxscape Milestone 2 — deck.gl Landscape Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `fluxscape --serve` computes a time series of flux landscapes in the background and serves them to a deck.gl page in the browser. Each landscape goes through these stages:
1. a Barnes–Hut force layout of the money-flow graph;
2. recursive-bisection lattice snapping with hysteresis;
3. an IDW raster of the signed-log hotness.

The page renders the terrain, stocks, flux arcs and portfolio, has a left control panel and a scrubber, and supports click-to-shock with a Δh landscape.

**Architecture:**
- **C++ units, all O(N log N) or O(N·k):**
  - `src/geom/` (layout, lattice, IDW), pure functions plus one stateful `LandscapeBuilder` for warm starts.
  - `src/server/frame_store` runs the pipeline and the builder on a worker thread, keeps up to 300 landscapes, and keeps a pipeline copy at bar T−2 for fast shocks.
  - `src/server/http_server` exposes REST and SSE through cpp-httplib, plus static `web/`.
- **Front end:** `web/index.html`, `app.js` and `style.css`, with deck.gl 9 as a UMD build from unpkg and no build step.
- **Verification:** a headless-Chrome smoke script loads the page in self-test mode and saves a screenshot.

**Tech Stack:** C++20 (GCC 13), OpenMP, cpp-httplib, nlohmann/json, DuckDB lake (existing), doctest; deck.gl 9 (UMD); google-chrome headless (`/usr/bin/google-chrome`) for the smoke test.

**Spec:** `docs/superpowers/specs/2026-10-01-fluxscape-design.md` §6 (geometry), §9.2b–9.4 (serve mode, API, UI), §5.3 (shocks).

## Global Constraints

- Commit directly on `master`. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Stage explicit paths only, and never the stray `*.whl` files in the repo root.
- C++20, namespace `fx`, `#pragma once`. The build must be warning-free under `-Wall -Wextra -Wpedantic`.
- **OpenMP determinism rule:** parallel loops write only index-owned slots, and floating-point sums across indices are serial and in index order. Results must be bit-identical for any thread count.
- **No O(N²) or O(N³) work per frame** (N up to 10,000; about 6,000 active nodes).
- Geometry defaults:
  - layout: Barnes–Hut θ = 0.8, repulsion 0.05 (normalized by active count), attraction 1.0, gravity 0.1, max step 0.05, 150 initial and 20 per-frame iterations, 5 layout edges per node;
  - lattice: hysteresis max shift 2 (Chebyshev);
  - IDW: subdivision 4, power 2, radius 3 cells;
  - height: signed-log, `sign(h)·log1p(|h|)`;
  - arcs: top 2,000.
- Serve defaults: host `127.0.0.1`, port 8080, web root `web`, up to 300 landscapes. The preset is money-flow unless `--legacy` is given.
- Never read or print `.env`. Only the final task runs against real data, in replay mode, and nothing runs `--mode alpaca`.
- Build: `cmake -S . -B build && cmake --build build -j`. Tests: `./build/fluxtests`.

## File Structure

```
src/geom/layout.hpp/.cpp        Barnes–Hut force layout, layout edges, initial positions
src/geom/lattice.hpp/.cpp       lattice size, recursive coordinate bisection, hysteresis
src/geom/idw.hpp/.cpp           IDW raster
src/geom/landscape.hpp/.cpp     HeightMode, LandscapeBuilder, top arcs, delta raster
src/server/frame_store.hpp/.cpp background computation, frame cache, fast shock
src/server/http_server.hpp/.cpp REST + SSE + static files
src/cli/args.*, src/main.cpp    --serve, --port, --host, --web
web/index.html, web/app.js, web/style.css
scripts/ui_smoke.sh             headless-Chrome smoke test + screenshot
tests/test_layout.cpp, test_lattice.cpp, test_idw.cpp, test_landscape.cpp, test_frame_store.cpp, test_server.cpp
```

---

### Task 1: Barnes–Hut force layout

**Files:** Create `src/geom/layout.hpp`, `src/geom/layout.cpp`. Test `tests/test_layout.cpp`.

**Interfaces:**
- Consumes: `Csr` (`src/graph/csr.hpp`: `n, row_ptr, col, val, raw`).
- Produces:
  - `struct fx::LayoutParams { int iterations_init = 150; int iterations_step = 20; double theta = 0.8; double repulsion = 0.05; double attraction = 1.0; double gravity = 0.1; double max_step = 0.05; std::uint64_t seed = 7; }`
  - `struct fx::LayoutEdge { std::uint32_t a, b; double w; }`
  - `std::vector<double> fx::initial_positions(std::size_t n, std::uint64_t seed)`: a 2n vector, a deterministic jittered grid in [−1, 1]².
  - `std::vector<LayoutEdge> fx::layout_edges(const Csr& P, const std::vector<bool>& active, std::size_t per_node)`: for each active row, the top `per_node` off-diagonal edges to active columns by `raw`. Ties go to the lower column. Rows are emitted in ascending order.
  - `std::vector<double> fx::exact_repulsion(const std::vector<double>& xy, const std::vector<bool>& active)` and `std::vector<double> fx::barnes_hut_repulsion(const std::vector<double>& xy, const std::vector<bool>& active, double theta)`: the per-node sum of `(x_i − x_j) / (d² + 1e-9)` over the other active nodes. They return 2n values, 0 for inactive nodes.
  - `void fx::run_layout(std::vector<double>& xy, const std::vector<bool>& active, const std::vector<LayoutEdge>& edges, const LayoutParams& p, int iterations)`

- [ ] **Step 1: Write the failing tests** in `tests/test_layout.cpp`:
```cpp
#include <doctest/doctest.h>
#include <omp.h>

#include <cmath>
#include <random>
#include <vector>

#include "geom/layout.hpp"

using namespace fx;

namespace {
std::vector<double> random_xy(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> xy(2 * n);
  for (auto& v : xy) v = u(rng);
  return xy;
}
}  // namespace

TEST_CASE("Barnes-Hut repulsion approximates the exact sum") {
  const std::size_t n = 400;
  auto xy = random_xy(n, 3);
  std::vector<bool> active(n, true);
  active[7] = false;
  auto exact = exact_repulsion(xy, active);
  auto bh = barnes_hut_repulsion(xy, active, 0.5);
  double num = 0, den = 0;
  for (std::size_t k = 0; k < 2 * n; ++k) {
    num += (bh[k] - exact[k]) * (bh[k] - exact[k]);
    den += exact[k] * exact[k];
  }
  CHECK(std::sqrt(num / den) < 0.02);
  CHECK(bh[14] == 0.0);
  CHECK(bh[15] == 0.0);
  auto bh0 = barnes_hut_repulsion(xy, active, 0.0);  // never approximates
  for (std::size_t k = 0; k < 2 * n; ++k) CHECK(bh0[k] == doctest::Approx(exact[k]).epsilon(1e-9));
}

TEST_CASE("layout pulls strongly connected clusters together") {
  const std::size_t n = 40;
  std::vector<bool> active(n, true);
  std::vector<LayoutEdge> edges;
  for (std::uint32_t a = 0; a < 20; ++a)
    for (std::uint32_t b = a + 1; b < 20; ++b) {
      edges.push_back({a, b, 1.0});
      edges.push_back({a + 20, b + 20, 1.0});
    }
  auto xy = initial_positions(n, 7);
  LayoutParams p;
  run_layout(xy, active, edges, p, 300);
  auto dist = [&](std::size_t i, std::size_t j) { return std::hypot(xy[2 * i] - xy[2 * j], xy[2 * i + 1] - xy[2 * j + 1]); };
  double intra = 0, inter = 0;
  int ni = 0, nx = 0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j) {
      if ((i < 20) == (j < 20)) intra += dist(i, j), ++ni;
      else inter += dist(i, j), ++nx;
    }
  CHECK(intra / ni < 0.5 * (inter / nx));
}

TEST_CASE("layout is deterministic across thread counts") {
  const std::size_t n = 600;
  std::vector<bool> active(n, true);
  std::vector<LayoutEdge> edges;
  for (std::uint32_t i = 0; i + 1 < n; ++i) edges.push_back({i, i + 1, 1.0 + (i % 7)});
  auto a = initial_positions(n, 7), b = a;
  const int saved = omp_get_max_threads();
  omp_set_num_threads(1);
  run_layout(a, active, edges, LayoutParams{}, 30);
  omp_set_num_threads(std::max(saved, 4));
  run_layout(b, active, edges, LayoutParams{}, 30);
  omp_set_num_threads(saved);
  CHECK(a == b);
}

TEST_CASE("layout_edges keeps the top raw edges per active row, off-diagonal") {
  Csr P;
  P.n = 3;
  P.row_ptr = {0, 3, 4, 5};
  P.col = {0, 1, 2, 2, 1};
  P.val = {0.2, 0.5, 0.3, 1.0, 1.0};
  P.raw = {9.0, 5.0, 3.0, 1.0, 2.0};
  auto e = layout_edges(P, {true, true, true}, 1);
  REQUIRE(e.size() == 3);
  CHECK(e[0].a == 0);
  CHECK(e[0].b == 1);   // self edge (raw 9) skipped
  CHECK(e[0].w == 5.0);
  CHECK(e[1].a == 1);
  CHECK(e[2].a == 2);
  CHECK(layout_edges(P, {true, false, true}, 5).size() == 1);  // 0->2 only (1 inactive; 2->1 dropped)
}

TEST_CASE("initial positions are deterministic and in range") {
  auto a = initial_positions(50, 7), b = initial_positions(50, 7);
  CHECK(a == b);
  for (double v : a) CHECK(std::abs(v) <= 1.0);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: FAIL with `geom/layout.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/geom/layout.hpp`:
```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "graph/csr.hpp"

namespace fx {

struct LayoutParams {
  int iterations_init = 150;
  int iterations_step = 20;
  double theta = 0.8;
  double repulsion = 0.05;  // multiplied by 1 / active count
  double attraction = 1.0;
  double gravity = 0.1;
  double max_step = 0.05;
  std::uint64_t seed = 7;
};

struct LayoutEdge {
  std::uint32_t a, b;
  double w;
};

std::vector<double> initial_positions(std::size_t n, std::uint64_t seed);
std::vector<LayoutEdge> layout_edges(const Csr& P, const std::vector<bool>& active, std::size_t per_node);
std::vector<double> exact_repulsion(const std::vector<double>& xy, const std::vector<bool>& active);
std::vector<double> barnes_hut_repulsion(const std::vector<double>& xy, const std::vector<bool>& active,
                                         double theta);
// Spec 6.1: springs on edges (weight / max weight), Barnes-Hut repulsion, gravity, cooling step cap.
void run_layout(std::vector<double>& xy, const std::vector<bool>& active,
                const std::vector<LayoutEdge>& edges, const LayoutParams& p, int iterations);

}  // namespace fx
```

`src/geom/layout.cpp`:
```cpp
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/fluxtests -tc="*Barnes*,*layout*,*initial*"`, then run `./build/fluxtests` and check for warnings.
If "layout pulls strongly connected clusters together" fails, tune only `LayoutParams` defaults within reason (gravity 0.02–0.2, repulsion 0.02–0.2). Report the values used. Don't weaken the assertion.

- [ ] **Step 5: Commit**
```bash
git add src/geom/layout.* tests/test_layout.cpp
git commit -m "feat: Barnes-Hut force layout with deterministic OpenMP forces

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Lattice — recursive coordinate bisection and hysteresis

**Files:** Create `src/geom/lattice.hpp`, `src/geom/lattice.cpp`. Test `tests/test_lattice.cpp`.

**Interfaces:**
- Produces:
  - `struct fx::LatticeSize { std::size_t cols = 0, rows = 0; std::size_t cells() const; bool operator==(const LatticeSize&) const = default; }`
  - `LatticeSize fx::lattice_size(std::size_t n_active)`: cols = ⌈√n⌉, rows = ⌈n/cols⌉; {0, 0} for n = 0.
  - `std::vector<std::int32_t> fx::rcb_assign(const std::vector<double>& xy, const std::vector<bool>& active, LatticeSize)`: cell = row·cols + col, and −1 for inactive nodes.
  - `std::vector<std::int32_t> fx::apply_hysteresis(const std::vector<std::int32_t>& prev, const std::vector<std::int32_t>& next, LatticeSize, int max_shift)`

- [ ] **Step 1: Write the failing tests** in `tests/test_lattice.cpp`:
```cpp
#include <doctest/doctest.h>

#include <algorithm>
#include <random>
#include <set>
#include <vector>

#include "geom/lattice.hpp"

using namespace fx;

namespace {
void check_bijection(const std::vector<std::int32_t>& cell, const std::vector<bool>& active, LatticeSize s) {
  std::set<std::int32_t> seen;
  for (std::size_t i = 0; i < cell.size(); ++i) {
    if (!active[i]) {
      CHECK(cell[i] == -1);
      continue;
    }
    REQUIRE(cell[i] >= 0);
    CHECK(static_cast<std::size_t>(cell[i]) < s.cells());
    CHECK(seen.insert(cell[i]).second);
  }
}
}  // namespace

TEST_CASE("lattice size is near-square and fits all active nodes") {
  CHECK(lattice_size(0).cells() == 0);
  auto s = lattice_size(6000);
  CHECK(s.cols == 78);
  CHECK(s.rows == 77);
  CHECK(s.cells() >= 6000);
  CHECK(lattice_size(1).cells() == 1);
}

TEST_CASE("rcb assigns one active node per cell and preserves x order across columns") {
  const std::size_t n = 103;
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> u(-1, 1);
  std::vector<double> xy(2 * n);
  for (auto& v : xy) v = u(rng);
  std::vector<bool> active(n, true);
  active[3] = active[50] = false;
  auto s = lattice_size(101);
  auto cell = rcb_assign(xy, active, s);
  check_bijection(cell, active, s);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      if (active[i] && active[j] && cell[i] % static_cast<int>(s.cols) < cell[j] % static_cast<int>(s.cols))
        CHECK(xy[2 * i] <= xy[2 * j]);
}

TEST_CASE("hysteresis keeps nearby previous cells and stays a bijection") {
  LatticeSize s{4, 4};
  std::vector<bool> active(5, true);
  std::vector<std::int32_t> prev = {0, 5, 10, 15, -1};
  std::vector<std::int32_t> next = {1, 5, 0, 3, 2};  // node 0 moved 1 cell; node 2 moved far
  auto out = apply_hysteresis(prev, next, s, 2);
  check_bijection(out, active, s);
  CHECK(out[0] == 0);   // kept (shift 1)
  CHECK(out[1] == 5);
  CHECK(out[2] != 10);  // shift > 2: takes its new cell (0 is taken by node 0 -> nearest free)
  CHECK(out[3] == 3);   // prev 15 is 3 rows away -> new cell
  CHECK(out[4] == 2);   // new node
}

TEST_CASE("hysteresis resolves conflicts deterministically") {
  LatticeSize s{3, 3};
  std::vector<bool> active(3, true);
  std::vector<std::int32_t> prev = {4, 4, 4};  // impossible prior state, still must yield a bijection
  std::vector<std::int32_t> next = {0, 1, 2};
  auto out = apply_hysteresis(prev, next, s, 2);
  check_bijection(out, active, s);
  CHECK(out[0] == 4);
  CHECK(out == apply_hysteresis(prev, next, s, 2));
}
```

- [ ] **Step 2: Run to verify failure.** Expected: `geom/lattice.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/geom/lattice.hpp`:
```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fx {

struct LatticeSize {
  std::size_t cols = 0, rows = 0;
  std::size_t cells() const { return cols * rows; }
  bool operator==(const LatticeSize&) const = default;
};

LatticeSize lattice_size(std::size_t n_active);
// Spec 6.2: sort by x into column bands of `rows` nodes, then by y within a band. cell = row*cols+col.
std::vector<std::int32_t> rcb_assign(const std::vector<double>& xy, const std::vector<bool>& active,
                                     LatticeSize size);
// Keep prev cell when within max_shift (Chebyshev) and free; else new cell if free; else nearest free.
std::vector<std::int32_t> apply_hysteresis(const std::vector<std::int32_t>& prev,
                                           const std::vector<std::int32_t>& next, LatticeSize size,
                                           int max_shift);

}  // namespace fx
```

`src/geom/lattice.cpp`:
```cpp
#include "geom/lattice.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace fx {

LatticeSize lattice_size(std::size_t n) {
  if (n == 0) return {0, 0};
  const auto cols = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
  return {cols, (n + cols - 1) / cols};
}

std::vector<std::int32_t> rcb_assign(const std::vector<double>& xy, const std::vector<bool>& active,
                                     LatticeSize size) {
  const std::size_t n = active.size();
  std::vector<std::int32_t> cell(n, -1);
  std::vector<std::uint32_t> ids;
  for (std::size_t i = 0; i < n; ++i)
    if (active[i]) ids.push_back(static_cast<std::uint32_t>(i));
  if (ids.empty() || size.rows == 0) return cell;
  std::sort(ids.begin(), ids.end(), [&](auto a, auto b) {
    return xy[2 * a] < xy[2 * b] || (xy[2 * a] == xy[2 * b] && a < b);
  });
  for (std::size_t band = 0; band * size.rows < ids.size(); ++band) {
    const std::size_t lo = band * size.rows, hi = std::min(ids.size(), lo + size.rows);
    std::sort(ids.begin() + static_cast<std::ptrdiff_t>(lo), ids.begin() + static_cast<std::ptrdiff_t>(hi),
              [&](auto a, auto b) { return xy[2 * a + 1] < xy[2 * b + 1] || (xy[2 * a + 1] == xy[2 * b + 1] && a < b); });
    for (std::size_t r = lo; r < hi; ++r)
      cell[ids[r]] = static_cast<std::int32_t>((r - lo) * size.cols + band);
  }
  return cell;
}

std::vector<std::int32_t> apply_hysteresis(const std::vector<std::int32_t>& prev,
                                           const std::vector<std::int32_t>& next, LatticeSize size,
                                           int max_shift) {
  const std::size_t n = next.size();
  const auto cols = static_cast<long>(size.cols), rows = static_cast<long>(size.rows);
  std::vector<std::int32_t> out(n, -1);
  std::vector<char> taken(size.cells(), 0);
  auto cheb = [&](long a, long b) { return std::max(std::labs(a % cols - b % cols), std::labs(a / cols - b / cols)); };
  for (std::size_t i = 0; i < n; ++i) {  // pass 1: keep nearby previous cells
    if (next[i] < 0 || i >= prev.size() || prev[i] < 0 || static_cast<std::size_t>(prev[i]) >= size.cells()) continue;
    if (cheb(prev[i], next[i]) <= max_shift && !taken[static_cast<std::size_t>(prev[i])]) {
      out[i] = prev[i];
      taken[static_cast<std::size_t>(prev[i])] = 1;
    }
  }
  for (std::size_t i = 0; i < n; ++i) {  // pass 2: new cell if free
    if (next[i] < 0 || out[i] >= 0 || taken[static_cast<std::size_t>(next[i])]) continue;
    out[i] = next[i];
    taken[static_cast<std::size_t>(next[i])] = 1;
  }
  for (std::size_t i = 0; i < n; ++i) {  // pass 3: nearest free cell, ring by ring
    if (next[i] < 0 || out[i] >= 0) continue;
    const long c0 = next[i] % cols, r0 = next[i] / cols;
    for (long r = 1; r <= std::max(cols, rows) && out[i] < 0; ++r) {
      for (long dr = -r; dr <= r && out[i] < 0; ++dr)
        for (long dc = -r; dc <= r && out[i] < 0; ++dc) {
          if (std::max(std::labs(dc), std::labs(dr)) != r) continue;
          const long c = c0 + dc, rr = r0 + dr;
          if (c < 0 || c >= cols || rr < 0 || rr >= rows) continue;
          const auto k = static_cast<std::size_t>(rr * cols + c);
          if (!taken[k]) {
            out[i] = static_cast<std::int32_t>(k);
            taken[k] = 1;
          }
        }
    }
  }
  return out;
}

}  // namespace fx
```

- [ ] **Step 4: Run the tests to verify they pass** (`-tc="*lattice*,*rcb*,*hysteresis*"`, then the full suite). The build must be warning-free.

- [ ] **Step 5: Commit**
```bash
git add src/geom/lattice.* tests/test_lattice.cpp
git commit -m "feat: lattice snap by recursive coordinate bisection with hysteresis

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: IDW raster

**Files:** Create `src/geom/idw.hpp`, `src/geom/idw.cpp`. Test `tests/test_idw.cpp`.

**Interfaces:**
- Consumes: `LatticeSize` (Task 2).
- Produces:
  - `struct fx::IdwParams { int subdivision = 4; double power = 2.0; int radius_cells = 3; }`
  - `struct fx::Raster { std::size_t w = 0, h = 0; std::vector<float> z; float zmin = 0, zmax = 0; }`
  - `Raster fx::idw_raster(const std::vector<std::int32_t>& cell, const std::vector<double>& value, LatticeSize, const IdwParams&)`: z is row-major `[py·w + px]`. Pixel centres are at lattice coordinates ((px+0.5)/s, (py+0.5)/s), and node centres at (col+0.5, row+0.5). Nodes with a non-finite value are ignored.

- [ ] **Step 1: Write the failing tests** in `tests/test_idw.cpp`:
```cpp
#include <doctest/doctest.h>
#include <omp.h>

#include <cmath>
#include <limits>
#include <vector>

#include "geom/idw.hpp"

using namespace fx;

TEST_CASE("IDW with subdivision 1 is exact at the nodes") {
  LatticeSize s{3, 2};
  std::vector<std::int32_t> cell = {0, 1, 2, 3, 4, 5};
  std::vector<double> v = {1, 2, 3, 4, 5, 6};
  IdwParams p;
  p.subdivision = 1;
  Raster r = idw_raster(cell, v, s, p);
  REQUIRE(r.w == 3);
  REQUIRE(r.h == 2);
  for (int k = 0; k < 6; ++k) CHECK(r.z[static_cast<std::size_t>(k)] == doctest::Approx(v[static_cast<std::size_t>(k)]));
  CHECK(r.zmin == doctest::Approx(1));
  CHECK(r.zmax == doctest::Approx(6));
}

TEST_CASE("IDW stays within the node value range and uses the nearest node beyond the radius") {
  LatticeSize s{20, 20};
  std::vector<std::int32_t> cell = {0, 399};
  std::vector<double> v = {-2.0, 5.0};
  IdwParams p;
  p.radius_cells = 2;
  Raster r = idw_raster(cell, v, s, p);
  REQUIRE(r.z.size() == 80 * 80);
  for (float z : r.z) {
    CHECK(z >= -2.0f - 1e-5f);
    CHECK(z <= 5.0f + 1e-5f);
  }
  CHECK(r.z[0] == doctest::Approx(-2.0));                  // next to node 0
  CHECK(r.z[80 * 79 + 79] == doctest::Approx(5.0));       // next to node 399
  CHECK(r.z[40 * 80 + 10] == doctest::Approx(-2.0));      // closer to node 0, outside both radii
}

TEST_CASE("IDW ignores non-finite values and is deterministic across threads") {
  LatticeSize s{10, 10};
  std::vector<std::int32_t> cell;
  std::vector<double> v;
  for (int i = 0; i < 100; ++i) {
    cell.push_back(i);
    v.push_back(i == 55 ? std::numeric_limits<double>::quiet_NaN() : std::sin(i * 0.3));
  }
  const int saved = omp_get_max_threads();
  omp_set_num_threads(1);
  Raster a = idw_raster(cell, v, s, IdwParams{});
  omp_set_num_threads(std::max(saved, 4));
  Raster b = idw_raster(cell, v, s, IdwParams{});
  omp_set_num_threads(saved);
  CHECK(a.z == b.z);
  for (float z : a.z) CHECK(std::isfinite(z));
}
```

- [ ] **Step 2: Run to verify failure.** Expected: `geom/idw.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/geom/idw.hpp`:
```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "geom/lattice.hpp"

namespace fx {

struct IdwParams {
  int subdivision = 4;
  double power = 2.0;
  int radius_cells = 3;
};

struct Raster {
  std::size_t w = 0, h = 0;
  std::vector<float> z;  // row-major [py * w + px]
  float zmin = 0, zmax = 0;
};

// Spec 6.3.
Raster idw_raster(const std::vector<std::int32_t>& cell, const std::vector<double>& value, LatticeSize size,
                  const IdwParams& p);

}  // namespace fx
```

`src/geom/idw.cpp`:
```cpp
#include "geom/idw.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fx {

Raster idw_raster(const std::vector<std::int32_t>& cell, const std::vector<double>& value, LatticeSize size,
                  const IdwParams& p) {
  Raster r;
  const long s = std::max(1, p.subdivision);
  const long cols = static_cast<long>(size.cols), rows = static_cast<long>(size.rows);
  r.w = static_cast<std::size_t>(cols * s);
  r.h = static_cast<std::size_t>(rows * s);
  r.z.assign(r.w * r.h, 0.0f);
  if (r.z.empty()) return r;
  std::vector<double> occ(size.cells(), std::numeric_limits<double>::quiet_NaN());
  bool any = false;
  for (std::size_t i = 0; i < cell.size(); ++i)
    if (cell[i] >= 0 && std::isfinite(value[i])) {
      occ[static_cast<std::size_t>(cell[i])] = value[i];
      any = true;
    }
  if (!any) return r;
  const long R = std::max(0, p.radius_cells);
#pragma omp parallel for schedule(static)
  for (long py = 0; py < static_cast<long>(r.h); ++py) {
    for (long px = 0; px < static_cast<long>(r.w); ++px) {
      const double u = (static_cast<double>(px) + 0.5) / static_cast<double>(s);
      const double v = (static_cast<double>(py) + 0.5) / static_cast<double>(s);
      const long c0 = static_cast<long>(u), r0 = static_cast<long>(v);
      double num = 0, den = 0, exact = std::numeric_limits<double>::quiet_NaN();
      for (long rr = std::max(0L, r0 - R); rr <= std::min(rows - 1, r0 + R) && std::isnan(exact); ++rr)
        for (long cc = std::max(0L, c0 - R); cc <= std::min(cols - 1, c0 + R); ++cc) {
          const double val = occ[static_cast<std::size_t>(rr * cols + cc)];
          if (std::isnan(val)) continue;
          const double d = std::hypot(u - (static_cast<double>(cc) + 0.5), v - (static_cast<double>(rr) + 0.5));
          if (d < 1e-9) {
            exact = val;
            break;
          }
          const double w = 1.0 / std::pow(d, p.power);
          num += w * val;
          den += w;
        }
      double z;
      if (!std::isnan(exact)) {
        z = exact;
      } else if (den > 0) {
        z = num / den;
      } else {  // nearest occupied cell, ring by ring beyond R
        double best_d = std::numeric_limits<double>::infinity();
        z = 0;
        for (long ring = R + 1; ring <= std::max(cols, rows) && !std::isfinite(best_d); ++ring)
          for (long dr = -ring; dr <= ring; ++dr)
            for (long dc = -ring; dc <= ring; ++dc) {
              if (std::max(std::labs(dc), std::labs(dr)) != ring) continue;
              const long cc = c0 + dc, rr = r0 + dr;
              if (cc < 0 || cc >= cols || rr < 0 || rr >= rows) continue;
              const double val = occ[static_cast<std::size_t>(rr * cols + cc)];
              if (std::isnan(val)) continue;
              const double d = std::hypot(u - (static_cast<double>(cc) + 0.5), v - (static_cast<double>(rr) + 0.5));
              if (d < best_d) {
                best_d = d;
                z = val;
              }
            }
      }
      r.z[static_cast<std::size_t>(py) * r.w + static_cast<std::size_t>(px)] = static_cast<float>(z);
    }
  }
  r.zmin = *std::min_element(r.z.begin(), r.z.end());
  r.zmax = *std::max_element(r.z.begin(), r.z.end());
  return r;
}

}  // namespace fx
```
Note: in the nearest-node fallback, ties keep the first cell found in ring order, so the result is deterministic.

- [ ] **Step 4: Run the tests to verify they pass.** The build must be warning-free.

- [ ] **Step 5: Commit**
```bash
git add src/geom/idw.* tests/test_idw.cpp
git commit -m "feat: IDW landscape raster

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Landscape builder

**Files:** Create `src/geom/landscape.hpp`, `src/geom/landscape.cpp`. Test `tests/test_landscape.cpp`.

**Interfaces:**
- Consumes: Tasks 1–3; `Frame` (`src/pipeline/core_pipeline.hpp`: `t, active, pi, h, forecasts, P`).
- Produces:
  - `enum class fx::HeightMode { SignedLog, Linear }`, plus `parse_height_mode("signed-log"|"linear")` and `to_string`.
  - `double fx::display_height(double h, HeightMode)`
  - `struct fx::LandscapeParams { LayoutParams layout; IdwParams idw; HeightMode height = HeightMode::SignedLog; int max_shift = 2; std::size_t edges_per_node = 5; std::size_t max_arcs = 2000; }`
  - `struct fx::LandscapeNode { std::uint32_t i; std::int32_t cell; float fx, fy; double h, hdisp, pi, score; }`
  - `struct fx::LandscapeArc { std::uint32_t a, b; double w; }`
  - `struct fx::LandscapeFrame { TimePoint t; LatticeSize size; std::vector<LandscapeNode> nodes; std::vector<LandscapeArc> arcs; Raster raster; double compute_ms; }`
  - `class fx::LandscapeBuilder { LandscapeBuilder(std::size_t n, LandscapeParams); LandscapeFrame build(const Frame&); }`
  - `std::vector<LandscapeArc> fx::top_arcs(const Csr&, const std::vector<bool>& active, std::size_t max_arcs)`, sorted by w descending; ties go to the lower (a, b).
  - `Raster fx::delta_raster(const LandscapeFrame& base, const std::vector<double>& delta, const LandscapeParams&)`: an IDW of `display_height(delta[i])` on the base frame's cells.

- [ ] **Step 1: Write the failing tests** in `tests/test_landscape.cpp`:
```cpp
#include <doctest/doctest.h>

#include <cmath>
#include <set>
#include <vector>

#include "geom/landscape.hpp"
#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "pipeline/core_pipeline.hpp"
#include "test_util.hpp"

using namespace fx;

namespace {
Frame synthetic_frame() {
  SyntheticConfig cfg;
  BarStore store(test::temp_dir("landscape"));
  auto secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : secs) tickers.push_back(s.ticker);
  return run_panel_last(build_panel(store, tickers, cfg.tf), CoreParams::money_flow());
}
}  // namespace

TEST_CASE("display height") {
  CHECK(display_height(0.0, HeightMode::SignedLog) == 0.0);
  CHECK(display_height(std::exp(1.0) - 1.0, HeightMode::SignedLog) == doctest::Approx(1.0));
  CHECK(display_height(-(std::exp(2.0) - 1.0), HeightMode::SignedLog) == doctest::Approx(-2.0));
  CHECK(display_height(-0.3, HeightMode::Linear) == -0.3);
  CHECK(parse_height_mode(to_string(HeightMode::Linear)) == HeightMode::Linear);
}

TEST_CASE("landscape build: one cell per active node, raster sized from the lattice, arcs sorted") {
  const Frame f = synthetic_frame();
  LandscapeParams p;
  p.max_arcs = 50;
  LandscapeBuilder b(f.active.size(), p);
  LandscapeFrame lf = b.build(f);
  std::size_t n_active = 0;
  for (bool a : f.active) n_active += a ? 1 : 0;
  CHECK(lf.nodes.size() == n_active);
  CHECK(lf.size == lattice_size(n_active));
  std::set<std::int32_t> cells;
  for (const auto& nd : lf.nodes) {
    CHECK(nd.cell >= 0);
    CHECK(cells.insert(nd.cell).second);
    CHECK(nd.hdisp == doctest::Approx(display_height(nd.h, HeightMode::SignedLog)));
    CHECK(nd.fx >= 0.0f);
    CHECK(nd.fx <= 1.0f);
  }
  CHECK(lf.raster.w == lf.size.cols * 4);
  CHECK(lf.raster.h == lf.size.rows * 4);
  CHECK(lf.arcs.size() <= 50);
  for (std::size_t k = 1; k < lf.arcs.size(); ++k) CHECK(lf.arcs[k - 1].w >= lf.arcs[k].w);
  CHECK(lf.t == f.t);
}

TEST_CASE("rebuilding on the same frame keeps most cells (warm start + hysteresis)") {
  const Frame f = synthetic_frame();
  LandscapeBuilder b(f.active.size(), LandscapeParams{});
  LandscapeFrame a = b.build(f);
  LandscapeFrame c = b.build(f);
  std::size_t same = 0;
  for (std::size_t k = 0; k < a.nodes.size(); ++k) same += a.nodes[k].cell == c.nodes[k].cell ? 1 : 0;
  CHECK(static_cast<double>(same) >= 0.8 * static_cast<double>(a.nodes.size()));
}

TEST_CASE("delta raster uses the base cells") {
  const Frame f = synthetic_frame();
  LandscapeBuilder b(f.active.size(), LandscapeParams{});
  LandscapeFrame lf = b.build(f);
  std::vector<double> delta(f.active.size(), 0.0);
  delta[lf.nodes.front().i] = 1.0;
  Raster r = delta_raster(lf, delta, LandscapeParams{});
  CHECK(r.w == lf.raster.w);
  CHECK(r.zmax > 0.0f);
  CHECK(r.zmin >= 0.0f);
}
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`src/geom/landscape.hpp`:
```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "core/types.hpp"
#include "geom/idw.hpp"
#include "geom/lattice.hpp"
#include "geom/layout.hpp"
#include "pipeline/core_pipeline.hpp"

namespace fx {

enum class HeightMode { SignedLog, Linear };
HeightMode parse_height_mode(std::string_view s);  // "signed-log" | "linear"
std::string_view to_string(HeightMode m);
double display_height(double h, HeightMode m);

struct LandscapeParams {
  LayoutParams layout;
  IdwParams idw;
  HeightMode height = HeightMode::SignedLog;
  int max_shift = 2;
  std::size_t edges_per_node = 5;
  std::size_t max_arcs = 2000;
};

struct LandscapeNode {
  std::uint32_t i;
  std::int32_t cell;
  float fx, fy;  // free layout position normalized to [0, 1]
  double h, hdisp, pi, score;
};

struct LandscapeArc {
  std::uint32_t a, b;
  double w;
};

struct LandscapeFrame {
  TimePoint t = 0;
  LatticeSize size;
  std::vector<LandscapeNode> nodes;  // active nodes, ascending i
  std::vector<LandscapeArc> arcs;
  Raster raster;
  double compute_ms = 0;
};

std::vector<LandscapeArc> top_arcs(const Csr& P, const std::vector<bool>& active, std::size_t max_arcs);
Raster delta_raster(const LandscapeFrame& base, const std::vector<double>& delta, const LandscapeParams& p);

// Stateful across frames: warm-started layout positions and hysteresis on the lattice cells.
class LandscapeBuilder {
 public:
  LandscapeBuilder(std::size_t n, LandscapeParams params);
  LandscapeFrame build(const Frame& f);

 private:
  std::size_t n_;
  LandscapeParams p_;
  std::vector<double> xy_;
  std::vector<std::int32_t> cells_;
  LatticeSize size_;
  bool first_ = true;
};

}  // namespace fx
```

`src/geom/landscape.cpp`:
```cpp
#include "geom/landscape.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace fx {

HeightMode parse_height_mode(std::string_view s) {
  if (s == "signed-log") return HeightMode::SignedLog;
  if (s == "linear") return HeightMode::Linear;
  throw std::invalid_argument("unknown height mode: " + std::string(s));
}

std::string_view to_string(HeightMode m) { return m == HeightMode::SignedLog ? "signed-log" : "linear"; }

double display_height(double h, HeightMode m) {
  if (!std::isfinite(h)) return std::numeric_limits<double>::quiet_NaN();
  if (m == HeightMode::Linear) return h;
  return h >= 0 ? std::log1p(h) : -std::log1p(-h);
}

std::vector<LandscapeArc> top_arcs(const Csr& P, const std::vector<bool>& active, std::size_t max_arcs) {
  std::vector<LandscapeArc> arcs;
  for (std::size_t i = 0; i < P.n; ++i) {
    if (!active[i]) continue;
    for (auto e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) {
      const auto j = P.col[e];
      if (j != i && active[j] && P.raw[e] > 0) arcs.push_back({static_cast<std::uint32_t>(i), j, P.raw[e]});
    }
  }
  auto better = [](const LandscapeArc& x, const LandscapeArc& y) {
    if (x.w != y.w) return x.w > y.w;
    if (x.a != y.a) return x.a < y.a;
    return x.b < y.b;
  };
  if (arcs.size() > max_arcs) {
    std::nth_element(arcs.begin(), arcs.begin() + static_cast<std::ptrdiff_t>(max_arcs), arcs.end(), better);
    arcs.resize(max_arcs);
  }
  std::sort(arcs.begin(), arcs.end(), better);
  return arcs;
}

Raster delta_raster(const LandscapeFrame& base, const std::vector<double>& delta, const LandscapeParams& p) {
  std::vector<std::int32_t> cell(delta.size(), -1);
  std::vector<double> v(delta.size(), std::numeric_limits<double>::quiet_NaN());
  for (const auto& nd : base.nodes) {
    cell[nd.i] = nd.cell;
    v[nd.i] = display_height(delta[nd.i], p.height);
  }
  return idw_raster(cell, v, base.size, p.idw);
}

LandscapeBuilder::LandscapeBuilder(std::size_t n, LandscapeParams params)
    : n_(n), p_(params), xy_(initial_positions(n, params.layout.seed)) {}

LandscapeFrame LandscapeBuilder::build(const Frame& f) {
  const auto t0 = std::chrono::steady_clock::now();
  if (f.active.size() != n_) throw std::invalid_argument("LandscapeBuilder: frame size mismatch");
  const auto edges = layout_edges(f.P, f.active, p_.edges_per_node);
  run_layout(xy_, f.active, edges, p_.layout, first_ ? p_.layout.iterations_init : p_.layout.iterations_step);
  std::size_t n_active = 0;
  double minx = 1e300, maxx = -1e300, miny = 1e300, maxy = -1e300;
  for (std::size_t i = 0; i < n_; ++i) {
    if (!f.active[i]) continue;
    ++n_active;
    minx = std::min(minx, xy_[2 * i]);
    maxx = std::max(maxx, xy_[2 * i]);
    miny = std::min(miny, xy_[2 * i + 1]);
    maxy = std::max(maxy, xy_[2 * i + 1]);
  }
  const LatticeSize size = lattice_size(n_active);
  auto next = rcb_assign(xy_, f.active, size);
  cells_ = (!first_ && size == size_) ? apply_hysteresis(cells_, next, size, p_.max_shift) : next;
  size_ = size;
  first_ = false;

  LandscapeFrame lf;
  lf.t = f.t;
  lf.size = size;
  std::vector<double> values(n_, std::numeric_limits<double>::quiet_NaN());
  const double rx = maxx > minx ? maxx - minx : 1.0, ry = maxy > miny ? maxy - miny : 1.0;
  const auto& score = f.forecasts.empty() ? f.h : f.forecasts.front().score;
  for (std::size_t i = 0; i < n_; ++i) {
    if (!f.active[i]) continue;
    const double hd = display_height(f.h[i], p_.height);
    values[i] = hd;
    lf.nodes.push_back({static_cast<std::uint32_t>(i), cells_[i], static_cast<float>((xy_[2 * i] - minx) / rx),
                        static_cast<float>((xy_[2 * i + 1] - miny) / ry), f.h[i], hd, f.pi[i], score[i]});
  }
  lf.raster = idw_raster(cells_, values, size, p_.idw);
  lf.arcs = top_arcs(f.P, f.active, p_.max_arcs);
  lf.compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return lf;
}

}  // namespace fx
```

- [ ] **Step 4: Run the tests to verify they pass** (full suite, warning-free).

- [ ] **Step 5: Commit**
```bash
git add src/geom/landscape.* tests/test_landscape.cpp
git commit -m "feat: landscape builder (layout, lattice, IDW, arcs, delta raster)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Frame store (background computation, cache, fast shocks)

**Files:** Create `src/server/frame_store.hpp`, `src/server/frame_store.cpp`. Test `tests/test_frame_store.cpp`.

**Interfaces:**
- Consumes: `CorePipeline` (copyable, `step(panel, t, shocks)`), `Shock`, `shock_response`/`ShockDelta` (`src/pipeline/shock.hpp`), `LandscapeBuilder`/`delta_raster` (Task 4), `Panel`, `Security`.
- Produces:
```cpp
class fx::FrameStore {
 public:
  struct Status { std::size_t computed = 0, total = 0; bool running = false, ready = false; std::string error; std::uint64_t generation = 0; };
  struct ShockResult { TimePoint t; ShockDelta delta; Raster raster; std::shared_ptr<const LandscapeFrame> base; };
  FrameStore(Panel panel, std::vector<Security> nodes, CoreParams core, LandscapeParams land, std::size_t max_frames = 300);
  ~FrameStore();
  void start();
  void set_params(CoreParams core, LandscapeParams land);
  Status status() const;
  std::vector<TimePoint> times() const;
  std::shared_ptr<const LandscapeFrame> landscape(std::optional<TimePoint> t) const;  // nullopt = latest; nullptr if absent
  ShockResult shock(const std::vector<Shock>& shocks);  // throws std::runtime_error if not ready
  std::uint64_t wait_for_change(std::uint64_t seen, std::chrono::milliseconds timeout) const;
  const Panel& panel() const; const std::vector<Security>& nodes() const;
  CoreParams core_params() const; LandscapeParams landscape_params() const;
};
```
Semantics:
- The worker computes core frames t = 1..T−1. Landscapes are built only for the last `max_frames` bars, warm-starting from the first landscape bar.
- Just before stepping t = T−1, the worker keeps a copy of the pipeline (`pre_last_`).
- `set_params` bumps the generation, joins the worker, clears the caches and restarts.
- A version counter increments on every progress change (each step and each stored landscape) and on completion, and `wait_for_change` blocks on it.
- Worker errors are caught into `Status::error`, which leaves `ready` false.

- [ ] **Step 1: Write the failing tests** in `tests/test_frame_store.cpp`:
```cpp
#include <doctest/doctest.h>

#include <chrono>
#include <thread>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "server/frame_store.hpp"
#include "test_util.hpp"

using namespace fx;
using namespace std::chrono_literals;

namespace {
struct Market {
  Panel panel;
  std::vector<Security> secs;
};
Market market() {
  SyntheticConfig cfg;
  cfg.bars = 120;
  cfg.rotation_start = 60;
  BarStore store(test::temp_dir("frame_store"));
  Market m;
  m.secs = generate_synthetic(cfg, store);
  std::vector<std::string> tickers;
  for (auto& s : m.secs) tickers.push_back(s.ticker);
  m.panel = build_panel(store, tickers, cfg.tf);
  return m;
}
void wait_ready(const FrameStore& fs) {
  for (int k = 0; k < 600 && !fs.status().ready; ++k) {
    REQUIRE(fs.status().error.empty());
    std::this_thread::sleep_for(50ms);
  }
  REQUIRE(fs.status().ready);
}
}  // namespace

TEST_CASE("frame store computes landscapes for the last max_frames bars") {
  Market m = market();
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 30);
  fs.start();
  wait_ready(fs);
  auto st = fs.status();
  CHECK(st.computed == st.total);
  CHECK(st.total == m.panel.T() - 1);
  auto times = fs.times();
  REQUIRE(times.size() == 30);
  CHECK(times.back() == m.panel.times.back());
  auto latest = fs.landscape(std::nullopt);
  REQUIRE(latest);
  CHECK(latest->t == m.panel.times.back());
  CHECK(fs.landscape(times.front()));
  CHECK_FALSE(fs.landscape(m.panel.times.front()));
}

TEST_CASE("frame store shock gives deltas and a delta raster on the latest landscape") {
  Market m = market();
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  fs.start();
  wait_ready(fs);
  auto latest = fs.landscape(std::nullopt);
  const std::size_t node = latest->nodes.front().i;
  auto r = fs.shock({{node, -10.0}});
  CHECK(r.t == m.panel.times.back());
  CHECK(r.delta.dh.size() == m.secs.size());
  CHECK(r.delta.l1_dpi > 0);
  CHECK(r.delta.dh[node] < 0);
  CHECK(r.raster.w == latest->raster.w);
}

TEST_CASE("frame store restarts on new parameters and is deterministic") {
  Market m = market();
  FrameStore a(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  FrameStore b(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  a.start();
  b.start();
  wait_ready(a);
  wait_ready(b);
  CHECK(a.landscape(std::nullopt)->raster.z == b.landscape(std::nullopt)->raster.z);
  const auto g = a.status().generation;
  a.set_params(CoreParams::legacy(), LandscapeParams{});
  wait_ready(a);
  CHECK(a.status().generation > g);
  CHECK(a.times().size() == 10);
}

TEST_CASE("wait_for_change returns when progress happens") {
  Market m = market();
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  const auto v0 = fs.wait_for_change(0, 1ms);
  fs.start();
  const auto v1 = fs.wait_for_change(v0, 5000ms);
  CHECK(v1 > v0);
  wait_ready(fs);
}

TEST_CASE("shock before ready throws") {
  Market m = market();
  FrameStore fs(m.panel, m.secs, CoreParams::money_flow(), LandscapeParams{}, 10);
  CHECK_THROWS_AS(fs.shock({{0, -1.0}}), std::runtime_error);
}
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`src/server/frame_store.hpp`:
```cpp
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

namespace fx {

class FrameStore {
 public:
  struct Status {
    std::size_t computed = 0, total = 0;
    bool running = false, ready = false;
    std::string error;
    std::uint64_t generation = 0;
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
  void set_params(CoreParams core, LandscapeParams land);
  Status status() const;
  std::vector<TimePoint> times() const;
  std::shared_ptr<const LandscapeFrame> landscape(std::optional<TimePoint> t) const;
  ShockResult shock(const std::vector<Shock>& shocks);
  std::uint64_t wait_for_change(std::uint64_t seen, std::chrono::milliseconds timeout) const;
  const Panel& panel() const { return panel_; }
  const std::vector<Security>& nodes() const { return nodes_; }
  CoreParams core_params() const;
  LandscapeParams landscape_params() const;

 private:
  void stop_worker();
  void run(std::uint64_t gen, CoreParams core, LandscapeParams land);
  void bump();  // version++ and notify (call with m_ held)

  const Panel panel_;
  const std::vector<Security> nodes_;
  const std::size_t max_frames_;
  mutable std::mutex m_;
  mutable std::condition_variable cv_;
  std::thread worker_;
  std::atomic<std::uint64_t> gen_{0};
  std::uint64_t version_ = 0;
  CoreParams core_;
  LandscapeParams land_;
  Status status_;
  std::map<TimePoint, std::shared_ptr<const LandscapeFrame>> frames_;
  std::optional<CorePipeline> pre_last_;
  std::shared_ptr<const Frame> last_core_;
};

}  // namespace fx
```

`src/server/frame_store.cpp`:
```cpp
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
```
Note: `start()` calls `stop_worker()` before taking the lock. `stop_worker` must never be called while holding `m_`, because the worker takes `m_` and joining it under the lock would deadlock.

- [ ] **Step 4: Run the tests to verify they pass** (full suite, warning-free). Run them twice to check for flakiness.

- [ ] **Step 5: Commit**
```bash
git add src/server/frame_store.* tests/test_frame_store.cpp
git commit -m "feat: background frame store with landscape cache and fast last-bar shocks

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: HTTP server and `--serve`

**Files:**
- Create `src/server/http_server.hpp`, `src/server/http_server.cpp`.
- Modify `src/cli/args.hpp`, `src/cli/args.cpp` and `src/main.cpp`.
- Test `tests/test_server.cpp` and `tests/test_cli_args.cpp`.

**Interfaces:**
- Consumes: `FrameStore` (Task 5), `PortfolioSpec`, `describe(CoreParams)`, `parse_*` helpers.
- Produces:
  - `struct fx::ServerOptions { std::string host = "127.0.0.1"; int port = 8080; std::filesystem::path web_root = "web"; }`
  - `class fx::FluxServer { FluxServer(FrameStore&, std::optional<PortfolioSpec>, std::string label); int bind(const ServerOptions&); void listen(); void stop(); }`. `bind` returns the bound port, and port 0 means an ephemeral port.
  - CliArgs gains `bool serve = false; int port = 8080; std::string host = "127.0.0.1"; std::filesystem::path web = "web";` and the flags `--serve`, `--port N`, `--host H` and `--web DIR`. With `--serve` and without `--legacy`, the base params are `CoreParams::money_flow()`; later flags still override them.
- Routes:
  - `GET /api/status` returns `{computed, total, running, ready, error, generation, params: describe(core), height, label, nodes: N}`.
  - `GET /api/times` returns `[t, ...]` (unix seconds).
  - `GET /api/frame[?t=]` returns:
    - `t`, `time` (RFC 3339), `lattice: {cols, rows}`, `raster: {w, h, zmin, zmax}`;
    - `nodes: [[i, ticker, sector, cell, fx, fy, h, hdisp, pi, score], ...]`, where non-finite numbers become null;
    - `arcs: [[a, b, w], ...]`;
    - `portfolio: [{ticker, weight, i|null}]`;
    - `params`.
    It returns 404 `{error}` if t is absent, and 503 if nothing is computed yet.
  - `GET /api/frame/grid[?t=]` returns `application/octet-stream`: the raster as little-endian float32 (w·h·4 bytes). 404 or 503 as for `/api/frame`.
  - `POST /api/params` takes a JSON body:
    - `preset` (`money-flow` | `defaults` | `legacy`);
    - optional `h_ref`, `pressure`, `lift`, `k_out`, `k_in`, `retention`, `lambda`, `vol_scale`;
    - landscape options `height`, `idw_power`, `idw_radius`, `subdivision`.
    It applies them, calls `set_params` and returns 202 `{generation}`. Invalid values return 400 `{error}`.
  - `POST /api/shock` takes `{shocks: [{ticker, size}]}`. It maps tickers to node indices (unknown → 400) and calls `FrameStore::shock`. Inactive or invalid → 400, and not ready → 503. It returns:
    - `{t, l1_dpi, raster: {w, h, zmin, zmax}}`;
    - `shocked: [{ticker, dh, dpi}]`;
    - `receivers: [{ticker, sector, dh, dpi, dscore}]`, the top 15 by dh;
    - `losers`, the bottom 15.
    It stores the shock raster for `GET /api/shock/grid` (float32), which returns 404 if there was no shock yet.
  - `GET /api/events` is an SSE stream: `event: status\ndata: <status json>\n\n` on every change, with a keepalive `: ping\n\n` after 15 s of quiet. Sends are throttled to at most one every 250 ms.
  - Static files: `set_mount_point("/", web_root)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_server.cpp`:
```cpp
#include <doctest/doctest.h>
#include <httplib.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <thread>

#include "market/panel.hpp"
#include "market/synthetic_market.hpp"
#include "server/http_server.hpp"
#include "test_util.hpp"

using namespace fx;
using namespace std::chrono_literals;
using nlohmann::json;

namespace {
struct Fixture {
  std::filesystem::path web = test::temp_dir("web");
  std::unique_ptr<FrameStore> store;
  std::unique_ptr<FluxServer> server;
  std::thread th;
  int port = 0;
  Fixture() {
    SyntheticConfig cfg;
    cfg.bars = 80;
    cfg.rotation_start = 40;
    BarStore bars(test::temp_dir("server_bars"));
    auto secs = generate_synthetic(cfg, bars);
    std::vector<std::string> tickers;
    for (auto& s : secs) tickers.push_back(s.ticker);
    store = std::make_unique<FrameStore>(build_panel(bars, tickers, cfg.tf), secs, CoreParams::money_flow(),
                                         LandscapeParams{}, 10);
    store->start();
    for (int k = 0; k < 600 && !store->status().ready; ++k) std::this_thread::sleep_for(50ms);
    test::write_file(web / "index.html", "hello");
    server = std::make_unique<FluxServer>(*store, std::nullopt, "synthetic 1d");
    ServerOptions o;
    o.port = 0;
    o.web_root = web;
    port = server->bind(o);
    th = std::thread([this] { server->listen(); });
  }
  ~Fixture() {
    server->stop();
    th.join();
  }
};
}  // namespace

TEST_CASE("server: status, times, frame and grid") {
  Fixture fx_;
  httplib::Client cli("127.0.0.1", fx_.port);
  auto st = cli.Get("/api/status");
  REQUIRE(st);
  CHECK(st->status == 200);
  CHECK(json::parse(st->body)["ready"] == true);
  auto times = json::parse(cli.Get("/api/times")->body);
  REQUIRE(times.size() == 10);
  auto fr = cli.Get("/api/frame");
  REQUIRE(fr->status == 200);
  auto f = json::parse(fr->body);
  CHECK(f["nodes"].size() > 0);
  CHECK(f["lattice"]["cols"].get<int>() > 0);
  const auto w = f["raster"]["w"].get<std::size_t>(), h = f["raster"]["h"].get<std::size_t>();
  auto grid = cli.Get("/api/frame/grid");
  REQUIRE(grid->status == 200);
  CHECK(grid->body.size() == w * h * 4);
  CHECK(cli.Get("/api/frame?t=1")->status == 404);
  CHECK(cli.Get("/api/frame?t=" + std::to_string(times[0].get<long long>()))->status == 200);
  CHECK(cli.Get("/")->body == "hello");
}

TEST_CASE("server: shock and params") {
  Fixture fx_;
  httplib::Client cli("127.0.0.1", fx_.port);
  auto f = json::parse(cli.Get("/api/frame")->body);
  const std::string ticker = f["nodes"][0][1];
  auto r = cli.Post("/api/shock", json{{"shocks", {{{"ticker", ticker}, {"size", -10}}}}}.dump(), "application/json");
  REQUIRE(r);
  CHECK(r->status == 200);
  auto body = json::parse(r->body);
  CHECK(body["receivers"].size() > 0);
  CHECK(body["shocked"][0]["dh"].get<double>() < 0);
  CHECK(cli.Get("/api/shock/grid")->status == 200);
  CHECK(cli.Post("/api/shock", R"({"shocks":[{"ticker":"NOPE","size":1}]})", "application/json")->status == 400);
  CHECK(cli.Post("/api/params", R"({"preset":"bogus"})", "application/json")->status == 400);
  const auto g = json::parse(cli.Get("/api/status")->body)["generation"].get<std::uint64_t>();
  auto p = cli.Post("/api/params", R"({"preset":"legacy","height":"linear"})", "application/json");
  CHECK(p->status == 202);
  CHECK(json::parse(cli.Get("/api/status")->body)["generation"].get<std::uint64_t>() > g);
}
```
Append to `tests/test_cli_args.cpp`:
```cpp
TEST_CASE("serve flags default to the money-flow preset") {
  CliArgs a = parse_cli({"--serve", "--port", "9000", "--host", "0.0.0.0", "--web", "/tmp/w"});
  CHECK(a.serve);
  CHECK(a.port == 9000);
  CHECK(a.host == "0.0.0.0");
  CHECK(a.web == "/tmp/w");
  CHECK(a.params.pressure == PressureMode::Dollar);  // money_flow()
  CHECK(a.params.h_ref == HotRef::Size);
  CliArgs b = parse_cli({"--serve", "--h-ref", "netflow"});
  CHECK(b.params.h_ref == HotRef::NetFlow);
  CliArgs c = parse_cli({"--serve", "--legacy"});
  CHECK(c.params.transition.lift == LiftMode::Off);
  CHECK(c.params.h_ref == HotRef::Uniform);
}
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`src/server/http_server.hpp`:
```cpp
#pragma once
#include <httplib.h>

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
  std::mutex shock_m_;
  std::optional<Raster> last_shock_;
};

}  // namespace fx
```

`src/server/http_server.cpp`:
```cpp
#include "server/http_server.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
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
void FluxServer::stop() { svr_.stop(); }

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
    auto seen = std::make_shared<std::uint64_t>(0);
    res.set_chunked_content_provider("text/event-stream", [this, seen](size_t, httplib::DataSink& sink) {
      const auto v = store_.wait_for_change(*seen, std::chrono::seconds(15));
      std::string msg;
      if (v == *seen) {
        msg = ": ping\n\n";
      } else {
        *seen = v;
        msg = "event: status\ndata: " + status_json(store_, label_).dump() + "\n\n";
      }
      if (!sink.write(msg.data(), msg.size())) return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(250));  // throttle
      return true;
    });
  });
}

}  // namespace fx
```

**CLI.**
- In `src/cli/args.hpp`, add the `CliArgs` fields from the interface list above.
- In `src/cli/args.cpp`:
  - Parse `--serve`, `--port` (`to_size`, then int; reject 0 or more than 65535 except for tests, which call `bind` directly), `--host` and `--web`.
  - Base preset: if `--serve` is present and `--legacy` is not, start from `CoreParams::money_flow()`, the same way `--money-flow` works. `--money-flow` alone keeps its current behaviour.
  - Add the flags to `cli_usage()`.

**main.** In `src/main.cpp`, after the universe and panel are built as usual:
- If `args.serve`:
  - create `FrameStore store(panel, universe.nodes(), args.params, LandscapeParams{});`, call `store.start()`, then create `FluxServer server(store, portfolio, label)`, where label is `mode + " " + to_string(tf)`;
  - call `bind({args.host, args.port, args.web})` and print `serving http://<host>:<port>  (Ctrl-C to stop)` to stderr;
  - call `server.listen()`, and return 0.
- Serve mode skips the rank and eval output. Include `server/http_server.hpp`.

- [ ] **Step 4: Run the tests to verify they pass** (full suite, warning-free). Then:
```bash
./build/fluxscape --serve --mode synthetic --port 18765 &
sleep 5; curl -s http://127.0.0.1:18765/api/status; echo; curl -s -o /dev/null -w "%{http_code} %{size_download}\n" http://127.0.0.1:18765/api/frame/grid
kill %1
```
Expected: the status JSON has `"ready":true` and the grid request returns 200. Paste the output into the report.

- [ ] **Step 5: Commit**
```bash
git add src/server/http_server.* src/cli/args.* src/main.cpp tests/test_server.cpp tests/test_cli_args.cpp
git commit -m "feat: --serve: REST + SSE API over the frame store, static web root

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: deck.gl front end and headless smoke test

**Files:** Create `web/index.html`, `web/app.js`, `web/style.css` and `scripts/ui_smoke.sh`.

**Interfaces:**
- Consumes the Task 6 API.
- Produces a page that sets `document.title` to `fluxscape-ok:<nodes>` after its first successful render when loaded with `?selftest=1`, or to `fluxscape-error:<message>` on failure.

- [ ] **Step 1: Pin deck.gl.** Run `curl -sI https://unpkg.com/deck.gl@9/dist.min.js | grep -i '^location'` to get the exact 9.x version unpkg resolves to, for example `/deck.gl@9.1.14/dist.min.js`. Use that exact version in `index.html` and note it in the report.

- [ ] **Step 2: Write `web/style.css`:**
```css
:root { --panel: #f6f6f4; --ink: #1d1d1b; --muted: #6b6b66; --line: #ddddd8; --accent: #b2182b; }
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; font: 13px/1.4 system-ui, -apple-system, Segoe UI, sans-serif; color: var(--ink); }
#app { display: grid; grid-template-columns: 320px 1fr; height: 100vh; }
#panel { background: var(--panel); border-right: 1px solid var(--line); overflow-y: auto; padding: 12px 14px; }
#panel h1 { font-size: 16px; margin: 0 0 4px; }
#panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 16px 0 6px; }
#panel label { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin: 4px 0; }
#panel input[type=range] { width: 150px; }
#panel select, #panel input[type=text], #panel input[type=number] { width: 150px; }
#panel button { margin: 6px 6px 0 0; padding: 4px 10px; }
#status { color: var(--muted); font-size: 12px; min-height: 2.8em; }
#canvas { position: relative; }
.list { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre; }
.pos { color: #b2182b; } .neg { color: #2166ac; }
@media (max-width: 700px) { #app { grid-template-columns: 1fr; grid-template-rows: auto 60vh; } #panel { border-right: 0; border-bottom: 1px solid var(--line); } }
```

- [ ] **Step 3: Write `web/index.html`** (replace `DECK_URL` with the pinned URL from Step 1):
```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fluxscape</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<div id="app">
  <aside id="panel">
    <h1>Fluxscape</h1>
    <div id="status">loading…</div>
    <h2>Data</h2>
    <label>Bar <input id="scrub" type="range" min="0" max="0" value="0"></label>
    <div id="tlabel"></div>
    <label>Follow latest <input id="follow" type="checkbox" checked></label>
    <button id="play">Play</button>
    <h2>Graph</h2>
    <label>Preset <select id="preset"><option>money-flow</option><option>defaults</option><option>legacy</option></select></label>
    <label>Hotness <select id="href"><option value="">preset</option><option>size</option><option>netflow</option><option>uniform</option><option>longrun</option></select></label>
    <h2>Landscape</h2>
    <label>Height <select id="height"><option>signed-log</option><option>linear</option></select></label>
    <label>IDW power <input id="idwPower" type="number" min="0.5" max="6" step="0.5" value="2"></label>
    <label>IDW radius <input id="idwRadius" type="number" min="1" max="10" step="1" value="3"></label>
    <label>Subdivision <input id="subdiv" type="number" min="1" max="8" step="1" value="4"></label>
    <button id="apply">Apply</button>
    <label>Height scale <input id="hscale" type="range" min="0.1" max="3" step="0.1" value="1"></label>
    <label>Flux arcs <input id="arcs" type="checkbox" checked></label>
    <label>Labels <input id="labels" type="checkbox" checked></label>
    <h2>Portfolio</h2>
    <div id="holdings" class="list"></div>
    <h2>Shock</h2>
    <label>Ticker <input id="shockTicker" type="text" list="tickers" placeholder="NVDA"></label>
    <datalist id="tickers"></datalist>
    <label>Size (%) <input id="shockSize" type="range" min="-20" max="20" step="1" value="-10"></label>
    <div id="shockSizeLabel">-10%</div>
    <button id="shockApply">Apply shock</button><button id="shockReset">Reset</button>
    <div id="shockOut" class="list"></div>
  </aside>
  <main id="canvas"></main>
</div>
<script src="DECK_URL"></script>
<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Write `web/app.js`:**
```js
/* global deck */
'use strict';
const S = {
  status: null, times: [], frame: null, raster: null, shock: null, shockRaster: null,
  heightBase: 1, deck: null, playing: null, loadedKey: '', selftest: new URLSearchParams(location.search).has('selftest'),
};
const $ = (id) => document.getElementById(id);

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.text();
  if (!r.ok) throw new Error(`${url}: ${r.status} ${body}`);
  return JSON.parse(body);
}
async function getFloat32(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return new Float32Array(await r.arrayBuffer());
}

function colormap(v, vmax) {
  const x = Math.max(-1, Math.min(1, vmax > 0 ? v / vmax : 0));
  const a = Math.abs(x), mid = [247, 247, 247], end = x < 0 ? [33, 102, 172] : [178, 24, 43];
  return [0, 1, 2].map((k) => Math.round(mid[k] + (end[k] - mid[k]) * a));
}

function buildTerrain(z, meta, lattice, scale) {
  const { w, h } = meta, sx = lattice.cols / w, sy = lattice.rows / h;
  const pos = new Float32Array(w * h * 3), nrm = new Float32Array(w * h * 3), uv = new Float32Array(w * h * 2);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const k = y * w + x;
    pos[3 * k] = (x + 0.5) * sx; pos[3 * k + 1] = (y + 0.5) * sy; pos[3 * k + 2] = z[k] * scale;
    uv[2 * k] = (x + 0.5) / w; uv[2 * k + 1] = (y + 0.5) / h;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const k = y * w + x;
    const dx = (z[y * w + Math.min(w - 1, x + 1)] - z[y * w + Math.max(0, x - 1)]) * scale / (2 * sx);
    const dy = (z[Math.min(h - 1, y + 1) * w + x] - z[Math.max(0, y - 1) * w + x]) * scale / (2 * sy);
    const len = Math.hypot(dx, dy, 1);
    nrm[3 * k] = -dx / len; nrm[3 * k + 1] = -dy / len; nrm[3 * k + 2] = 1 / len;
  }
  const idx = new Uint32Array((w - 1) * (h - 1) * 6);
  let p = 0;
  for (let y = 0; y < h - 1; y++) for (let x = 0; x < w - 1; x++) {
    const a = y * w + x, b = a + 1, c = a + w, d = c + 1;
    idx[p++] = a; idx[p++] = b; idx[p++] = d; idx[p++] = a; idx[p++] = d; idx[p++] = c;
  }
  return {
    topology: 'triangle-list', mode: 4,
    attributes: { POSITION: { value: pos, size: 3 }, NORMAL: { value: nrm, size: 3 }, TEXCOORD_0: { value: uv, size: 2 } },
    indices: { value: idx, size: 1 },
  };
}

function buildTexture(z, meta) {
  const c = document.createElement('canvas');
  c.width = meta.w; c.height = meta.h;
  const ctx = c.getContext('2d'), img = ctx.createImageData(meta.w, meta.h);
  const vmax = Math.max(Math.abs(meta.zmin), Math.abs(meta.zmax), 1e-9);
  for (let k = 0; k < meta.w * meta.h; k++) {
    const [r, g, b] = colormap(z[k], vmax);
    img.data[4 * k] = r; img.data[4 * k + 1] = g; img.data[4 * k + 2] = b; img.data[4 * k + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function sample(z, meta, lattice, col, row) {
  const px = Math.min(meta.w - 1, Math.floor((col + 0.5) * meta.w / lattice.cols));
  const py = Math.min(meta.h - 1, Math.floor((row + 0.5) * meta.h / lattice.rows));
  return z[py * meta.w + px];
}

function render() {
  const f = S.frame;
  if (!f || !S.deck) return;
  const L = f.lattice;
  const meta = S.shock ? S.shock.raster : f.raster, z = S.shock ? S.shockRaster : S.raster;
  const scale = S.heightBase * parseFloat($('hscale').value);
  const nodes = f.nodes.map((n) => ({ i: n[0], ticker: n[1], sector: n[2], col: n[3] % L.cols, row: Math.floor(n[3] / L.cols), h: n[6], pi: n[8], score: n[9] }));
  const byI = new Map(nodes.map((n) => [n.i, n]));
  const pos = (n, lift = 0.3) => [n.col + 0.5, n.row + 0.5, sample(z, meta, L, n.col, n.row) * scale + lift];
  const ranked = nodes.filter((n) => n.h !== null).sort((a, b) => b.h - a.h);
  const labeled = ranked.slice(0, 30).concat(ranked.slice(-30));
  const arcs = $('arcs').checked ? f.arcs.slice(0, 400).map((a) => ({ s: byI.get(a[0]), t: byI.get(a[1]), w: a[2] })).filter((a) => a.s && a.t) : [];
  const wmax = arcs.reduce((m, a) => Math.max(m, a.w), 1e-12);
  const holdings = (f.portfolio || []).filter((p) => p.i !== null && byI.has(p.i)).map((p) => ({ ...byI.get(p.i), weight: p.weight }));
  const layers = [
    new deck.SimpleMeshLayer({
      id: 'terrain', data: [{}], mesh: buildTerrain(z, meta, L, scale), texture: buildTexture(z, meta),
      getPosition: [0, 0, 0], getColor: [255, 255, 255], material: { ambient: 0.55, diffuse: 0.55, shininess: 12, specularColor: [30, 30, 30] },
    }),
    new deck.ScatterplotLayer({
      id: 'nodes', data: nodes, getPosition: (n) => pos(n), getRadius: 0.16, radiusUnits: 'common',
      getFillColor: (n) => (n.h >= 0 ? [178, 24, 43, 220] : [33, 102, 172, 220]), pickable: true,
    }),
    new deck.ArcLayer({
      id: 'arcs', data: arcs, getSourcePosition: (a) => pos(a.s), getTargetPosition: (a) => pos(a.t),
      getWidth: (a) => 0.5 + 3 * a.w / wmax, widthUnits: 'pixels', getSourceColor: [255, 140, 0, 150], getTargetColor: [255, 215, 0, 150],
    }),
    new deck.ScatterplotLayer({
      id: 'portfolio', data: holdings, getPosition: (n) => pos(n, 0.4), getRadius: (n) => 0.35 + 1.5 * n.weight, radiusUnits: 'common',
      stroked: true, filled: false, getLineColor: [0, 150, 80], getLineWidth: 3, lineWidthUnits: 'pixels',
    }),
  ];
  if ($('labels').checked) {
    layers.push(new deck.TextLayer({
      id: 'labels', data: labeled, getPosition: (n) => pos(n, 0.9), getText: (n) => n.ticker, getSize: 12,
      getColor: [25, 25, 25], getTextAnchor: 'middle', getAlignmentBaseline: 'bottom', billboard: true,
    }));
  }
  S.deck.setProps({ layers });
}

function initDeck(lattice) {
  const el = $('canvas');
  const span = Math.max(lattice.cols, lattice.rows);
  S.deck = new deck.Deck({
    parent: el,
    views: new deck.OrbitView({ orbitAxis: 'Z', fovy: 40 }),
    initialViewState: { target: [lattice.cols / 2, lattice.rows / 2, 0], rotationX: 45, rotationOrbit: -25, zoom: Math.log2(Math.min(el.clientWidth, el.clientHeight) / (1.6 * span)), minZoom: -6, maxZoom: 12 },
    controller: true,
    getTooltip: ({ object }) => (object && object.ticker
      ? `${object.ticker} · ${object.sector}\nh ${object.h === null ? 'n/a' : object.h.toFixed(3)}  π ${object.pi === null ? 'n/a' : object.pi.toExponential(2)}\nscore+1 ${object.score === null ? 'n/a' : object.score.toFixed(3)}`
      : null),
  });
}

async function loadFrame(t) {
  const q = t === undefined || t === null ? '' : `?t=${t}`;
  const [f, z] = await Promise.all([getJSON(`/api/frame${q}`), getFloat32(`/api/frame/grid${q}`)]);
  S.frame = f; S.raster = z;
  if (!S.deck) {
    const zmax = Math.max(Math.abs(f.raster.zmin), Math.abs(f.raster.zmax), 1e-9);
    S.heightBase = 0.2 * Math.max(f.lattice.cols, f.lattice.rows) / zmax;
    initDeck(f.lattice);
    $('tickers').innerHTML = f.nodes.map((n) => `<option value="${n[1]}">`).join('');
  }
  $('tlabel').textContent = `${f.time}  ·  ${f.nodes.length} active stocks  ·  ${f.params}`;
  $('holdings').innerHTML = (f.portfolio || []).map((p) => {
    const n = p.i === null ? null : f.nodes.find((x) => x[0] === p.i);
    const h = n && n[6] !== null ? n[6] : null;
    const cls = h === null ? '' : (h >= 0 ? 'pos' : 'neg');
    return `${p.ticker.padEnd(6)} ${(100 * p.weight).toFixed(1).padStart(5)}%  <span class="${cls}">${h === null ? 'n/a' : h.toFixed(3)}</span>`;
  }).join('\n');
  render();
}

async function refreshTimes() {
  S.times = await getJSON('/api/times');
  const s = $('scrub');
  s.max = Math.max(0, S.times.length - 1);
  if ($('follow').checked) s.value = s.max;
}

function showStatus() {
  const st = S.status;
  if (!st) return;
  $('status').textContent = st.error ? `error: ${st.error}` : `${st.label} · ${st.ready ? 'ready' : `computing ${st.computed}/${st.total}`} · ${st.nodes} stocks`;
}

async function onStatus(st) {
  S.status = st;
  showStatus();
  const key = `${st.generation}:${st.ready}`;
  if (st.ready && key !== S.loadedKey) {
    S.loadedKey = key;
    S.shock = null;
    await refreshTimes();
    await loadFrame(S.times[Number($('scrub').value)]);
    if (S.selftest) setTimeout(() => { document.title = `fluxscape-ok:${S.frame.nodes.length}`; }, 1500);
  }
}

function wire() {
  $('scrub').addEventListener('input', () => { $('follow').checked = false; S.shock = null; loadFrame(S.times[Number($('scrub').value)]).catch(fail); });
  $('play').addEventListener('click', () => {
    if (S.playing) { clearInterval(S.playing); S.playing = null; $('play').textContent = 'Play'; return; }
    $('play').textContent = 'Pause';
    S.playing = setInterval(() => {
      const s = $('scrub'); s.value = (Number(s.value) + 1) % (Number(s.max) + 1);
      loadFrame(S.times[Number(s.value)]).catch(fail);
    }, 700);
  });
  ['hscale', 'arcs', 'labels'].forEach((id) => $(id).addEventListener('input', render));
  $('apply').addEventListener('click', async () => {
    const body = { preset: $('preset').value, height: $('height').value, idw_power: Number($('idwPower').value), idw_radius: Number($('idwRadius').value), subdivision: Number($('subdiv').value) };
    if ($('href').value) body.h_ref = $('href').value;
    try { await getJSON('/api/params', { method: 'POST', body: JSON.stringify(body) }); S.deck && S.deck.finalize(); S.deck = null; } catch (e) { fail(e); }
  });
  $('shockSize').addEventListener('input', () => { $('shockSizeLabel').textContent = `${$('shockSize').value}%`; });
  $('shockApply').addEventListener('click', async () => {
    try {
      const body = { shocks: [{ ticker: $('shockTicker').value.trim().toUpperCase(), size: Number($('shockSize').value) }] };
      const r = await getJSON('/api/shock', { method: 'POST', body: JSON.stringify(body) });
      S.shockRaster = await getFloat32('/api/shock/grid'); S.shock = r;
      const fmt = (x) => `${x.ticker.padEnd(6)} ${x.dh >= 0 ? '+' : ''}${x.dh.toFixed(4)}`;
      $('shockOut').innerHTML = `Δh landscape · L1 Δπ ${r.l1_dpi.toExponential(2)}\n` +
        `shocked: ${r.shocked.map(fmt).join(', ')}\n\nreceivers\n${r.receivers.slice(0, 10).map(fmt).join('\n')}\n\nlosers\n${r.losers.slice(0, 10).map(fmt).join('\n')}`;
      render();
    } catch (e) { $('shockOut').textContent = String(e.message || e); }
  });
  $('shockReset').addEventListener('click', () => { S.shock = null; $('shockOut').textContent = ''; render(); });
}

function fail(e) {
  console.error(e);
  $('status').textContent = `error: ${e.message || e}`;
  if (S.selftest) document.title = `fluxscape-error:${e.message || e}`;
}

(async function main() {
  try {
    if (typeof deck === 'undefined') throw new Error('deck.gl failed to load');
    wire();
    await onStatus(await getJSON('/api/status'));
    const es = new EventSource('/api/events');
    es.addEventListener('status', (ev) => onStatus(JSON.parse(ev.data)).catch(fail));
  } catch (e) { fail(e); }
}());
```

- [ ] **Step 5: Write `scripts/ui_smoke.sh`** (`chmod +x`):
```bash
#!/usr/bin/env bash
# Headless-Chrome smoke test of the landscape UI. Usage: scripts/ui_smoke.sh [extra fluxscape args...]
set -euo pipefail
PORT=${PORT:-18765}
OUT=${OUT:-/tmp/fluxscape_ui.png}
CHROME=${CHROME:-google-chrome}
LOG=${LOG:-/tmp/fluxscape_serve.log}
./build/fluxscape --serve --port "$PORT" --mode "${MODE:-synthetic}" "$@" >"$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for _ in $(seq 1 ${WAIT_TICKS:-240}); do
  if curl -s "http://127.0.0.1:$PORT/api/status" | grep -q '"ready":true'; then break; fi
  sleep 0.5
done
FLAGS=(--headless=new --no-sandbox --enable-unsafe-swiftshader --use-angle=swiftshader --window-size=1400,900 --virtual-time-budget=25000)
"$CHROME" "${FLAGS[@]}" --screenshot="$OUT" "http://127.0.0.1:$PORT/?selftest=1" >/dev/null 2>&1 || true
TITLE=$("$CHROME" "${FLAGS[@]}" --dump-dom "http://127.0.0.1:$PORT/?selftest=1" 2>/dev/null | grep -o '<title>[^<]*</title>' || true)
echo "title: $TITLE"
echo "screenshot: $OUT"
echo "$TITLE" | grep -q 'fluxscape-ok'
```

- [ ] **Step 6: Verify the page with the smoke test.** Run `scripts/ui_smoke.sh`.
  - Expected: `title: <title>fluxscape-ok:<N></title>` and exit 0.
  - Open the screenshot (`/tmp/fluxscape_ui.png`) with your image-reading tool and check it visually:
    - the left panel is visible;
    - there is a coloured 3D terrain with red hills and blue valleys;
    - stock dots and labels are drawn on the surface;
    - the arcs are visible.
  - **If the terrain is missing or black**, the mesh format didn't render. Try these in order:
    - (1) drop `topology` and `mode` from the mesh object;
    - (2) pass `mesh` with lower-case attribute names (`positions`, `normals`, `texCoords`);
    - (3) fall back to a `deck.ColumnLayer` of raster cells: square disks, `diskResolution: 4`, `extruded: true`, `getElevation` from z, `getFillColor` from colormap, at most one column per pixel of a 2×-downsampled raster.
    Report which approach rendered.
  - **If the colours are vertically flipped relative to the heights** (red texture on valleys), flip the texture rows (`uv[2*k+1] = 1 - (y+0.5)/h`).
  - Iterate until the screenshot shows a correct landscape. Include the final screenshot path and a short description in the report.

- [ ] **Step 7: Commit**
```bash
git add web scripts/ui_smoke.sh
git commit -m "feat: deck.gl landscape UI with control panel, scrubber and shock mode; headless smoke test

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Real-data landscape run and README

**Files:** Modify `README.md`.

- [ ] **Step 1: Serve real data.** Run `MODE=replay WAIT_TICKS=2400 OUT=/tmp/fluxscape_ui_real.png scripts/ui_smoke.sh --timeframe 1d` with a Bash timeout of 20 minutes, in the background if needed. This uses the 10K lake and the latest snapshot.
  - Record the time to ready from the serve log (`/tmp/fluxscape_serve.log`), or time the wait.
  - Record the `/api/frame` JSON size and the grid size: start the server again, then `curl -s -o /dev/null -w '%{size_download}'`.
  - Record the mean landscape `compute_ms` from `/api/frame`.
  - Look at the screenshot and describe it. Check that labels such as AAPL and NVDA appear if they are in the top or bottom 30, and that the portfolio rings are visible.
- [ ] **Step 2: Shock on real data:**
```bash
./build/fluxscape --serve --mode replay --port 18766 >/tmp/s.log 2>&1 &
# wait for ready, then:
curl -s -X POST http://127.0.0.1:18766/api/shock -d '{"shocks":[{"ticker":"NVDA","size":-10}]}' | head -c 1500
```
  Record the response time and the top receivers and losers.
- [ ] **Step 3: README.** Add a "Landscape UI" section after `## Run`:
````markdown
## Landscape UI

```bash
./build/fluxscape --serve --mode replay            # http://127.0.0.1:8080 (money-flow preset)
./build/fluxscape --serve --mode synthetic --port 9000
scripts/ui_smoke.sh                                # headless-Chrome smoke test + screenshot
```

The page shows the IDW terrain of signed-log hotness, with hills where money settles relative to size and valleys where it drains. It draws the stocks on their lattice cells, the strongest flux arcs, and your portfolio as green rings. The left panel switches preset and hotness reference, sets the IDW and height parameters, scrubs or plays through the last 300 bars, and applies a shock (`TICKER`, ±%) to show the Δh landscape of who absorbs and who loses.
````
- [ ] **Step 4: Commit**
```bash
git add README.md
git commit -m "docs: landscape UI usage

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
Report all measurements and both screenshot descriptions.
