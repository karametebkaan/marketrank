#include "geom/embedding.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "geom/landscape.hpp"

namespace fx {

namespace {

// Power iteration on the symmetric dims x dims matrix C. `against` (optional) is orthogonalized out of the start.
std::vector<double> top_eigenvector(const std::vector<double>& C, std::size_t dims, const std::vector<double>* against,
                                    double& lambda) {
  std::vector<double> v(dims, 1.0 / std::sqrt(static_cast<double>(dims))), w(dims);
  auto normalize = [&](std::vector<double>& x) {
    double nn = 0;
    for (double e : x) nn += e * e;
    nn = std::sqrt(nn);
    if (nn > 0)
      for (double& e : x) e /= nn;
    return nn;
  };
  if (against) {
    double dot = 0;
    for (std::size_t k = 0; k < dims; ++k) dot += v[k] * (*against)[k];
    for (std::size_t k = 0; k < dims; ++k) v[k] -= dot * (*against)[k];
    normalize(v);
  }
  lambda = 0;
  for (int it = 0; it < 300; ++it) {
    for (std::size_t r = 0; r < dims; ++r) {
      double s = 0;
      for (std::size_t c = 0; c < dims; ++c) s += C[r * dims + c] * v[c];
      w[r] = s;
    }
    if (against) {  // keep v2 orthogonal to v1 against round-off drift
      double dot = 0;
      for (std::size_t k = 0; k < dims; ++k) dot += w[k] * (*against)[k];
      for (std::size_t k = 0; k < dims; ++k) w[k] -= dot * (*against)[k];
    }
    lambda = normalize(w);
    if (lambda == 0) break;
    v = w;
  }
  // sign rule: largest-magnitude entry positive (lowest index on ties)
  std::size_t best = 0;
  for (std::size_t k = 1; k < dims; ++k)
    if (std::fabs(v[k]) > std::fabs(v[best])) best = k;
  if (v[best] < 0)
    for (double& e : v) e = -e;
  return v;
}

struct Fit {
  double score;
  double c, s;  // cos, sin of the rotation
};

// Best rotation for M = sum x prev^T; `flip` negates the second row of M (reflection about the x axis first).
Fit best_rotation(const double M[2][2], bool flip) {
  double m00 = M[0][0], m01 = M[0][1], m10 = flip ? -M[1][0] : M[1][0], m11 = flip ? -M[1][1] : M[1][1];
  double th = std::atan2(m01 - m10, m00 + m11);
  double c = std::cos(th), s = std::sin(th);
  return {m00 * c + m11 * c + (m01 - m10) * s, c, s};
}

}  // namespace

std::vector<double> solve_features(const Frame& f, const EmbeddingParams& p, std::size_t& dims_out) {
  const std::size_t n = f.active.size();
  if (f.h.size() != n || f.pi.size() != n) throw std::invalid_argument("solve_features: h/pi size mismatch");
  for (const auto& fc : f.forecasts)
    if (fc.score.size() != n) throw std::invalid_argument("solve_features: forecast score size mismatch");
  const std::size_t F = 2 + f.forecasts.size();
  dims_out = F;
  std::vector<double> Y(n * F, 0.0), col(n), tmp;
  for (std::size_t d = 0; d < F; ++d) {
    for (std::size_t i = 0; i < n; ++i) {
      if (d == 0)
        col[i] = display_height(f.h[i], HeightMode::SignedLog);
      else if (d == 1)
        col[i] = std::log(f.pi[i]);
      else
        col[i] = display_height(f.forecasts[d - 2].score[i], HeightMode::SignedLog);
    }
    tmp.clear();
    for (std::size_t i = 0; i < n; ++i)
      if (f.active[i] && std::isfinite(col[i])) tmp.push_back(col[i]);
    if (tmp.empty()) continue;
    auto median = [](std::vector<double>& v) {
      const std::size_t m = v.size() / 2;
      std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m), v.end());
      double hi = v[m];
      if (v.size() % 2) return hi;
      double lo = *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m));
      return 0.5 * (lo + hi);
    };
    const double med = median(tmp);
    for (double& v : tmp) v = std::fabs(v - med);
    const double mad = 1.4826 * median(tmp);
    if (!(mad > 0) || !std::isfinite(mad)) continue;  // degenerate column stays 0
    for (std::size_t i = 0; i < n; ++i) {
      if (!f.active[i] || !std::isfinite(col[i])) continue;
      Y[i * F + d] = std::clamp((col[i] - med) / mad, -p.clip, p.clip);
    }
  }
  return Y;
}

std::vector<double> pca2(const std::vector<double>& Y, std::size_t dims, const std::vector<bool>& active) {
  const std::size_t n = active.size();
  std::vector<double> xy(2 * n, 0.0);
  std::size_t cnt = 0;
  for (std::size_t i = 0; i < n; ++i) cnt += active[i] ? 1 : 0;
  if (cnt < 2 || dims == 0) return xy;

  std::vector<double> mean(dims, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    if (active[i])
      for (std::size_t d = 0; d < dims; ++d) mean[d] += Y[i * dims + d];
  for (double& m : mean) m /= static_cast<double>(cnt);

  std::vector<double> C(dims * dims, 0.0), yc(dims);
  for (std::size_t i = 0; i < n; ++i) {
    if (!active[i]) continue;
    for (std::size_t d = 0; d < dims; ++d) yc[d] = Y[i * dims + d] - mean[d];
    for (std::size_t r = 0; r < dims; ++r)
      for (std::size_t c = 0; c < dims; ++c) C[r * dims + c] += yc[r] * yc[c];
  }
  for (double& c : C) c /= static_cast<double>(cnt);

  double l1 = 0, l2 = 0;
  std::vector<double> v1 = top_eigenvector(C, dims, nullptr, l1);
  for (std::size_t r = 0; r < dims; ++r)
    for (std::size_t c = 0; c < dims; ++c) C[r * dims + c] -= l1 * v1[r] * v1[c];
  std::vector<double> v2 = top_eigenvector(C, dims, &v1, l2);

  double ss = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!active[i]) continue;
    double a = 0, b = 0;
    for (std::size_t d = 0; d < dims; ++d) {
      double y = Y[i * dims + d] - mean[d];
      a += y * v1[d];
      b += y * v2[d];
    }
    xy[2 * i] = a;
    xy[2 * i + 1] = b;
    ss += a * a + b * b;
  }
  const double rms = std::sqrt(ss / static_cast<double>(cnt));
  if (rms > 0)
    for (std::size_t i = 0; i < n; ++i)
      if (active[i]) {
        xy[2 * i] /= rms;
        xy[2 * i + 1] /= rms;
      }
  return xy;
}

void align_to(std::vector<double>& xy, const std::vector<double>& prev, const std::vector<bool>& mask) {
  const std::size_t n = xy.size() / 2;
  double M[2][2] = {{0, 0}, {0, 0}};
  bool any = false;
  for (std::size_t i = 0; i < n; ++i) {
    if (!mask[i]) continue;
    if (!std::isfinite(xy[2 * i]) || !std::isfinite(xy[2 * i + 1]) || !std::isfinite(prev[2 * i]) ||
        !std::isfinite(prev[2 * i + 1]))
      continue;
    any = true;
    M[0][0] += xy[2 * i] * prev[2 * i];
    M[0][1] += xy[2 * i] * prev[2 * i + 1];
    M[1][0] += xy[2 * i + 1] * prev[2 * i];
    M[1][1] += xy[2 * i + 1] * prev[2 * i + 1];
  }
  if (!any) return;
  Fit rot = best_rotation(M, false), refl = best_rotation(M, true);
  const bool flip = refl.score > rot.score;
  const Fit& f = flip ? refl : rot;
  for (std::size_t i = 0; i < n; ++i) {
    const double x = xy[2 * i], y = flip ? -xy[2 * i + 1] : xy[2 * i + 1];
    xy[2 * i] = f.c * x - f.s * y;
    xy[2 * i + 1] = f.s * x + f.c * y;
  }
}

SolveEmbedding::SolveEmbedding(std::size_t n, EmbeddingParams p)
    : n_(n), p_(p), prev_(2 * n, 0.0), prev_active_(n, false) {}

const std::vector<double>& SolveEmbedding::positions(const Frame& f) {
  if (f.active.size() != n_) throw std::invalid_argument("SolveEmbedding: frame size mismatch");
  const std::vector<bool>& active = f.active;
  std::size_t F = 0;
  std::vector<double> Y = solve_features(f, p_, F);
  std::vector<double> next = pca2(Y, F, active);
  if (have_prev_) {
    std::vector<bool> both(n_, false);
    for (std::size_t i = 0; i < n_; ++i) both[i] = active[i] && prev_active_[i];
    align_to(next, prev_, both);
    for (std::size_t i = 0; i < n_; ++i)  // non-finite entries fall back to the previous position
      if (active[i] && !(std::isfinite(next[2 * i]) && std::isfinite(next[2 * i + 1]))) {
        next[2 * i] = both[i] ? prev_[2 * i] : 0.0;
        next[2 * i + 1] = both[i] ? prev_[2 * i + 1] : 0.0;
      }
    for (std::size_t i = 0; i < n_; ++i) {
      if (!active[i]) continue;  // inactive keep previous position
      if (both[i]) {
        prev_[2 * i] = (1.0 - p_.smoothing) * next[2 * i] + p_.smoothing * prev_[2 * i];
        prev_[2 * i + 1] = (1.0 - p_.smoothing) * next[2 * i + 1] + p_.smoothing * prev_[2 * i + 1];
      } else {
        prev_[2 * i] = next[2 * i];
        prev_[2 * i + 1] = next[2 * i + 1];
      }
    }
  } else {
    for (std::size_t i = 0; i < n_; ++i)
      if (active[i]) {
        if (!(std::isfinite(next[2 * i]) && std::isfinite(next[2 * i + 1]))) next[2 * i] = next[2 * i + 1] = 0.0;
        prev_[2 * i] = next[2 * i];
        prev_[2 * i + 1] = next[2 * i + 1];
      }
  }
  for (std::size_t i = 0; i < n_; ++i) prev_active_[i] = active[i];
  have_prev_ = true;
  return prev_;
}

}  // namespace fx
