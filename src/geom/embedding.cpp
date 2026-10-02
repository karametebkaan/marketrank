#include "geom/embedding.hpp"

#include <algorithm>
#include <cmath>

namespace fx {

namespace {

std::uint64_t splitmix64(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Standard normal that depends only on (seed, i, d).
double gauss(std::uint64_t seed, std::size_t i, std::size_t d) {
  std::uint64_t s = seed ^ (static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ULL) ^
                    (static_cast<std::uint64_t>(d) * 0xBF58476D1CE4E5B9ULL);
  const double inv = 1.0 / 9007199254740992.0;  // 2^-53
  double u1 = static_cast<double>((splitmix64(s) >> 11) + 1) * inv;
  double u2 = static_cast<double>((splitmix64(s) >> 11) + 1) * inv;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

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

std::vector<double> destination_signatures(const Csr& P, const std::vector<bool>& active, const EmbeddingParams& p) {
  const std::size_t n = P.n, D = static_cast<std::size_t>(p.dims);
  std::vector<double> Y(n * D, 0.0), Z(n * D, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (!active[i]) continue;
    for (std::size_t d = 0; d < D; ++d) Y[i * D + d] = gauss(p.seed, i, d);
  }
  // P_off row i: off-diagonal active columns, renormalized; absorbing when nothing remains.
  std::vector<double> inv(n, 0.0);
  std::vector<char> absorbing(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    if (!active[i]) continue;
    double s = 0;
    bool finite = true;
    for (std::size_t e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) {
      if (!std::isfinite(P.val[e])) finite = false;
      std::uint32_t j = P.col[e];
      if (j != i && active[j]) s += P.val[e];
    }
    if (finite && s > 0 && std::isfinite(s))
      inv[i] = 1.0 / s;
    else
      absorbing[i] = 1;
  }
  for (int step = 0; step < p.steps; ++step) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
      double* z = &Z[i * D];
      if (!active[i]) continue;  // stays zero
      if (absorbing[i]) {
        for (std::size_t d = 0; d < D; ++d) z[d] = Y[i * D + d];
        continue;
      }
      for (std::size_t d = 0; d < D; ++d) z[d] = 0.0;
      for (std::size_t e = P.row_ptr[i]; e < P.row_ptr[i + 1]; ++e) {
        std::uint32_t j = P.col[e];
        if (j == i || !active[j]) continue;
        const double w = P.val[e] * inv[i];
        const double* y = &Y[static_cast<std::size_t>(j) * D];
        for (std::size_t d = 0; d < D; ++d) z[d] += w * y[d];
      }
    }
    Y.swap(Z);
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

const std::vector<double>& SolveEmbedding::positions(const Csr& P, const std::vector<bool>& active) {
  std::vector<double> next = pca2(destination_signatures(P, active, p_), static_cast<std::size_t>(p_.dims), active);
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
