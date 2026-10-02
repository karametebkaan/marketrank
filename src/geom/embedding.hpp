#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "pipeline/core_pipeline.hpp"

namespace fx {

struct EmbeddingParams {
  double clip = 4.0;       // robust z-score clip
  double smoothing = 0.5;  // weight on the previous frame's position
};

// Row-major n x F solver-output features. Feature columns, in this order:
//   0: display_height(h[i], SignedLog);
//   1: log(pi[i]);
//   2..: display_height(forecasts[j].score[i], SignedLog) for each forecast j, in order.
// Each column is robust-standardized over active rows: (x - median) / (1.4826 * MAD), clipped to [-clip, clip].
// A column whose MAD is 0 or non-finite becomes all 0. Non-finite entries become 0, the column median.
// Inactive rows are all 0. Medians and MADs are computed serially with std::nth_element on a copy.
// Throws std::invalid_argument unless h, pi and every forecast score have size n = f.active.size().
std::vector<double> solve_features(const Frame& f, const EmbeddingParams& p, std::size_t& dims_out);

// Top-2 principal components of the active rows (centered), projected and scaled to unit RMS radius.
// Returns interleaved xy (size 2n). Inactive entries are 0.
std::vector<double> pca2(const std::vector<double>& Y, std::size_t dims, const std::vector<bool>& active);

// Orthogonal Procrustes in 2D: the rotation or reflection R minimizing sum over `mask` of |R x_i - prev_i|^2,
// applied to every entry of xy in place.
void align_to(std::vector<double>& xy, const std::vector<double>& prev, const std::vector<bool>& mask);

class SolveEmbedding {
 public:
  SolveEmbedding(std::size_t n, EmbeddingParams p = {});
  // Interleaved xy (size 2n), the input format of rcb_assign (src/geom/lattice.hpp).
  const std::vector<double>& positions(const Frame& f);  // pca2(solve_features) -> align -> smooth

 private:
  std::size_t n_;
  EmbeddingParams p_;
  std::vector<double> prev_;
  std::vector<bool> prev_active_;
  bool have_prev_ = false;
};

}  // namespace fx
