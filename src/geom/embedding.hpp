#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "graph/csr.hpp"

namespace fx {

struct EmbeddingParams {
  int steps = 4;            // k: hops of P_off
  int dims = 16;            // random projection width
  std::uint64_t seed = 11;
  double smoothing = 0.5;   // weight on the previous frame's position
};

// Y = P_off^k * G, row-major n x dims. P_off: diagonal removed, columns restricted to active,
// rows renormalized; an active row with no remaining mass becomes absorbing (P_off[i][i] = 1).
// Inactive rows are all zero.
std::vector<double> destination_signatures(const Csr& P, const std::vector<bool>& active, const EmbeddingParams& p);

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
  const std::vector<double>& positions(const Csr& P, const std::vector<bool>& active);

 private:
  std::size_t n_;
  EmbeddingParams p_;
  std::vector<double> prev_;
  std::vector<bool> prev_active_;
  bool have_prev_ = false;
};

}  // namespace fx
