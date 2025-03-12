#pragma once
#ifdef __AVX2__
#include "./type.hpp"
#include "../kernel_forward.hpp"

namespace nda::simd {

  template <>
  inline void kernel_transpose(std::array<simd_i8, 8> &simd_block) {
    __m256 a0b0a1b1a4b4a5b5 = _mm256_unpacklo_ps(_mm256_castsi256_ps(simd_block[0]), _mm256_castsi256_ps(simd_block[1]));
    __m256 a2b2a3b3a6b6a7b7 = _mm256_unpackhi_ps(_mm256_castsi256_ps(simd_block[0]), _mm256_castsi256_ps(simd_block[1]));
    __m256 c0d0c1d1c4d4c5d5 = _mm256_unpacklo_ps(_mm256_castsi256_ps(simd_block[2]), _mm256_castsi256_ps(simd_block[3]));
    __m256 c2d2c3d3c6d6c7d7 = _mm256_unpackhi_ps(_mm256_castsi256_ps(simd_block[2]), _mm256_castsi256_ps(simd_block[3]));
    __m256 e0f0e1f1e4f4e5f5 = _mm256_unpacklo_ps(_mm256_castsi256_ps(simd_block[4]), _mm256_castsi256_ps(simd_block[5]));
    __m256 e2f2e3f3e6f6e7f7 = _mm256_unpackhi_ps(_mm256_castsi256_ps(simd_block[4]), _mm256_castsi256_ps(simd_block[5]));
    __m256 g0h0g1h1g4h4g5h5 = _mm256_unpacklo_ps(_mm256_castsi256_ps(simd_block[6]), _mm256_castsi256_ps(simd_block[7]));
    __m256 g2h2g3h3g6h6g7h7 = _mm256_unpackhi_ps(_mm256_castsi256_ps(simd_block[6]), _mm256_castsi256_ps(simd_block[7]));

    __m256 a0b0c0d0a4b4c4d4 = _mm256_shuffle_ps(a0b0a1b1a4b4a5b5, c0d0c1d1c4d4c5d5, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 a1b1c1d1a5b5c5d5 = _mm256_shuffle_ps(a0b0a1b1a4b4a5b5, c0d0c1d1c4d4c5d5, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m256 a2b2c2d3a6b6c6d6 = _mm256_shuffle_ps(a2b2a3b3a6b6a7b7, c2d2c3d3c6d6c7d7, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 a3b3c3d3a7b7c7d7 = _mm256_shuffle_ps(a2b2a3b3a6b6a7b7, c2d2c3d3c6d6c7d7, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m256 e0f0g0h0e4f4g4h4 = _mm256_shuffle_ps(e0f0e1f1e4f4e5f5, g0h0g1h1g4h4g5h5, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 e1f1g1h1e5f5g5h5 = _mm256_shuffle_ps(e0f0e1f1e4f4e5f5, g0h0g1h1g4h4g5h5, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m256 e2f2g2h2e6f6g6h6 = _mm256_shuffle_ps(e2f2e3f3e6f6e7f7, g2h2g3h3g6h6g7h7, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 e3f3g3h3e7f7g7h7 = _mm256_shuffle_ps(e2f2e3f3e6f6e7f7, g2h2g3h3g6h6g7h7, NDA_SHUFFLE_MASK4(2, 3, 2, 3));

    simd_block[0] = simd_i8(_mm256_castps_si256(_mm256_permute2f128_ps(a0b0c0d0a4b4c4d4, e0f0g0h0e4f4g4h4, 0x20)));
    simd_block[4] = simd_i8(_mm256_castps_si256(_mm256_permute2f128_ps(a0b0c0d0a4b4c4d4, e0f0g0h0e4f4g4h4, 0x31)));
    simd_block[2] = simd_i8(_mm256_castps_si256(_mm256_permute2f128_ps(a2b2c2d3a6b6c6d6, e2f2g2h2e6f6g6h6, 0x20)));
    simd_block[6] = simd_i8(_mm256_castps_si256(_mm256_permute2f128_ps(a2b2c2d3a6b6c6d6, e2f2g2h2e6f6g6h6, 0x31)));
    simd_block[1] = simd_i8(_mm256_castps_si256(_mm256_permute2f128_ps(a1b1c1d1a5b5c5d5, e1f1g1h1e5f5g5h5, 0x20)));
    simd_block[5] = simd_i8(_mm256_castps_si256(_mm256_permute2f128_ps(a1b1c1d1a5b5c5d5, e1f1g1h1e5f5g5h5, 0x31)));
    simd_block[3] = simd_i8(_mm256_castps_si256(_mm256_permute2f128_ps(a3b3c3d3a7b7c7d7, e3f3g3h3e7f7g7h7, 0x20)));
    simd_block[7] = simd_i8(_mm256_castps_si256(_mm256_permute2f128_ps(a3b3c3d3a7b7c7d7, e3f3g3h3e7f7g7h7, 0x31)));
  }

  template <>
  inline void kernel_transpose(std::array<simd_l4, 4> &simd_block) {
    __m256i a0b0a2b2 = _mm256_unpacklo_epi64(simd_block[0], simd_block[1]);
    __m256i a1b1a3b3 = _mm256_unpackhi_epi64(simd_block[0], simd_block[1]);
    __m256i c0d0c2d2 = _mm256_unpacklo_epi64(simd_block[2], simd_block[3]);
    __m256i c1d1c3d3 = _mm256_unpackhi_epi64(simd_block[2], simd_block[3]);

    simd_block[0] = simd_l4(_mm256_permute2x128_si256(a0b0a2b2, c0d0c2d2, 0x20));
    simd_block[2] = simd_l4(_mm256_permute2x128_si256(a0b0a2b2, c0d0c2d2, 0x31));
    simd_block[1] = simd_l4(_mm256_permute2x128_si256(a1b1a3b3, c1d1c3d3, 0x20));
    simd_block[3] = simd_l4(_mm256_permute2x128_si256(a1b1a3b3, c1d1c3d3, 0x31));
  }

  template <>
  inline void kernel_transpose(std::array<simd_f8, 8> &simd_block) {
    __m256 a0b0a1b1a4b4a5b5 = _mm256_unpacklo_ps(simd_block[0], simd_block[1]);
    __m256 a2b2a3b3a6b6a7b7 = _mm256_unpackhi_ps(simd_block[0], simd_block[1]);
    __m256 c0d0c1d1c4d4c5d5 = _mm256_unpacklo_ps(simd_block[2], simd_block[3]);
    __m256 c2d2c3d3c6d6c7d7 = _mm256_unpackhi_ps(simd_block[2], simd_block[3]);
    __m256 e0f0e1f1e4f4e5f5 = _mm256_unpacklo_ps(simd_block[4], simd_block[5]);
    __m256 e2f2e3f3e6f6e7f7 = _mm256_unpackhi_ps(simd_block[4], simd_block[5]);
    __m256 g0h0g1h1g4h4g5h5 = _mm256_unpacklo_ps(simd_block[6], simd_block[7]);
    __m256 g2h2g3h3g6h6g7h7 = _mm256_unpackhi_ps(simd_block[6], simd_block[7]);

    __m256 a0b0c0d0a4b4c4d4 = _mm256_shuffle_ps(a0b0a1b1a4b4a5b5, c0d0c1d1c4d4c5d5, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 a1b1c1d1a5b5c5d5 = _mm256_shuffle_ps(a0b0a1b1a4b4a5b5, c0d0c1d1c4d4c5d5, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m256 a2b2c2d3a6b6c6d6 = _mm256_shuffle_ps(a2b2a3b3a6b6a7b7, c2d2c3d3c6d6c7d7, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 a3b3c3d3a7b7c7d7 = _mm256_shuffle_ps(a2b2a3b3a6b6a7b7, c2d2c3d3c6d6c7d7, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m256 e0f0g0h0e4f4g4h4 = _mm256_shuffle_ps(e0f0e1f1e4f4e5f5, g0h0g1h1g4h4g5h5, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 e1f1g1h1e5f5g5h5 = _mm256_shuffle_ps(e0f0e1f1e4f4e5f5, g0h0g1h1g4h4g5h5, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m256 e2f2g2h2e6f6g6h6 = _mm256_shuffle_ps(e2f2e3f3e6f6e7f7, g2h2g3h3g6h6g7h7, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 e3f3g3h3e7f7g7h7 = _mm256_shuffle_ps(e2f2e3f3e6f6e7f7, g2h2g3h3g6h6g7h7, NDA_SHUFFLE_MASK4(2, 3, 2, 3));

    simd_block[0] = simd_f8(_mm256_permute2f128_ps(a0b0c0d0a4b4c4d4, e0f0g0h0e4f4g4h4, 0x20));
    simd_block[4] = simd_f8(_mm256_permute2f128_ps(a0b0c0d0a4b4c4d4, e0f0g0h0e4f4g4h4, 0x31));
    simd_block[2] = simd_f8(_mm256_permute2f128_ps(a2b2c2d3a6b6c6d6, e2f2g2h2e6f6g6h6, 0x20));
    simd_block[6] = simd_f8(_mm256_permute2f128_ps(a2b2c2d3a6b6c6d6, e2f2g2h2e6f6g6h6, 0x31));
    simd_block[1] = simd_f8(_mm256_permute2f128_ps(a1b1c1d1a5b5c5d5, e1f1g1h1e5f5g5h5, 0x20));
    simd_block[5] = simd_f8(_mm256_permute2f128_ps(a1b1c1d1a5b5c5d5, e1f1g1h1e5f5g5h5, 0x31));
    simd_block[3] = simd_f8(_mm256_permute2f128_ps(a3b3c3d3a7b7c7d7, e3f3g3h3e7f7g7h7, 0x20));
    simd_block[7] = simd_f8(_mm256_permute2f128_ps(a3b3c3d3a7b7c7d7, e3f3g3h3e7f7g7h7, 0x31));
  }

  template <>
  inline void kernel_transpose(std::array<simd_d4, 4> &simd_block) {
    __m256d a0b0a2b2 = _mm256_unpacklo_pd(simd_block[0], simd_block[1]);
    __m256d a1b1a3b3 = _mm256_unpackhi_pd(simd_block[0], simd_block[1]);
    __m256d c0d0c2d2 = _mm256_unpacklo_pd(simd_block[2], simd_block[3]);
    __m256d c1d1c3d3 = _mm256_unpackhi_pd(simd_block[2], simd_block[3]);

    simd_block[0] = simd_d4(_mm256_permute2f128_pd(a0b0a2b2, c0d0c2d2, 0x20));
    simd_block[2] = simd_d4(_mm256_permute2f128_pd(a0b0a2b2, c0d0c2d2, 0x31));
    simd_block[1] = simd_d4(_mm256_permute2f128_pd(a1b1a3b3, c1d1c3d3, 0x20));
    simd_block[3] = simd_d4(_mm256_permute2f128_pd(a1b1a3b3, c1d1c3d3, 0x31));
  }

  template <>
  inline void kernel_transpose(std::array<simd_cf4, 4> &simd_block) {
    __m256d a0b0a2b2 = _mm256_unpacklo_pd(_mm256_castps_pd(simd_block[0]), _mm256_castps_pd(simd_block[1]));
    __m256d a1b1a3b3 = _mm256_unpackhi_pd(_mm256_castps_pd(simd_block[0]), _mm256_castps_pd(simd_block[1]));
    __m256d c0d0c2d2 = _mm256_unpacklo_pd(_mm256_castps_pd(simd_block[2]), _mm256_castps_pd(simd_block[3]));
    __m256d c1d1c3d3 = _mm256_unpackhi_pd(_mm256_castps_pd(simd_block[2]), _mm256_castps_pd(simd_block[3]));

    simd_block[0] = simd_cf4(_mm256_castpd_ps(_mm256_permute2f128_pd(a0b0a2b2, c0d0c2d2, 0x20)));
    simd_block[2] = simd_cf4(_mm256_castpd_ps(_mm256_permute2f128_pd(a0b0a2b2, c0d0c2d2, 0x31)));
    simd_block[1] = simd_cf4(_mm256_castpd_ps(_mm256_permute2f128_pd(a1b1a3b3, c1d1c3d3, 0x20)));
    simd_block[3] = simd_cf4(_mm256_castpd_ps(_mm256_permute2f128_pd(a1b1a3b3, c1d1c3d3, 0x31)));
  }

  inline void kernel_transpose(std::array<simd_cd2, 2> &simd_block) {
    simd_cd2 a0a1 = simd_block[0];
    simd_cd2 b0b1 = simd_block[1];
    simd_block[0] = simd_cd2(_mm256_permute2f128_pd(a0a1, b0b1, 0x20));
    simd_block[1] = simd_cd2(_mm256_permute2f128_pd(a0a1, b0b1, 0x31));
  }

} // namespace nda::simd
#endif
