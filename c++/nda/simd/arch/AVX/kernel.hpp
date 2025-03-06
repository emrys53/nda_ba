#pragma once
#ifdef __AVX2__
#include "./type.hpp"
#include "../kernel_forward.hpp"

namespace nda::simd {

  template <>
  inline std::array<simd_i8, 8> kernel_transpose(const std::array<simd_i8, 8> &simd_block) {
    simd_i8 a0a1a2a3a4a5a6a7 = simd_block[0];
    simd_i8 b0b1b2b3b4b5b6b7 = simd_block[1];
    simd_i8 c0c1c2c3c4c5c6c7 = simd_block[2];
    simd_i8 d0d1d2d3d4d5d6d7 = simd_block[3];
    simd_i8 e0e1e2e3e4e5e6e7 = simd_block[4];
    simd_i8 f0f1f2f3f4f5f6f  = simd_block[5];
    simd_i8 g0g1g2g3g4g5g6g7 = simd_block[6];
    simd_i8 h0h1h2h3h4h5h6h7 = simd_block[7];

    __m256 a0b0a1b1a4b4a5b5 = _mm256_unpacklo_ps(_mm256_castsi256_ps(a0a1a2a3a4a5a6a7), _mm256_castsi256_ps(b0b1b2b3b4b5b6b7));
    __m256 a2b2a3b3a6b6a7b7 = _mm256_unpackhi_ps(_mm256_castsi256_ps(a0a1a2a3a4a5a6a7), _mm256_castsi256_ps(b0b1b2b3b4b5b6b7));
    __m256 c0d0c1d1c4d4c5d5 = _mm256_unpacklo_ps(_mm256_castsi256_ps(c0c1c2c3c4c5c6c7), _mm256_castsi256_ps(d0d1d2d3d4d5d6d7));
    __m256 c2d2c3d3c6d6c7d7 = _mm256_unpackhi_ps(_mm256_castsi256_ps(c0c1c2c3c4c5c6c7), _mm256_castsi256_ps(d0d1d2d3d4d5d6d7));
    __m256 e0f0e1f1e4f4e5f5 = _mm256_unpacklo_ps(_mm256_castsi256_ps(e0e1e2e3e4e5e6e7), _mm256_castsi256_ps(f0f1f2f3f4f5f6f));
    __m256 e2f2e3f3e6f6e7f7 = _mm256_unpackhi_ps(_mm256_castsi256_ps(e0e1e2e3e4e5e6e7), _mm256_castsi256_ps(f0f1f2f3f4f5f6f));
    __m256 g0h0g1h1g4h4g5h5 = _mm256_unpacklo_ps(_mm256_castsi256_ps(g0g1g2g3g4g5g6g7), _mm256_castsi256_ps(h0h1h2h3h4h5h6h7));
    __m256 g2h2g3h3g6h6g7h7 = _mm256_unpackhi_ps(_mm256_castsi256_ps(g0g1g2g3g4g5g6g7), _mm256_castsi256_ps(h0h1h2h3h4h5h6h7));

    __m256 a0b0c0d0a4b4c4d4 = _mm256_shuffle_ps(a0b0a1b1a4b4a5b5, c0d0c1d1c4d4c5d5, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 a1b1c1d1a5b5c5d5 = _mm256_shuffle_ps(a0b0a1b1a4b4a5b5, c0d0c1d1c4d4c5d5, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m256 a2b2c2d3a6b6c6d6 = _mm256_shuffle_ps(a2b2a3b3a6b6a7b7, c2d2c3d3c6d6c7d7, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 a3b3c3d3a7b7c7d7 = _mm256_shuffle_ps(a2b2a3b3a6b6a7b7, c2d2c3d3c6d6c7d7, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m256 e0f0g0h0e4f4g4h4 = _mm256_shuffle_ps(e0f0e1f1e4f4e5f5, g0h0g1h1g4h4g5h5, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 e1f1g1h1e5f5g5h5 = _mm256_shuffle_ps(e0f0e1f1e4f4e5f5, g0h0g1h1g4h4g5h5, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m256 e2f2g2h2e6f6g6h6 = _mm256_shuffle_ps(e2f2e3f3e6f6e7f7, g2h2g3h3g6h6g7h7, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 e3f3g3h3e7f7g7h7 = _mm256_shuffle_ps(e2f2e3f3e6f6e7f7, g2h2g3h3g6h6g7h7, NDA_SHUFFLE_MASK4(2, 3, 2, 3));

    simd_i8 a0b0c0d0e0f0g0h0(_mm256_castps_si256(_mm256_permute2f128_ps(a0b0c0d0a4b4c4d4, e0f0g0h0e4f4g4h4, 0x20)));
    simd_i8 a4b4c4d4e4f4g4h4(_mm256_castps_si256(_mm256_permute2f128_ps(a0b0c0d0a4b4c4d4, e0f0g0h0e4f4g4h4, 0x31)));
    simd_i8 a2b2c2d2e2f2g2h2(_mm256_castps_si256(_mm256_permute2f128_ps(a2b2c2d3a6b6c6d6, e2f2g2h2e6f6g6h6, 0x20)));
    simd_i8 a6b6c6d6e6f6g6h6(_mm256_castps_si256(_mm256_permute2f128_ps(a2b2c2d3a6b6c6d6, e2f2g2h2e6f6g6h6, 0x31)));
    simd_i8 a1b1c1d1e1f1g1h1(_mm256_castps_si256(_mm256_permute2f128_ps(a1b1c1d1a5b5c5d5, e1f1g1h1e5f5g5h5, 0x20)));
    simd_i8 a5b5c5d5e5f5g5h5(_mm256_castps_si256(_mm256_permute2f128_ps(a1b1c1d1a5b5c5d5, e1f1g1h1e5f5g5h5, 0x31)));
    simd_i8 a3b3c3d3e3f3g3h3(_mm256_castps_si256(_mm256_permute2f128_ps(a3b3c3d3a7b7c7d7, e3f3g3h3e7f7g7h7, 0x20)));
    simd_i8 a7b7c7d7e7f7g7h7(_mm256_castps_si256(_mm256_permute2f128_ps(a3b3c3d3a7b7c7d7, e3f3g3h3e7f7g7h7, 0x31)));

    return {a0b0c0d0e0f0g0h0, a1b1c1d1e1f1g1h1, a2b2c2d2e2f2g2h2, a3b3c3d3e3f3g3h3,
            a4b4c4d4e4f4g4h4, a5b5c5d5e5f5g5h5, a6b6c6d6e6f6g6h6, a7b7c7d7e7f7g7h7};
  }

  template <>
  inline std::array<simd_l4, 4> kernel_transpose(const std::array<simd_l4, 4> &simd_block) {
    simd_l4 a0a1a2a3 = simd_block[0];
    simd_l4 b0b1b2b3 = simd_block[1];
    simd_l4 c0c1c2c3 = simd_block[2];
    simd_l4 d0d1d2d3 = simd_block[3];

    __m256i a0b0a2b2 = _mm256_unpacklo_epi64(a0a1a2a3, b0b1b2b3);
    __m256i a1b1a3b3 = _mm256_unpackhi_epi64(a0a1a2a3, b0b1b2b3);
    __m256i c0d0c2d2 = _mm256_unpacklo_epi64(c0c1c2c3, d0d1d2d3);
    __m256i c1d1c3d3 = _mm256_unpackhi_epi64(c0c1c2c3, d0d1d2d3);

    simd_l4 a0b0c0d0(_mm256_permute2x128_si256(a0b0a2b2, c0d0c2d2, 0x20));
    simd_l4 a2b2c2d2(_mm256_permute2x128_si256(a0b0a2b2, c0d0c2d2, 0x31));
    simd_l4 a1b1c1d1(_mm256_permute2x128_si256(a1b1a3b3, c1d1c3d3, 0x20));
    simd_l4 a3b3c3d3(_mm256_permute2x128_si256(a1b1a3b3, c1d1c3d3, 0x31));

    return {a0b0c0d0, a1b1c1d1, a2b2c2d2, a3b3c3d3};
  }

  template <>
  inline std::array<simd_f8, 8> kernel_transpose(const std::array<simd_f8, 8> &simd_block) {
    simd_f8 a0a1a2a3a4a5a6a7 = simd_block[0];
    simd_f8 b0b1b2b3b4b5b6b7 = simd_block[1];
    simd_f8 c0c1c2c3c4c5c6c7 = simd_block[2];
    simd_f8 d0d1d2d3d4d5d6d7 = simd_block[3];
    simd_f8 e0e1e2e3e4e5e6e7 = simd_block[4];
    simd_f8 f0f1f2f3f4f5f6f  = simd_block[5];
    simd_f8 g0g1g2g3g4g5g6g7 = simd_block[6];
    simd_f8 h0h1h2h3h4h5h6h7 = simd_block[7];

    __m256 a0b0a1b1a4b4a5b5 = _mm256_unpacklo_ps(a0a1a2a3a4a5a6a7, b0b1b2b3b4b5b6b7);
    __m256 a2b2a3b3a6b6a7b7 = _mm256_unpackhi_ps(a0a1a2a3a4a5a6a7, b0b1b2b3b4b5b6b7);
    __m256 c0d0c1d1c4d4c5d5 = _mm256_unpacklo_ps(c0c1c2c3c4c5c6c7, d0d1d2d3d4d5d6d7);
    __m256 c2d2c3d3c6d6c7d7 = _mm256_unpackhi_ps(c0c1c2c3c4c5c6c7, d0d1d2d3d4d5d6d7);
    __m256 e0f0e1f1e4f4e5f5 = _mm256_unpacklo_ps(e0e1e2e3e4e5e6e7, f0f1f2f3f4f5f6f);
    __m256 e2f2e3f3e6f6e7f7 = _mm256_unpackhi_ps(e0e1e2e3e4e5e6e7, f0f1f2f3f4f5f6f);
    __m256 g0h0g1h1g4h4g5h5 = _mm256_unpacklo_ps(g0g1g2g3g4g5g6g7, h0h1h2h3h4h5h6h7);
    __m256 g2h2g3h3g6h6g7h7 = _mm256_unpackhi_ps(g0g1g2g3g4g5g6g7, h0h1h2h3h4h5h6h7);

    __m256 a0b0c0d0a4b4c4d4 = _mm256_shuffle_ps(a0b0a1b1a4b4a5b5, c0d0c1d1c4d4c5d5, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 a1b1c1d1a5b5c5d5 = _mm256_shuffle_ps(a0b0a1b1a4b4a5b5, c0d0c1d1c4d4c5d5, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m256 a2b2c2d3a6b6c6d6 = _mm256_shuffle_ps(a2b2a3b3a6b6a7b7, c2d2c3d3c6d6c7d7, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 a3b3c3d3a7b7c7d7 = _mm256_shuffle_ps(a2b2a3b3a6b6a7b7, c2d2c3d3c6d6c7d7, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m256 e0f0g0h0e4f4g4h4 = _mm256_shuffle_ps(e0f0e1f1e4f4e5f5, g0h0g1h1g4h4g5h5, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 e1f1g1h1e5f5g5h5 = _mm256_shuffle_ps(e0f0e1f1e4f4e5f5, g0h0g1h1g4h4g5h5, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m256 e2f2g2h2e6f6g6h6 = _mm256_shuffle_ps(e2f2e3f3e6f6e7f7, g2h2g3h3g6h6g7h7, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m256 e3f3g3h3e7f7g7h7 = _mm256_shuffle_ps(e2f2e3f3e6f6e7f7, g2h2g3h3g6h6g7h7, NDA_SHUFFLE_MASK4(2, 3, 2, 3));

    simd_f8 a0b0c0d0e0f0g0h0(_mm256_permute2f128_ps(a0b0c0d0a4b4c4d4, e0f0g0h0e4f4g4h4, 0x20));
    simd_f8 a4b4c4d4e4f4g4h4(_mm256_permute2f128_ps(a0b0c0d0a4b4c4d4, e0f0g0h0e4f4g4h4, 0x31));
    simd_f8 a2b2c2d2e2f2g2h2(_mm256_permute2f128_ps(a2b2c2d3a6b6c6d6, e2f2g2h2e6f6g6h6, 0x20));
    simd_f8 a6b6c6d6e6f6g6h6(_mm256_permute2f128_ps(a2b2c2d3a6b6c6d6, e2f2g2h2e6f6g6h6, 0x31));
    simd_f8 a1b1c1d1e1f1g1h1(_mm256_permute2f128_ps(a1b1c1d1a5b5c5d5, e1f1g1h1e5f5g5h5, 0x20));
    simd_f8 a5b5c5d5e5f5g5h5(_mm256_permute2f128_ps(a1b1c1d1a5b5c5d5, e1f1g1h1e5f5g5h5, 0x31));
    simd_f8 a3b3c3d3e3f3g3h3(_mm256_permute2f128_ps(a3b3c3d3a7b7c7d7, e3f3g3h3e7f7g7h7, 0x20));
    simd_f8 a7b7c7d7e7f7g7h7(_mm256_permute2f128_ps(a3b3c3d3a7b7c7d7, e3f3g3h3e7f7g7h7, 0x31));

    return {a0b0c0d0e0f0g0h0, a1b1c1d1e1f1g1h1, a2b2c2d2e2f2g2h2, a3b3c3d3e3f3g3h3,
            a4b4c4d4e4f4g4h4, a5b5c5d5e5f5g5h5, a6b6c6d6e6f6g6h6, a7b7c7d7e7f7g7h7};
  }

  template <>
  inline std::array<simd_d4, 4> kernel_transpose(const std::array<simd_d4, 4> &simd_block) {
    simd_d4 a0a1a2a3 = simd_block[0];
    simd_d4 b0b1b2b3 = simd_block[1];
    simd_d4 c0c1c2c3 = simd_block[2];
    simd_d4 d0d1d2d3 = simd_block[3];

    __m256d a0b0a2b2 = _mm256_unpacklo_pd(a0a1a2a3, b0b1b2b3);
    __m256d a1b1a3b3 = _mm256_unpackhi_pd(a0a1a2a3, b0b1b2b3);
    __m256d c0d0c2d2 = _mm256_unpacklo_pd(c0c1c2c3, d0d1d2d3);
    __m256d c1d1c3d3 = _mm256_unpackhi_pd(c0c1c2c3, d0d1d2d3);

    simd_d4 a0b0c0d0(_mm256_permute2f128_pd(a0b0a2b2, c0d0c2d2, 0x20));
    simd_d4 a2b2c2d2(_mm256_permute2f128_pd(a0b0a2b2, c0d0c2d2, 0x31));
    simd_d4 a1b1c1d1(_mm256_permute2f128_pd(a1b1a3b3, c1d1c3d3, 0x20));
    simd_d4 a3b3c3d3(_mm256_permute2f128_pd(a1b1a3b3, c1d1c3d3, 0x31));
    return {a0b0c0d0, a1b1c1d1, a2b2c2d2, a3b3c3d3};
  }

  template <>
  inline std::array<simd_cf4, 4> kernel_transpose(const std::array<simd_cf4, 4> &simd_block) {
    simd_cf4 a0a1a2a3 = simd_block[0];
    simd_cf4 b0b1b2b3 = simd_block[1];
    simd_cf4 c0c1c2c3 = simd_block[2];
    simd_cf4 d0d1d2d3 = simd_block[3];

    __m256d a0b0a2b2  = _mm256_unpacklo_pd(_mm256_castps_pd(a0a1a2a3), _mm256_castps_pd(b0b1b2b3));
    __m256d a1b1a3b3 = _mm256_unpackhi_pd(_mm256_castps_pd(a0a1a2a3), _mm256_castps_pd(b0b1b2b3));
    __m256d c0d0c2d2  = _mm256_unpacklo_pd(_mm256_castps_pd(c0c1c2c3), _mm256_castps_pd(d0d1d2d3));
    __m256d c1d1c3d3 = _mm256_unpackhi_pd(_mm256_castps_pd(c0c1c2c3), _mm256_castps_pd(d0d1d2d3));

    simd_cf4 a0b0c0d0(_mm256_castpd_ps(_mm256_permute2f128_pd(a0b0a2b2, c0d0c2d2, 0x20)));
    simd_cf4 a2b2c2d2(_mm256_castpd_ps(_mm256_permute2f128_pd(a0b0a2b2, c0d0c2d2, 0x31)));
    simd_cf4 a1b1c1d1(_mm256_castpd_ps(_mm256_permute2f128_pd(a1b1a3b3, c1d1c3d3, 0x20)));
    simd_cf4 a3b3c3d3(_mm256_castpd_ps(_mm256_permute2f128_pd(a1b1a3b3, c1d1c3d3, 0x31)));

    return {a0b0c0d0, a1b1c1d1, a2b2c2d2, a3b3c3d3};
  }

  inline std::array<simd_cd2, 2> kernel_transpose(const std::array<simd_cd2, 2> &simd_block) {
    simd_cd2 a0a1 = simd_block[0];
    simd_cd2 b0b1 = simd_block[1];
    simd_cd2 a0b0(_mm256_permute2f128_pd(a0a1, b0b1, 0x20));
    simd_cd2 a1b1(_mm256_permute2f128_pd(a0a1, b0b1, 0x31));

    return {a0b0, a1b1};
  }

} // namespace nda::simd
#endif
