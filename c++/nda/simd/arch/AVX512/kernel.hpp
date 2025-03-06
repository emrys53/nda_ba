#pragma once
#ifdef __AVX512F__
#include "./type.hpp"
#include "./functions.hpp"
#include "../kernel_forward.hpp"
namespace nda::simd {
  template <>
  inline std::array<simd_i16, 16> kernel_transpose(const std::array<simd_i16, 16> &simd_block) {
    return simd_block;
  }


  template <>
  inline std::array<simd_l8, 8> kernel_transpose(const std::array<simd_l8, 8> &simd_block) {
    return simd_block;
  }

  template <>
  inline std::array<simd_f16, 16> kernel_transpose(const std::array<simd_f16, 16> &simd_block) {
    return simd_block;
  }

  template <>
  inline std::array<simd_d8, 8> kernel_transpose(const std::array<simd_d8, 8> &simd_block) {
    return simd_block;
  }

  template <>
  inline std::array<simd_cf8, 8> kernel_transpose(const std::array<simd_cf8, 8> &simd_block) {
    simd_cf8 a0a1a2a3a4a5a6a7 = simd_block[0];
    simd_cf8 b0b1b2b3b4b5b6b7 = simd_block[1];
    simd_cf8 c0c1c2c3c4c5c6c7 = simd_block[2];
    simd_cf8 d0d1d2d3d4d5d6d7 = simd_block[3];
    simd_cf8 e0e1e2e3e4e5e6e7 = simd_block[4];
    simd_cf8 f0f1f2f3f4f5f6f7 = simd_block[5];
    simd_cf8 g0g1g2g3g4g5g6g7 = simd_block[6];
    simd_cf8 h0h1h2h3h4h5h6h7 = simd_block[7];

    __m512d a0b0a2b2a4b4a6b6 = _mm512_unpacklo_pd(_mm512_castps_pd(a0a1a2a3a4a5a6a7), _mm512_castps_pd(b0b1b2b3b4b5b6b7));
    __m512d a1b1a3b3a5b5a7b7 = _mm512_unpackhi_pd(_mm512_castps_pd(a0a1a2a3a4a5a6a7), _mm512_castps_pd(b0b1b2b3b4b5b6b7));
    __m512d c0d0c2d2c4d4c6d6 = _mm512_unpacklo_pd(_mm512_castps_pd(c0c1c2c3c4c5c6c7), _mm512_castps_pd(d0d1d2d3d4d5d6d7));
    __m512d c1d1c3d3c5d5c7d7 = _mm512_unpackhi_pd(_mm512_castps_pd(c0c1c2c3c4c5c6c7), _mm512_castps_pd(d0d1d2d3d4d5d6d7));
    __m512d e0f0e2f2e4f4e6f6 = _mm512_unpacklo_pd(_mm512_castps_pd(e0e1e2e3e4e5e6e7), _mm512_castps_pd(f0f1f2f3f4f5f6f7));
    __m512d e1f1e3f3e5f5e5f7 = _mm512_unpackhi_pd(_mm512_castps_pd(e0e1e2e3e4e5e6e7), _mm512_castps_pd(f0f1f2f3f4f5f6f7));
    __m512d g0h0g2h2g4h4g6h6 = _mm512_unpacklo_pd(_mm512_castps_pd(g0g1g2g3g4g5g6g7), _mm512_castps_pd(h0h1h2h3h4h5h6h7));
    __m512d g1h1g3h3g5h5g7h7 = _mm512_unpackhi_pd(_mm512_castps_pd(g0g1g2g3g4g5g6g7), _mm512_castps_pd(h0h1h2h3h4h5h6h7));

    __m512d a2b2a0b0a6b6a4b4 = _mm512_shuffle_f64x2(a0b0a2b2a4b4a6b6, a0b0a2b2a4b4a6b6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d a3b3a1b1a7b7a5b5 = _mm512_shuffle_f64x2(a1b1a3b3a5b5a7b7, a1b1a3b3a5b5a7b7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d c2d2c0d0c6d6c4d4 = _mm512_shuffle_f64x2(c0d0c2d2c4d4c6d6, c0d0c2d2c4d4c6d6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d c3d3c1d1c7d7c5d5 = _mm512_shuffle_f64x2(c1d1c3d3c5d5c7d7, c1d1c3d3c5d5c7d7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d e2f2e0f0e6f6e4f4 = _mm512_shuffle_f64x2(e0f0e2f2e4f4e6f6, e0f0e2f2e4f4e6f6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d e3f3e1f1e7f7e5f5 = _mm512_shuffle_f64x2(e1f1e3f3e5f5e5f7, e1f1e3f3e5f5e5f7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d g2h2g0h0g6h6g4h4 = _mm512_shuffle_f64x2(g0h0g2h2g4h4g6h6, g0h0g2h2g4h4g6h6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d g3h3g1h1g7h7g5h5 = _mm512_shuffle_f64x2(g1h1g3h3g5h5g7h7, g1h1g3h3g5h5g7h7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));

    __m512d a2b2c2d2a6b6c6d6 = _mm512_mask_blend_pd(0b11001100, a2b2a0b0a6b6a4b4, c0d0c2d2c4d4c6d6);
    __m512d a3b3c3d3a7b7c7d7 = _mm512_mask_blend_pd(0b11001100, a3b3a1b1a7b7a5b5, c1d1c3d3c5d5c7d7);
    __m512d a0b0c0d0a4b4c4d4 = _mm512_mask_blend_pd(0b11001100, a0b0a2b2a4b4a6b6, c2d2c0d0c6d6c4d4);
    __m512d a1b1c1d1a5b5c5d5 = _mm512_mask_blend_pd(0b11001100, a1b1a3b3a5b5a7b7, c3d3c1d1c7d7c5d5);
    __m512d e2f2g2h2e6f6g6h6 = _mm512_mask_blend_pd(0b11001100, e2f2e0f0e6f6e4f4, g0h0g2h2g4h4g6h6);
    __m512d e3f3g3h3e7f7g7h7 = _mm512_mask_blend_pd(0b11001100, e3f3e1f1e7f7e5f5, g1h1g3h3g5h5g7h7);
    __m512d e0f0g0h0e4f4g4h4 = _mm512_mask_blend_pd(0b11001100, e0f0e2f2e4f4e6f6, g2h2g0h0g6h6g4h4);
    __m512d e1f1g1h1e5f5g5h5 = _mm512_mask_blend_pd(0b11001100, e1f1e3f3e5f5e5f7, g3h3g1h1g7h7g5h5);

    __m512d e0f0g0h0e0f0g0h0 = _mm512_shuffle_f64x2(e0f0g0h0e4f4g4h4, e0f0g0h0e4f4g4h4, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d e1f1g1h1e1f1g1h1 = _mm512_shuffle_f64x2(e1f1g1h1e5f5g5h5, e1f1g1h1e5f5g5h5, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d e2f2g2h2e2f2g2h2 = _mm512_shuffle_f64x2(e2f2g2h2e6f6g6h6, e2f2g2h2e6f6g6h6, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d e3f3g3h3e3f3g3h3 = _mm512_shuffle_f64x2(e3f3g3h3e7f7g7h7, e3f3g3h3e7f7g7h7, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d a4b4c4d4a4b4c4d4 = _mm512_shuffle_f64x2(a0b0c0d0a4b4c4d4, a0b0c0d0a4b4c4d4, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512d a5b5c5d5a5b5c5d5 = _mm512_shuffle_f64x2(a1b1c1d1a5b5c5d5, a1b1c1d1a5b5c5d5, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512d a6b6c6d6a6b6c6d6 = _mm512_shuffle_f64x2(a2b2c2d2a6b6c6d6, a2b2c2d2a6b6c6d6, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512d a7b7c7d7a7b7c7d7 = _mm512_shuffle_f64x2(a3b3c3d3a7b7c7d7, a3b3c3d3a7b7c7d7, NDA_SHUFFLE_MASK4(2, 3, 2, 3));

    simd_cf8 a0b0c0d0e0f0g0h0(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a0b0c0d0a4b4c4d4, e0f0g0h0e0f0g0h0)));

    simd_cf8 a1b1c1d1e1f1g1h1(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a1b1c1d1a5b5c5d5, e1f1g1h1e1f1g1h1)));

    simd_cf8 a2b2c2d2e2f2g2h2(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a2b2c2d2a6b6c6d6, e2f2g2h2e2f2g2h2)));

    simd_cf8 a3b3c3d3e3f3g3h3(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a3b3c3d3a7b7c7d7, e3f3g3h3e3f3g3h3)));

    simd_cf8 a4b4c4d4e4f4g4h4(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a4b4c4d4a4b4c4d4, e0f0g0h0e4f4g4h4)));

    simd_cf8 a5b5c5d5e5f5g5h5(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a5b5c5d5a5b5c5d5, e1f1g1h1e5f5g5h5)));

    simd_cf8 a6b6c6d6e6f6g6h6(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a6b6c6d6a6b6c6d6, e2f2g2h2e6f6g6h6)));

    simd_cf8 a7b7c7d7e7f7g7h7(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a7b7c7d7a7b7c7d7, e3f3g3h3e7f7g7h7)));

    return {a0b0c0d0e0f0g0h0, a1b1c1d1e1f1g1h1, a2b2c2d2e2f2g2h2, a3b3c3d3e3f3g3h3,
            a4b4c4d4e4f4g4h4, a5b5c5d5e5f5g5h5, a6b6c6d6e6f6g6h6, a7b7c7d7e7f7g7h7};
  }

  template <>
  inline std::array<simd_cd4, 4> kernel_transpose(const std::array<simd_cd4, 4> &simd_block) {
    simd_cd4 a0a1a2a3 = simd_block[0];
    simd_cd4 b0b1b2b3 = simd_block[1];
    simd_cd4 c0c1c2c3 = simd_block[2];
    simd_cd4 d0d1d2d3 = simd_block[3];

    __m512d a0a1b0b1 = _mm512_shuffle_f64x2(a0a1a2a3, b0b1b2b3, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d c0c1d0d1 = _mm512_shuffle_f64x2(c0c1c2c3, d0d1d2d3, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d a2a3b2b3 = _mm512_shuffle_f64x2(a0a1a2a3, b0b1b2b3, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512d c2c3d2d3 = _mm512_shuffle_f64x2(c0c1c2c3, d0d1d2d3, NDA_SHUFFLE_MASK4(2, 3, 2, 3));

    simd_cd4 a0b0c0d0(_mm512_shuffle_f64x2(a0a1b0b1, c0c1d0d1, NDA_SHUFFLE_MASK4(0, 2, 0, 2)));
    simd_cd4 a1b1c1d1(_mm512_shuffle_f64x2(a0a1b0b1, c0c1d0d1, NDA_SHUFFLE_MASK4(1, 3, 1, 3)));
    simd_cd4 a2b2c2d2(_mm512_shuffle_f64x2(a2a3b2b3, c2c3d2d3, NDA_SHUFFLE_MASK4(0, 2, 0, 2)));
    simd_cd4 a3b3c3d3(_mm512_shuffle_f64x2(a2a3b2b3, c2c3d2d3, NDA_SHUFFLE_MASK4(1, 3, 1, 3)));

    return {a0b0c0d0, a1b1c1d1, a2b2c2d2, a3b3c3d3};
  }
} // namespace nda::simd

#endif