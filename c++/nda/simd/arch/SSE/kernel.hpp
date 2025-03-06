#pragma once
#ifdef __SSE2__
#include "./type.hpp"
#include "../kernel_forward.hpp"

namespace nda::simd {

  template <>
  inline std::array<simd_i4, 4> kernel_transpose(const std::array<simd_i4, 4> &simd_block) {
    simd_i4 a0a1a2a3 = simd_block[0];
    simd_i4 b0b1b2b3 = simd_block[1];
    simd_i4 c0c1c2c3 = simd_block[2];
    simd_i4 d0d1d2d3 = simd_block[3];

    __m128i a0b0a1b1 = _mm_unpacklo_epi32(a0a1a2a3, b0b1b2b3);
    __m128i a2b2a3b3 = _mm_unpackhi_epi32(a0a1a2a3, b0b1b2b3);
    __m128i c0d0c1d1 = _mm_unpacklo_epi32(c0c1c2c3, d0d1d2d3);
    __m128i c2d2c3d3 = _mm_unpackhi_epi32(c0c1c2c3, d0d1d2d3);

    simd_i4 a0b0c0d0(_mm_unpacklo_epi64(a0b0a1b1, c0d0c1d1));
    simd_i4 a1b1c1d1(_mm_unpackhi_epi64(a0b0a1b1, c0d0c1d1));
    simd_i4 a2b2c2d2(_mm_unpacklo_epi64(a2b2a3b3, c2d2c3d3));
    simd_i4 a3b3c3d3(_mm_unpackhi_epi64(a2b2a3b3, c2d2c3d3));
    return {a0b0c0d0, a1b1c1d1, a2b2c2d2, a3b3c3d3};
  }

  template <>
  inline std::array<simd_l2, 2> kernel_transpose(const std::array<simd_l2, 2> &simd_block) {
    simd_l2 a0a1 = simd_block[0];
    simd_l2 b0b1 = simd_block[1];
    simd_l2 a0b0(_mm_unpacklo_epi64(a0a1, b0b1));
    simd_l2 a1b1(_mm_unpackhi_epi64(a0a1, b0b1));
    return {a0b0, a1b1};
  }

  template <>
  inline std::array<simd_f4, 4> kernel_transpose(const std::array<simd_f4, 4> &simd_block) {
    simd_f4 a0a1a2a3 = simd_block[0];
    simd_f4 b0b1b2b3 = simd_block[1];
    simd_f4 c0c1c2c3 = simd_block[2];
    simd_f4 d0d1d2d3 = simd_block[3];

    __m128 a0b0a1b1 = _mm_unpacklo_ps(a0a1a2a3, b0b1b2b3);
    __m128 a2b2a3b3 = _mm_unpackhi_ps(a0a1a2a3, b0b1b2b3);
    __m128 c0d0c1d1 = _mm_unpacklo_ps(c0c1c2c3, d0d1d2d3);
    __m128 c2d2c3d3 = _mm_unpackhi_ps(c0c1c2c3, d0d1d2d3);

    simd_f4 a0b0c0d0(_mm_movelh_ps(a0b0a1b1, c0d0c1d1));
    simd_f4 a1b1c1d1(_mm_movehl_ps(c0d0c1d1, a0b0a1b1));
    simd_f4 a2b2c2d2(_mm_movelh_ps(a2b2a3b3, c2d2c3d3));
    simd_f4 a3b3c3d3(_mm_movehl_ps(c2d2c3d3, a2b2a3b3));
    return {a0b0c0d0, a1b1c1d1, a2b2c2d2, a3b3c3d3};
  }

  template <>
  inline std::array<simd_d2, 2> kernel_transpose(const std::array<simd_d2, 2> &simd_block) {
    simd_d2 a0a1 = simd_block[0];
    simd_d2 b0b1 = simd_block[1];
    simd_d2 a0b0(_mm_unpacklo_pd(a0a1, b0b1));
    simd_d2 a1b1(_mm_unpackhi_pd(a0a1, b0b1));
    return {a0b0, a1b1};
  }

  template <>
  inline std::array<simd_cf2, 2> kernel_transpose(const std::array<simd_cf2, 2> &simd_block) {
    simd_cf2 a0a1 = simd_block[0];
    simd_cf2 b0b1 = simd_block[1];
    simd_cf2 a0b0(_mm_castpd_ps((_mm_unpacklo_pd(_mm_castps_pd(a0a1), _mm_castps_pd(b0b1)))));
    simd_cf2 a1b1(_mm_castpd_ps((_mm_unpackhi_pd(_mm_castps_pd(a0a1), _mm_castps_pd(b0b1)))));
    return {a0b0, a1b1};
  }

  template <>
  inline std::array<simd_cd1, 1> kernel_transpose(const std::array<simd_cd1, 1> &simd_block) {
    return simd_block;
  }

} // namespace nda::simd
#endif