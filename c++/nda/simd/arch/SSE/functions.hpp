#pragma once
#ifdef __SSE2__
#include "./type.hpp"
#include "../macros.hpp"
#include "../functions_forward.hpp"
#include "../../../macros.hpp"
#include "immintrin.h"

namespace nda::simd {

  // Absolute value functions:
  template <>
  inline simd_i4 abs(const simd_i4 &x) {
#ifdef __SSE3__
    return simd_i4(_mm_abs_epi32(x));
#else
    simd_i4 sign(_mm_srai_epi32(x, 31));
    return simd_i4(_mm_sub_epi32(_mm_xor_si128(x, sign), sign));
#endif
  }

  template <>
  inline simd_l2 abs(const simd_l2 &x) {
    __m128i sign = _mm_srai_epi32(x, 31);
    sign         = _mm_shuffle_epi32(sign, NDA_SHUFFLE_MASK4(1, 1, 3, 3));
    return simd_l2(_mm_sub_epi64(_mm_xor_si128(x, sign), sign));
  }

  template <>
  inline simd_f4 abs(const simd_f4 &x) {
    const __m128i mask = _mm_setr_epi32(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF);
    return simd_f4(_mm_castsi128_ps(_mm_and_si128(mask, _mm_castps_si128(x))));
  }

  template <>
  inline simd_d2 abs(const simd_d2 &x) {
    const __m128i mask = _mm_setr_epi32(0xFFFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF);
    return simd_d2(_mm_castsi128_pd(_mm_and_si128(mask, _mm_castpd_si128(x))));
  }

  // Conj functions
  template <>
  inline simd_i4 conj(const simd_i4 &x) {
    return simd_i4(x);
  }

  template <>
  inline simd_f4 conj(const simd_f4 &x) {
    return simd_f4(x);
  }

  template <>
  inline simd_l2 conj(const simd_l2 &x) {
    return simd_l2(x);
  }

  template <>
  inline simd_d2 conj(const simd_d2 &x) {
    return simd_d2(x);
  }

  template <>
  inline simd_cf2 conj(const simd_cf2 &x) {
    const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0x00000000, 0x80000000, 0x00000000, 0x80000000));
    return simd_cf2(_mm_xor_ps(x, mask));
  }
  template <>
  inline simd_cd1 conj(const simd_cd1 &x) {
    const __m128d mask = _mm_castsi128_pd(_mm_set_epi32(0x80000000, 0x0, 0x0, 0x0));
    return simd_cd1(_mm_xor_pd(x, mask));
  }

  //Sqrt functions

  template <>
  inline simd_f4 sqrt(const simd_f4 &x) {
    return simd_f4(_mm_sqrt_ps(x));
  }

  template <>
  inline simd_d2 sqrt(const simd_d2 &x) {
    return simd_d2(_mm_sqrt_pd(x));
  }

} // namespace nda::simd
#endif