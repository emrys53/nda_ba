#pragma once
#ifdef __AVX2__
#include "./type.hpp"
#include "../functions_forward.hpp"
#include "../../../macros.hpp"
#include "immintrin.h"

namespace nda::simd {

  //Absolute value functions
  template <>
  inline simd_i8 abs(const simd_i8 &x) {
#ifdef __AVX2__
    return simd_i8(_mm256_abs_epi32(x));
#else
    __m128i lo = _mm_abs_epi32(_mm256_extractf128_si256(a, 0));
    __m128i hi = _mm_abs_epi32(_mm256_extractf128_si256(a, 1));
    return simd_i8(_mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1));
#endif
  }
  inline simd_l4 abs(const simd_l4 &x) {
    const __m256i zero = _mm256_setzero_si256();
    simd_l4 mask(_mm256_cmpgt_epi64(x, zero));
    return mask - simd_l4(_mm256_xor_si256(x, mask));
  }

  template <>
  inline simd_f8 abs(const simd_f8 &x) {
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    return simd_f8(_mm256_and_ps(x, mask));
  }

  template <>
  inline simd_d4 abs(const simd_d4 &x) {
    const __m256d mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
    return simd_d4(_mm256_and_pd(x, mask));
  }

  // Conjugate functions
  template <>
  inline simd_i8 conj(const simd_i8 &x) {
    return simd_i8(x);
  }

  template <>
  inline simd_l4 conj(const simd_l4 &x) {
    return simd_l4(x);
  }

  template <>
  inline simd_f8 conj(const simd_f8 &x) {
    return simd_f8(x);
  }

  template <>
  inline simd_d4 conj(const simd_d4 &x) {
    return simd_d4(x);
  }
  template <>
  inline simd_cf4 conj(const simd_cf4 &x) {
    const __m256 mask =
       _mm256_castsi256_ps(_mm256_setr_epi32(0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000));
    return simd_cf4(_mm256_xor_ps(x, mask));
  }
  template <>
  inline simd_cd2 conj(const simd_cd2 &x) {
    const __m256d mask = _mm256_castsi256_pd(_mm256_set_epi32(0x80000000, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x0));
    return simd_cd2(_mm256_xor_pd(x, mask));
  }

  //Sqrt functions
  template <>
  inline simd_f8 sqrt(const simd_f8 &x) {
    return simd_f8(_mm256_sqrt_ps(x));
  }

  template <>
  inline simd_d4 sqrt(const simd_d4 &x) {
    return simd_d4(_mm256_sqrt_pd(x));
  }

} // namespace nda::simd
#endif