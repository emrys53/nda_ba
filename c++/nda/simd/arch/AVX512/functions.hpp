#pragma once
#ifdef __AVX512F__
#include "./type.hpp"
#include "../functions_forward.hpp"
#include "../../../macros.hpp"
#include <immintrin.h>

namespace nda::simd {

  //Absolute value functions
  template <>
 inline  simd_i16 abs(const simd_i16 &x) {
    return simd_i16(_mm512_abs_epi32(x));
  }
    template <>
  inline simd_l8 abs(const simd_l8 &x) {
    return simd_l8(_mm512_abs_epi32(x));
  }

  template <>
 inline  simd_f16 abs(const simd_f16 &x) {
    return simd_f16(_mm512_abs_ps(x));
  }

  template <>
 inline  simd_d8 abs(const simd_d8 &x) {
   return simd_d8(_mm512_abs_pd(x));
  }

  // Conjugate functions
  template <>
  inline simd_i16 conj(const simd_i16 &x) {
    return simd_i16(x);
  }

  template <>
 inline  simd_l8 conj(const simd_l8 &x) {
    return simd_l8(x);
  }

  template <>
 inline  simd_f16 conj(const simd_f16 &x) {
    return simd_f16(x);
  }

  template <>
 inline  simd_d8 conj(const simd_d8 &x) {
    return simd_d8(x);
  }
  template <>
  inline simd_cf8 conj(const simd_cf8 &x) {
    const __m512 mask = _mm512_castsi512_ps(_mm512_setr_epi32(
      0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000,
      0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000));
#ifdef __AVX512DQ__
    return simd_cf8(_mm512_xor_ps(x,mask));
#else
    return simd_cf8(_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(x),_mm512_castps_si512(mask))));
#endif

  }
  template <>
  inline simd_cd4 conj(const simd_cd4 &x) {
    const __m512d mask = _mm512_castsi512_pd(_mm512_set_epi32(0x80000000, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x0,
                                                            0x80000000, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x0));
#ifdef __AVX512DQ__
    return simd_cd4(_mm512_xor_pd(x,mask));
 #else
    return simd_cd4(_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castps_si512(x),_mm512_castpd_si512(mask))));
 #endif
  }

  //Sqrt functions
  template <>
 inline  simd_f16 sqrt(const simd_f16 &x) {
    return simd_f16(_mm512_sqrt_ps(x));
  }

  template <>
  inline simd_d8 sqrt(const simd_d8 &x) {
    return simd_d8(_mm512_sqrt_pd(x));
  }

  // Min functions
  template <>
  inline simd_i16 min(const simd_i16 &x, const simd_i16 &y) {
    return simd_i16(_mm512_min_epi32(x,y));
  }

  template <>
  inline simd_l8 min(const simd_l8 &x, const simd_l8 &y) {
    return simd_l8(_mm512_min_epi64(x,y));
  }

  template <>
  inline simd_f16 min(const simd_f16 &x, const simd_f16 &y) {
    return simd_f16(_mm512_min_ps(y,x));
  }

  template <>
  inline simd_d8 min(const simd_d8 &x, const simd_d8 &y) {
    return simd_d8(_mm512_min_pd(y,x));
  }

  //Max functions
  template <>
  inline simd_i16 max(const simd_i16 &x, const simd_i16 &y) {
    return simd_i16(_mm512_max_epi32(x,y));
  }

  template <>
  inline simd_l8 max(const simd_l8 &x, const simd_l8 &y) {
    return simd_l8(_mm512_max_epi64(x,y));
  }

  template <>
  inline simd_f16 max(const simd_f16 &x, const simd_f16 &y) {
    return simd_f16(_mm512_max_ps(y,x));
  }

  template <>
  inline simd_d8 max(const simd_d8 &x, const simd_d8 &y) {
    return simd_d8(_mm512_max_pd(y,x));
  }

  // Reduce Min
  template <>
  inline simd_i16::value_t reduce_min(const simd_i16 &x) {
    return _mm512_reduce_min_epi32(x);
   }

  template <>
  inline simd_l8::value_t reduce_min(const simd_l8 &x) {
    return _mm512_reduce_min_epi64(x);
  }

  template <>
  inline simd_f16::value_t reduce_min(const simd_f16 &x) {
    //Eigen
    __m128 lane0 = _mm512_extractf32x4_ps(x, 0);
    __m128 lane1 = _mm512_extractf32x4_ps(x, 1);
    __m128 lane2 = _mm512_extractf32x4_ps(x, 2);
    __m128 lane3 = _mm512_extractf32x4_ps(x, 3);
    simd_f4 min  = min(min(simd_f4{lane0}, simd_f4{lane1}), min(simd_f4{lane2}, simd_f4{lane3}));
    return reduce_min(min);

  }

  template <>
  inline simd_d8::value_t reduce_min(const simd_d8 &x) {
    __m256d lane0 = _mm512_extractf64x4_pd(x, 0);
    __m256d lane1 = _mm512_extractf64x4_pd(x, 1);
    simd_d4 min = min(simd_d4{lane0}, simd_d4{lane1});
    return reduce_min(min);
  }

  // Reduce Max
  template <>
  inline simd_i16::value_t reduce_max(const simd_i16 &x) {
    return _mm512_reduce_max_epi32(x);
  }

  template <>
  inline simd_l8::value_t reduce_max(const simd_l8 &x) {
    _mm512_reduce_max_epi64(x)
  }

  template <>
  inline simd_f16::value_t reduce_max(const simd_f16 &x) {
    __m128 lane0 = _mm512_extractf32x4_ps(x, 0);
    __m128 lane1 = _mm512_extractf32x4_ps(x, 1);
    __m128 lane2 = _mm512_extractf32x4_ps(x, 2);
    __m128 lane3 = _mm512_extractf32x4_ps(x, 3);
    simd_f4 max  = max(max(simd_f4{lane0}, simd_f4{lane1}), max(simd_f4{lane2}, simd_f4{lane3}));
    return reduce_max(max);
  }

  template <>
  inline simd_d8::value_t reduce_max(const simd_d8 &x) {
    __m256d lane0 = _mm512_extractf64x4_pd(x, 0);
    __m256d lane1 = _mm512_extractf64x4_pd(x, 1);
    simd_d4 max = max(simd_d4{lane0}, simd_d4{lane1});
    return reduce_max(max);
  }

} // namespace nda::simd
#endif