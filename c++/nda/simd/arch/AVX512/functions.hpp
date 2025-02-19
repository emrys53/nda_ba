#pragma once
#ifndef __AVX512F__
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

  //Reduce_sum
  template <>
  inline simd_i16::value_t reduce_sum(const simd_i16 &x) {
    return _mm512_reduce_add_epi32(x);
  }

  template <>
  inline simd_l8::value_t reduce_sum(const simd_l8 &x) {
    return _mm512_reduce_add_epi64(x);
  }

  template <>
  inline simd_f16::value_t reduce_sum(const simd_f16 &x) {
#ifdef __AVX512DQ__
    simd_f8 lo{_mm512_extractf32x8_ps(x, 0)};
    simd_f8 hi{_mm512_extractf32x8_ps(x, 1)};
    return reduce_sum(lo + hi);
#else
    simd_f4 lo_1{_mm512_extractf32x4_ps(x,0)};
    simd_f4 lo_2{_mm512_extractf32x4_ps(x,1)};
    simd_f4 hi_1{_mm512_extractf32x4_ps(x,2)};
    simd_f4 hi_2{_mm512_extractf32x4_ps(x,3)};
    return reduce_sum(lo_1 + lo_2 + hi_1 + hi_2);
#endif
  }

  template <>
  inline simd_d8::value_t reduce_sum(const simd_d8 &x) {
    simd_d8 lo{_mm512_extractf64x4_pd(x, 0)};
    simd_d8 hi{_mm512_extractf64x4_pd(x, 1)};
    return reduce_sum(lo + hi);
  }

  template <>
  inline simd_cf8::value_t reduce_sum(const simd_cf8 &x) {
#ifdef __AVX512DQ__
    simd_cf4 lo{_mm512_extractf32x8_ps(x, 0)};
    simd_cf4 hi{_mm512_extractf32x4_ps(x, 1)};
    return reduce_sum(lo + hi);
#else
    simd_cf4 lo{_mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(x), 0));
    simd_cf4 hi{_mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(x), 1));
      return reduce_sum(lo + hi)
#endif
  }

  template <>
  inline simd_cd4::value_t reduce_sum(const simd_cd4 &x) {
    simd_cd2 lo{_mm512_extractf64x4_pd(x, 0)};
    simd_cd2 hi{_mm512_extractf64x4_pd(x, 1)};
    return reduce_sum(lo + hi);
  }

  //Reduce_mul
  template <>
  inline simd_i16::value_t reduce_mul(const simd_i16 &x) {
    return _mm512_reduce_mul_epi32(x);
  }

  template <>
  inline simd_l8::value_t reduce_mul(const simd_l8 &x) {
    return _mm512_reduce_mul_epi64(x);
  }

  template <>
  inline simd_f16::value_t reduce_mul(const simd_f16 &x) {
    simd_f4 lo_1{_mm512_extractf32x4_ps(x,0)};
    simd_f4 lo_2{_mm512_extractf32x4_ps(x,1)};
    simd_f4 hi_1{_mm512_extractf32x4_ps(x,2)};
    simd_f4 hi_2{_mm512_extractf32x4_ps(x,3)};
    return reduce_mul(lo_1 * lo_2 * hi_1 * hi_2);
  }

  template <>
  inline simd_d8::value_t reduce_mul(const simd_d8 &x) {
    simd_d8 lo{_mm512_extractf64x4_pd(x, 0)};
    simd_d8 hi{_mm512_extractf64x4_pd(x, 1)};
    return reduce_mul(lo * hi);
  }

  template <>
  inline simd_cf8::value_t reduce_mul(const simd_cf8 &x) {
#ifdef __AVX512DQ__
    simd_cf4 lo{_mm512_extractf32x8_ps(x, 0)};
    simd_cf4 hi{_mm512_extractf32x4_ps(x, 1)};
    return reduce_mul(lo * hi);
#else
    simd_cf4 lo{_mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(x), 0));
    simd_cf4 hi{_mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(x), 1));
    return reduce_mul(lo * hi)
#endif
  }

  template <>
  inline simd_cd4::value_t reduce_mul(const simd_cd4 &x) {
    simd_cd2 lo{_mm512_extractf64x4_pd(x, 0)};
    simd_cd2 hi{_mm512_extractf64x4_pd(x, 1)};
    return reduce_mul(lo * hi);
  }

  #ifdef __FMA__
  // FMA ADD
  template <>
  inline simd_f16 fma_add(const simd_f16 &x, const simd_f16 &y, const simd_f16 &z) {
    return simd_f16(_mm512_fmadd_ps(x, y, z));
  }

  template <>
  inline simd_d8 fma_add(const simd_d8 &x, const simd_d8 &y, const simd_d8 &z) {
    return simd_d8(_mm512_fmadd_pd(x, y, z));
  }

  template <>
  inline simd_cf8 fma_add(const simd_cf8 &x, const simd_cf8 &y, const simd_cf8 &z) {
    return x * y + z;
  }

  template <>
  inline simd_cd4 fma_add(const simd_cd4 &x, const simd_cd4 &y, const simd_cd4 &z) {
    return x * y + z;
  }

  //FMA_SUB
  template <>
  inline simd_f16 fma_sub(const simd_f16 &x, const simd_f16 &y, const simd_f16 &z) {
    return simd_f16(_mm512_fmsub_ps(x, y, z));
  }

  template <>
  inline simd_d8 fma_sub(const simd_d8 &x, const simd_d8 &y, const simd_d8 &z) {
    return simd_d8(_mm512_fmsub_pd(x, y, z));
  }

  template <>
  inline simd_cf8 fma_sub(const simd_cf8 &x, const simd_cf8 &y, const simd_cf8 &z) {
    return x * y + z;
  }

  template <>
  inline simd_cd4 fma_sub(const simd_cd4 &x, const simd_cd4 &y, const simd_cd4 &z) {
    return x * y + z;
  }

  //FMA_NADD
  template <>
  inline simd_f16 fma_nadd(const simd_f16 &x, const simd_f16 &y, const simd_f16 &z) {
    return simd_f16(_mm512_fnmadd_ps(x, y, z));
  }

  template <>
  inline simd_d8 fma_nadd(const simd_d8 &x, const simd_d8 &y, const simd_d8 &z) {
    return simd_d8(_mm512_fnmadd_pd(x, y, z));
  }

  template <>
  inline simd_cf8 fma_nadd(const simd_cf8 &x, const simd_cf8 &y, const simd_cf8 &z) {
    return x * y + z;
  }

  template <>
  inline simd_cd4 fma_nadd(const simd_cd4 &x, const simd_cd4 &y, const simd_cd4 &z) {
    return x * y + z;
  }

  //FMA_NSUB
  template <>
  inline simd_f16 fma_nsub(const simd_f16 &x, const simd_f16 &y, const simd_f16 &z) {
    return simd_f16(_mm512_fnmsub_ps(x, y, z));
  }

  template <>
  inline simd_d8 fma_nsub(const simd_d8 &x, const simd_d8 &y, const simd_d8 &z) {
    return simd_d8(_mm512_fnmsub_pd(x, y, z));
  }

  template <>
  inline simd_cf8 fma_nsub(const simd_cf8 &x, const simd_cf8 &y, const simd_cf8 &z) {
    return x * y + z;
  }

  template <>
  inline simd_cd4 fma_nsub(const simd_cd4 &x, const simd_cd4 &y, const simd_cd4 &z) {
    return x * y + z;
  }
#endif

} // namespace nda::simd
#endif