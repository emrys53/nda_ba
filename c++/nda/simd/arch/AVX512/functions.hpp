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

} // namespace nda::simd
#endif