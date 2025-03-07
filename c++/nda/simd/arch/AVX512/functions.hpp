#pragma once
#ifdef __AVX512F__
#include "./type.hpp"
#include "../functions_forward.hpp"
#include "../AVX/functions.hpp"
#include "../SSE/functions.hpp"
#include "immintrin.h"

namespace nda::simd {

  //Absolute value functions
  template <>
  inline simd_i16 abs(const simd_i16 &x) {
    return simd_i16(_mm512_abs_epi32(x));
  }
  template <>
  inline simd_l8 abs(const simd_l8 &x) {
    return simd_l8(_mm512_abs_epi64(x));
  }

  template <>
  inline simd_f16 abs(const simd_f16 &x) {
    return simd_f16(_mm512_abs_ps(x));
  }

  template <>
  inline simd_d8 abs(const simd_d8 &x) {
    return simd_d8(_mm512_abs_pd(x));
  }

  // Conjugate functions
  template <>
  inline simd_i16 conj(const simd_i16 &x) {
    return simd_i16(x);
  }

  template <>
  inline simd_l8 conj(const simd_l8 &x) {
    return simd_l8(x);
  }

  template <>
  inline simd_f16 conj(const simd_f16 &x) {
    return simd_f16(x);
  }

  template <>
  inline simd_d8 conj(const simd_d8 &x) {
    return simd_d8(x);
  }
  template <>
  inline simd_cf8 conj(const simd_cf8 &x) {
    const __m512 mask =
       _mm512_castsi512_ps(_mm512_setr_epi32(0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000,
                                             0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000));
#ifdef __AVX512DQ__
    return simd_cf8(_mm512_xor_ps(x, mask));
#else
    return simd_cf8(_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(x), _mm512_castps_si512(mask))));
#endif
  }
  template <>
  inline simd_cd4 conj(const simd_cd4 &x) {
    const __m512d mask = _mm512_castsi512_pd(
       _mm512_set_epi32(0x80000000, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x0));
#ifdef __AVX512DQ__
    return simd_cd4(_mm512_xor_pd(x, mask));
#else
    return simd_cd4(_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castps_si512(x), _mm512_castpd_si512(mask))));
#endif
  }

  //Sqrt functions
  template <>
  inline simd_f16 sqrt(const simd_f16 &x) {
    return simd_f16(_mm512_sqrt_ps(x));
  }

  template <>
  inline simd_d8 sqrt(const simd_d8 &x) {
    return simd_d8(_mm512_sqrt_pd(x));
  }

  // Min functions
  template <>
  inline simd_i16 min(const simd_i16 &x, const simd_i16 &y) {
    return simd_i16(_mm512_min_epi32(x, y));
  }

  template <>
  inline simd_l8 min(const simd_l8 &x, const simd_l8 &y) {
    return simd_l8(_mm512_min_epi64(x, y));
  }

  template <>
  inline simd_f16 min(const simd_f16 &x, const simd_f16 &y) {
    return simd_f16(_mm512_min_ps(y, x));
  }

  template <>
  inline simd_d8 min(const simd_d8 &x, const simd_d8 &y) {
    return simd_d8(_mm512_min_pd(y, x));
  }

  //Max functions
  template <>
  inline simd_i16 max(const simd_i16 &x, const simd_i16 &y) {
    return simd_i16(_mm512_max_epi32(x, y));
  }

  template <>
  inline simd_l8 max(const simd_l8 &x, const simd_l8 &y) {
    return simd_l8(_mm512_max_epi64(x, y));
  }

  template <>
  inline simd_f16 max(const simd_f16 &x, const simd_f16 &y) {
    return simd_f16(_mm512_max_ps(y, x));
  }

  template <>
  inline simd_d8 max(const simd_d8 &x, const simd_d8 &y) {
    return simd_d8(_mm512_max_pd(y, x));
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
    __m128 lane0  = _mm512_extractf32x4_ps(x, 0);
    __m128 lane1  = _mm512_extractf32x4_ps(x, 1);
    __m128 lane2  = _mm512_extractf32x4_ps(x, 2);
    __m128 lane3  = _mm512_extractf32x4_ps(x, 3);
    simd_f4 min_v = min(min(simd_f4(lane0), simd_f4(lane1)), min(simd_f4(lane2), simd_f4(lane3)));
    return reduce_min(min_v);
  }

  template <>
  inline simd_d8::value_t reduce_min(const simd_d8 &x) {
    __m256d lane0 = _mm512_extractf64x4_pd(x, 0);
    __m256d lane1 = _mm512_extractf64x4_pd(x, 1);
    simd_d4 min_v = min(simd_d4(lane0), simd_d4(lane1));
    return reduce_min(min_v);
  }

  // Reduce Max
  template <>
  inline simd_i16::value_t reduce_max(const simd_i16 &x) {
    return _mm512_reduce_max_epi32(x);
  }

  template <>
  inline simd_l8::value_t reduce_max(const simd_l8 &x) {
    return _mm512_reduce_max_epi64(x);
  }

  template <>
  inline simd_f16::value_t reduce_max(const simd_f16 &x) {
    __m128 lane0  = _mm512_extractf32x4_ps(x, 0);
    __m128 lane1  = _mm512_extractf32x4_ps(x, 1);
    __m128 lane2  = _mm512_extractf32x4_ps(x, 2);
    __m128 lane3  = _mm512_extractf32x4_ps(x, 3);
    simd_f4 max_v = max(max(simd_f4(lane0), simd_f4(lane1)), max(simd_f4(lane2), simd_f4(lane3)));
    return reduce_max(max_v);
  }

  template <>
  inline simd_d8::value_t reduce_max(const simd_d8 &x) {
    __m256d lane0 = _mm512_extractf64x4_pd(x, 0);
    __m256d lane1 = _mm512_extractf64x4_pd(x, 1);
    simd_d4 max_v = max(simd_d4(lane0), simd_d4(lane1));
    return reduce_max(max_v);
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
    simd_f8 lo(_mm512_extractf32x8_ps(x, 0));
    simd_f8 hi(_mm512_extractf32x8_ps(x, 1));
    return reduce_sum(lo + hi);
#else
    simd_f4 lo_1(_mm512_extractf32x4_ps(x, 0));
    simd_f4 lo_2(_mm512_extractf32x4_ps(x, 1));
    simd_f4 hi_1(_mm512_extractf32x4_ps(x, 2));
    simd_f4 hi_2(_mm512_extractf32x4_ps(x, 3));
    return reduce_sum(lo_1 + lo_2 + hi_1 + hi_2);
#endif
  }

  template <>
  inline simd_d8::value_t reduce_sum(const simd_d8 &x) {
    simd_d4 lo(_mm512_extractf64x4_pd(x, 0));
    simd_d4 hi(_mm512_extractf64x4_pd(x, 1));
    return reduce_sum(lo + hi);
  }

  template <>
  inline simd_cf8::value_t reduce_sum(const simd_cf8 &x) {
#ifdef __AVX512DQ__
    simd_cf4 lo(_mm512_extractf32x8_ps(x, 0));
    simd_cf4 hi(_mm512_extractf32x8_ps(x, 1));
    return reduce_sum(lo + hi);
#else
    simd_cf4 lo(_mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(x), 0));
    simd_cf4 hi(_mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(x), 1));
      return reduce_sum(lo + hi)
#endif
  }

  template <>
  inline simd_cd4::value_t reduce_sum(const simd_cd4 &x) {
    simd_cd2 lo(_mm512_extractf64x4_pd(x, 0));
    simd_cd2 hi(_mm512_extractf64x4_pd(x, 1));
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
    simd_f4 lo_1(_mm512_extractf32x4_ps(x, 0));
    simd_f4 lo_2(_mm512_extractf32x4_ps(x, 1));
    simd_f4 hi_1(_mm512_extractf32x4_ps(x, 2));
    simd_f4 hi_2(_mm512_extractf32x4_ps(x, 3));
    return reduce_mul(lo_1 * lo_2 * hi_1 * hi_2);
  }

  template <>
  inline simd_d8::value_t reduce_mul(const simd_d8 &x) {
    simd_d4 lo(_mm512_extractf64x4_pd(x, 0));
    simd_d4 hi(_mm512_extractf64x4_pd(x, 1));
    return reduce_mul(lo * hi);
  }

  template <>
  inline simd_cf8::value_t reduce_mul(const simd_cf8 &x) {
#ifdef __AVX512DQ__
    simd_cf4 lo(_mm512_extractf32x8_ps(x, 0));
    simd_cf4 hi(_mm512_extractf32x8_ps(x, 1));
    return reduce_mul(lo * hi);
#else
    simd_cf4 lo(_mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(x), 0));
    simd_cf4 hi(_mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(x), 1));
    return reduce_mul(lo * hi)
#endif
  }

  template <>
  inline simd_cd4::value_t reduce_mul(const simd_cd4 &x) {
    simd_cd2 lo(_mm512_extractf64x4_pd(x, 0));
    simd_cd2 hi(_mm512_extractf64x4_pd(x, 1));
    return reduce_mul(lo * hi);
  }

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
    __m512 x_odd  = _mm512_movehdup_ps(x);
    __m512 x_even = _mm512_moveldup_ps(x);
    __m512 y_swap = _mm512_permute_ps(y, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512 result = _mm512_fmaddsub_ps(x_even, y, _mm512_fmaddsub_ps(x_odd, y_swap, z));
    return simd_cf8(result);
  }

  template <>
  inline simd_cd4 fma_add(const simd_cd4 &x, const simd_cd4 &y, const simd_cd4 &z) {
    __m512d x_odd  = _mm512_permute_pd(x, 0xFF);
    __m512d x_even = _mm512_movedup_pd(x);
    __m512d y_swap = _mm512_permute_pd(y, 0x55);
    __m512d result = _mm512_fmaddsub_pd(x_even, y, _mm512_fmaddsub_pd(x_odd, y_swap, z));
    return simd_cd4(result);
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
    __m512 x_odd  = _mm512_movehdup_ps(x);
    __m512 x_even = _mm512_moveldup_ps(x);
    __m512 y_swap = _mm512_permute_ps(y, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512 result = _mm512_fmaddsub_ps(x_even, y, _mm512_fmsubadd_ps(x_odd, y_swap, z));
    return simd_cf8(result);
  }

  template <>
  inline simd_cd4 fma_sub(const simd_cd4 &x, const simd_cd4 &y, const simd_cd4 &z) {
    __m512d x_odd  = _mm512_permute_pd(x, 0xFF);
    __m512d x_even = _mm512_movedup_pd(x);
    __m512d y_swap = _mm512_permute_pd(y, 0x55);
    __m512d result = _mm512_fmaddsub_pd(x_even, y, _mm512_fmsubadd_pd(x_odd, y_swap, z));
    return simd_cd4(result);
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
    __m512 x_odd         = _mm512_movehdup_ps(x);
    __m512 x_even        = _mm512_moveldup_ps(x);
    __m512 y_swap        = _mm512_permute_ps(y, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    simd_cf8 y_swap_conj = conj(simd_cf8(y_swap));
    __m512 result        = _mm512_fmsub_ps(x_odd, y_swap_conj, _mm512_fmsub_ps(x_even, y, z));
    return simd_cf8(result);
  }

  template <>
  inline simd_cd4 fma_nadd(const simd_cd4 &x, const simd_cd4 &y, const simd_cd4 &z) {
    __m512d x_odd        = _mm512_permute_pd(x, 0xFF);
    __m512d x_even       = _mm512_movedup_pd(x);
    __m512d y_swap       = _mm512_permute_pd(y, 0x55);
    simd_cd4 y_swap_conj = conj(simd_cd4(y_swap));
    __m512d result       = _mm512_fmsub_pd(x_odd, y_swap_conj, _mm512_fmsub_pd(x_even, y, z));
    return simd_cd4(result);
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
    __m512 x_odd         = _mm512_movehdup_ps(x);
    __m512 x_even        = _mm512_moveldup_ps(x);
    __m512 y_swap        = _mm512_permute_ps(y, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    simd_cf8 y_swap_conj = conj(simd_cf8(y_swap));
    __m512 result        = _mm512_fmsub_ps(x_odd, y_swap_conj, _mm512_fmadd_ps(x_even, y, z));
    return simd_cf8(result);
  }

  template <>
  inline simd_cd4 fma_nsub(const simd_cd4 &x, const simd_cd4 &y, const simd_cd4 &z) {
    __m512d x_odd        = _mm512_permute_pd(x, 0xFF);
    __m512d x_even       = _mm512_movedup_pd(x);
    __m512d y_swap       = _mm512_permute_pd(y, 0x55);
    simd_cd4 y_swap_conj = conj(simd_cd4(y_swap));
    __m512d result       = _mm512_fmsub_pd(x_odd, y_swap_conj, _mm512_fmadd_pd(x_even, y, z));
    return simd_cd4(result);
  }
  // FMA functions for integral types

  // FMA ADD
  template <>
  inline simd_i16 fma_add(const simd_i16 &x, const simd_i16 &y, const simd_i16 &z) {
    return x * y + z;
  }

  template <>
  inline simd_l8 fma_add(const simd_l8 &x, const simd_l8 &y, const simd_l8 &z) {
    return x * y + z;
  }

  // FMA SUB
  template <>
  inline simd_i16 fma_sub(const simd_i16 &x, const simd_i16 &y, const simd_i16 &z) {
    return x * y - z;
  }

  template <>
  inline simd_l8 fma_sub(const simd_l8 &x, const simd_l8 &y, const simd_l8 &z) {
    return x * y - z;
  }

  // FMA NADD
  template <>
  inline simd_i16 fma_nadd(const simd_i16 &x, const simd_i16 &y, const simd_i16 &z) {
    return z - (x * y);
  }

  template <>
  inline simd_l8 fma_nadd(const simd_l8 &x, const simd_l8 &y, const simd_l8 &z) {
    return z - (x * y);
  }

  // FMA NSUB
  template <>
  inline simd_i16 fma_nsub(const simd_i16 &x, const simd_i16 &y, const simd_i16 &z) {
    return -(x * y + z);
  }

  template <>
  inline simd_l8 fma_nsub(const simd_l8 &x, const simd_l8 &y, const simd_l8 &z) {
    return -(x * y + z);
  }

  //Gather functions.
  template <>
  inline simd_i16 gather(const simd_i16::value_t *from, const long stride) {
    simd_i16 simd_stride(static_cast<int32_t>(stride));
    const simd_i16 multiplier({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    simd_i16 vindex = simd_stride * multiplier;
    return simd_i16(_mm512_i32gather_epi32(vindex, from, sizeof(simd_i16::value_t)));
  }

  template <>
  inline simd_l8 gather(const simd_l8::value_t *from, const long stride) {
    simd_i8 simd_stride(stride);
    const simd_i8 multiplier({0, 1, 2, 3, 4, 5, 6, 7});
    simd_i8 vindex = simd_stride * multiplier;
    return simd_l8(_mm512_i32gather_epi64(vindex, from, sizeof(simd_l8::value_t)));
  }

  template <>
  inline simd_f16 gather(const simd_f16::value_t *from, const long stride) {
    simd_i16 simd_stride(static_cast<int32_t>(stride));
    const simd_i16 multiplier({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    simd_i16 vindex = simd_stride * multiplier;
    return simd_f16(_mm512_i32gather_ps(vindex, from, sizeof(simd_f16::value_t)));
  }

  template <>
  inline simd_d8 gather(const simd_d8::value_t *from, const long stride) {
    simd_i8 simd_stride(stride);
    const simd_i8 multiplier({0, 1, 2, 3, 4, 5, 6, 7});
    simd_i8 vindex = simd_stride * multiplier;
    return simd_d8(_mm512_i32gather_pd(vindex, from, sizeof(simd_d8::value_t)));
  }

  template <>
  inline simd_cf8 gather(const simd_cf8::value_t *from, const long stride) {
    return simd_cf8(_mm512_castpd_ps(gather<simd_d8>(reinterpret_cast<const simd_d8::value_t *>(from), stride)));
  }

  template <>
  inline simd_cd4 gather(const simd_cd4::value_t *from, const long stride) {
    simd_cd1 a, b, c, d;
    a.load_unaligned(from);
    b.load_unaligned(from + stride);
    c.load_unaligned(from + 2 * stride);
    d.load_unaligned(from + 3 * stride);
    __m256d ab = _mm256_insertf128_pd(_mm256_castpd128_pd256(a), b, 1);
    __m256d cd = _mm256_insertf128_pd(_mm256_castpd128_pd256(c), d, 1);
    return simd_cd4(_mm512_insertf64x4(_mm512_castpd256_pd512(ab), cd, 1));
  }

  //Scatter functions
  template <>
  inline void scatter(const simd_i16 &v, simd_i16::value_t *to, const long stride) {
    simd_i16 simd_stride(stride);
    const simd_i16 multiplier({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    simd_i16 vindex = simd_stride * multiplier;
    _mm512_i32scatter_epi32(to, vindex, v, sizeof(simd_i16::value_t));
  }

  template <>
  inline void scatter(const simd_l8 &v, simd_l8::value_t *to, const long stride) {
    simd_i8 simd_stride(stride);
    const simd_i8 multiplier({0, 1, 2, 3, 4, 5, 6, 7});
    simd_i8 vindex = simd_stride * multiplier;
    _mm512_i32scatter_epi64(to, vindex, v, sizeof(simd_l8::value_t));
  }

  template <>
  inline void scatter(const simd_f16 &v, simd_f16::value_t *to, const long stride) {
    simd_i16 simd_stride(stride);
    const simd_i16 multiplier({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    simd_i16 vindex = simd_stride * multiplier;
    _mm512_i32scatter_ps(to, vindex, v, sizeof(simd_f16::value_t));
  }

  template <>
  inline void scatter(const simd_d8 &v, simd_d8::value_t *to, const long stride) {
    simd_i8 simd_stride(stride);
    simd_i8 multiplier({0, 1, 2, 3, 4, 5, 6, 7});
    simd_i8 vindex = simd_stride * multiplier;
    _mm512_i32scatter_pd(to, vindex, v, sizeof(simd_d8::value_t));
  }

  template <>
  inline void scatter(const simd_cf8 &v, simd_cf8::value_t *to, const long stride) {
    simd_i8 simd_stride(stride);
    simd_i8 multiplier({0, 1, 2, 3, 4, 5, 6, 7});
    simd_i8 vindex = simd_stride * multiplier;
    _mm512_i32scatter_pd(reinterpret_cast<double *>(to), vindex, _mm512_castps_pd(v), sizeof(simd_d8::value_t));
  }

  template <>
  inline void scatter(const simd_cd4 &v, simd_cd4::value_t *to, const long stride) {
    simd_f16 x(_mm512_castpd_ps(v));
    double *p = reinterpret_cast<double *>(to);
    _mm_storeu_pd(p, _mm_castps_pd(_mm512_extractf32x4_ps(x, 0x0)));
    _mm_storeu_pd(p + 2 * stride, _mm_castps_pd(_mm512_extractf32x4_ps(x, 0x1)));
    _mm_storeu_pd(p + 4 * stride, _mm_castps_pd(_mm512_extractf32x4_ps(x, 0x2)));
    _mm_storeu_pd(p + 6 * stride, _mm_castps_pd(_mm512_extractf32x4_ps(x, 0x3)));
  }

} // namespace nda::simd
#endif