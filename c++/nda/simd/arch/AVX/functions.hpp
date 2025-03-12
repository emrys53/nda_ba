#pragma once
#ifdef __AVX2__
#include "./type.hpp"
#include "../functions_forward.hpp"
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
    return x;
  }

  template <>
  inline simd_l4 conj(const simd_l4 &x) {
    return x;
  }

  template <>
  inline simd_f8 conj(const simd_f8 &x) {
    return x;
  }

  template <>
  inline simd_d4 conj(const simd_d4 &x) {
    return x;
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

  // Min functions
  template <>
  inline simd_i8 min(const simd_i8 &x, const simd_i8 &y) {
#ifdef __AVX2__
    return simd_i8{_mm256_min_epi32(x, y)};
#else
    simd_i4 lo_x{_mm256_extractf128_si256(x, 0)};
    simd_i4 lo_y{_mm256_extractf128_si256(y, 0)};
    simd_i4 hi_x{_mm256_extractf128_si256(x, 1)};
    simd_i4 hi_y{_mm256_extractf128_si256(y, 1)};
    simd_i4 lo_min = min(lo_x, lo_y);
    simd_i4 hi_min = min(hi_x, hi_y);
    return simd_i8{_mm256_insertf128_si256(_mm256_castsi128_si256(lo_min), hi_min, 1)};
#endif
  }

  template <>
  inline simd_l4 min(const simd_l4 &x, const simd_l4 &y) {
    __m256i mask = _mm256_cmpgt_epi64(x, y);
    return simd_l4{_mm256_or_si256(_mm256_andnot_si256(mask, x), _mm256_and_si256(mask, y))};
  }

  template <>
  inline simd_f8 min(const simd_f8 &x, const simd_f8 &y) {
    return simd_f8(_mm256_min_ps(y, x));
  }

  template <>
  inline simd_d4 min(const simd_d4 &x, const simd_d4 &y) {
    return simd_d4{_mm256_min_pd(y, x)};
  }

  //Max functions
  template <>
  inline simd_i8 max(const simd_i8 &x, const simd_i8 &y) {
#ifdef __AVX2__
    return simd_i8{_mm256_max_epi32(x, y)};
#else
    simd_i4 lo_x{_mm256_extractf128_si256(x, 0)};
    simd_i4 lo_y{_mm256_extractf128_si256(y, 0)};
    simd_i4 hi_x{_mm256_extractf128_si256(x, 1)};
    simd_i4 hi_y{_mm256_extractf128_si256(y, 1)};
    simd_i4 lo_max = max(lo_x, lo_y);
    simd_i4 hi_max = max(hi_x, hi_y);
    return simd_i8{_mm256_insertf128_si256(_mm256_castsi128_si256(lo_max), hi_max, 1)};
#endif
  }

  template <>
  inline simd_l4 max(const simd_l4 &x, const simd_l4 &y) {
    __m256i mask = _mm256_cmpgt_epi64(x, y);
    return simd_l4{_mm256_or_si256(_mm256_and_si256(mask, x), _mm256_andnot_si256(mask, y))};
  }

  template <>
  inline simd_f8 max(const simd_f8 &x, const simd_f8 &y) {
    return simd_f8{_mm256_max_ps(y, x)};
  }

  template <>
  inline simd_d4 max(const simd_d4 &x, const simd_d4 &y) {
    return simd_d4{_mm256_max_pd(y, x)};
  }

  // Reduce Min
  template <>
  inline simd_i8::value_t reduce_min(const simd_i8 &x) {
    simd_i4 lo_x{_mm256_extractf128_si256(x, 0)};
    simd_i4 hi_x{_mm256_extractf128_si256(x, 1)};
    simd_i8::value_t lo_min = reduce_min(lo_x);
    simd_i8::value_t hi_min = reduce_min(hi_x);
    return lo_min < hi_min ? lo_min : hi_min;
  }

  template <>
  inline simd_l4::value_t reduce_min(const simd_l4 &x) {
    alignas(simd_l4::alignment()) std::array<int64_t, 4> a;
    x.store(a.data());
    simd_l4::value_t lo_min = a[0] < a[1] ? a[0] : a[1];
    simd_l4::value_t hi_min = a[2] < a[3] ? a[2] : a[3];
    return lo_min < hi_min ? lo_min : hi_min;
  }

  template <>
  inline simd_f8::value_t reduce_min(const simd_f8 &x) {
    simd_f8 m_04_15_26_37 = min(x, simd_f8{_mm256_permute2f128_ps(x, x, 0x1)});
    simd_f8 m_0437_1526   = min(m_04_15_26_37, simd_f8{_mm256_shuffle_ps(m_04_15_26_37, m_04_15_26_37, NDA_SHUFFLE_MASK4(3, 2, 1, 0))});
    simd_f8 m_04371526    = min(m_0437_1526, simd_f8(_mm256_shuffle_ps(m_0437_1526, m_0437_1526, 0x1)));
    return _mm256_cvtss_f32(m_04371526);
  }

  template <>
  inline simd_d4::value_t reduce_min(const simd_d4 &x) {
    simd_d4 m_02_13 = min(x, simd_d4{_mm256_permute2f128_pd(x, x, 0x1)});
    simd_d4 m_0213  = min(m_02_13, simd_d4{_mm256_shuffle_pd(m_02_13, m_02_13, 0x1)});
    return _mm256_cvtsd_f64(m_0213);
  }

  // Reduce Max
  template <>
  inline simd_i8::value_t reduce_max(const simd_i8 &x) {
    simd_i4 lo_x{_mm256_extractf128_si256(x, 0)};
    simd_i4 hi_x{_mm256_extractf128_si256(x, 1)};
    simd_i8::value_t lo_max = reduce_max(lo_x);
    simd_i8::value_t hi_max = reduce_max(hi_x);
    return lo_max < hi_max ? hi_max : lo_max;
  }

  template <>
  inline simd_l4::value_t reduce_max(const simd_l4 &x) {
    alignas(simd_l4::alignment()) std::array<int64_t, 4> a;
    x.store(a.data());
    simd_l4::value_t lo_max = a[0] < a[1] ? a[1] : a[0];
    simd_l4::value_t hi_max = a[2] < a[3] ? a[3] : a[2];
    return lo_max < hi_max ? hi_max : lo_max;
  }

  template <>
  inline simd_f8::value_t reduce_max(const simd_f8 &x) {
    simd_f8 m_04_15_26_37 = max(x, simd_f8{_mm256_permute2f128_ps(x, x, 0x1)});
    simd_f8 m_0437_1526   = max(m_04_15_26_37, simd_f8{_mm256_shuffle_ps(m_04_15_26_37, m_04_15_26_37, NDA_SHUFFLE_MASK4(3, 2, 1, 0))});
    simd_f8 m_04371526    = max(m_0437_1526, simd_f8(_mm256_shuffle_ps(m_0437_1526, m_0437_1526, 0x1)));
    return _mm256_cvtss_f32(m_04371526);
  }

  template <>
  inline simd_d4::value_t reduce_max(const simd_d4 &x) {
    simd_d4 m_02_13 = max(x, simd_d4{_mm256_permute2f128_pd(x, x, 0x1)});
    simd_d4 m_0213  = max(m_02_13, simd_d4{_mm256_shuffle_pd(m_02_13, m_02_13, 0x1)});
    return _mm256_cvtsd_f64(m_0213);
  }

  //Reduce_sum
  template <>
  inline simd_i8::value_t reduce_sum(const simd_i8 &x) {
    simd_i4 lo{_mm256_extractf128_si256(x, 0)};
    simd_i4 hi{_mm256_extractf128_si256(x, 1)};
    return reduce_sum(lo + hi);
  }

  template <>
  inline simd_l4::value_t reduce_sum(const simd_l4 &x) {
    __m128i r = _mm_add_epi64(_mm256_extractf128_si256(x, 0), _mm256_extractf128_si256(x, 1));
    return _mm_extract_epi64(r, 0) + _mm_extract_epi64(r, 1);
  }

  template <>
  inline simd_f8::value_t reduce_sum(const simd_f8 &x) {
    simd_f4 lo{_mm256_extractf128_ps(x, 0)};
    simd_f4 hi{_mm256_extractf128_ps(x, 1)};
    return reduce_sum(lo + hi);
  }

  template <>
  inline simd_d4::value_t reduce_sum(const simd_d4 &x) {
    simd_d2 lo{_mm256_extractf128_pd(x, 0)};
    simd_d2 hi{_mm256_extractf128_pd(x, 1)};
    return reduce_sum(lo + hi);
  }

  template <>
  inline simd_cf4::value_t reduce_sum(const simd_cf4 &x) {
    simd_cf2 lo{_mm256_extractf128_ps(x, 0)};
    simd_cf2 hi{_mm256_extractf128_ps(x, 1)};
    return reduce_sum(lo + hi);
  }

  template <>
  inline simd_cd2::value_t reduce_sum(const simd_cd2 &x) {
    simd_cd1 lo{_mm256_extractf128_pd(x, 0)};
    simd_cd1 hi{_mm256_extractf128_pd(x, 1)};
    return reduce_sum(lo + hi);
  }

  //Reduce_mul
  template <>
  inline simd_i8::value_t reduce_mul(const simd_i8 &x) {
    simd_i4 lo{_mm256_extractf128_si256(x, 0)};
    simd_i4 hi{_mm256_extractf128_si256(x, 1)};
    return reduce_mul(lo * hi);
  }

  template <>
  inline simd_l4::value_t reduce_mul(const simd_l4 &x) {
    simd_l2 lo{_mm256_extractf128_si256(x, 0)};
    simd_l2 hi{_mm256_extractf128_si256(x, 1)};
    return reduce_mul(lo * hi);
  }

  template <>
  inline simd_f8::value_t reduce_mul(const simd_f8 &x) {
    __m256 m_04_15_26_37 = _mm256_mul_ps(x, _mm256_permute2f128_ps(x, x, 0x1));
    __m256 m_0437_1526   = _mm256_mul_ps(m_04_15_26_37, _mm256_shuffle_ps(m_04_15_26_37, m_04_15_26_37, NDA_SHUFFLE_MASK4(3, 2, 1, 0)));
    __m256 m_04371526    = _mm256_mul_ps(m_0437_1526, _mm256_shuffle_ps(m_0437_1526, m_0437_1526, 0x1));
    return _mm256_cvtss_f32(m_04371526);
  }

  template <>
  inline simd_d4::value_t reduce_mul(const simd_d4 &x) {
    __m256d m_02_13 = _mm256_mul_pd(x, _mm256_permute2f128_pd(x, x, 0x1));
    __m256d m_0213  = _mm256_mul_pd(m_02_13, _mm256_shuffle_pd(m_02_13, m_02_13, 0x1));
    return _mm256_cvtsd_f64(m_0213);
  }

  template <>
  inline simd_cf4::value_t reduce_mul(const simd_cf4 &x) {
    simd_cf2 lo{_mm256_extractf128_ps(x, 0)};
    simd_cf2 hi{_mm256_extractf128_ps(x, 1)};
    return reduce_mul(lo * hi);
  }

  template <>
  inline simd_cd2::value_t reduce_mul(const simd_cd2 &x) {
    simd_cd1 lo{_mm256_extractf128_pd(x, 0)};
    simd_cd1 hi{_mm256_extractf128_pd(x, 1)};
    return reduce_mul(lo * hi);
  }

//See Eigen library for implementation
#ifdef __FMA__
  // FMA ADD
  template <>
  inline simd_f8 fma_add(const simd_f8 &x, const simd_f8 &y, const simd_f8 &z) {
    return simd_f8(_mm256_fmadd_ps(x, y, z));
  }

  template <>
  inline simd_d4 fma_add(const simd_d4 &x, const simd_d4 &y, const simd_d4 &z) {
    return simd_d4(_mm256_fmadd_pd(x, y, z));
  }

  template <>
  inline simd_cf4 fma_add(const simd_cf4 &x, const simd_cf4 &y, const simd_cf4 &z) {
    __m256 x_odd  = _mm256_movehdup_ps(x);
    __m256 x_even = _mm256_moveldup_ps(x);
    __m256 y_swap = _mm256_permute_ps(y, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m256 result = _mm256_fmaddsub_ps(x_even, y, _mm256_fmaddsub_ps(x_odd, y_swap, z));
    return simd_cf4(result);
  }

  template <>
  inline simd_cd2 fma_add(const simd_cd2 &x, const simd_cd2 &y, const simd_cd2 &z) {
    __m256d x_odd  = _mm256_permute_pd(x, 0xF);
    __m256d x_even = _mm256_movedup_pd(x);
    __m256d y_swap = _mm256_permute_pd(y, 0x5);
    __m256d result = _mm256_fmaddsub_pd(x_even, y, _mm256_fmaddsub_pd(x_odd, y_swap, z));
    return simd_cd2(result);
  }

  //FMA_SUB
  template <>
  inline simd_f8 fma_sub(const simd_f8 &x, const simd_f8 &y, const simd_f8 &z) {
    return simd_f8(_mm256_fmsub_ps(x, y, z));
  }

  template <>
  inline simd_d4 fma_sub(const simd_d4 &x, const simd_d4 &y, const simd_d4 &z) {
    return simd_d4(_mm256_fmsub_pd(x, y, z));
  }

  template <>
  inline simd_cf4 fma_sub(const simd_cf4 &x, const simd_cf4 &y, const simd_cf4 &z) {
    __m256 x_odd  = _mm256_movehdup_ps(x);
    __m256 x_even = _mm256_moveldup_ps(x);
    __m256 y_swap = _mm256_permute_ps(y, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m256 result = _mm256_fmaddsub_ps(x_even, y, _mm256_fmsubadd_ps(x_odd, y_swap, z));
    return simd_cf4(result);
  }

  template <>
  inline simd_cd2 fma_sub(const simd_cd2 &x, const simd_cd2 &y, const simd_cd2 &z) {
    __m256d x_odd  = _mm256_permute_pd(x, 0xF);
    __m256d x_even = _mm256_movedup_pd(x);
    __m256d y_swap = _mm256_permute_pd(y, 0x5);
    __m256d result = _mm256_fmaddsub_pd(x_even, y, _mm256_fmsubadd_pd(x_odd, y_swap, z));
    return simd_cd2(result);
  }

  //FMA_NADD
  template <>
  inline simd_f8 fma_nadd(const simd_f8 &x, const simd_f8 &y, const simd_f8 &z) {
    return simd_f8(_mm256_fnmadd_ps(x, y, z));
  }

  template <>
  inline simd_d4 fma_nadd(const simd_d4 &x, const simd_d4 &y, const simd_d4 &z) {
    return simd_d4(_mm256_fnmadd_pd(x, y, z));
  }

  template <>
  inline simd_cf4 fma_nadd(const simd_cf4 &x, const simd_cf4 &y, const simd_cf4 &z) {
    __m256 x_odd         = _mm256_movehdup_ps(x);
    __m256 x_even        = _mm256_moveldup_ps(x);
    __m256 y_swap        = _mm256_permute_ps(y, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    simd_cf4 y_swap_conj = conj(simd_cf4(y_swap));
    __m256 result        = _mm256_fmsub_ps(x_odd, y_swap_conj, _mm256_fmsub_ps(x_even, y, z));
    return simd_cf4(result);
  }

  template <>
  inline simd_cd2 fma_nadd(const simd_cd2 &x, const simd_cd2 &y, const simd_cd2 &z) {
    __m256d x_odd        = _mm256_permute_pd(x, 0xF);
    __m256d x_even       = _mm256_movedup_pd(x);
    __m256d y_swap       = _mm256_permute_pd(y, 0x5);
    simd_cd2 y_swap_conj = conj(simd_cd2(y_swap));
    __m256d result       = _mm256_fmsub_pd(x_odd, y_swap_conj, _mm256_fmsub_pd(x_even, y, z));
    return simd_cd2(result);
  }

  //FMA_NSUB
  template <>
  inline simd_f8 fma_nsub(const simd_f8 &x, const simd_f8 &y, const simd_f8 &z) {
    return simd_f8(_mm256_fnmsub_ps(x, y, z));
  }

  template <>
  inline simd_d4 fma_nsub(const simd_d4 &x, const simd_d4 &y, const simd_d4 &z) {
    return simd_d4(_mm256_fnmsub_pd(x, y, z));
  }

  template <>
  inline simd_cf4 fma_nsub(const simd_cf4 &x, const simd_cf4 &y, const simd_cf4 &z) {
    __m256 x_odd         = _mm256_movehdup_ps(x);
    __m256 x_even        = _mm256_moveldup_ps(x);
    __m256 y_swap        = _mm256_permute_ps(y, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    simd_cf4 y_swap_conj = conj(simd_cf4(y_swap));
    __m256 result        = _mm256_fmsub_ps(x_odd, y_swap_conj, _mm256_fmadd_ps(x_even, y, z));
    return simd_cf4(result);
  }

  template <>
  inline simd_cd2 fma_nsub(const simd_cd2 &x, const simd_cd2 &y, const simd_cd2 &z) {
    __m256d x_odd        = _mm256_permute_pd(x, 0xF);
    __m256d x_even       = _mm256_movedup_pd(x);
    __m256d y_swap       = _mm256_permute_pd(y, 0x5);
    simd_cd2 y_swap_conj = conj(simd_cd2(y_swap));
    __m256d result       = _mm256_fmsub_pd(x_odd, y_swap_conj, _mm256_fmadd_pd(x_even, y, z));
    return simd_cd2(result);
  }
#else
  // FMA ADD
  template <>
  inline simd_f8 fma_add(const simd_f8 &x, const simd_f8 &y, const simd_f8 &z) {
    return x * y + z;
  }

  template <>
  inline simd_d4 fma_add(const simd_d4 &x, const simd_d4 &y, const simd_d4 &z) {
    return x * y + z;
  }

  template <>
  inline simd_cf4 fma_add(const simd_cf4 &x, const simd_cf4 &y, const simd_cf4 &z) {
    return x * y + z;
  }

  template <>
  inline simd_cd2 fma_add(const simd_cd2 &x, const simd_cd2 &y, const simd_cd2 &z) {
    return x * y + z;
  }

  //FMA_SUB
  template <>
  inline simd_f8 fma_sub(const simd_f8 &x, const simd_f8 &y, const simd_f8 &z) {
    return x * y - z;
  }

  template <>
  inline simd_d4 fma_sub(const simd_d4 &x, const simd_d4 &y, const simd_d4 &z) {
    return x * y - z;
  }

  template <>
  inline simd_cf4 fma_sub(const simd_cf4 &x, const simd_cf4 &y, const simd_cf4 &z) {
    return x * y - z;
  }

  template <>
  inline simd_cd2 fma_sub(const simd_cd2 &x, const simd_cd2 &y, const simd_cd2 &z) {
    return x * y - z;
  }

  //FMA_NADD
  template <>
  inline simd_f8 fma_nadd(const simd_f8 &x, const simd_f8 &y, const simd_f8 &z) {
    return z - (x * y);
  }

  template <>
  inline simd_d4 fma_nadd(const simd_d4 &x, const simd_d4 &y, const simd_d4 &z) {
    return z - (x * y);
  }

  template <>
  inline simd_cf4 fma_nadd(const simd_cf4 &x, const simd_cf4 &y, const simd_cf4 &z) {
    return z - (x * y);
  }

  template <>
  inline simd_cd2 fma_nadd(const simd_cd2 &x, const simd_cd2 &y, const simd_cd2 &z) {
    return z - (x * y);
  }

  //FMA_NSUB
  template <>
  inline simd_f8 fma_nsub(const simd_f8 &x, const simd_f8 &y, const simd_f8 &z) {
    return -(x * y + z);
  }

  template <>
  inline simd_d4 fma_nsub(const simd_d4 &x, const simd_d4 &y, const simd_d4 &z) {
    return -(x * y + z);
  }

  template <>
  inline simd_cf4 fma_nsub(const simd_cf4 &x, const simd_cf4 &y, const simd_cf4 &z) {
    return -(x * y + z);
  }

  template <>
  inline simd_cd2 fma_nsub(const simd_cd2 &x, const simd_cd2 &y, const simd_cd2 &z) {
    return -(x * y + z);
  }
#endif
  // FMA simulation for integral types.
  // FMA ADD
  template <>
  inline simd_i8 fma_add(const simd_i8 &x, const simd_i8 &y, const simd_i8 &z) {
    return x * y + z;
  }

  template <>
  inline simd_l4 fma_add(const simd_l4 &x, const simd_l4 &y, const simd_l4 &z) {
    return x * y + z;
  }

  // FMA SUB
  template <>
  inline simd_i8 fma_sub(const simd_i8 &x, const simd_i8 &y, const simd_i8 &z) {
    return x * y - z;
  }

  template <>
  inline simd_l4 fma_sub(const simd_l4 &x, const simd_l4 &y, const simd_l4 &z) {
    return x * y - z;
  }

  // FMA NADD
  template <>
  inline simd_i8 fma_nadd(const simd_i8 &x, const simd_i8 &y, const simd_i8 &z) {
    return z - (x * y);
  }

  template <>
  inline simd_l4 fma_nadd(const simd_l4 &x, const simd_l4 &y, const simd_l4 &z) {
    return z - (x * y);
  }

  // FMA NSUB
  template <>
  inline simd_i8 fma_nsub(const simd_i8 &x, const simd_i8 &y, const simd_i8 &z) {
    return -(x * y + z);
  }

  template <>
  inline simd_l4 fma_nsub(const simd_l4 &x, const simd_l4 &y, const simd_l4 &z) {
    return -(x * y + z);
  }

  //Gather functions.
  template <>
  inline simd_i8 gather(const simd_i8::value_t *from, const long stride) {
    simd_i8 simd_stride(static_cast<int32_t>(stride));
    const simd_i8 multiplier({0, 1, 2, 3, 4, 5, 6, 7});
    simd_i8 vindex = simd_stride * multiplier;
    return simd_i8(_mm256_i32gather_epi32(from, vindex, sizeof(simd_i8::value_t)));
  }

  template <>
  inline simd_l4 gather(const simd_l4::value_t *from, const long stride) {
    simd_l4 simd_stride(stride);
    const simd_l4 multiplier({0, 1, 2, 3});
    simd_l4 vindex = simd_stride * multiplier;
    return simd_l4(_mm256_i64gather_epi64(reinterpret_cast<const long long int *>(from), vindex, sizeof(simd_l4::value_t)));
  }

  template <>
  inline simd_f8 gather(const simd_f8::value_t *from, const long stride) {
    simd_i8 simd_stride(static_cast<int32_t>(stride));
    const simd_i8 multiplier({0, 1, 2, 3, 4, 5, 6, 7});
    simd_i8 vindex = simd_stride * multiplier;
    return simd_f8(_mm256_i32gather_ps(from, vindex, sizeof(simd_f8::value_t)));
  }

  template <>
  inline simd_d4 gather(const simd_d4::value_t *from, const long stride) {
    simd_l4 simd_stride(stride);
    const simd_l4 multiplier({0, 1, 2, 3});
    simd_l4 vindex = simd_stride * multiplier;
    return simd_d4(_mm256_i64gather_pd(from, vindex, sizeof(simd_d4::value_t)));
  }

  template <>
  inline simd_cf4 gather(const simd_cf4::value_t *from, const long stride) {
    return simd_cf4(_mm256_castpd_ps(gather<simd_d4>(reinterpret_cast<const simd_d4::value_t *>(from), stride)));
  }

  template <>
  inline simd_cd2 gather(const simd_cd2::value_t *from, const long stride) {
    return simd_cd2(_mm256_set_pd(from[stride].imag(), from[stride].real(), from[0].imag(), from[0].real()));
  }
  //Scatter functions:

  template <>
  inline void scatter(const simd_i8 &v, simd_i8::value_t *to, const long stride) {
    __m128i low    = _mm256_extractf128_si256(v, 0);
    __m128i high   = _mm256_extractf128_si256(v, 1);
    to[0]          = _mm_extract_epi32(low, 0);
    to[stride]     = _mm_extract_epi32(low, 1);
    to[2 * stride] = _mm_extract_epi32(low, 2);
    to[3 * stride] = _mm_extract_epi32(low, 3);
    to[4 * stride] = _mm_extract_epi32(high, 0);
    to[5 * stride] = _mm_extract_epi32(high, 1);
    to[6 * stride] = _mm_extract_epi32(high, 2);
    to[7 * stride] = _mm_extract_epi32(high, 3);
  }

  template <>
  inline void scatter(const simd_l4 &v, simd_l4::value_t *to, const long stride) {
    __m128i low    = _mm256_extractf128_si256(v, 0);
    __m128i high   = _mm256_extractf128_si256(v, 1);
    to[0]          = _mm_extract_epi64(low, 0);
    to[stride]     = _mm_extract_epi64(low, 1);
    to[2 * stride] = _mm_extract_epi64(high, 0);
    to[3 * stride] = _mm_extract_epi64(high, 1);
  }

  template <>
  inline void scatter(const simd_f8 &v, simd_f8::value_t *to, const long stride) {
    __m128 low     = _mm256_extractf128_ps(v, 0);
    __m128 high    = _mm256_extractf128_ps(v, 1);
    to[0]          = _mm_cvtss_f32(low);
    to[stride]     = _mm_cvtss_f32(_mm_shuffle_ps(low, low, 0x1));
    to[2 * stride] = _mm_cvtss_f32(_mm_shuffle_ps(low, low, 0x2));
    to[3 * stride] = _mm_cvtss_f32(_mm_shuffle_ps(low, low, 0x3));
    to[4 * stride] = _mm_cvtss_f32(high);
    to[5 * stride] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 0x1));
    to[6 * stride] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 0x2));
    to[7 * stride] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 0x3));
  }

  template <>
  inline void scatter(const simd_d4 &v, simd_d4::value_t *to, const long stride) {
    __m128d low    = _mm256_extractf128_pd(v, 0);
    __m128d high   = _mm256_extractf128_pd(v, 1);
    to[0]          = _mm_cvtsd_f64(low);
    to[stride]     = _mm_cvtsd_f64(_mm_shuffle_pd(low, low, 0x1));
    to[2 * stride] = _mm_cvtsd_f64(high);
    to[3 * stride] = _mm_cvtsd_f64(_mm_shuffle_pd(high, high, 0x1));
  }

  template <>
  inline void scatter(const simd_cf4 &v, simd_cf4::value_t *to, const long stride) {
    __m128 low     = _mm256_extractf128_ps(v, 0);
    __m128 high    = _mm256_extractf128_ps(v, 1);
    to[0]          = simd_cf4::value_t(_mm_cvtss_f32(low), _mm_cvtss_f32(_mm_shuffle_ps(low, low, 0x1)));
    to[stride]     = simd_cf4::value_t(_mm_cvtss_f32(_mm_shuffle_ps(low, low, 0x2)), _mm_cvtss_f32(_mm_shuffle_ps(low, low, 0x3)));
    to[2 * stride] = simd_cf4::value_t(_mm_cvtss_f32(high), _mm_cvtss_f32(_mm_shuffle_ps(high, high, 0x1)));
    to[3 * stride] = simd_cf4::value_t(_mm_cvtss_f32(_mm_shuffle_ps(high, high, 0x2)), _mm_cvtss_f32(_mm_shuffle_ps(high, high, 0x3)));
  }

  template <>
  inline void scatter(const simd_cd2 &v, simd_cd2::value_t *to, const long stride) {
    __m128d low  = _mm256_extractf128_pd(v, 0);
    __m128d high = _mm256_extractf128_pd(v, 1);
    to[0]        = simd_cd2::value_t(_mm_cvtsd_f64(low), _mm_cvtsd_f64(_mm_shuffle_pd(low, low, 0x1)));
    to[stride]   = simd_cd2::value_t(_mm_cvtsd_f64(high), _mm_cvtsd_f64(_mm_shuffle_pd(high, high, 0x1)));
  }

} // namespace nda::simd
#endif