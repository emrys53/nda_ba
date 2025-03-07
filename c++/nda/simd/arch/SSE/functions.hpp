#pragma once
#ifdef __SSE2__

#include "./type.hpp"
#include "../macros.hpp"
#include "../functions_forward.hpp"
#include "../../../macros.hpp"
#include "immintrin.h"

#include <concepts>

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

  // Min functions
  template <>
  inline simd_i4 min(const simd_i4 &x, const simd_i4 &y) {
#ifdef __SSE4_1__
    return simd_i4{_mm_min_epi32(x, y)};
#else
    simd_i4 mask{_mm_cmplt_epi32(x, y)};
    return simd_i4{_mm_or_si128(_mm_and_si128(mask, x), _mm_andnot_si128(mask, y))};
#endif
  }

  template <>
  inline simd_l2 min(const simd_l2 &x, const simd_l2 &y) {
    //TODO: Do Vectorized version in the future.
    alignas(simd_l2::alignment()) std::array<int64_t, 2> a, b, c;
    x.store(a.data());
    y.store(b.data());
    c[0] = std::min(a[0], b[0]);
    c[1] = std::min(a[1], b[1]);
    return simd_l2{c.data()};
  }

  template <>
  inline simd_f4 min(const simd_f4 &x, const simd_f4 &y) {
    return simd_f4{_mm_min_ps(y, x)};
  }

  template <>
  inline simd_d2 min(const simd_d2 &x, const simd_d2 &y) {
    return simd_d2{_mm_min_pd(y, x)};
  }

  //Max functions
  template <>
  inline simd_i4 max(const simd_i4 &x, const simd_i4 &y) {
#ifdef __SSE4_1__
    return simd_i4{_mm_max_epi32(x, y)};
#else
    simd_i4 mask{_mm_cmpgt_epi32(x, y)};
    return simd_i4{_mm_or_si128(_mm_and_si128(mask, x), _mm_andnot_si128(mask, y))};
#endif
  }

  template <>
  inline simd_l2 max(const simd_l2 &x, const simd_l2 &y) {
    //TODO: Do Vectorized version in the future.
    alignas(simd_l2::alignment()) std::array<int64_t, 2> a, b, c;
    x.store(a.data());
    y.store(b.data());
    c[0] = std::max(a[0], b[0]);
    c[1] = std::max(a[1], b[1]);
    return simd_l2{c.data()};
  }

  template <>
  inline simd_f4 max(const simd_f4 &x, const simd_f4 &y) {
    return simd_f4{_mm_max_ps(y, x)};
  }

  template <>
  inline simd_d2 max(const simd_d2 &x, const simd_d2 &y) {
    return simd_d2{_mm_max_pd(y, x)};
  }

  // Reduce Min
  template <>
  inline simd_i4::value_t reduce_min(const simd_i4 &x) {
    simd_i4 m_3{_mm_shuffle_epi32(x, NDA_SHUFFLE_MASK4(3, 2, 1, 0))};
    simd_i4 m_03   = min(x, m_3);
    simd_i4 m_0321 = min(m_03, simd_i4{_mm_shuffle_epi32(m_03, 0x1)});
    return _mm_cvtsi128_si32(m_0321);
  }

  template <>
  inline simd_l2::value_t reduce_min(const simd_l2 &x) {
    alignas(simd_l2::alignment()) std::array<int64_t, 2> a;
    x.store(a.data());
    return a[0] < a[1] ? a[0] : a[1];
  }

  template <>
  inline simd_f4::value_t reduce_min(const simd_f4 &x) {
    simd_f4 m_2{_mm_movehl_ps(x, x)};
    simd_f4 m_02   = min(x, m_2);
    simd_f4 m_0231 = min(m_02, simd_f4{_mm_shuffle_ps(m_02, m_02, 1)});
    return _mm_cvtss_f32(m_0231);
  }

  template <>
  inline simd_d2::value_t reduce_min(const simd_d2 &x) {
    return _mm_cvtsd_f64(min(x, simd_d2{_mm_unpackhi_pd(x, x)}));
  }

  // Reduce Max
  template <>
  inline simd_i4::value_t reduce_max(const simd_i4 &x) {
    simd_i4 m_3{_mm_shuffle_epi32(x, NDA_SHUFFLE_MASK4(3, 2, 1, 0))};
    simd_i4 m_03   = max(x, m_3);
    simd_i4 m_0321 = max(m_03, simd_i4{_mm_shuffle_epi32(m_03, 0x1)});
    return _mm_cvtsi128_si32(m_0321);
  }

  template <>
  inline simd_l2::value_t reduce_max(const simd_l2 &x) {
    alignas(simd_l2::alignment()) std::array<int64_t, 2> a;
    x.store(a.data());
    return a[0] < a[1] ? a[1] : a[0];
  }

  template <>
  inline simd_f4::value_t reduce_max(const simd_f4 &x) {
    simd_f4 m_2{_mm_movehl_ps(x, x)};
    simd_f4 m_02   = max(x, m_2);
    simd_f4 m_0231 = max(m_02, simd_f4{_mm_shuffle_ps(m_02, m_02, 1)});
    return _mm_cvtss_f32(m_0231);
  }

  template <>
  inline simd_d2::value_t reduce_max(const simd_d2 &x) {
    return _mm_cvtsd_f64(max(x, simd_d2{_mm_unpackhi_pd(x, x)}));
  }

  //Reduce_sum
  template <>
  inline simd_i4::value_t reduce_sum(const simd_i4 &x) {
#ifndef __SSE3__
    __m128i m_01_23 = _mm_hadd_epi32(x, x);
    __m128i m_0123  = _mm_hadd_epi32(m_01_23, m_01_23);
    return _mm_cvtsi128_si32(m_0123);
#else
    __m128i m_02_13 = _mm_add_epi32(x, _mm_unpackhi_epi64(x, x));
    int32_t m_02    = _mm_cvtsi128_si32(m_02_13);
    int32_t m_13    = _mm_cvtsi128_si32(_mm_shuffle_epi32(m_02_13, 0x1));
    return m_02 + m_13;
#endif
  }

  template <>
  inline simd_l2::value_t reduce_sum(const simd_l2 &x) {
    return _mm_cvtsi128_si64(_mm_add_epi64(x, _mm_unpackhi_epi64(x, x)));
  }

  template <>
  inline simd_f4::value_t reduce_sum(const simd_f4 &x) {
    __m128 m_02_13 = _mm_add_ps(x, _mm_movehl_ps(x, x));
    __m128 m_0213  = _mm_add_ps(m_02_13, _mm_shuffle_ps(m_02_13, m_02_13, 0x1));
    return _mm_cvtss_f32(m_0213);
  }

  template <>
  inline simd_d2::value_t reduce_sum(const simd_d2 &x) {
    return _mm_cvtsd_f64(_mm_add_pd(x, _mm_unpackhi_pd(x, x)));
  }

  template <>
  inline simd_cf2::value_t reduce_sum(const simd_cf2 &x) {
    __m128 m_01 = _mm_add_ps(x, _mm_movehl_ps(x, x));
    // Now get the std::complex<float> from m_01
    alignas(alignof(__m64)) std::complex<float> res;
    _mm_storel_pi(reinterpret_cast<__m64 *>(&res), m_01);
    return res;
  }

  template <>
  inline simd_cd1::value_t reduce_sum(const simd_cd1 &x) {
    alignas(simd_cd1::alignment()) std::array<simd_cd1::scalar_t, 2> res;
    x.store(res.data());
    return simd_cd1::value_t{res[0], res[1]};
  }

  //Reduce_mul
  template <>
  inline simd_i4::value_t reduce_mul(const simd_i4 &x) {
    alignas(simd_i4::alignment()) std::array<simd_i4::value_t, 4> a;
    x.store(a.data());
    return a[0] * a[1] * a[2] * a[3];
  }

  template <>
  inline simd_l2::value_t reduce_mul(const simd_l2 &x) {
    alignas(simd_l2::alignment()) std::array<simd_l2::value_t, 2> a;
    x.store(a.data());
    return a[0] * a[1];
  }

  template <>
  inline simd_f4::value_t reduce_mul(const simd_f4 &x) {
    __m128 m_02_13 = _mm_mul_ps(x, _mm_movehl_ps(x, x));
    __m128 m_0213  = _mm_mul_ps(m_02_13, _mm_shuffle_ps(m_02_13, m_02_13, 0x1));
    return _mm_cvtss_f32(m_0213);
  }

  template <>
  inline simd_d2::value_t reduce_mul(const simd_d2 &x) {
    return _mm_cvtsd_f64(_mm_mul_pd(x, _mm_unpackhi_pd(x, x)));
  }

  template <>
  inline simd_cf2::value_t reduce_mul(const simd_cf2 &x) {
    simd_cf2 m_13{_mm_movehl_ps(x, x)};
    simd_cf2 tmp = x * m_13;
    alignas(alignof(__m64)) std::complex<float> res;
    _mm_storel_pi(reinterpret_cast<__m64 *>(&res), tmp);
    return res;
  }

  template <>
  inline simd_cd1::value_t reduce_mul(const simd_cd1 &x) {
    alignas(simd_cd1::alignment()) std::array<simd_cd1::scalar_t, 2> res;
    x.store(res.data());
    return simd_cd1::value_t{res[0], res[1]};
  }
// See Eigen library
#ifdef __FMA__
  // FMA ADD
  template <>
  inline simd_f4 fma_add(const simd_f4 &x, const simd_f4 &y, const simd_f4 &z) {
    return simd_f4(_mm_fmadd_ps(x, y, z));
  }

  template <>
  inline simd_d2 fma_add(const simd_d2 &x, const simd_d2 &y, const simd_d2 &z) {
    return simd_d2(_mm_fmadd_pd(x, y, z));
  }

  template <>
  inline simd_cf2 fma_add(const simd_cf2 &x, const simd_cf2 &y, const simd_cf2 &z) {
    __m128 x_odd  = _mm_movehdup_ps(x);
    __m128 x_even = _mm_moveldup_ps(x);
    __m128 y_swap = _mm_permute_ps(y, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m128 result = _mm_fmaddsub_ps(x_even, y, _mm_fmaddsub_ps(x_odd, y_swap, z));
    return simd_cf2(result);
  }

  template <>
  inline simd_cd1 fma_add(const simd_cd1 &x, const simd_cd1 &y, const simd_cd1 &z) {
    __m128d x_odd  = _mm_permute_pd(x, 0x3);
    __m128d x_even = _mm_movedup_pd(x);
    __m128d y_swap = _mm_permute_pd(y, 0x1);
    __m128d result = _mm_fmaddsub_pd(x_even, y, _mm_fmaddsub_pd(x_odd, y_swap, z));
    return simd_cd1(result);
  }

  //FMA_SUB
  template <>
  inline simd_f4 fma_sub(const simd_f4 &x, const simd_f4 &y, const simd_f4 &z) {
    return simd_f4(_mm_fmsub_ps(x, y, z));
  }

  template <>
  inline simd_d2 fma_sub(const simd_d2 &x, const simd_d2 &y, const simd_d2 &z) {
    return simd_d2(_mm_fmsub_pd(x, y, z));
  }

  template <>
  inline simd_cf2 fma_sub(const simd_cf2 &x, const simd_cf2 &y, const simd_cf2 &z) {
    __m128 x_odd  = _mm_movehdup_ps(x);
    __m128 x_even = _mm_moveldup_ps(x);
    __m128 y_swap = _mm_permute_ps(y, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m128 result = _mm_fmaddsub_ps(x_even, y, _mm_fmsubadd_ps(x_odd, y_swap, z));
    return simd_cf2(result);
  }

  template <>
  inline simd_cd1 fma_sub(const simd_cd1 &x, const simd_cd1 &y, const simd_cd1 &z) {
    __m128d x_odd  = _mm_permute_pd(x, 0x3);
    __m128d x_even = _mm_movedup_pd(x);
    __m128d y_swap = _mm_permute_pd(y, 0x1);
    __m128d result = _mm_fmaddsub_pd(x_even, y, _mm_fmsubadd_pd(x_odd, y_swap, z));
    return simd_cd1(result);
  }

  //FMA_NADD
  template <>
  inline simd_f4 fma_nadd(const simd_f4 &x, const simd_f4 &y, const simd_f4 &z) {
    return simd_f4(_mm_fnmadd_ps(x, y, z));
  }

  template <>
  inline simd_d2 fma_nadd(const simd_d2 &x, const simd_d2 &y, const simd_d2 &z) {
    return simd_d2(_mm_fnmadd_pd(x, y, z));
  }

  template <>
  inline simd_cf2 fma_nadd(const simd_cf2 &x, const simd_cf2 &y, const simd_cf2 &z) {
    __m128 x_odd         = _mm_movehdup_ps(x);
    __m128 x_even        = _mm_moveldup_ps(x);
    __m128 y_swap        = _mm_permute_ps(y, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    simd_cf2 y_swap_conj = conj(simd_cf2(y_swap)); // TODO: Eigen bug create issue maybe in eigen.
    __m128 result        = _mm_fmsub_ps(x_odd, y_swap_conj, _mm_fmsub_ps(x_even, y, z));
    return simd_cf2(result);
  }

  template <>
  inline simd_cd1 fma_nadd(const simd_cd1 &x, const simd_cd1 &y, const simd_cd1 &z) {
    __m128d x_odd        = _mm_permute_pd(x, 0x3);
    __m128d x_even       = _mm_movedup_pd(x);
    __m128d y_swap       = _mm_permute_pd(y, 0x1);
    simd_cd1 y_swap_conj = conj(simd_cd1(y_swap));
    __m128d result       = _mm_fmsub_pd(x_odd, y_swap_conj, _mm_fmsub_pd(x_even, y, z));
    return simd_cd1(result);
  }

  //FMA_NSUB
  template <>
  inline simd_f4 fma_nsub(const simd_f4 &x, const simd_f4 &y, const simd_f4 &z) {
    return simd_f4(_mm_fnmsub_ps(x, y, z));
  }

  template <>
  inline simd_d2 fma_nsub(const simd_d2 &x, const simd_d2 &y, const simd_d2 &z) {
    return simd_d2(_mm_fnmsub_pd(x, y, z));
  }

  template <>
  inline simd_cf2 fma_nsub(const simd_cf2 &x, const simd_cf2 &y, const simd_cf2 &z) {
    __m128 x_odd         = _mm_movehdup_ps(x);
    __m128 x_even        = _mm_moveldup_ps(x);
    __m128 y_swap        = _mm_permute_ps(y, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    simd_cf2 y_swap_conj = conj(simd_cf2(y_swap));
    __m128 result        = _mm_fmsub_ps(x_odd, y_swap_conj, _mm_fmadd_ps(x_even, y, z));
    return simd_cf2(result);
  }

  template <>
  inline simd_cd1 fma_nsub(const simd_cd1 &x, const simd_cd1 &y, const simd_cd1 &z) {
    __m128d x_odd        = _mm_permute_pd(x, 0x3);
    __m128d x_even       = _mm_movedup_pd(x);
    __m128d y_swap       = _mm_permute_pd(y, 0x1);
    simd_cd1 y_swap_conj = conj(simd_cd1(y_swap));

    __m128d result = _mm_fmsub_pd(x_odd, y_swap_conj, _mm_fmadd_pd(x_even, y, z));
    return simd_cd1(result);
  }
#else
  // FMA ADD
  template <>
  inline simd_f4 fma_add(const simd_f4 &x, const simd_f4 &y, const simd_f4 &z) {
    return x * y + z;
  }

  template <>
  inline simd_d2 fma_add(const simd_d2 &x, const simd_d2 &y, const simd_d2 &z) {
    return x * y + z;
  }

  template <>
  inline simd_cf2 fma_add(const simd_cf2 &x, const simd_cf2 &y, const simd_cf2 &z) {
    return x * y + z;
  }

  template <>
  inline simd_cd1 fma_add(const simd_cd1 &x, const simd_cd1 &y, const simd_cd1 &z) {
    return x * y + z;
  }

  //FMA_SUB
  template <>
  inline simd_f4 fma_sub(const simd_f4 &x, const simd_f4 &y, const simd_f4 &z) {
    return x * y - z;
  }

  template <>
  inline simd_d2 fma_sub(const simd_d2 &x, const simd_d2 &y, const simd_d2 &z) {
    return x * y - z;
  }

  template <>
  inline simd_cf2 fma_sub(const simd_cf2 &x, const simd_cf2 &y, const simd_cf2 &z) {
    return x * y - z;
  }

  template <>
  inline simd_cd1 fma_sub(const simd_cd1 &x, const simd_cd1 &y, const simd_cd1 &z) {
    return x * y - z;
  }

  //FMA_NADD
  template <>
  inline simd_f4 fma_nadd(const simd_f4 &x, const simd_f4 &y, const simd_f4 &z) {
    return z - (x * y);
  }

  template <>
  inline simd_d2 fma_nadd(const simd_d2 &x, const simd_d2 &y, const simd_d2 &z) {
    return z - (x * y);
  }

  template <>
  inline simd_cf2 fma_nadd(const simd_cf2 &x, const simd_cf2 &y, const simd_cf2 &z) {
    return z - (x * y);
  }

  template <>
  inline simd_cd1 fma_nadd(const simd_cd1 &x, const simd_cd1 &y, const simd_cd1 &z) {
    return z - (x * y);
  }

  //FMA_NSUB
  template <>
  inline simd_f4 fma_nsub(const simd_f4 &x, const simd_f4 &y, const simd_f4 &z) {
    return -(x * y + z);
  }

  template <>
  inline simd_d2 fma_nsub(const simd_d2 &x, const simd_d2 &y, const simd_d2 &z) {
    return -(x * y + z);
  }

  template <>
  inline simd_cf2 fma_nsub(const simd_cf2 &x, const simd_cf2 &y, const simd_cf2 &z) {
    return -(x * y + z);
  }

  template <>
  inline simd_cd1 fma_nsub(const simd_cd1 &x, const simd_cd1 &y, const simd_cd1 &z) {
    return -(x * y + z);
  }

#endif
  // Fma functions for integer simulations.
  // FMA ADD
  template <>
  inline simd_i4 fma_add(const simd_i4 &x, const simd_i4 &y, const simd_i4 &z) {
    return x * y + z;
  }

  template <>
  inline simd_l2 fma_add(const simd_l2 &x, const simd_l2 &y, const simd_l2 &z) {
    return x * y + z;
  }

  // FMA SUB
  template <>
  inline simd_i4 fma_sub(const simd_i4 &x, const simd_i4 &y, const simd_i4 &z) {
    return x * y - z;
  }

  template <>
  inline simd_l2 fma_sub(const simd_l2 &x, const simd_l2 &y, const simd_l2 &z) {
    return x * y - z;
  }

  // FMA NADD
  template <>
  inline simd_i4 fma_nadd(const simd_i4 &x, const simd_i4 &y, const simd_i4 &z) {
    return z - (x * y);
  }

  template <>
  inline simd_l2 fma_nadd(const simd_l2 &x, const simd_l2 &y, const simd_l2 &z) {
    return z - (x * y);
  }

  // FMA NSUB
  template <>
  inline simd_i4 fma_nsub(const simd_i4 &x, const simd_i4 &y, const simd_i4 &z) {
    return -(x * y + z);
  }

  template <>
  inline simd_l2 fma_nsub(const simd_l2 &x, const simd_l2 &y, const simd_l2 &z) {
    return -(x * y + z);
  }
  // Gather Functions with given strides in vindex.
#ifdef __AVX2__
  template <>
  inline simd_i4 gather(const simd_i4::value_t *from, const long stride) {
    simd_i4 simd_stride(static_cast<int32_t>(stride));
    const simd_i4 multiplier({0, 1, 2, 3});
    simd_i4 vindex = simd_stride * multiplier;
    return simd_i4(_mm_i32gather_epi32(from, vindex, sizeof(simd_i4::value_t)));
  }

  template <>
  inline simd_l2 gather(const simd_l2::value_t *from, const long stride) {
    simd_l2 simd_stride(stride);
    const simd_l2 multiplier({0, 1});
    simd_l2 vindex = simd_stride * multiplier;
    return simd_l2(_mm_i64gather_epi64(reinterpret_cast<const long long int *>(from), vindex, sizeof(simd_l2::value_t)));
  }

  template <>
  inline simd_f4 gather(const simd_f4::value_t *from, const long stride) {
    simd_i4 simd_stride(static_cast<int32_t>(stride));
    const simd_i4 multiplier({0, 1, 2, 3});
    simd_i4 vindex = simd_stride * multiplier;
    return simd_f4(_mm_i32gather_ps(from, vindex, sizeof(simd_f4::value_t)));
  }

  template <>
  inline simd_d2 gather(const simd_d2::value_t *from, const long stride) {
    simd_l2 simd_stride(stride);
    const simd_l2 multiplier({0, 1});
    simd_l2 vindex = simd_stride * multiplier;
    return simd_d2(_mm_i64gather_pd(from, vindex, sizeof(simd_d2::value_t)));
  }

  template <>
  inline simd_cf2 gather(const simd_cf2::value_t *from, const long stride) {
    return simd_cf2(_mm_castpd_ps(gather<simd_d2>(reinterpret_cast<const simd_d2::value_t *>(from), stride)));
  }

  template <>
  inline simd_cd1 gather(const simd_cd1::value_t *from, [[maybe_unused]] const long stride) {
    simd_cd1 tmp;
    tmp.load_unaligned(from);
    return tmp;
  }
#else
  template <>
  inline simd_i4 gather<simd_i4>(const simd_i4::value_t *from, const long stride) {
    return simd_i4(_mm_set_epi32(from[3 * stride], from[2 * stride], from[stride], from[0]));
  }

  template <>
  inline simd_l2 gather<simd_l2>(const simd_l2::value_t *from, const long stride) {
    return simd_l2(_mm_set_epi64x(from[stride], from[0]));
  }

  template <>
  inline simd_f4 gather<simd_f4>(const simd_f4::value_t *from, const long stride) {
    return simd_f4(_mm_set_ps(from[3 * stride], from[2 * stride], from[stride], from[0]));
  }

  template <>
  inline simd_d2 gather<simd_d2>(const simd_d2::value_t *from, const long stride) {
    return simd_d2(_mm_set_pd(from[stride], from[0]));
  }

  template <>
  inline simd_cf2 gather<simd_cf2>(const simd_cf2::value_t *from, const long stride) {
    return simd_cf2(_mm_set_ps(from[stride].imag(), from[stride].real(), from[0].imag(), from[0].real()));
  }

  template <>
  inline simd_cd1 gather<simd_cd1>(const simd_cd1::value_t *from, [[maybe_unused]] const long stride) {
    simd_cd1 tmp;
    tmp.load_unaligned(from);
    return tmp;
  }
#endif
  template <>
  inline void scatter(const simd_i4 &v, simd_i4::value_t *to, const long stride) {
    to[0]          = _mm_cvtsi128_si32(v);
    to[stride]     = _mm_cvtsi128_si32(_mm_shuffle_epi32(v, 0x1));
    to[2 * stride] = _mm_cvtsi128_si32(_mm_shuffle_epi32(v, 0x2));
    to[3 * stride] = _mm_cvtsi128_si32(_mm_shuffle_epi32(v, 0x3));
  }

  template <>
  inline void scatter(const simd_l2 &v, simd_l2::value_t *to, const long stride) {
    to[0]      = _mm_cvtsi128_si64(v);
    to[stride] = _mm_cvtsi128_si64(_mm_unpackhi_epi64(v, v));
  }

  template <>
  inline void scatter(const simd_f4 &v, simd_f4::value_t *to, const long stride) {
    to[0]          = _mm_cvtss_f32(v);
    to[stride]     = _mm_cvtss_f32(_mm_shuffle_ps(v, v, 0x1));
    to[2 * stride] = _mm_cvtss_f32(_mm_shuffle_ps(v, v, 0x2));
    to[3 * stride] = _mm_cvtss_f32(_mm_shuffle_ps(v, v, 0x3));
  }

  template <>
  inline void scatter(const simd_d2 &v, simd_d2::value_t *to, const long stride) {
    to[0]      = _mm_cvtsd_f64(v);
    to[stride] = _mm_cvtsd_f64(_mm_shuffle_pd(v, v, 0x1));
  }

  template <>
  inline void scatter(const simd_cf2 &v, simd_cf2::value_t *to, const long stride) {
    to[0]      = simd_cf2::value_t(_mm_cvtss_f32(v), _mm_cvtss_f32(_mm_shuffle_ps(v, v, 0x1)));
    to[stride] = simd_cf2::value_t(_mm_cvtss_f32(_mm_shuffle_ps(v, v, 0x2)), _mm_cvtss_f32(_mm_shuffle_ps(v, v, 0x3)));
  }

  template <>
  inline void scatter(const simd_cd1 &v, simd_cd1::value_t *to, [[maybe_unused]] const long stride) {
    to[0] = simd_cd1::value_t(_mm_cvtsd_f64(v), _mm_cvtsd_f64(_mm_shuffle_pd(v, v, 0x1)));
  }
} // namespace nda::simd
#endif