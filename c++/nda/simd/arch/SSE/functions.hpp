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
    alignas(x.alignment()) std::array<int64_t, 2> a, b, c;
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
    alignas(x.alignment()) std::array<int64_t, 2> a, b, c;
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
    alignas(x.alignment()) std::array<int64_t, 2> a;
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
    alignas(x.alignment()) std::array<int64_t, 2> a;
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

} // namespace nda::simd
#endif