#pragma once


#if defined(__x86_64__)
#include <immintrin.h>
#endif
//TODO ask what about macos

// Default Simd with in Bytes.
#if defined(__AVX512F__)
#define SIMD_WIDTH 64
#elif defined(__AVX2__) || defined(__AVX__)
#define SIMD_WIDTH 32
#elif defined(__SSE4_2__)
#define SIMD_WIDTH 16
#else
#define SIMD_WIDTH 4
#endif

namespace nda {
  template <typename T, int WIDTH>
  struct invalid_simd_type {};

  template <typename T, int WIDTH>
  struct simd_type {
    using intrinsic_type       = typename invalid_simd_type<T, WIDTH>::intrinsic_type;
    using base_type            = typename invalid_simd_type<T, WIDTH>::base_type;
    static constexpr int width = WIDTH;
  };

  template <>
  struct simd_type<int, 1> {
    using intrinsic_type = int;
    using base_type      = int;
  };

  template <>
  struct simd_type<float, 1> {
    using intrinsic_type = float;
    using base_type      = float;
  };

  template <>
  struct simd_type<double, 1> {
    using intrinsic_type = double;
    using base_type      = double;
  };

  template <>
  struct simd_type<long long, 1> {
    using intrinsic_type = long long;
    using base_type      = long long;
  };

#if defined(__SSE4_2__)
  template <>
  struct simd_type<int, 4> {
    using intrinsic_type = __m128i;
    using base_type      = int;
  };
  template <>
  struct simd_type<float, 4> {
    using intrinsic_type = __m128;
    using base_type      = float;
  };
  template <>
  struct simd_type<double, 2> {
    using intrinsic_type = __m128d;
    using base_type      = double;
  };
  template <>
  struct simd_type<long long, 2> {
    using intrinsic_type = __m128i;
    using base_type      = long long;
  };
#endif

#if defined(__AVX2__) || defined(__AVX__)
  template <>
  struct simd_type<int, 8> {
    using intrinsic_type = __m256i;
    using base_type      = int;
  };
  template <>
  struct simd_type<float, 8> {
    using intrinsic_type = __m256;
    using base_type      = float;
  };
  template <>
  struct simd_type<double, 4> {
    using intrinsic_type = __m256d;
    using base_type      = double;
  };
  template <>
  struct simd_type<long long, 4> {
    using intrinsic_type = __m256i;
    using base_type      = long long;
  };

#endif

#if defined(__AVX512F__)
  template <>
  struct simd_type<int, 16> {
    using intrinsic_type = __m512i;
    using base_type      = int;
  };
  template <>
  struct simd_type<float, 16> {
    using intrinsic_type = __m512;
    using base_type      = float;
  };
  template <>
  struct simd_type<double, 8> {
    using intrinsic_type = __m512d;
    using base_type      = double;
  };
  template <>
  struct simd_type<long long, 8> {
    using intrinsic_type = __m512i;
    using base_type      = long long;
  };

#endif
} // namespace nda