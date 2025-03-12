#pragma once
#include <complex>
#include "abi.hpp"
#include "../../macros.hpp"
#include "../../traits.hpp"

namespace nda {

  template <typename T, size_t Width, abi_tag ABI>
  class simd_type;

  struct simd_aligned_memory {};
  struct simd_unaligned_memory {};
  struct simd_zero_initialize {};

  static constexpr inline simd_aligned_memory simd_aligned_memory_t;
  static constexpr inline simd_unaligned_memory simd_unaligned_memory_t;
  static constexpr inline simd_zero_initialize simd_zero_initialize_t;

  using simd_f1    = simd_type<float, 1, abi_tag::Default>;
  using simd_d1    = simd_type<double, 1, abi_tag::Default>;
  using simd_i1    = simd_type<int32_t, 1, abi_tag::Default>;
  using simd_l1    = simd_type<int64_t, 1, abi_tag::Default>;
  using simd_cf1   = simd_type<std::complex<float>, 1, abi_tag::Default>;
  using simd_cd1_d = simd_type<std::complex<double>, 1, abi_tag::Default>;
#ifdef __SSE2__
  using simd_f4  = simd_type<float, 4, abi_tag::SSE>;
  using simd_d2  = simd_type<double, 2, abi_tag::SSE>;
  using simd_i4  = simd_type<int32_t, 4, abi_tag::SSE>;
  using simd_l2  = simd_type<int64_t, 2, abi_tag::SSE>;
  using simd_cf2 = simd_type<std::complex<float>, 2, abi_tag::SSE>;
  using simd_cd1 = simd_type<std::complex<double>, 1, abi_tag::SSE>;
#endif
#ifdef __AVX__
  using simd_f8  = simd_type<float, 8, abi_tag::AVX>;
  using simd_d4  = simd_type<double, 4, abi_tag::AVX>;
  using simd_i8  = simd_type<int32_t, 8, abi_tag::AVX>;
  using simd_l4  = simd_type<int64_t, 4, abi_tag::AVX>;
  using simd_cf4 = simd_type<std::complex<float>, 4, abi_tag::AVX>;
  using simd_cd2 = simd_type<std::complex<double>, 2, abi_tag::AVX>;
#endif
#ifdef __AVX512F__
  using simd_f16 = simd_type<float, 16, abi_tag::AVX512>;
  using simd_d8  = simd_type<double, 8, abi_tag::AVX512>;
  using simd_i16 = simd_type<int32_t, 16, abi_tag::AVX512>;
  using simd_l8  = simd_type<int64_t, 8, abi_tag::AVX512>;
  using simd_cf8 = simd_type<std::complex<float>, 8, abi_tag::AVX512>;
  using simd_cd4 = simd_type<std::complex<double>, 4, abi_tag::AVX512>;
#endif

}; // namespace nda