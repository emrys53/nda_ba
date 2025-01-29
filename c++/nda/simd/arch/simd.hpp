#pragma once
#include "type.hpp"
#ifdef __AVX__
#include "AVX/type.hpp"
#endif
#ifdef __SSE2__
#include "SSE/type.hpp"
#endif
namespace nda {
  // Need wrappers for default ones.
  using simd_f1    = float;
  using simd_d1    = double;
  using simd_i1    = int32_t;
  using simd_l1    = int64_t;
  using simd_cf1   = std::complex<float>;
  using simd_cd1_d = std::complex<double>;

  using simd_f4  = simd_type<float, 4, abi_tag::SSE>;
  using simd_d2  = simd_type<double, 2, abi_tag::SSE>;
  using simd_i4  = simd_type<int32_t, 4, abi_tag::SSE>;
  using simd_l2  = simd_type<int64_t, 2, abi_tag::SSE>;
  using simd_cf2 = simd_type<std::complex<float>, 2, abi_tag::SSE>;
  using simd_cd1 = simd_type<std::complex<double>, 1, abi_tag::SSE>;

  using simd_f8  = simd_type<float, 8, abi_tag::AVX>;
  using simd_d4  = simd_type<double, 4, abi_tag::AVX>;
  using simd_i8  = simd_type<int32_t, 8, abi_tag::AVX>;
  using simd_l4  = simd_type<int64_t, 4, abi_tag::AVX>;
  using simd_cf4 = simd_type<std::complex<float>, 4, abi_tag::AVX>;
  using simd_cd2 = simd_type<std::complex<double>, 2, abi_tag::AVX>;

  using simd_f16 = simd_type<float, 16, abi_tag::AVX512>;
  using simd_d8  = simd_type<double, 8, abi_tag::AVX512>;
  using simd_i16 = simd_type<int32_t, 16, abi_tag::AVX512>;
  using simd_l8  = simd_type<int64_t, 8, abi_tag::AVX512>;
  using simd_cf8 = simd_type<std::complex<float>, 8, abi_tag::AVX512>;
  using simd_cd4 = simd_type<std::complex<double>, 4, abi_tag::AVX512>;

}; // namespace nda
