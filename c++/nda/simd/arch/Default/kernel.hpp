#pragma once
#include "./type.hpp"
#include "../kernel_forward.hpp"

namespace nda::simd {

  template <>
  inline std::array<simd_i1, 1> kernel_transpose(const std::array<simd_i1, 1> &simd_block) {
    return simd_block;
  }

  template <>
  inline std::array<simd_l1, 1> kernel_transpose(const std::array<simd_l1, 1> &simd_block) {
    return simd_block;
  }

  template <>
  inline std::array<simd_f1, 1> kernel_transpose(const std::array<simd_f1, 1> &simd_block) {
    return simd_block;
  }

  template <>
  inline std::array<simd_d1, 1> kernel_transpose(const std::array<simd_d1, 1> &simd_block) {
    return simd_block;
  }

  template <>
  inline std::array<simd_cf1, 1> kernel_transpose(const std::array<simd_cf1, 1> &simd_block) {
    return simd_block;
  }

  template <>
  inline std::array<simd_cd1_d, 1> kernel_transpose(const std::array<simd_cd1_d, 1> &simd_block) {
    return simd_block;
  }

} // namespace nda::simd