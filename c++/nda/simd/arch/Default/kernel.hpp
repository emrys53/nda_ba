#pragma once
#include "./type.hpp"
#include "../kernel_forward.hpp"

namespace nda::simd {

  template <>
  inline void kernel_transpose([[maybe_unused]] std::array<simd_i1, 1> &simd_block) {}

  template <>
  inline void kernel_transpose([[maybe_unused]] std::array<simd_l1, 1> &simd_block) {}

  template <>
  inline void kernel_transpose([[maybe_unused]] std::array<simd_f1, 1> &simd_block) {}

  template <>
  inline void kernel_transpose([[maybe_unused]] std::array<simd_d1, 1> &simd_block) {}

  template <>
  inline void kernel_transpose([[maybe_unused]] std::array<simd_cf1, 1> &simd_block) {}

  template <>
  inline void kernel_transpose([[maybe_unused]] std::array<simd_cd1_d, 1> &simd_block) {}

} // namespace nda::simd