#pragma once
#include <cstddef>
#include <array>
namespace nda::simd {
  template <typename T>
  inline std::array<T, T::size()> kernel_transpose(const std::array<T, T::size()> &simd_block);
}