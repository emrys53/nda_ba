#pragma once
#include <cstddef>
#include <array>
namespace nda::simd {
  template <typename T>
  inline void kernel_transpose(std::array<T, T::size()> &simd_block);
}