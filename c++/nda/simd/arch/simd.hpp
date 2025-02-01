#pragma once
#include "type.hpp"

namespace nda {
  template <typename T>
  using native_simd = simd_type<std::conditional_t<std::is_same_v<T, long long>, int64_t, T>, abi::get_native_width<T>(), abi::get_native_abi_tag()>;

} // namespace nda
