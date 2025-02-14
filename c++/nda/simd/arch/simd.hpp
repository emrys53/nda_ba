#pragma once
#include "type.hpp"
#include "./Default/functions.hpp"
#ifdef __SSE2__
#include "./SSE/functions.hpp"
#endif
#ifdef __AVX__
#include "./AVX/functions.hpp"
#endif
#ifdef __AVX512F__
#include "./AVX512/functions.hpp"
#endif

namespace nda {
  template <typename T>
  using native_simd = simd_type<std::conditional_t<std::is_same_v<T, long long>, int64_t, T>, abi::get_native_width<T>(), abi::get_native_abi_tag()>;

} // namespace nda
