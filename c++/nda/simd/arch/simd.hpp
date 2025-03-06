#pragma once
#include "type.hpp"
#include "./Default/functions.hpp"
#include "./Default/kernel.hpp"
#ifdef __SSE2__
#include "./SSE/functions.hpp"
#include "./SSE/kernel.hpp"
#endif
#ifdef __AVX__
#include "./AVX/functions.hpp"
#include "./AVX/kernel.hpp"
#endif
#ifdef __AVX512F__
#include "./AVX512/functions.hpp"
#include "./AVX512/kernel.hpp"
#endif

#include <algorithm>

namespace nda {
  template <typename T>
  using native_simd = simd_type<std::conditional_t<std::is_same_v<std::remove_cvref_t<T>, long long>, int64_t, std::remove_cvref_t<T>>,
                                abi::get_native_width<T>(), abi::get_native_abi_tag()>;

  template <typename T, bool Aligned>
  struct simd_block {
    using simd_t = native_simd<T>;

    simd_t value;
    T *const data;
    const size_t valid_field;

    void store() {
      if (valid_field == simd_t::size()) {
        if constexpr (Aligned) {
          value.store(data);
        } else {
          value.store_unaligned(data);
        }
      } else {
        alignas(simd_t::alignment()) std::array<T, simd_t::size()> tmp;
        value.store(tmp.data());
        std::copy_n(tmp.data(), valid_field, data);
      }
    }
  };

} // namespace nda
