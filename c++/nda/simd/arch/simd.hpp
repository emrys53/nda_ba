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

#include <algorithm>

namespace nda {
  template <typename T>
  using native_simd = simd_type<std::conditional_t<std::is_same_v<T, long long>, int64_t, T>, abi::get_native_width<T>(), abi::get_native_abi_tag()>;

  template <typename T, bool Aligned>
  struct simd_block {
    using simd_t = native_simd<T>;

    simd_t value;
    T *data;
    size_t valid_field;

    simd_block(const simd_t &v, const T *d, const size_t valid) : value(v), data(d), valid_field(valid) {}

    void store() {
      if (valid_field == simd_t::size()) [[likely]] {
        if constexpr (Aligned) {
          value.store(data);
        } else {
          value.store_unaligned(data);
        }
      } else {
        alignas(simd_t::alignment()) std::array<T, simd_t::size()> tmp;
        value.store(tmp);
        std::copy_n(data, valid_field, tmp);
      }
    }
  };

} // namespace nda
