#pragma once
#include "type.hpp"
#include "abi.hpp"

namespace nda {

  template <typename T>
  using native_simd = simd_type<T, abi::get_native_width<T>(), abi::get_native_abi_tag<T>()>;

}
