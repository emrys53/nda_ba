#pragma once
namespace nda {
  enum class  abi_tag { Default, SSE, AVX, AVX512};
  namespace abi {
    // template <typename T>
    // static constexpr abi_tag get_native_abi_tag() {
    // return abi_tag::Default;
    // }
    static constexpr abi_tag get_native_abi_tag() {
#ifdef __AVX512F__
      return abi_tag::AVX512;
#else
#ifdef __AVX__
      return abi_tag::AVX;
#else
#ifdef __SSE2__
      return abi_tag::SSE;
#else
#endif
#endif
#endif
      return abi_tag::Default;
    }

    template <typename T>
    static constexpr size_t get_native_width() {
      return 0;
    }

    template <typename T>
    requires requires (T x) {sizeof(T);}
    static constexpr size_t get_native_width() {
      constexpr abi_tag abi = get_native_abi_tag();

      constexpr auto is_abi = [](abi_tag lhs, abi_tag rhs) {
        return static_cast<std::underlying_type_t<abi_tag>>(lhs) == static_cast<std::underlying_type_t<abi_tag>>(rhs);
      };

      if constexpr (is_abi(abi, abi_tag::SSE)) {
        return 16 / sizeof(T);
      } else if constexpr (is_abi(abi, abi_tag::AVX)) {
        return 32 / sizeof(T);
      } else if constexpr (is_abi(abi, abi_tag::AVX512)) {
        return 64 / sizeof(T);
      }
      return 1;
    }

  }
}// namespace  nda