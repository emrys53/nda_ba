#pragma once
namespace nda {
  enum class abi_tag { Default, SSE, AVX, AVX512 };
  namespace abi {
    inline static constexpr abi_tag get_native_abi_tag() {
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


    template <typename T, size_t Width>
    inline static constexpr abi_tag get_abi_tag_with_width() {
      constexpr size_t total_size = sizeof(T) * Width;
      if constexpr (total_size == 64) {
#ifdef __AVX512F__
        return abi_tag::AVX512;
#else
        static_assert(false, "There is no architecture support for given type and Width");
#endif
      } else if constexpr (total_size == 32) {
#ifdef __AVX__
        return abi_tag::AVX;
#else
        static_assert(false, "There is no architecture support for given type and Width");
#endif
      } else if constexpr (total_size == 16) {
#ifdef __SSE2__
        return abi_tag::SSE;
#else
        static_assert(false, "There is no architecture support for given type and Width");
#endif
      } else if constexpr (Width == 1) {
        return abi_tag::Default;
      } else {
        static_assert(false, "There is no architecture support for given type and Width");
      }
    }

    template <typename T>
    inline static constexpr size_t get_native_width() {
      return 0;
    }

    template <typename T>
      requires requires(T x) { sizeof(T); }
    inline static constexpr size_t get_native_width() {
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

  } // namespace abi
} // namespace  nda