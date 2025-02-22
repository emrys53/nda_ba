#pragma once
#include "./arch/simd.hpp"
#include "../concepts.hpp"

namespace nda::simd {
  template <typename Derived, Vectorizable T>
  struct mock_simd {
    using value_t = T;
    using simd_t  = native_simd<T>;

    template <typename... Args>
    value_t operator()(Args &&...args) const {
      static_assert((std::is_same_v<value_t, std::remove_cvref_t<Args>> and ...), "All types have to be the same.");
      return static_cast<Derived *>(this)->operator()(std::forward<Args>(args)...);
    }

    private:
    [[gnu::always_inline]] std::array<value_t, simd_t::size()> convert_simd_to_array(const simd_t &a) {
      alignas(simd_t::alignment()) std::array<T, simd_t::size()> result;
      a.store(result.data());
      return result;
    }

    template <size_t... Is, typename... Args>
    [[gnu::always_inline]] auto make_array_tuple(std::index_sequence<Is...>, const std::tuple<Args...> &args_tuple) {
      return std::make_tuple(convert_simd_to_array(std::get<Is>(args_tuple))...);
    }

    template <size_t... Is, typename... Args>
    [[gnu::always_inline]] auto apply_function(std::index_sequence<Is...>, const std::tuple<Args...> &array_tuple) {
      alignas(simd_t::alignment()) std::array<T, simd_t::size()> result_array;
      for (int i = 0; i < simd_t::size(); ++i) { result_array[i] = static_cast<Derived *>(this)->operator()(std::get<Is>(array_tuple)[i]...); }
      return result_array;
    }

    public:
    template <typename... Args>
    simd_t load(Args &&...args) {
      static_assert((std::is_same_v<simd_t, std::remove_cvref_t<Args>> and ...), "All types have to be the same.");
      constexpr size_t args_size        = sizeof...(Args);
      std::tuple<Args &&...> args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);
      auto array_tuple                  = make_array_tuple(std::make_index_sequence<args_size>{}, args_tuple);
      alignas(simd_t::alignment()) std::array<value_t, simd_t::size()> result_array;
      result_array = apply_function(std::make_index_sequence<args_size>{}, array_tuple);
      simd_t result;
      result.load(result_array.data());
      return result;
    }
  };
} // namespace nda::simd