#pragma once
#include "../abi.hpp"
#include "../type_forward.hpp"

#include <cstddef>
#include <initializer_list>
#include <concepts>
#include <bit>

namespace nda {

  template <typename T>
    requires(std::same_as<std::remove_cvref_t<T>, int32_t> or std::same_as<std::remove_cvref_t<T>, int64_t>
             or std::same_as<std::remove_cvref_t<T>, long long> or std::same_as<std::remove_cvref_t<T>, float>
             or std::same_as<std::remove_cvref_t<T>, double>)
  class simd_type<T, 1, abi_tag::Default> {
    public:
    using intrinsic_t = std::remove_cvref_t<T>;
    using value_t     = T;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(*v) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(*v) {}

    simd_type(simd_zero_initialize) : value(0) {}

    simd_type(const std::array<value_t, 1> &v) : value(v[0]) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 1UL; };
    static constexpr size_t alignment() { return 0UL; };

    void load(const value_t *from) { value = *from; }
    void load_unaligned(const value_t *from) { value = *from; }
    void store(value_t *to) const { *to = value; }
    void store_unaligned(value_t *to) const { *to = value; }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{lhs.value + rhs.value}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{lhs.value - rhs.value}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) { return simd_type{lhs.value * rhs.value}; }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) { return simd_type{lhs.value / rhs.value}; }

    simd_type &operator+=(const simd_type &other) {
      *this = *this + other;
      return *this;
    }

    simd_type &operator-=(const simd_type &other) {
      *this = *this - other;
      return *this;
    }

    simd_type &operator*=(const simd_type &other) {
      *this = *this * other;
      return *this;
    }

    simd_type &operator/=(const simd_type &other) {
      *this = *this / other;
      return *this;
    }

    simd_type operator-() const { return simd_type{-value}; }

    bool operator==(const simd_type &other) const { return value == other.value; }

    bool operator!=(const simd_type &other) const { return not(*this == other); };

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) {
      if constexpr (sizeof(T) == 4) {
        return simd_type{std::bit_cast<T>(std::bit_cast<int32_t>(lhs.value) ^ std::bit_cast<int32_t>(rhs.value))};
      } else {
        return simd_type{std::bit_cast<T>(std::bit_cast<int64_t>(lhs.value) ^ std::bit_cast<int64_t>(rhs.value))};
      }
    }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) {
      if constexpr (sizeof(T) == 4) {
        return simd_type{std::bit_cast<T>(std::bit_cast<int32_t>(lhs.value) & std::bit_cast<int32_t>(rhs.value))};
      } else {
        return simd_type{std::bit_cast<T>(std::bit_cast<int64_t>(lhs.value) & std::bit_cast<int64_t>(rhs.value))};
      }
    }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) {
      if constexpr (sizeof(T) == 4) {
        return simd_type{std::bit_cast<T>(std::bit_cast<int32_t>(lhs.value) | std::bit_cast<int32_t>(rhs.value))};
      } else {
        return simd_type{std::bit_cast<T>(std::bit_cast<int64_t>(lhs.value) | std::bit_cast<int64_t>(rhs.value))};
      }
    }

    simd_type &operator^=(const simd_type &other) {
      *this = *this ^ other;
      return *this;
    }

    simd_type &operator&=(const simd_type &other) {
      *this = *this & other;
      return *this;
    }

    simd_type &operator|=(const simd_type &other) {
      *this = *this | other;
      return *this;
    }

    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator+(const simd_type &lhs, const U &rhs) {
      return simd_type(lhs.value + static_cast<T>(rhs));
    }

    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator-(const simd_type &lhs, const U &rhs) {
      return simd_type(lhs.value - static_cast<T>(rhs));
    }

    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator*(const simd_type &lhs, const U &rhs) {
      return simd_type(lhs.value * static_cast<T>(rhs));
    }

    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator/(const simd_type &lhs, const U &rhs) {
      return simd_type(lhs.value / static_cast<T>(rhs));
    }

    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator+(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<T>(lhs) + rhs.value);
    }

    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator-(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<T>(lhs) - rhs.value);
    }

    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator*(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<T>(lhs) * rhs.value);
    }

    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator/(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<T>(lhs) / rhs.value);
    }

    template <typename U>
      requires std::is_arithmetic_v<U>
    simd_type &operator+=(const U &other) {
      *this = *this + other;
      return *this;
    }

    template <typename U>
      requires std::is_arithmetic_v<U>
    simd_type &operator-=(const U &other) {
      *this = *this - other;
      return *this;
    }

    template <typename U>
      requires std::is_arithmetic_v<U>
    simd_type &operator*=(const U &other) {
      *this = *this * other;
      return *this;
    }

    template <typename U>
      requires std::is_arithmetic_v<U>
    simd_type &operator/=(const U &other) {
      *this = *this / other;
      return *this;
    }
  };

  template <typename T>
  class simd_type<std::complex<T>, 1, abi_tag::Default> {
    public:
    using intrinsic_t = std::complex<std::remove_cvref_t<T>>;
    using value_t     = std::complex<std::remove_cvref_t<T>>;
    using scalar_t    = std::remove_cvref_t<T>;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(*v) {}

    simd_type(const scalar_t *v, simd_aligned_memory) : value(v[0], v[1]) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(*v) {}

    simd_type(const scalar_t *v, simd_unaligned_memory) : value(v[0], v[1]) {}

    simd_type(simd_zero_initialize) : value({}, {}) {}

    simd_type(const std::array<value_t, 1> &v) : value(v[0]) {}

    simd_type(const std::array<scalar_t, 2> &v) : value(v[0], v[1]) {}

    simd_type(const scalar_t v) : value(v, v) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 1UL; };

    static constexpr size_t alignment() { return 0UL; }

    void load(const scalar_t *from) { value = intrinsic_t(from[0], from[1]); }
    void load(const value_t *from) { value = *from; }
    void load_unaligned(const scalar_t *from) { value = intrinsic_t(from[0], from[1]); }
    void load_unaligned(const value_t *from) { value = *from; }

    void store(scalar_t *to) const {
      to[0] = value.real();
      to[1] = value.imag();
    }
    void store(value_t *to) const { *to = value; }
    void store_unaligned(scalar_t *to) const {
      to[0] = value.real();
      to[1] = value.imag();
    }
    void store_unaligned(value_t *to) const { *to = value; }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{lhs.value + rhs.value}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{lhs.value - rhs.value}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) { return simd_type{lhs.value * rhs.value}; }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) { return simd_type{lhs.value / rhs.value}; }

    simd_type &operator+=(const simd_type &other) {
      *this = *this + other;
      return *this;
    }

    simd_type &operator-=(const simd_type &other) {
      *this = *this - other;
      return *this;
    }

    simd_type &operator*=(const simd_type &other) {
      *this = *this * other;
      return *this;
    }

    simd_type &operator/=(const simd_type &other) {
      *this = *this / other;
      return *this;
    }

    simd_type operator-() const { return simd_type{-value}; }

    bool operator==(const simd_type &other) const { return value == other.value; }

    bool operator!=(const simd_type &other) const { return not(*this == other); };

    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) {
      if constexpr (sizeof(scalar_t) == 4) {
        T real = std::bit_cast<T>(std::bit_cast<int32_t>(lhs.value.real()) ^ std::bit_cast<int32_t>(rhs.value.real()));
        T imag = std::bit_cast<T>(std::bit_cast<int32_t>(lhs.value.imag()) ^ std::bit_cast<int32_t>(rhs.value.imag()));
        return simd_type{value_t{real, imag}};
      } else {
        T real = std::bit_cast<T>(std::bit_cast<int64_t>(lhs.value.real()) ^ std::bit_cast<int64_t>(rhs.value.real()));
        T imag = std::bit_cast<T>(std::bit_cast<int64_t>(lhs.value.imag()) ^ std::bit_cast<int64_t>(rhs.value.imag()));
        return simd_type{value_t{real, imag}};
      }
    }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) {
      if constexpr (sizeof(scalar_t) == 4) {
        T real = std::bit_cast<T>(std::bit_cast<int32_t>(lhs.value.real()) & std::bit_cast<int32_t>(rhs.value.real()));
        T imag = std::bit_cast<T>(std::bit_cast<int32_t>(lhs.value.imag()) & std::bit_cast<int32_t>(rhs.value.imag()));
        return simd_type{value_t{real, imag}};
      } else {
        T real = std::bit_cast<T>(std::bit_cast<int64_t>(lhs.value.real()) & std::bit_cast<int64_t>(rhs.value.real()));
        T imag = std::bit_cast<T>(std::bit_cast<int64_t>(lhs.value.imag()) & std::bit_cast<int64_t>(rhs.value.imag()));
        return simd_type{value_t{real, imag}};
      }
    }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) {
      if constexpr (sizeof(scalar_t) == 4) {
        T real = std::bit_cast<T>(std::bit_cast<int32_t>(lhs.value.real()) | std::bit_cast<int32_t>(rhs.value.real()));
        T imag = std::bit_cast<T>(std::bit_cast<int32_t>(lhs.value.imag()) | std::bit_cast<int32_t>(rhs.value.imag()));
        return simd_type{value_t{real, imag}};
      } else {
        T real = std::bit_cast<T>(std::bit_cast<int64_t>(lhs.value.real()) | std::bit_cast<int64_t>(rhs.value.real()));
        T imag = std::bit_cast<T>(std::bit_cast<int64_t>(lhs.value.imag()) | std::bit_cast<int64_t>(rhs.value.imag()));
        return simd_type{value_t{real, imag}};
      }
    }

    simd_type &operator^=(const simd_type &other) {
      *this = *this ^ other;
      return *this;
    }

    simd_type &operator&=(const simd_type &other) {
      *this = *this & other;
      return *this;
    }

    simd_type &operator|=(const simd_type &other) {
      *this = *this | other;
      return *this;
    }

    template <typename U>
      requires(std::is_arithmetic_v<U> || is_complex_v<U>)
    friend simd_type operator+(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return lhs + simd_type(value_t(rhs, U{}));
      } else {
        return lhs + simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires(std::is_arithmetic_v<U> || is_complex_v<U>)
    friend simd_type operator+(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(value_t(lhs, U{})) + rhs;
      } else {
        return simd_type(static_cast<value_t>(lhs)) + rhs;
      }
    }

    template <typename U>
      requires(std::is_arithmetic_v<U> || is_complex_v<U>)
    friend simd_type operator-(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return lhs - simd_type(value_t(rhs, U{}));
      } else {
        return lhs - simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires(std::is_arithmetic_v<U> || is_complex_v<U>)
    friend simd_type operator-(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(value_t(lhs, U{})) - rhs;
      } else {
        return simd_type(static_cast<value_t>(lhs)) - rhs;
      }
    }

    template <typename U>
      requires(std::is_arithmetic_v<U> || is_complex_v<U>)
    friend simd_type operator*(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(lhs.value * static_cast<scalar_t>(rhs));
      } else {
        return lhs * simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires(std::is_arithmetic_v<U> || is_complex_v<U>)
    friend simd_type operator*(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(static_cast<scalar_t>(lhs) * rhs.value);
      } else {
        return simd_type(static_cast<value_t>(lhs)) * rhs;
      }
    }

    template <typename U>
      requires(std::is_arithmetic_v<U> || is_complex_v<U>)
    friend simd_type operator/(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(lhs.value / static_cast<scalar_t>(rhs));
      } else {
        return lhs / simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires(std::is_arithmetic_v<U> || is_complex_v<U>)
    friend simd_type operator/(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(static_cast<scalar_t>(lhs) / rhs.value);
      } else {
        return simd_type(static_cast<value_t>(lhs)) / rhs;
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    simd_type &operator+=(const U &other) {
      *this = *this + other;
      return *this;
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    simd_type &operator-=(const U &other) {
      *this = *this - other;
      return *this;
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    simd_type &operator*=(const U &other) {
      *this = *this * other;
      return *this;
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    simd_type &operator/=(const U &other) {
      *this = *this / other;
      return *this;
    }
  };

} // namespace nda