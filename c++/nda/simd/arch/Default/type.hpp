#pragma once
#include "../abi.hpp"
#include "../type_forward.hpp"

#include <cstddef>
#include <initializer_list>
#include <concepts>
#include <bit>

namespace nda {

  template <typename T>
    requires(std::same_as<T, int32_t> or std::same_as<T, int64_t> or std::same_as<T, long long> or std::same_as<T, float> or std::same_as<T, double>)
  class simd_type<T, 1, abi_tag::Default> {
    public:
    using intrinsic_t = T;
    using value_t     = T;

    private:
    intrinsic_t value{};

    public:
    explicit simd_type(intrinsic_t v) : value(v) {}

    static constexpr size_t size() { return 1UL; };
    static constexpr size_t alignment() { return 0UL; };

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const value_t *from) { value = *from; }
    void load_unaligned(const value_t *from) { value = *from; }
    void store(value_t *to) const { *to = value; }
    void store_unaligned(value_t *to) const { *to = value; }

    simd_type() : value(0) {}


    simd_type(std::initializer_list<value_t> l) {
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.size() != size()) {
        throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
                                 + " is not equal to size of register: " + std::to_string(size()));
      }
#endif
      load_unaligned(l.begin());
    }

    explicit simd_type(const value_t *v) { load(v); }


    simd_type operator+(const simd_type &other) const { return simd_type{value + other.value}; }

    simd_type operator-(const simd_type &other) const { return simd_type{value - other.value}; }

    simd_type operator*(const simd_type &other) const { return simd_type{value * other.value}; }

    simd_type operator/(const simd_type &other) const { return simd_type{value / other.value}; }

    simd_type &operator+=(const simd_type &other) {
      value = (*this + other).value;
      return *this;
    }

    simd_type &operator-=(const simd_type &other) {
      value = (*this - other).value;
      return *this;
    }

    simd_type &operator*=(const simd_type &other) {
      value = (*this * other).value;
      return *this;
    }

    simd_type &operator/=(const simd_type &other) {
      value = (*this / other).value;
      return *this;
    }

    simd_type operator-() const { return simd_type{-value}; }

    bool operator==(const simd_type &other) const { return value == other.value; }

    bool operator!=(const simd_type &other) const { return not(*this == other); };

    // Bitwise operators
    simd_type operator^(const simd_type &other) const {
      if constexpr (sizeof(T) == 4) {
        return simd_type{std::bit_cast<T>(std::bit_cast<int32_t>(value) ^ std::bit_cast<int32_t>(other.value))};
      } else {
        return simd_type{std::bit_cast<T>(std::bit_cast<int64_t>(value) ^ std::bit_cast<int64_t>(other.value))};
      }
    }
    simd_type operator&(const simd_type &other) const {
      if constexpr (sizeof(T) == 4) {
        return simd_type{std::bit_cast<T>(std::bit_cast<int32_t>(value) & std::bit_cast<int32_t>(other.value))};
      } else {
        return simd_type{std::bit_cast<T>(std::bit_cast<int64_t>(value) & std::bit_cast<int64_t>(other.value))};
      }
    }
    simd_type operator|(const simd_type &other) const {
      if constexpr (sizeof(T) == 4) {
        return simd_type{std::bit_cast<T>(std::bit_cast<int32_t>(value) | std::bit_cast<int32_t>(other.value))};
      } else {
        return simd_type{std::bit_cast<T>(std::bit_cast<int64_t>(value) | std::bit_cast<int64_t>(other.value))};
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

    operator intrinsic_t() const { return value; }
  };

  template <typename T>
  class simd_type<std::complex<T>, 1, abi_tag::Default> {
    public:
    using intrinsic_t = std::complex<T>;
    using value_t     = std::complex<T>;
    using scalar_t    = T;

    private:
    intrinsic_t value;

    public:
    explicit simd_type(intrinsic_t v) : value(v) {}

    static constexpr size_t size() { return 1UL; };

    static constexpr size_t alignment() { return 0UL; }

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

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

    simd_type() : value(0, 0) {}

    explicit simd_type(const scalar_t v) : value(v, v) {}

    explicit simd_type(const value_t *v) { load(v); }

    explicit simd_type(const scalar_t *v) { load(v); }

    simd_type(std::initializer_list<value_t> l) {
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.size() != size()) {
        throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
                                 + " is not equal to size of register: " + std::to_string(size()));
      }
#endif
      load_unaligned(l.begin());
    }

    simd_type(std::initializer_list<scalar_t> l) {
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.size() != 2 * size()) {
        throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
                                 + " is not equal to size of register: " + std::to_string(size() * 2));
      }
#endif
      load_unaligned(l.begin());
    }

    simd_type operator+(const simd_type &other) const { return simd_type{value + other.value}; }

    simd_type operator-(const simd_type &other) const { return simd_type{value - other.value}; }

    simd_type operator*(const simd_type &other) const { return simd_type{value * other.value}; }

    simd_type operator/(const simd_type &other) const { return simd_type{value / other.value}; }

    simd_type &operator+=(const simd_type &other) {
      value = (*this + other).value;
      return *this;
    }

    simd_type &operator-=(const simd_type &other) {
      value = (*this - other).value;
      return *this;
    }

    simd_type &operator*=(const simd_type &other) {
      value = (*this * other).value;
      return *this;
    }

    simd_type &operator/=(const simd_type &other) {
      value = (*this / other).value;
      return *this;
    }

    simd_type operator-() const { return simd_type{-value}; }

    bool operator==(const simd_type &other) const { return value == other.value; }

    bool operator!=(const simd_type &other) const { return not(*this == other); };

    // Bitwise operators
    simd_type operator^(const simd_type &other) const {
      if constexpr (sizeof(scalar_t) == 4) {
        T real = std::bit_cast<T>(std::bit_cast<int32_t>(value.real()) ^ std::bit_cast<int32_t>(other.value.real()));
        T imag = std::bit_cast<T>(std::bit_cast<int32_t>(value.imag()) ^ std::bit_cast<int32_t>(other.value.imag()));
        return simd_type{value_t{real, imag}};
      } else {
        T real = std::bit_cast<T>(std::bit_cast<int64_t>(value.real()) ^ std::bit_cast<int64_t>(other.value.real()));
        T imag = std::bit_cast<T>(std::bit_cast<int64_t>(value.imag()) ^ std::bit_cast<int64_t>(other.value.imag()));
        return simd_type{value_t{real, imag}};
      }
    }
    simd_type operator&(const simd_type &other) const {
      if constexpr (sizeof(scalar_t) == 4) {
        T real = std::bit_cast<T>(std::bit_cast<int32_t>(value.real()) & std::bit_cast<int32_t>(other.value.real()));
        T imag = std::bit_cast<T>(std::bit_cast<int32_t>(value.imag()) & std::bit_cast<int32_t>(other.value.imag()));
        return simd_type{value_t{real, imag}};
      } else {
        T real = std::bit_cast<T>(std::bit_cast<int64_t>(value.real()) & std::bit_cast<int64_t>(other.value.real()));
        T imag = std::bit_cast<T>(std::bit_cast<int64_t>(value.imag()) & std::bit_cast<int64_t>(other.value.imag()));
        return simd_type{value_t{real, imag}};
      }
    }
    simd_type operator|(const simd_type &other) const {
      if constexpr (sizeof(scalar_t) == 4) {
        T real = std::bit_cast<T>(std::bit_cast<int32_t>(value.real()) | std::bit_cast<int32_t>(other.value.real()));
        T imag = std::bit_cast<T>(std::bit_cast<int32_t>(value.imag()) | std::bit_cast<int32_t>(other.value.imag()));
        return simd_type{value_t{real, imag}};
      } else {
        T real = std::bit_cast<T>(std::bit_cast<int64_t>(value.real()) | std::bit_cast<int64_t>(other.value.real()));
        T imag = std::bit_cast<T>(std::bit_cast<int64_t>(value.imag()) | std::bit_cast<int64_t>(other.value.imag()));
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

    operator intrinsic_t() const { return value; }
  };

} // namespace nda