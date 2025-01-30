#pragma once
#include "../abi.hpp"

#include <cstddef>
#include <initializer_list>

namespace nda {

  template <typename  T>
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

    bool operator==(const simd_type &other) const {
      return value == other.value;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); };

    operator intrinsic_t() const { return value; }
  };

} // namespace nda