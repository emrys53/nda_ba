#pragma once
#include <experimental/simd>
#include <complex>
#include <array>
#include <utility>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace nda {
  /*
   * Width in bytes.
   */
  template <typename T>
  class simd {
    std::experimental::native_simd<T> value;

    // Proxy class for overloading [] operator.
    class simd_proxy {
      std::experimental::native_simd<T> &parent;
      size_t index;

      public:
      simd_proxy(std::experimental::native_simd<T> &parent, size_t index) : parent(parent), index(index) {}

      simd_proxy &operator=(const T &v) {
        parent[index] = v;
        return *this;
      }

      operator T() const { return parent[index]; }
    };

    public:
    simd() : value() {}

    explicit simd(T v) : value(v) {}

    explicit simd(T *v) { this->copy_from_aligned(v); }

    simd(const simd<T> &other) noexcept   = default; // Copy constructor
    simd(simd<T> &&other) noexcept        = default; // Move constructor
    simd &operator=(const simd<T> &other) = default; // Copy assignment
    simd &operator=(simd<T> &&other)      = default; // Move assignment
    ~simd()                               = default; // Destructor

    static constexpr size_t size() { return std::experimental::native_simd<T>::size(); }

    void copy_from_aligned(T *data) { value.copy_from(data, std::experimental::vector_aligned); }

    void copy_to_aligned(T *data) { value.copy_to(data, std::experimental::vector_aligned); }

    void copy_from_not_aligned(T *data) { value.copy_from(data, std::experimental::element_aligned); }

    void copy_to_not_aligned(T *data) { value.copy_to(data, std::experimental::element_aligned); }

    /// Operator Overloads

    simd_proxy operator[](size_t index) { return simd_proxy(value, index); }

    T operator[](size_t index) const { return value[index]; }

    simd<T> operator+(const simd<T> &other) { return simd<T>(value + other.value); }

    simd<T> operator-(const simd<T> &other) { return simd<T>(value - other.value); }

    simd<T> operator*(const simd<T> &other) { return simd<T>(value * other.value); }

    simd<T> operator/(const simd<T> &other) { return simd<T>(value / other.value); }

    simd<T> &operator+=(const simd<T> &other) {
      value += other.value;
      return *this;
    }

    simd<T> &operator-=(const simd<T> &other) {
      value -= other.value;
      return *this;
    }

    simd<T> &operator*=(const simd<T> &other) {
      value *= other.value;
      return *this;
    }

    simd<T> &operator/=(const simd<T> &other) {
      value /= other.value;
      return *this;
    }
  };

  template <typename T>
  class simd<std::complex<T>> {
    std::experimental::native_simd<T> value;

    class simd_complex_proxy {
      std::experimental::native_simd<T> &parent;
      size_t index;

      public:
      simd_complex_proxy(std::experimental::native_simd<T> &parent, size_t index) : parent(parent), index(index) {}

      simd_complex_proxy &operator=(const std::complex<T> &v) {
        parent[index]     = v.real();
        parent[index + 1] = v.imag();
        return *this;
      }

      operator std::complex<T>() const { return {parent[index], parent[index + 1]}; }
    };

    public:
    simd() : value() {};

    explicit simd(std::experimental::native_simd<T> v) : value(v) {}

    explicit simd(std::complex<T> v) {
      for (int i = 0; i < size(); ++i) {
        value[2 * i]     = v.real();
        value[2 * i + 1] = v.imag();
      }
    }

    explicit simd(T *v) { this->copy_from_aligned(v); }

    simd(const simd<std::complex<T>> &other) noexcept                    = default; // Copy constructor
    simd(simd<std::complex<T>> &&other) noexcept                         = default; // Move constructor
    simd<std::complex<T>> &operator=(const simd<std::complex<T>> &other) = default; // Copy assignment
    simd<std::complex<T>> &operator=(simd<std::complex<T>> &&other)      = default; // Move assignment
    ~simd()                                                              = default; // Destructor

    static constexpr size_t size() { return std::experimental::native_simd<T>::size() / 2; }

    void copy_from_aligned(T *data) { value.copy_from(data, std::experimental::vector_aligned); }

    void copy_to_aligned(T *data) { value.copy_to(data, std::experimental::vector_aligned); }

    void copy_from_not_aligned(T *data) { value.copy_from(data, std::experimental::element_aligned); }

    void copy_to_not_aligned(T *data) { value.copy_to(data, std::experimental::element_aligned); }

    simd_complex_proxy operator[](size_t index) { return {value, 2 * index}; }

    std::complex<T> operator[](size_t index) const { return {value[2 * index], value[2 * index + 1]}; }

    simd<std::complex<T>> operator-(const simd<std::complex<T>> &other) { return simd<std::complex<T>>(value - other.value); }

    simd<std::complex<T>> operator+(const simd<std::complex<T>> &other) { return simd<std::complex<T>>(value + other.value); }

    /// (a+bi) * (c+di) = ac + adi + bci -bd =  (ac-bd) + (ad + bc)i
    /// a b c d e f g h
    /// q w e r t y u i
    /// (aq-bw) (aw+bq) (ce-dr) (cr+de) (et-fy) (ey+ft) (gu-hi) (gi+hu)
    simd<std::complex<T>> operator*(const simd<std::complex<T>> &other) {
      std::array<T, size() * 2> first, second, third, fourth, fifth;
      value.copy_to(first.data(), std::experimental::element_aligned);
      other.value.copy_to(second.data(), std::experimental::element_aligned);
      for (int i = 0; i < size() * 2; i += 2) {
        third[i]      = first[i];
        third[i + 1]  = first[i];
        fourth[i]     = -first[i + 1];
        fourth[i + 1] = first[i + 1];
        fifth[i]      = second[i + 1];
        fifth[i + 1]  = second[i];
      }
      std::experimental::native_simd<T> y(third.data(), std::experimental::element_aligned);
      std::experimental::native_simd<T> z(fourth.data(), std::experimental::element_aligned);
      std::experimental::native_simd<T> w(fifth.data(), std::experimental::element_aligned);
      std::experimental::native_simd<T> result = other.value * y + z * w;
      return simd<std::complex<T>>(result);
    }

    /// (a+bi)/(c+di) = ((a+bi)(c-di)) / c^2+d^2
    simd<std::complex<T>> operator/(const simd<std::complex<T>> &other) {
      std::experimental::native_simd<T> b([](int i) { return i % 2 == 0 ? 1 : -1; });

      simd<std::complex<T>> temp(b * other.value); // (c-di)

      simd<std::complex<T>> dividend = temp * *this; // (a+bi) * (c-di)
      std::array<T, size() * 2> first{};

      other.value.copy_to(first.data(), std::experimental::element_aligned);
      for (int i = 0; i < size() * 2; i += 2) { std::swap(first[i], first[i + 1]); }
      b.copy_from(first.data(), std::experimental::element_aligned);
      std::experimental::native_simd<T> c2      = other.value * other.value;
      std::experimental::native_simd<T> d2      = b * b;
      std::experimental::native_simd<T> divisor = c2 + d2;
      simd<std::complex<T>> result(dividend.value / divisor);
      return result;
    }

    simd<std::complex<T>> &operator+=(const simd<std::complex<T>> &other) {
      value += other.value;
      return *this;
    }

    simd<std::complex<T>> &operator-=(const simd<std::complex<T>> &other) {
      value -= other.value;
      return *this;
    }

    simd<std::complex<T>> &operator*=(const simd<std::complex<T>> &other) {
      *this = *this * other;
      return *this;
    }

    simd<std::complex<T>> &operator/=(const simd<std::complex<T>> &other) {
      *this = *this / other;
      return *this;
    }
  };
} // namespace nda
