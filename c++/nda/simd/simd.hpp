#pragma once
/// To include experimental headers with libc++ and clang
#define _LIBCPP_ENABLE_EXPERIMENTAL

#include <experimental/simd>
#include <complex>
#include <array>
#include <type_traits>

#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include "arch/SSE/type.hpp"
#include "arch/type.hpp"

#include "../concepts.hpp"
#ifndef __clang__
namespace nda {
  /*
   * Width in bytes.
   */
  template <Vectorizable T>
  class simd {
    std::experimental::native_simd<T> value;

    // Proxy class for overloading [] operator.
    class simd_proxy {
      std::experimental::native_simd<T> &parent;
      size_t index;

      public:
      simd_proxy(std::experimental::native_simd<T> &parent, size_t index)
        : parent(parent),
          index(index) {}

      simd_proxy &operator=(const T &val) {
        parent[index] = val;
        return *this;
      }

      operator T() const { return parent[index]; }
    };

    public:
    simd()
      : value() {}

    explicit simd(T v)
      : value(v) {}

    explicit simd(T *v) { this->copy_from_aligned(v); }

    explicit simd(std::experimental::native_simd<T> v) : value(v) {}

    simd(const simd<T> &other) noexcept   = default; // Copy constructor
    simd(simd<T> &&other) noexcept        = default; // Move constructor
    simd &operator=(const simd<T> &other) = default; // Copy assignment
    simd &operator=(simd<T> &&other)      = default; // Move assignment
    ~simd()                               = default; // Destructor

    static constexpr size_t size() { return std::experimental::native_simd<T>::size(); }

    static constexpr size_t alignment(){return size() * sizeof(T); }

    void copy_from_aligned(T *data) { value.copy_from(data, std::experimental::vector_aligned); }

    void copy_to_aligned(T *data) const { value.copy_to(data, std::experimental::vector_aligned); }

    void copy_from_not_aligned(T *data) { value.copy_from(data, std::experimental::element_aligned); }

    void copy_to_not_aligned(T *data) const { value.copy_to(data, std::experimental::element_aligned); }

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

  template <Vectorizable T>
  class simd<std::complex<T>> {
    std::experimental::native_simd<T> value;

    class simd_complex_proxy {
      std::experimental::native_simd<T> &parent;
      size_t index;

      public:
      simd_complex_proxy(std::experimental::native_simd<T> &parent, size_t index)
        : parent(parent),
          index(index) {}

      simd_complex_proxy &operator=(const std::complex<T> &c) {
        parent[index]     = c.real();
        parent[index + 1] = c.imag();
        return *this;
      }

      operator std::complex<T>() const { return {parent[index], parent[index + 1]}; }
    };

    public:
    simd()
      : value() {};

    explicit simd(std::experimental::native_simd<T> v)
      : value(v) {}

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

    static constexpr size_t alignment() { return simd<T>::alignment();}

    void copy_from_aligned(T *data) { value.copy_from(data, std::experimental::vector_aligned); }

    void copy_to_aligned(T *data) const { value.copy_to(data, std::experimental::vector_aligned); }

    void copy_from_not_aligned(T *data) { value.copy_from(data, std::experimental::element_aligned); }

    void copy_to_not_aligned(T *data) const { value.copy_to(data, std::experimental::element_aligned); }

    simd_complex_proxy operator[](size_t index) { return {value, 2 * index}; }

    std::complex<T> operator[](size_t index) const { return {value[2 * index], value[2 * index + 1]}; }

    simd<std::complex<T>> operator-(const simd<std::complex<T>> &other) { return simd<std::complex<T>>(value - other.value); }

    simd<std::complex<T>> operator+(const simd<std::complex<T>> &other) { return simd<std::complex<T>>(value + other.value); }

    /// (a+bi) * (c+di) = ac + adi + bci -bd =  (ac-bd) + (ad + bc)i
    /// a b c d e f g h
    /// q w e r t y u i
    /// (aq-bw) (aw+bq) (ce-dr) (cr+de) (et-fy) (ey+ft) (gu-hi) (gi+hu)
    simd<std::complex<T>> operator*(const simd<std::complex<T>> &other) {
      // https://bitbucket.org/blaze-lib/blaze/src/master/blaze/math/simd/Mult.h
      alignas(64) simd<std::complex<T>> result;
#if defined(__x86_64__)
      if constexpr (std::is_same_v<T, double>) {
        if constexpr (size() == 1) {
          /// SSE3
          alignas(16) __m128d a, b, x, y, z;
          a = _mm_load_pd((double *)&this->value);
          b = _mm_load_pd((double *)&other.value);
          x = _mm_shuffle_pd(a, a, 0);
          z = _mm_mul_pd(x, b);
          x = _mm_shuffle_pd(a, a, 3);
          y = _mm_shuffle_pd(b, b, 1);
          y = _mm_mul_pd(x, y);
          z = _mm_addsub_pd(z, y);
          _mm_store_pd((double *)&result, z);
          return result;
        } else if constexpr (size() == 2) {
          /// AVX
          alignas(32) __m256d a, b, x, y, z;
          a = _mm256_load_pd((double *)&this->value);
          b = _mm256_load_pd((double *)&other.value);
          x = _mm256_shuffle_pd(a, a, 0);
          z = _mm256_mul_pd(x, b);
          x = _mm256_shuffle_pd(a, a, 15);
          y = _mm256_shuffle_pd(b, b, 5);
          y = _mm256_mul_pd(x, y);
          z = _mm256_addsub_pd(z, y);
          _mm256_store_pd((double *)&result, z);
          return result;
        } else if constexpr (size() == 4) {
          /// AVX512
          alignas(64) __m512d a, b;
          a                  = _mm512_load_pd((double *)&this->value);
          b                  = _mm512_load_pd((double *)&other.value);
          const __m512d a_ii = _mm512_permute_pd(a, 0b1'1'1'1'1'1'1'1);
          const __m512d b_ri = _mm512_permute_pd(b, 0b0'1'0'1'0'1'0'1);
          const __m512d a_rr = _mm512_permute_pd(a, 0);
          a                  = _mm512_fmaddsub_pd(a_rr, b, _mm512_mul_pd(a_ii, b_ri));
          _mm512_store_pd((double *)&result, a);
          return result;
        }
      } else if constexpr (std::is_same_v<T, float>) {
        if constexpr (size() == 2) {
          /// SSE3
          alignas(16) __m128 a, b, x, y, z;
          a = _mm_load_ps((float *)&this->value);
          b = _mm_load_ps((float *)&other.value);
          x = _mm_shuffle_ps(a, a, 0xA0);
          z = _mm_mul_ps(x, b);
          x = _mm_shuffle_ps(a, a, 0xF5);
          y = _mm_shuffle_ps(b, b, 0xB1);
          y = _mm_mul_ps(x, y);
          z = _mm_addsub_ps(z, y);
          _mm_store_ps((float *)&result, z);
          return result;
        } else if constexpr (size() == 4) {
          /// AVX
          alignas(32) __m256 a, b, x, y, z;
          a = _mm256_load_ps((float *)&this->value);
          b = _mm256_load_ps((float *)&other.value);
          x = _mm256_shuffle_ps(a, a, 0xA0);
          z = _mm256_mul_ps(x, b);
          x = _mm256_shuffle_ps(a, a, 0xF5);
          y = _mm256_shuffle_ps(b, b, 0xB1);
          y = _mm256_mul_ps(x, y);
          z = _mm256_addsub_ps(z, y);
          _mm256_store_ps((float *)&result, z);
          return result;
        } else if constexpr (size() == 8) {
          /// AVX512
          alignas(64) __m512 a, b;
          a                 = _mm512_load_ps((float *)&this->value);
          b                 = _mm512_load_ps((float *)&other.value);
          const __m512 a_ii = _mm512_permute_ps(a, 0b11'11'01'01);
          const __m512 b_ri = _mm512_permute_ps(b, 0b10'11'00'01);
          const __m512 a_rr = _mm512_permute_ps(a, 0b10'10'00'00);
          a                 = _mm512_fmaddsub_ps(a_rr, b, _mm512_mul_ps(a_ii, b_ri));
          _mm512_store_ps((float *)&result, a);
          return result;
        }
      }
#endif
      std::array<T, size() * 2> first, second, third, fourth, fifth;
      this->copy_to_not_aligned(first.data());
      other.copy_to_not_aligned(second.data());
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
      result = simd<std::complex<T>>{other.value * y + z * w};
      return result;
    }

    /// (a+bi)/(c+di) = ((a+bi)(c-di)) / c^2+d^2
    simd<std::complex<T>> operator/(const simd<std::complex<T>> &other) {
      std::experimental::native_simd<T> b([](int i) { return i % 2 == 0 ? 1 : -1; });

      simd<std::complex<T>> temp(b * other.value); // (c-di)

      simd<std::complex<T>> dividend = temp * *this; // (a+bi) * (c-di)
      std::array<T, size() * 2> first{};

      other.copy_to_not_aligned(first.data());
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
#endif