#pragma once
#ifdef __AVX512F__

#include <cstddef>
#include <complex>
#include "immintrin.h"
#include "../type_forward.hpp"
#include "../AVX/type.hpp"

namespace nda {
  template <>
  class simd_type<int32_t, 16, abi_tag::AVX512> {
    public:
    using intrinsic_t = __m512i;
    using value_t     = int32_t;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm512_load_epi32(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm512_load_epi64(v)) {}

    simd_type(simd_zero_initialize) : value(_mm512_setzero_epi32()) {}

    simd_type(const std::array<value_t, 16> &v) : value(_mm512_loadu_epi32(v.data())) {}

    simd_type(const value_t v) : value(_mm512_set1_epi32(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 16UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); };

    void load(const value_t *from) { value = _mm512_load_epi32(from); }
    void load_unaligned(const value_t *from) { value = _mm512_loadu_epi32(from); }
    void store(value_t *to) const { _mm512_store_epi32(to, value); }
    void store_unaligned(value_t *to) const { _mm512_storeu_epi32(to, value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_add_epi32(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_sub_epi32(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_mullo_epi32(lhs.value, rhs.value)}; }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) {
      const simd_i8 lo_1(_mm512_extracti64x4_epi64(lhs.value, 0));
      const simd_i8 hi_1(_mm512_extracti64x4_epi64(lhs.value, 1));
      const simd_i8 lo_2(_mm512_extracti64x4_epi64(rhs.value, 0));
      const simd_i8 hi_2(_mm512_extracti64x4_epi64(rhs.value, 1));
      const simd_i8 lo = lo_1 / lo_2;
      const simd_i8 hi = hi_1 / hi_2;
      return simd_type{_mm512_inserti64x4(_mm512_castsi256_si512(lo.value), hi.value, 1)};
    }

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

    bool operator==(const simd_type &other) const {
      const int cmp = _mm512_cmp_epi32_mask(value, other.value, 0x00);
      return cmp == 0xFFFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_xor_si512(lhs.value, rhs.value)}; }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_and_si512(lhs.value, rhs.value)}; }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_or_si512(lhs.value, rhs.value)}; }

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

    // Friend functions for mixed arithmetic: scalar on the right.
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator+(const simd_type &lhs, const U &rhs) {
      return lhs + simd_type(static_cast<value_t>(rhs));
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator-(const simd_type &lhs, const U &rhs) {
      return lhs - simd_type(static_cast<value_t>(rhs));
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator*(const simd_type &lhs, const U &rhs) {
      return lhs * simd_type(static_cast<value_t>(rhs));
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator/(const simd_type &lhs, const U &rhs) {
      return lhs / simd_type(static_cast<value_t>(rhs));
    }

    // Friend functions for mixed arithmetic: scalar on the left.
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator+(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) + rhs;
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator-(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) - rhs;
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator*(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) * rhs;
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator/(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) / rhs;
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
    simd_type operator-() const { return simd_type{simd_zero_initialize_t} - *this; }
  };

  template <>
  class simd_type<int64_t, 8, abi_tag::AVX512> {
    public:
    using intrinsic_t = __m512i;
    using value_t     = int64_t;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm512_load_epi64(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm512_loadu_epi64(v)) {}

    simd_type(simd_zero_initialize) : value(_mm512_setzero_si512()) {}

    simd_type(const std::array<value_t, 8> &v) : value(_mm512_loadu_epi64(v.data())) {}

    simd_type(const value_t v) : value(_mm512_set1_epi64(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 8UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); };

    void load(const value_t *from) { value = _mm512_load_epi64(from); }
    void load_unaligned(const value_t *from) { value = _mm512_loadu_epi64(from); }
    void store(value_t *to) const { _mm512_store_epi64(to, value); }
    void store_unaligned(value_t *to) const { _mm512_storeu_epi64(to, value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_add_epi64(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_sub_epi64(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512DQ__
      return simd_type{_mm512_mullo_epi64(lhs.value, rhs.value)};
#else
      return simd_type{_mm512_mullox_epi64(lhs.value, rhs.value)};
#endif
    }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) {
      alignas(alignment()) std::array<value_t, size()> x{};
      alignas(alignment()) std::array<value_t, size()> y{};
      lhs.store(x.data());
      rhs.store(y.data());
      for (int i = 0; i < static_cast<int>(size()); i++) { x[i] = x[i] / y[i]; }
      return simd_type{x.data(), 0};
    }

    simd_type operator/(const simd_type &other) const {
      alignas(alignment()) std::array<value_t, size()> x{};
      alignas(alignment()) std::array<value_t, size()> y{};
      this->store(x.data());
      other.store(y.data());
      for (int i = 0; i < size(); i++) { x[i] = x[i] / y[i]; }
      return simd_type{x.data(), simd_aligned_memory_t};
    }

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

    bool operator==(const simd_type &other) const {
      const int cmp = _mm512_cmp_epi64_mask(value, other.value, 0x00);
      return cmp == 0xFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_xor_si512(lhs.value, rhs.value)}; }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_and_si512(lhs.value, rhs.value)}; }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_or_si512(lhs.value, rhs.value)}; }

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

    // Friend functions for mixed arithmetic: scalar on the right.
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator+(const simd_type &lhs, const U &rhs) {
      return lhs + simd_type(static_cast<value_t>(rhs));
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator-(const simd_type &lhs, const U &rhs) {
      return lhs - simd_type(static_cast<value_t>(rhs));
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator*(const simd_type &lhs, const U &rhs) {
      return lhs * simd_type(static_cast<value_t>(rhs));
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator/(const simd_type &lhs, const U &rhs) {
      return lhs / simd_type(static_cast<value_t>(rhs));
    }

    // Friend functions for mixed arithmetic: scalar on the left.
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator+(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) + rhs;
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator-(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) - rhs;
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator*(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) * rhs;
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator/(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) / rhs;
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

    simd_type operator-() const { return simd_type{simd_zero_initialize_t} - *this; }
  };

  template <>
  class simd_type<float, 16, abi_tag::AVX512> {
    public:
    using intrinsic_t = __m512;
    using value_t     = float;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm512_load_ps(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm512_loadu_ps(v)) {}

    simd_type(simd_zero_initialize) : value(_mm512_setzero_ps()) {}

    simd_type(const std::array<value_t, 16> &v) : value(_mm512_loadu_ps(v.data())) {}

    simd_type(const value_t v) : value(_mm512_set1_ps(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 16UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); };

    void load(const value_t *from) { value = _mm512_load_ps(from); }
    void load_unaligned(const value_t *from) { value = _mm512_loadu_ps(from); }
    void store(value_t *to) const { _mm512_store_ps(to, value); }
    void store_unaligned(value_t *to) const { _mm512_storeu_ps(to, value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_add_ps(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_sub_ps(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_mul_ps(lhs.value, rhs.value)}; }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_div_ps(lhs.value, rhs.value)}; }

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

    bool operator==(const simd_type &other) const {
      const int cmp = _mm512_cmp_ps_mask(value, other.value, 0x00);
      return cmp == 0xFFFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512DQ__
      return simd_type{_mm512_xor_ps(lhs.value, rhs.value)};
#else
      return simd_type{_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(lhs.value), _mm512_castps_si512(rhs.value)))};
#endif
    }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512DQ__
      return simd_type{_mm512_and_ps(lhs.value, rhs.value)};
#else
      return simd_type{_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(lhs.value), _mm512_castps_si512(rhs.value)))};
#endif
    }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512DQ__
      return simd_type{_mm512_or_ps(lhs.value, rhs.value)};
#else
      return simd_type{_mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(lhs.value), _mm512_castps_si512(rhs.value)))};
#endif
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

    // Friend functions for mixed arithmetic: scalar on the right.
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator+(const simd_type &lhs, const U &rhs) {
      return lhs + simd_type(static_cast<value_t>(rhs));
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator-(const simd_type &lhs, const U &rhs) {
      return lhs - simd_type(static_cast<value_t>(rhs));
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator*(const simd_type &lhs, const U &rhs) {
      return lhs * simd_type(static_cast<value_t>(rhs));
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator/(const simd_type &lhs, const U &rhs) {
      return lhs / simd_type(static_cast<value_t>(rhs));
    }

    // Friend functions for mixed arithmetic: scalar on the left.
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator+(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) + rhs;
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator-(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) - rhs;
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator*(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) * rhs;
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator/(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) / rhs;
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

    simd_type operator-() const {
      const intrinsic_t mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));
      return *this ^ simd_type { mask };
    }
  };

  template <>
  class simd_type<double, 8, abi_tag::AVX512> {
    public:
    using intrinsic_t = __m512d;
    using value_t     = double;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm512_load_pd(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm512_loadu_pd(v)) {}

    simd_type(simd_zero_initialize) : value(_mm512_setzero_pd()) {}

    simd_type(const std::array<value_t, 8> &v) : value(_mm512_loadu_pd(v.data())) {}

    simd_type(const value_t v) : value(_mm512_set1_pd(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 8UL; }
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    void load(const value_t *from) { value = _mm512_load_pd(from); }
    void load_unaligned(const value_t *from) { value = _mm512_loadu_pd(from); }
    void store(value_t *to) const { _mm512_store_pd(to, value); }
    void store_unaligned(value_t *to) const { _mm512_storeu_pd(to, value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_add_pd(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_sub_pd(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_mul_pd(lhs.value, rhs.value)}; }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_div_pd(lhs.value, rhs.value)}; }

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

    bool operator==(const simd_type &other) const {
      const int cmp = _mm512_cmp_pd_mask(value, other.value, 0x00);
      return cmp == 0xFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512DQ__
      return simd_type{_mm512_xor_pd(lhs.value, rhs.value)};
#else
      return simd_type{_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(lhs.value), _mm512_castpd_si512(rhs.value)))};
#endif
    }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512DQ__
      return simd_type{_mm512_and_pd(lhs.value, rhs.value)};
#else
      return simd_type{_mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(lhs.value), _mm512_castpd_si512(rhs.value)))};
#endif
    }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512DQ__
      return simd_type{_mm512_or_pd(lhs.value, rhs.value)};
#else
      return simd_type{_mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(lhs.value), _mm512_castpd_si512(rhs.value)))};
#endif
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

    // Friend functions for mixed arithmetic: scalar on the right.
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator+(const simd_type &lhs, const U &rhs) {
      return lhs + simd_type(static_cast<value_t>(rhs));
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator-(const simd_type &lhs, const U &rhs) {
      return lhs - simd_type(static_cast<value_t>(rhs));
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator*(const simd_type &lhs, const U &rhs) {
      return lhs * simd_type(static_cast<value_t>(rhs));
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator/(const simd_type &lhs, const U &rhs) {
      return lhs / simd_type(static_cast<value_t>(rhs));
    }

    // Friend functions for mixed arithmetic: scalar on the left.
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator+(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) + rhs;
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator-(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) - rhs;
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator*(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) * rhs;
    }
    template <typename U>
      requires std::is_arithmetic_v<U>
    friend simd_type operator/(const U &lhs, const simd_type &rhs) {
      return simd_type(static_cast<value_t>(lhs)) / rhs;
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

    simd_type operator-() const {
      const intrinsic_t mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x8000000000000000ULL));
      return *this ^ simd_type { mask };
    }
  };

  template <>
  class simd_type<std::complex<float>, 8, abi_tag::AVX512> {
    public:
    using intrinsic_t = __m512;
    using value_t     = std::complex<float>;
    using scalar_t    = float;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm512_load_ps(reinterpret_cast<const scalar_t *>(v))) {}

    simd_type(const scalar_t *v, simd_aligned_memory) : value(_mm512_load_ps(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm512_loadu_ps(reinterpret_cast<const scalar_t *>(v))) {}

    simd_type(const scalar_t *v, simd_unaligned_memory) : value(_mm512_loadu_ps(v)) {}

    simd_type(simd_zero_initialize) : value(_mm512_setzero_ps()) {}

    simd_type(const std::array<value_t, 8> &v) : value(_mm512_loadu_ps(reinterpret_cast<const scalar_t *>(v.data()))) {}

    simd_type(const std::array<scalar_t, 16> &v) : value(_mm512_loadu_ps(v.data())) {}

    explicit simd_type(const value_t v)
       : value(_mm512_set_ps(v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real(),
                             v.imag(), v.real(), v.imag(), v.real())) {}

    simd_type(const scalar_t v) : value(_mm512_set1_ps(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 8UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    void load(const scalar_t *from) { value = _mm512_load_ps(from); }
    void load(const value_t *from) { value = _mm512_load_ps(from); }
    void load_unaligned(const scalar_t *from) { value = _mm512_loadu_ps(from); }
    void load_unaligned(const value_t *from) { value = _mm512_loadu_ps(from); }

    void store(scalar_t *to) const { _mm512_store_ps(to, value); }
    void store(value_t *to) const { _mm512_store_ps(to, value); }
    void store_unaligned(scalar_t *to) const { _mm512_storeu_ps(to, value); }
    void store_unaligned(value_t *to) const { _mm512_storeu_ps(to, value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_add_ps(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_sub_ps(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) {
      const intrinsic_t tmp2   = _mm512_mul_ps(_mm512_movehdup_ps(lhs.value), _mm512_permute_ps(rhs.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2)));
      const intrinsic_t result = _mm512_fmaddsub_ps(_mm512_moveldup_ps(lhs.value), rhs.value, tmp2);
      return simd_type{result};
    }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) {
      const intrinsic_t mask =
         _mm512_castsi512_ps(_mm512_setr_epi32(0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000,
                                               0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000));
#ifdef __AVX512DQ__
      const simd_type conj(_mm512_xor_ps(rhs.value, mask));
#else
      const simd_type conj(_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(rhs.value), _mm512_castps_si512(mask))));
#endif
      const simd_type upper  = lhs * conj;
      const intrinsic_t flip = _mm512_permute_ps(rhs.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
#ifdef __FMA__
      const intrinsic_t lower = _mm512_fmadd_ps(rhs.value, rhs.value, _mm512_mul_ps(flip, flip));
#else
      const intrinsic_t lower = _mm512_add_ps(_mm512_mul_ps(rhs.value, rhs.value), _mm512_mul_ps(flip, flip));
#endif
      return simd_type{_mm512_div_ps(upper.value, lower)};
    }

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

    bool operator==(const simd_type &other) const {
      const int cmp = _mm512_cmp_ps_mask(value, other.value, 0x00);
      return cmp == 0xFFFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512DQ__
      return simd_type{_mm512_xor_ps(lhs.value, rhs.value)};
#else
      return simd_type{_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(lhs.value), _mm512_castps_si512(rhs.value)))};
#endif
    }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512DQ__
      return simd_type{_mm512_and_ps(lhs.value, rhs.value)};
#else
      return simd_type{_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(lhs.value), _mm512_castps_si512(rhs.value)))};
#endif
    }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512DQ__
      return simd_type{_mm512_or_ps(lhs.value, rhs.value)};
#else
      return simd_type{_mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(lhs.value), _mm512_castps_si512(rhs.value)))};
#endif
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
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator+(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return lhs + simd_type(value_t(rhs, U{}));
      } else {
        return lhs + simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator+(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(value_t(lhs, U{})) + rhs;
      } else {
        return simd_type(static_cast<value_t>(lhs)) + rhs;
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator-(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return lhs - simd_type(value_t(rhs, U{}));
      } else {
        return lhs - simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator-(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(value_t(lhs, U{})) - rhs;
      } else {
        return simd_type(static_cast<value_t>(lhs)) - rhs;
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator*(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm512_mul_ps(lhs.value, _mm512_set1_ps(static_cast<scalar_t>(rhs))));
      } else {
        return lhs * simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator*(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm512_mul_ps(_mm512_set1_ps(static_cast<scalar_t>(lhs)), rhs.value));
      } else {
        return simd_type(static_cast<value_t>(lhs)) * rhs;
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator/(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm512_div_ps(lhs.value, _mm512_set1_ps(static_cast<scalar_t>(rhs))));
      } else {
        return lhs / simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator/(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(value_t(lhs, U{})) / rhs;
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

    simd_type operator-() const {
      const intrinsic_t mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));
      return *this ^ simd_type { mask };
    }
  };

  template <>
  class simd_type<std::complex<double>, 4, abi_tag::AVX512> {
    public:
    using intrinsic_t = __m512d;
    using value_t     = std::complex<double>;
    using scalar_t    = double;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm512_load_pd(v)) {}

    simd_type(const scalar_t *v, simd_aligned_memory) : value(_mm512_load_pd(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm512_loadu_pd(v)) {}

    simd_type(const scalar_t *v, simd_unaligned_memory) : value(_mm512_loadu_pd(v)) {}

    simd_type(simd_zero_initialize) : value(_mm512_setzero_pd()) {}

    simd_type(const std::array<value_t, 4> &v) : value(_mm512_loadu_pd(v.data())) {}

    simd_type(const std::array<scalar_t, 8> &v) : value(_mm512_loadu_pd(v.data())) {}

    simd_type(const value_t v) : value(_mm512_set_pd(v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real())) {}

    simd_type(const scalar_t v) : value(_mm512_set1_pd(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 4UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    void load(const scalar_t *from) { value = _mm512_load_pd(from); }
    void load(const value_t *from) { value = _mm512_load_pd(from); }

    void load_unaligned(const scalar_t *from) { value = _mm512_loadu_pd(from); }
    void load_unaligned(const value_t *from) { value = _mm512_loadu_pd(from); }

    void store(scalar_t *to) const { _mm512_store_pd(to, value); }
    void store(value_t *to) const { _mm512_store_pd(to, value); }

    void store_unaligned(scalar_t *to) const { _mm512_storeu_pd(to, value); }
    void store_unaligned(value_t *to) const { _mm512_storeu_pd(to, value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_add_pd(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm512_sub_pd(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) {
      const intrinsic_t tmp1   = _mm512_shuffle_pd(lhs.value, lhs.value, 0x0);
      const intrinsic_t tmp2   = _mm512_shuffle_pd(lhs.value, lhs.value, 0xFF);
      const intrinsic_t tmp3   = _mm512_shuffle_pd(rhs.value, rhs.value, 0x55);
      const intrinsic_t odd    = _mm512_mul_pd(tmp2, tmp3);
      const intrinsic_t result = _mm512_fmaddsub_pd(tmp1, rhs.value, odd);
      return simd_type{result};
    }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) {
      const intrinsic_t mask = _mm512_castsi512_pd(
         _mm512_set_epi32(0x80000000, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x0));
#ifdef __AVX512DQ__
      const simd_type conj(_mm512_xor_pd(rhs.value, mask));
#else
      const simd_type conj(_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castps_si512(rhs.value), _mm512_castpd_si512(mask))));
#endif
      const simd_type upper  = lhs * conj;
      const intrinsic_t flip = _mm512_permute_pd(rhs.value, 0x55);
#ifdef __FMA__
      const intrinsic_t lower = _mm512_fmadd_pd(rhs.value, rhs.value, _mm512_mul_pd(flip, flip));
#else
      const intrinsic_t lower = _mm512_add_pd(_mm512_mul_pd(rhs.value, rhs.value), _mm512_mul_pd(flip, flip));
#endif
      return simd_type{_mm512_div_pd(upper.value, lower)};
    }

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

    bool operator==(const simd_type &other) const {
      const int cmp = _mm512_cmp_pd_mask(value, other.value, 0x00);
      return cmp == 0xFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512DQ__
      return simd_type{_mm512_xor_pd(lhs.value, rhs.value)};
#else
      return simd_type{_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(lhs.value), _mm512_castpd_si512(rhs.value)))};
#endif
    }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512DQ__
      return simd_type{_mm512_and_pd(lhs.value, rhs.value)};
#else
      return simd_type{_mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(lhs.value), _mm512_castpd_si512(rhs.value)))};
#endif
    }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512DQ__
      return simd_type{_mm512_or_pd(lhs.value, rhs.value)};
#else
      return simd_type{_mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(lhs.value), _mm512_castpd_si512(rhs.value)))};
#endif
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
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator+(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(value_t(lhs, U{})) + rhs;
      } else {
        return simd_type(static_cast<value_t>(lhs)) + rhs;
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator-(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return lhs - simd_type(value_t(rhs, U{}));
      } else {
        return lhs - simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator-(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(value_t(lhs, U{})) - rhs;
      } else {
        return simd_type(static_cast<value_t>(lhs)) - rhs;
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator*(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm512_mul_pd(lhs.value, _mm512_set1_pd(static_cast<scalar_t>(rhs))));
      } else {
        return lhs * simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator*(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm512_mul_pd(_mm512_set1_pd(static_cast<scalar_t>(lhs)), rhs.value));
      } else {
        return simd_type(static_cast<value_t>(lhs)) * rhs;
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator/(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm512_div_pd(lhs.value, _mm512_set1_pd(static_cast<scalar_t>(rhs))));
      } else {
        return lhs / simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires(std::is_arithmetic_v<U> || is_complex_v<U>)
    friend simd_type operator/(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(value_t(lhs, U{})) / rhs;
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

    simd_type operator-() const {
      const intrinsic_t mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x8000000000000000ULL));
      return *this ^ simd_type { mask };
    }
  };
} // namespace nda

#endif
