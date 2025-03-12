#pragma once
#ifdef __AVX__
#include <cstddef>
#include <complex>
#include "immintrin.h"

#include "../type_forward.hpp"
#include "../SSE/type.hpp"
#include "../macros.hpp"
#include "../abi.hpp"

namespace nda {
  template <>
  class simd_type<int32_t, 8, abi_tag::AVX> {
    public:
    using intrinsic_t = __m256i;
    using value_t     = int32_t;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm256_load_si256(reinterpret_cast<const intrinsic_t *>(v))) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm256_loadu_si256(reinterpret_cast<const intrinsic_t *>(v))) {}

    simd_type(simd_zero_initialize) : value(_mm256_setzero_si256()) {}

    simd_type(const std::array<value_t, 8> &v) : value(_mm256_loadu_si256(reinterpret_cast<const intrinsic_t *>(v.data()))) {}

    simd_type(const value_t v) : value(_mm256_set1_epi32(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 8UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); };

    void load(const value_t *from) { value = _mm256_load_si256(reinterpret_cast<const intrinsic_t *>(from)); }
    void load_unaligned(const value_t *from) { value = _mm256_loadu_si256(reinterpret_cast<const intrinsic_t *>(from)); }
    void store(value_t *to) const { _mm256_store_si256(reinterpret_cast<intrinsic_t *>(to), value); }
    void store_unaligned(value_t *to) const { _mm256_storeu_si256(reinterpret_cast<intrinsic_t *>(to), value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX2__
      return simd_type{_mm256_add_epi32(lhs.value, rhs.value)};
#else
      const simd_i4 lo{_mm_add_epi32(_mm256_extractf128_si256(lhs.value, 0), _mm256_extractf128_si256(rhs.value, 0))};
      const simd_i4 hi{_mm_add_epi32(_mm256_extractf128_si256(lhs.value, 1), _mm256_extractf128_si256(rhs.value, 1))};
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo.value), hi.value, 1)};
#endif
    }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX2__
      return simd_type{_mm256_sub_epi32(lhs.value, rhs.value)};
#else
      const simd_i4 lo{_mm_sub_epi32(_mm256_extractf128_si256(lhs.value, 0), _mm256_extractf128_si256(rhs.value, 0))};
      const simd_i4 hi{_mm_sub_epi32(_mm256_extractf128_si256(lhs.value, 1), _mm256_extractf128_si256(rhs.value, 1))};
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo.value), hi.value, 1)};
#endif
    }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX2__
      return simd_type{_mm256_mullo_epi32(lhs.value, rhs.value)};
#else
      const simd_i4 lo{_mm_mullo_epi32(_mm256_extractf128_si256(lhs.value, 0), _mm256_extractf128_si256(rhs.value, 0))};
      const simd_i4 hi{_mm_mullo_epi32(_mm256_extractf128_si256(lhs.value, 1), _mm256_extractf128_si256(rhs.value, 1))};
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo.value), hi.value, 1)};
#endif
    }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512F__
      return simd_type{_mm512_cvttpd_epi32(_mm512_div_pd(_mm512_cvtepi32_pd(lhs.value), _mm512_cvtepi32_pd(rhs.value)))};
#else
      const simd_i4 lo = simd_i4{_mm256_extractf128_si256(lhs.value, 0)} / simd_i4{_mm256_extractf128_si256(rhs.value, 0)};
      const simd_i4 hi = simd_i4{_mm256_extractf128_si256(lhs.value, 1)} / simd_i4{_mm256_extractf128_si256(rhs.value, 1)};
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1)};
#endif
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
#ifdef __AVX2__
      const intrinsic_t cmp = _mm256_cmpeq_epi32(value, other.value);
      return _mm256_movemask_epi8(cmp) == 0xFFFFFFFF;
#else
      const simd_i4 lo_1{_mm256_extractf128_si256(value, 0)};
      const simd_i4 hi_1{_mm256_extractf128_si256(value, 1)};
      const simd_i4 lo_2{_mm256_extractf128_si256(other.value, 0)};
      const simd_i4 hi_2{_mm256_extractf128_si256(other.value, 1)};
      return lo_1 == lo_2 and hi_1 == hi_2;
#endif
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX2__
      return simd_type{_mm256_xor_si256(lhs.value, rhs.value)};
#else
      return simd_type{_mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(lhs.value), _mm256_castsi256_ps(rhs.value)))};
#endif
    }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX2__
      return simd_type{_mm256_and_si256(lhs.value, rhs.value)};
#else
      return simd_type{_mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(lhs.value), _mm256_castsi256_ps(rhs.value)))};
#endif
    }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX2__
      return simd_type{_mm256_or_si256(lhs.value, rhs.value)};
#else
      return simd_type{_mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(lhs.value), _mm256_castsi256_ps(rhs.value)))};
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

    simd_type operator-() const { return simd_type{simd_zero_initialize_t} - *this; }
  };

  template <>
  class simd_type<int64_t, 4, abi_tag::AVX> {
    public:
    using intrinsic_t = __m256i;
    using value_t     = int64_t;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm256_load_si256(reinterpret_cast<const intrinsic_t *>(v))) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm256_loadu_si256(reinterpret_cast<const intrinsic_t *>(v))) {}

    simd_type(simd_zero_initialize) : value(_mm256_setzero_si256()) {}

    simd_type(const std::array<value_t, 4> &v) : value(_mm256_loadu_si256(reinterpret_cast<const intrinsic_t *>(v.data()))) {}

    simd_type(const value_t v) : value(_mm256_set1_epi64x(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 4UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); };

    void load(const value_t *from) { value = _mm256_load_si256(reinterpret_cast<const intrinsic_t *>(from)); }
    void load_unaligned(const value_t *from) { value = _mm256_loadu_si256(reinterpret_cast<const intrinsic_t *>(from)); }
    void store(value_t *to) const { _mm256_store_si256(reinterpret_cast<intrinsic_t *>(to), value); }
    void store_unaligned(value_t *to) const { _mm256_storeu_si256(reinterpret_cast<intrinsic_t *>(to), value); }

    // Friend addition operator.
    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX2__
      return simd_type{_mm256_add_epi64(lhs.value, rhs.value)};
#else
      const simd_l2 lo{_mm_add_epi64(_mm256_extractf128_si256(lhs.value, 0), _mm256_extractf128_si256(rhs.value, 0))};
      const simd_l2 hi{_mm_add_epi64(_mm256_extractf128_si256(lhs.value, 1), _mm256_extractf128_si256(rhs.value, 1))};
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)};
#endif
    }

    // Friend subtraction operator.
    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX2__
      return simd_type{_mm256_sub_epi64(lhs.value, rhs.value)};
#else
      const simd_l2 lo{_mm_sub_epi64(_mm256_extractf128_si256(lhs.value, 0), _mm256_extractf128_si256(rhs.value, 0))};
      const simd_l2 hi{_mm_sub_epi64(_mm256_extractf128_si256(lhs.value, 1), _mm256_extractf128_si256(rhs.value, 1))};
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)};
#endif
    }

    // Friend multiplication operator.
    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX2__
      const intrinsic_t upper32_a = _mm256_srli_epi64(lhs.value, 32);
      const intrinsic_t upper32_b = _mm256_srli_epi64(rhs.value, 32);

      // upper * lower
      const intrinsic_t mul1 = _mm256_mul_epu32(upper32_a, rhs.value);
      const intrinsic_t mul2 = _mm256_mul_epu32(upper32_b, lhs.value);
      // Gives us both upper*upper and lower*lower
      const intrinsic_t mul3 = _mm256_mul_epu32(lhs.value, rhs.value);

      const intrinsic_t high = _mm256_slli_epi64(_mm256_add_epi64(mul1, mul2), 32);
      return simd_type{_mm256_add_epi64(high, mul3)};
#else
      const simd_l2 lo_1{_mm256_extractf128_si256(lhs.value, 0)};
      const simd_l2 hi_1{_mm256_extractf128_si256(lhs.value, 1)};
      const simd_l2 lo_2{_mm256_extractf128_si256(rhs.value, 0)};
      const simd_l2 hi_2{_mm256_extractf128_si256(rhs.value, 1)};
      const simd_l2 lo = lo_1 * lo_2;
      const simd_l2 hi = hi_1 * hi_2;
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)};
#endif
    }

    // Friend division operator.
    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) {
      alignas(alignment()) std::array<value_t, size()> x{};
      alignas(alignment()) std::array<value_t, size()> y{};
      lhs.store(x.data());
      rhs.store(y.data());
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
#ifdef __AVX2__
      const intrinsic_t cmp = _mm256_cmpeq_epi64(value, other.value);
      return _mm256_movemask_epi8(cmp) == 0xFFFFFFFF;
#else
      const simd_l2 lo_1{_mm256_extractf128_si256(value, 0)};
      const simd_l2 hi_1{_mm256_extractf128_si256(value, 1)};
      const simd_l2 lo_2{_mm256_extractf128_si256(other.value, 0)};
      const simd_l2 hi_2{_mm256_extractf128_si256(other.value, 1)};
      return lo_1 == lo_2 and hi_1 == hi_2;
#endif
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX2__
      return simd_type{_mm256_xor_si256(lhs.value, rhs.value)};
#else
      return simd_type{_mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(lhs.value), _mm256_castsi256_ps(rhs.value)))};
#endif
    }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX2__
      return simd_type{_mm256_and_si256(lhs.value, rhs.value)};
#else
      return simd_type{_mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(lhs.value), _mm256_castsi256_ps(rhs.value)))};
#endif
    }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX2__
      return simd_type{_mm256_or_si256(lhs.value, rhs.value)};
#else
      return simd_type{_mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(lhs.value), _mm256_castsi256_ps(rhs.value)))};
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

    simd_type operator-() const { return simd_type{simd_zero_initialize_t} - *this; }
  };

  template <>
  class simd_type<float, 8, abi_tag::AVX> {
    public:
    using intrinsic_t = __m256;
    using value_t     = float;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm256_load_ps(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm256_loadu_ps(v)) {}

    simd_type(simd_zero_initialize) : value(_mm256_setzero_ps()) {}

    simd_type(const std::array<value_t, 8> &v) : value(_mm256_loadu_ps(v.data())) {}

    simd_type(const value_t v) : value(_mm256_set1_ps(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 8UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); };

    void load(const value_t *from) { value = _mm256_load_ps(from); }
    void load_unaligned(const value_t *from) { value = _mm256_loadu_ps(from); }
    void store(value_t *to) const { _mm256_store_ps(to, value); }
    void store_unaligned(value_t *to) const { _mm256_storeu_ps(to, value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_add_ps(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_sub_ps(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_mul_ps(lhs.value, rhs.value)}; }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_div_ps(lhs.value, rhs.value)}; }

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
      const intrinsic_t cmp = _mm256_cmp_ps(value, other.value, 0x00);
      return _mm256_movemask_ps(cmp) == 0xFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_xor_ps(lhs.value, rhs.value)}; }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_and_ps(lhs.value, rhs.value)}; }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_or_ps(lhs.value, rhs.value)}; }

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
      const intrinsic_t mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
      return *this ^ simd_type { mask };
    }
  };

  template <>
  class simd_type<double, 4, abi_tag::AVX> {
    public:
    using intrinsic_t = __m256d;
    using value_t     = double;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm256_load_pd(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm256_loadu_pd(v)) {}

    simd_type(simd_zero_initialize) : value(_mm256_setzero_pd()) {}

    simd_type(const std::array<value_t, 4> &v) : value(_mm256_loadu_pd(v.data())) {}

    simd_type(const value_t v) : value(_mm256_set1_pd(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 4UL; }
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    void load(const value_t *from) { value = _mm256_load_pd(from); }
    void load_unaligned(const value_t *from) { value = _mm256_loadu_pd(from); }
    void store(value_t *to) const { _mm256_store_pd(to, value); }
    void store_unaligned(value_t *to) const { _mm256_storeu_pd(to, value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_add_pd(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_sub_pd(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_mul_pd(lhs.value, rhs.value)}; }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_div_pd(lhs.value, rhs.value)}; }

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
      const intrinsic_t cmp = _mm256_cmp_pd(value, other.value, 0x00);
      return _mm256_movemask_pd(cmp) == 0xF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_xor_pd(lhs.value, rhs.value)}; }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_and_pd(lhs.value, rhs.value)}; }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_or_pd(lhs.value, rhs.value)}; }

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
      const intrinsic_t mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000ULL));
      return *this ^ simd_type { mask };
    }
  };

  template <>
  class simd_type<std::complex<float>, 4, abi_tag::AVX> {
    public:
    using intrinsic_t = __m256;
    using value_t     = std::complex<float>;
    using scalar_t    = float;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm256_load_ps(reinterpret_cast<const scalar_t *>(v))) {}

    simd_type(const scalar_t *v, simd_aligned_memory) : value(_mm256_load_ps(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm256_loadu_ps(reinterpret_cast<const scalar_t *>(v))) {}

    simd_type(const scalar_t *v, simd_unaligned_memory) : value(_mm256_loadu_ps(v)) {}

    simd_type(simd_zero_initialize) : value(_mm256_setzero_ps()) {}

    simd_type(const std::array<value_t, 4> &v) : value(_mm256_loadu_ps(reinterpret_cast<const scalar_t *>(v.data()))) {}

    simd_type(const std::array<scalar_t, 8> &v) : value(_mm256_loadu_ps(v.data())) {}

    simd_type(const value_t v) : value(_mm256_set_ps(v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real())) {}

    simd_type(const scalar_t v) : value(_mm256_set1_ps(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 4UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    void load(const scalar_t *from) { value = _mm256_load_ps(from); }
    void load(const value_t *from) { value = _mm256_load_ps(reinterpret_cast<const scalar_t *>(from)); }
    void load_unaligned(const scalar_t *from) { value = _mm256_loadu_ps(from); }
    void load_unaligned(const value_t *from) { value = _mm256_loadu_ps(reinterpret_cast<const scalar_t *>(from)); }

    void store(scalar_t *to) const { _mm256_store_ps(to, value); }
    void store(value_t *to) const { _mm256_store_ps(reinterpret_cast<scalar_t *>(to), value); }
    void store_unaligned(scalar_t *to) const { _mm256_storeu_ps(to, value); }
    void store_unaligned(value_t *to) const { _mm256_storeu_ps(reinterpret_cast<scalar_t *>(to), value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_add_ps(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_sub_ps(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) {
      const intrinsic_t tmp1 = _mm256_mul_ps(_mm256_movehdup_ps(lhs.value), _mm256_permute_ps(rhs.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2)));
      const intrinsic_t tmp2 = _mm256_moveldup_ps(lhs.value);
#ifdef __FMA__
      return simd_type{_mm256_fmaddsub_ps(tmp2, rhs.value, tmp1)};
#else
      return simd_type{_mm256_addsub_ps(_mm256_mul_ps(tmp2, rhs.value), tmp1)};
#endif
    }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) {
      const intrinsic_t mask = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
      const simd_type conj   = simd_type{_mm256_xor_ps(rhs.value, mask)};
      const simd_type upper  = lhs * conj;
      const intrinsic_t flip = _mm256_permute_ps(rhs.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
#ifdef __FMA__
      const intrinsic_t lower = _mm256_fmadd_ps(rhs.value, rhs.value, _mm256_mul_ps(flip, flip));
#else
      const intrinsic_t lower = _mm256_add_ps(_mm256_mul_ps(rhs.value, rhs.value), _mm256_mul_ps(flip, flip));
#endif
      return simd_type{_mm256_div_ps(upper.value, lower)};
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
      const intrinsic_t cmp = _mm256_cmp_ps(value, other.value, 0x00);
      return _mm256_movemask_ps(cmp) == 0xFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_xor_ps(lhs.value, rhs.value)}; }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_and_ps(lhs.value, rhs.value)}; }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_or_ps(lhs.value, rhs.value)}; }

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
        return simd_type(_mm256_mul_ps(lhs.value, _mm256_set1_ps(static_cast<scalar_t>(rhs))));
      } else {
        return lhs * simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator*(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm256_mul_ps(_mm256_set1_ps(static_cast<scalar_t>(lhs)), rhs.value));
      } else {
        return simd_type(static_cast<value_t>(lhs)) * rhs;
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator/(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm256_div_ps(lhs.value, _mm256_set1_ps(static_cast<scalar_t>(rhs))));
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
      const intrinsic_t mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
      return *this ^ simd_type { mask };
    }
  };

  template <>
  class simd_type<std::complex<double>, 2, abi_tag::AVX> {
    public:
    using intrinsic_t = __m256d;
    using value_t     = std::complex<double>;
    using scalar_t    = double;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm256_load_pd(reinterpret_cast<const scalar_t *>(v))) {}

    simd_type(const scalar_t *v, simd_aligned_memory) : value(_mm256_load_pd(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm256_loadu_pd(reinterpret_cast<const scalar_t *>(v))) {}

    simd_type(const scalar_t *v, simd_unaligned_memory) : value(_mm256_loadu_pd(v)) {}

    simd_type(simd_zero_initialize) : value(_mm256_setzero_pd()) {}

    simd_type(const std::array<value_t, 2> &v) : value(_mm256_loadu_pd(reinterpret_cast<const scalar_t *>(v.data()))) {}

    simd_type(const std::array<scalar_t, 4> &v) : value(_mm256_loadu_pd(v.data())) {}

    simd_type(const value_t v) : value(_mm256_set_pd(v.imag(), v.real(), v.imag(), v.real())) {}

    simd_type(const scalar_t v) : value(_mm256_set1_pd(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 2UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    void load(const scalar_t *from) { value = _mm256_load_pd(from); }
    void load(const value_t *from) { value = _mm256_load_pd(reinterpret_cast<const scalar_t *>(from)); }
    void load_unaligned(const scalar_t *from) { value = _mm256_loadu_pd(from); }
    void load_unaligned(const value_t *from) { value = _mm256_loadu_pd(reinterpret_cast<const scalar_t *>(from)); }

    void store(scalar_t *to) const { _mm256_store_pd(to, value); }
    void store(value_t *to) const { _mm256_store_pd(reinterpret_cast<scalar_t *>(to), value); }
    void store_unaligned(scalar_t *to) const { _mm256_storeu_pd(to, value); }
    void store_unaligned(value_t *to) const { _mm256_storeu_pd(reinterpret_cast<scalar_t *>(to), value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_add_pd(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_sub_pd(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) {
      const intrinsic_t tmp1 = _mm256_mul_pd(_mm256_permute_pd(lhs.value, 0xF), _mm256_permute_pd(rhs.value, 0x5));
      const intrinsic_t tmp2 = _mm256_movedup_pd(lhs.value);
#ifdef __FMA__
      return simd_type{_mm256_fmaddsub_pd(tmp2, rhs.value, tmp1)};
#else
      return simd_type{_mm256_addsub_pd(_mm256_mul_pd(tmp2, rhs.value), tmp1)};
#endif
    }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) {
      const intrinsic_t mask = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
      const simd_type conj   = simd_type{_mm256_xor_pd(rhs.value, mask)};
      const simd_type upper  = lhs * conj;
      const intrinsic_t flip = _mm256_permute_pd(rhs.value, 0x5);
#ifdef __FMA__
      const intrinsic_t lower = _mm256_fmadd_pd(rhs.value, rhs.value, _mm256_mul_pd(flip, flip));
#else
      const intrinsic_t lower = _mm256_add_pd(_mm256_mul_pd(rhs.value, rhs.value), _mm256_mul_pd(flip, flip));
#endif
      return simd_type{_mm256_div_pd(upper.value, lower)};
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
      const intrinsic_t cmp = _mm256_cmp_pd(value, other.value, 0x00);
      return _mm256_movemask_pd(cmp) == 0xF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_xor_pd(lhs.value, rhs.value)}; }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_and_pd(lhs.value, rhs.value)}; }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm256_or_pd(lhs.value, rhs.value)}; }

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
        return simd_type(_mm256_mul_pd(lhs.value, _mm256_set1_pd(static_cast<scalar_t>(rhs))));
      } else {
        return lhs * simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator*(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm256_mul_pd(_mm256_set1_pd(static_cast<scalar_t>(lhs)), rhs.value));
      } else {
        return simd_type(static_cast<value_t>(lhs)) * rhs;
      }
    }

    template <typename U>
      requires std::is_arithmetic_v<U> or is_complex_v<U>
    friend simd_type operator/(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm256_div_pd(lhs.value, _mm256_set1_pd(static_cast<scalar_t>(rhs))));
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
      const intrinsic_t mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000ULL));
      return *this ^ simd_type { mask };
    }
  };
} // namespace nda

#endif