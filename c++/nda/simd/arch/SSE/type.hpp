#pragma once
#ifdef __SSE2__
#include "../abi.hpp"
#include "../macros.hpp"
#include "../type_forward.hpp"
#include "immintrin.h"

#include <complex>
#include <cstddef>
#include <initializer_list>
#include <array>

namespace nda {
  template <>
  class simd_type<int32_t, 4, abi_tag::SSE> {
    public:
    using intrinsic_t = __m128i;
    using value_t     = int32_t;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm_load_si128(reinterpret_cast<const intrinsic_t *>(v))) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm_loadu_si128(reinterpret_cast<const intrinsic_t *>(v))) {}

    simd_type(simd_zero_initialize) : value(_mm_setzero_si128()) {}

    simd_type(const std::array<value_t, 4> &v) : value(_mm_loadu_si128(reinterpret_cast<const intrinsic_t *>(v.data()))) {}

    simd_type(const value_t v) : value(_mm_set1_epi32(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 4UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); };

    void load(const value_t *from) { value = _mm_load_si128(reinterpret_cast<const intrinsic_t *>(from)); }
    void load_unaligned(const value_t *from) { value = _mm_loadu_si128(reinterpret_cast<const intrinsic_t *>(from)); }
    void store(value_t *to) const { _mm_store_si128(reinterpret_cast<intrinsic_t *>(to), value); }
    void store_unaligned(value_t *to) const { _mm_storeu_si128(reinterpret_cast<intrinsic_t *>(to), value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_add_epi32(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_sub_epi32(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) {
#ifdef __SSE4_1__
      return simd_type{_mm_mullo_epi32(lhs.value, rhs.value)};
#else
      // Fallback multiplication (see: https://stackoverflow.com/a/17264400)
      const intrinsic_t a13    = _mm_shuffle_epi32(lhs.value, 0xF5);  // (-, a3, -, a1)
      const intrinsic_t b13    = _mm_shuffle_epi32(rhs.value, 0xF5);  // (-, b3, -, b1)
      const intrinsic_t prod02 = _mm_mul_epu32(lhs.value, rhs.value); // (-, a2*b2, -, a0*b0)
      const intrinsic_t prod13 = _mm_mul_epu32(a13, b13);             // (-, a3*b3, -, a1*b1)
      const intrinsic_t prod01 = _mm_unpacklo_epi32(prod02, prod13);  // (-, -, a1*b1, a0*b0)
      const intrinsic_t prod23 = _mm_unpackhi_epi32(prod02, prod13);  // (-, -, a3*b3, a2*b2)
      return simd_type{_mm_unpacklo_epi64(prod01, prod23)};           // (a3*b3, a2*b2, a1*b1, a0*b0)
#endif
    }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX__
      // Use AVX intrinsics if available.
      return simd_type{_mm256_cvttpd_epi32(_mm256_div_pd(_mm256_cvtepi32_pd(lhs.value), _mm256_cvtepi32_pd(rhs.value)))};
#else
      // Fallback: perform division by converting elements to double,
      // then back to int32_t via truncation.
      const intrinsic_t q_lo = _mm_cvttpd_epi32(_mm_div_pd(_mm_cvtepi32_pd(lhs.value), _mm_cvtepi32_pd(rhs.value)));
      const intrinsic_t q_hi = _mm_cvttpd_epi32(_mm_div_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(lhs.value, NDA_SHUFFLE_MASK4(2, 3, 0, 1))),
                                                           _mm_cvtepi32_pd(_mm_shuffle_epi32(rhs.value, NDA_SHUFFLE_MASK4(2, 3, 0, 1)))));
      return simd_type{_mm_shuffle_epi32(_mm_unpacklo_epi32(q_lo, q_hi), NDA_SHUFFLE_MASK4(2, 1, 0, 3))};
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
      const intrinsic_t cmp = _mm_cmpeq_epi32(value, other.value);
      return _mm_movemask_epi8(cmp) == 0xFFFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); };

    //Bitwise Operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_xor_si128(lhs.value, rhs.value)}; }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_and_si128(lhs.value, rhs.value)}; }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_or_si128(lhs.value, rhs.value)}; }

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

    simd_type operator-() const { return simd_type(simd_zero_initialize_t) - *this; }
  };

  template <>
  class simd_type<int64_t, 2, abi_tag::SSE> {
    public:
    using intrinsic_t = __m128i;
    using value_t     = int64_t;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm_load_si128(reinterpret_cast<const intrinsic_t *>(v))) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm_loadu_si128(reinterpret_cast<const intrinsic_t *>(v))) {}

    simd_type(simd_zero_initialize) : value(_mm_setzero_si128()) {}

    simd_type(const std::array<value_t, 2> &v) : value(_mm_loadu_si128(reinterpret_cast<const intrinsic_t *>(v.data()))) {}

    simd_type(const value_t v) : value(_mm_set1_epi64x(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 2UL; }
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    void load(const value_t *from) { value = _mm_load_si128(reinterpret_cast<const intrinsic_t *>(from)); }
    void load_unaligned(const value_t *from) { value = _mm_loadu_si128(reinterpret_cast<const intrinsic_t *>(from)); }
    void store(value_t *to) const { _mm_store_si128(reinterpret_cast<intrinsic_t *>(to), value); }
    void store_unaligned(value_t *to) const { _mm_storeu_si128(reinterpret_cast<intrinsic_t *>(to), value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_add_epi64(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_sub_epi64(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) {
#ifdef __AVX512F__
      return simd_type{_mm_mullo_epi64(lhs.value, rhs.value)};
#else
      const intrinsic_t upper32_lhs = _mm_srli_epi64(lhs.value, 32);
      const intrinsic_t upper32_rhs = _mm_srli_epi64(rhs.value, 32);
      const intrinsic_t mul1        = _mm_mul_epu32(upper32_lhs, rhs.value);
      const intrinsic_t mul2        = _mm_mul_epu32(upper32_rhs, lhs.value);
      const intrinsic_t mul3        = _mm_mul_epu32(lhs.value, rhs.value);
      const intrinsic_t high        = _mm_slli_epi64(_mm_add_epi64(mul1, mul2), 32);
      return simd_type{_mm_add_epi64(high, mul3)};
#endif
    }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) {
      alignas(alignment()) std::array<value_t, size()> x, y;
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
      const intrinsic_t cmp = _mm_cmpeq_epi32(value, other.value);
      return _mm_movemask_epi8(cmp) == 0xFFFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_xor_si128(lhs.value, rhs.value)}; }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_and_si128(lhs.value, rhs.value)}; }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_or_si128(lhs.value, rhs.value)}; }

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
  class simd_type<float, 4, abi_tag::SSE> {
    public:
    using intrinsic_t = __m128;
    using value_t     = float;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm_load_ps(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm_loadu_ps(v)) {}

    simd_type(simd_zero_initialize) : value(_mm_setzero_ps()) {}

    simd_type(const std::array<value_t, 4> &v) : value(_mm_loadu_ps(v.data())) {}

    simd_type(const value_t v) : value(_mm_set1_ps(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 4UL; }
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    void load(const value_t *from) { value = _mm_load_ps(from); }
    void load_unaligned(const value_t *from) { value = _mm_loadu_ps(from); }
    void store(value_t *to) const { _mm_store_ps(to, value); }
    void store_unaligned(value_t *to) const { _mm_storeu_ps(to, value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_add_ps(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_sub_ps(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_mul_ps(lhs.value, rhs.value)}; }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_div_ps(lhs.value, rhs.value)}; }

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
      const intrinsic_t cmp = _mm_cmpeq_ps(value, other.value);
      return _mm_movemask_ps(cmp) == 0xF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    // Friend bitwise XOR operator.
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_xor_ps(lhs.value, rhs.value)}; }

    // Friend bitwise AND operator.
    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_and_ps(lhs.value, rhs.value)}; }

    // Friend bitwise OR operator.
    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_or_ps(lhs.value, rhs.value)}; }

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
      const intrinsic_t mask = _mm_castsi128_ps(_mm_setr_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
      return *this ^ simd_type { mask };
    }
  };

  template <>
  class simd_type<double, 2, abi_tag::SSE> {
    public:
    using intrinsic_t = __m128d;
    using value_t     = double;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm_load_pd(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm_loadu_pd(v)) {}

    simd_type(simd_zero_initialize) : value(_mm_setzero_pd()) {}

    simd_type(const std::array<value_t, 2> &v) : value(_mm_loadu_pd(v.data())) {}

    simd_type(const value_t v) : value(_mm_set1_pd(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 2UL; }
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    void load(const value_t *from) { value = _mm_load_pd(from); }
    void load_unaligned(const value_t *from) { value = _mm_loadu_pd(from); }
    void store(value_t *to) const { _mm_store_pd(to, value); }
    void store_unaligned(value_t *to) const { _mm_storeu_pd(to, value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_add_pd(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_sub_pd(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_mul_pd(lhs.value, rhs.value)}; }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_div_pd(lhs.value, rhs.value)}; }

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
      const intrinsic_t cmp = _mm_cmpeq_pd(value, other.value);
      return _mm_movemask_pd(cmp) == 0x3;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_xor_pd(lhs.value, rhs.value)}; }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_and_pd(lhs.value, rhs.value)}; }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_or_pd(lhs.value, rhs.value)}; }

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
      const intrinsic_t mask = _mm_castsi128_pd(_mm_setr_epi32(0x0, 0x80000000, 0x0, 0x80000000));
      return *this ^ simd_type { mask };
    }
  };

  template <>
  class simd_type<std::complex<float>, 2, abi_tag::SSE> {
    public:
    using intrinsic_t = __m128;
    using value_t     = std::complex<float>;
    using scalar_t    = float;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm_load_ps(reinterpret_cast<const scalar_t *>(v))) {}

    simd_type(const scalar_t *v, simd_aligned_memory) : value(_mm_load_ps(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm_loadu_ps(reinterpret_cast<const scalar_t *>(v))) {}

    simd_type(const scalar_t *v, simd_unaligned_memory) : value(_mm_loadu_ps(v)) {}

    simd_type(simd_zero_initialize) : value(_mm_setzero_ps()) {}

    simd_type(const std::array<value_t, 2> &v) : value(_mm_loadu_ps(reinterpret_cast<const scalar_t *>(v.data()))) {}

    simd_type(const std::array<scalar_t, 4> &v) : value(_mm_loadu_ps(v.data())) {}

    simd_type(const value_t v) : value(_mm_set_ps(v.imag(), v.real(), v.imag(), v.real())) {}

    simd_type(const scalar_t v) : value(_mm_set1_ps(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 2UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    void load(const scalar_t *from) { value = _mm_load_ps(from); }
    void load(const value_t *from) { value = _mm_load_ps(reinterpret_cast<const scalar_t *>(from)); }
    void load_unaligned(const scalar_t *from) { value = _mm_loadu_ps(from); }
    void load_unaligned(const value_t *from) { value = _mm_loadu_ps(reinterpret_cast<const scalar_t *>(from)); }

    void store(scalar_t *to) const { _mm_store_ps(to, value); }
    void store(value_t *to) const { _mm_store_ps(reinterpret_cast<scalar_t *>(to), value); }
    void store_unaligned(scalar_t *to) const { _mm_storeu_ps(to, value); }
    void store_unaligned(value_t *to) const { _mm_storeu_ps(reinterpret_cast<scalar_t *>(to), value); }

    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_add_ps(lhs.value, rhs.value)}; }

    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_sub_ps(lhs.value, rhs.value)}; }

    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) {
#ifdef __SSE3__
      const intrinsic_t tmp1 = _mm_mul_ps(_mm_movehdup_ps(lhs.value), _mm_shuffle_ps(rhs.value, rhs.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2)));
      const intrinsic_t tmp2 = _mm_moveldup_ps(lhs.value);
#else
      const intrinsic_t tmp1 = _mm_mul_ps(_mm_shuffle_ps(lhs.value, lhs.value, NDA_SHUFFLE_MASK4(1, 1, 3, 3)),
                                          _mm_shuffle_ps(rhs.value, rhs.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2)));
      const intrinsic_t tmp2 = _mm_shuffle_ps(lhs.value, lhs.value, NDA_SHUFFLE_MASK4(0, 0, 2, 2));
#endif
#ifdef __FMA__
      return simd_type{_mm_fmaddsub_ps(tmp2, rhs.value, tmp1)};
#else
#ifdef __SSE3__
      return simd_type{_mm_addsub_ps(_mm_mul_ps(tmp2, rhs.value), tmp1)};
#else
      const intrinsic_t mask = _mm_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f);
      return simd_type{_mm_add_ps(_mm_mul_ps(tmp2, rhs.value), _mm_xor_ps(tmp1, mask))};
#endif
#endif
    }

    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) {
      const intrinsic_t mask = _mm_setr_ps(0.0f, -0.0f, 0.0f, -0.0f);
      const simd_type conj   = simd_type{_mm_xor_ps(rhs.value, mask)};
      const simd_type upper  = lhs * conj;
      const intrinsic_t flip = _mm_shuffle_ps(rhs.value, rhs.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
#ifdef __FMA__
      const intrinsic_t lower = _mm_fmadd_ps(rhs.value, rhs.value, _mm_mul_ps(flip, flip));
#else
      const intrinsic_t lower = _mm_add_ps(_mm_mul_ps(rhs.value, rhs.value), _mm_mul_ps(flip, flip));
#endif
      return simd_type{_mm_div_ps(upper.value, lower)};
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
      const intrinsic_t cmp = _mm_cmpeq_ps(value, other.value);
      return _mm_movemask_ps(cmp) == 0xF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_xor_ps(lhs.value, rhs.value)}; }
    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_and_ps(lhs.value, rhs.value)}; }
    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_or_ps(lhs.value, rhs.value)}; }

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

    // Operator+ with scalar on left, SIMD on right.
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
        return simd_type(_mm_mul_ps(lhs.value, _mm_set1_ps(static_cast<scalar_t>(rhs))));
      } else {
        return lhs * simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires(std::is_arithmetic_v<U> || is_complex_v<U>)
    friend simd_type operator*(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm_mul_ps(_mm_set1_ps(static_cast<scalar_t>(lhs)), rhs.value));
      } else {
        return simd_type(static_cast<value_t>(lhs)) * rhs;
      }
    }

    template <typename U>
      requires(std::is_arithmetic_v<U> || is_complex_v<U>)
    friend simd_type operator/(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm_div_ps(lhs.value, _mm_set1_ps(static_cast<scalar_t>(rhs))));
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
      const intrinsic_t mask = _mm_castsi128_ps(_mm_setr_epi32(0x80000000, 0x80000000, 0x80000000, 0x80000000));
      return *this ^ simd_type { mask };
    }
  };

  template <>
  class simd_type<std::complex<double>, 1, abi_tag::SSE> {
    public:
    using intrinsic_t = __m128d;
    using value_t     = std::complex<double>;
    using scalar_t    = double;

    private:
    intrinsic_t value;

    public:
    simd_type() {} //NOLINT

    explicit simd_type(const intrinsic_t &v) : value(v) {}

    simd_type(const value_t *v, simd_aligned_memory) : value(_mm_load_pd(reinterpret_cast<const scalar_t *>(v))) {}

    simd_type(const scalar_t *v, simd_aligned_memory) : value(_mm_load_pd(v)) {}

    simd_type(const value_t *v, simd_unaligned_memory) : value(_mm_loadu_pd(reinterpret_cast<const scalar_t *>(v))) {}

    simd_type(const scalar_t *v, simd_unaligned_memory) : value(_mm_loadu_pd(v)) {}

    simd_type(simd_zero_initialize) : value(_mm_setzero_pd()) {}

    simd_type(const std::array<value_t, 1> &v) : value(_mm_loadu_pd(reinterpret_cast<const scalar_t *>(v.data()))) {}

    simd_type(const std::array<scalar_t, 2> &v) : value(_mm_loadu_pd(v.data())) {}

    simd_type(const value_t v) : value(_mm_set_pd(v.imag(), v.real())) {}

    simd_type(const scalar_t v) : value(_mm_set1_pd(v)) {}

    operator intrinsic_t() const { return value; }

    static constexpr size_t size() { return 1UL; };

    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    void load(const scalar_t *from) { value = _mm_load_pd(from); }
    void load(const value_t *from) { value = _mm_load_pd(reinterpret_cast<const scalar_t *>(from)); }
    void load_unaligned(const scalar_t *from) { value = _mm_loadu_pd(from); }
    void load_unaligned(const value_t *from) { value = _mm_loadu_pd(reinterpret_cast<const scalar_t *>(from)); }

    void store(scalar_t *to) const { _mm_store_pd(to, value); }
    void store(value_t *to) const { _mm_store_pd(reinterpret_cast<scalar_t *>(to), value); }
    void store_unaligned(scalar_t *to) const { _mm_storeu_pd(to, value); }
    void store_unaligned(value_t *to) const { _mm_storeu_pd(reinterpret_cast<scalar_t *>(to), value); }

    // Friend operator+ using _mm_add_pd.
    friend simd_type operator+(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_add_pd(lhs.value, rhs.value)}; }

    // Friend operator- using _mm_sub_pd.
    friend simd_type operator-(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_sub_pd(lhs.value, rhs.value)}; }

    // Friend operator* with conditional implementations.
    friend simd_type operator*(const simd_type &lhs, const simd_type &rhs) {
      // Multiply upper halves after shuffling.
      const intrinsic_t tmp1 = _mm_mul_pd(_mm_unpackhi_pd(lhs.value, lhs.value), _mm_shuffle_pd(rhs.value, rhs.value, NDA_SHUFFLE_MASK2(1, 0)));
#ifdef __SSE3__
      const intrinsic_t tmp2 = _mm_movedup_pd(lhs.value); // Duplicate each element.
#else
      const intrinsic_t tmp2 = _mm_unpacklo_pd(lhs.value, lhs.value);
#endif
#ifdef __FMA__
      return simd_type{_mm_fmaddsub_pd(tmp2, rhs.value, tmp1)};
#else
#ifdef __SSE3__
      return simd_type{_mm_addsub_pd(_mm_mul_pd(tmp2, rhs.value), tmp1)};
#else
      const intrinsic_t mask = _mm_setr_pd(-0.0, 0.0);
      return simd_type{_mm_add_pd(_mm_mul_pd(tmp2, rhs.value), _mm_xor_pd(tmp1, mask))};
#endif
#endif
    }

    // Friend operator/ using a conjugate-based division approach.
    friend simd_type operator/(const simd_type &lhs, const simd_type &rhs) {
      const intrinsic_t mask = _mm_setr_pd(0.0, -0.0);
      const simd_type conj   = simd_type{_mm_xor_pd(rhs.value, mask)};
      const simd_type upper  = lhs * conj;
      const intrinsic_t flip = _mm_shuffle_pd(rhs.value, rhs.value, 0x1);
#ifdef __FMA__
      const intrinsic_t lower = _mm_fmadd_pd(rhs.value, rhs.value, _mm_mul_pd(flip, flip));
#else
      const intrinsic_t lower = _mm_add_pd(_mm_mul_pd(rhs.value, rhs.value), _mm_mul_pd(flip, flip));
#endif
      return simd_type{_mm_div_pd(upper.value, lower)};
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
      const intrinsic_t cmp = _mm_cmpeq_pd(value, other.value);
      return _mm_movemask_pd(cmp) == 0x3;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    // Bitwise operators
    friend simd_type operator^(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_xor_pd(lhs.value, rhs.value)}; }

    friend simd_type operator&(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_and_pd(lhs.value, rhs.value)}; }

    friend simd_type operator|(const simd_type &lhs, const simd_type &rhs) { return simd_type{_mm_or_pd(lhs.value, rhs.value)}; }

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
        return simd_type(_mm_mul_pd(lhs.value, _mm_set1_pd(static_cast<scalar_t>(rhs))));
      } else {
        return lhs * simd_type(static_cast<value_t>(rhs));
      }
    }

    template <typename U>
      requires(std::is_arithmetic_v<U> || is_complex_v<U>)
    friend simd_type operator*(const U &lhs, const simd_type &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm_mul_pd(_mm_set1_pd(static_cast<scalar_t>(lhs)), rhs.value));
      } else {
        return simd_type(static_cast<value_t>(lhs)) * rhs;
      }
    }

    template <typename U>
      requires(std::is_arithmetic_v<U> || is_complex_v<U>)
    friend simd_type operator/(const simd_type &lhs, const U &rhs) {
      if constexpr (std::is_arithmetic_v<U>) {
        return simd_type(_mm_div_pd(lhs.value, _mm_set1_pd(static_cast<scalar_t>(rhs))));
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
      const intrinsic_t mask = _mm_castsi128_pd(_mm_setr_epi32(0x0, 0x80000000, 0x0, 0x80000000));
      return *this ^ simd_type { mask };
    }
  };

} // namespace nda
#endif