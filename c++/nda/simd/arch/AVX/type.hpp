#pragma once
#ifdef __AVX__
#include <cstddef>
#include <complex>
#include "immintrin.h"

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
    intrinsic_t value{};

    public:
    explicit simd_type(intrinsic_t v) : value(v) {}

    static constexpr size_t size() { return 8UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); };

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const value_t *from) { value = _mm256_load_si256(reinterpret_cast<const intrinsic_t *>(from)); }
    void load_unaligned(const value_t *from) { value = _mm256_loadu_si256(reinterpret_cast<const intrinsic_t *>(from)); }
    void store(value_t *to) const { _mm256_store_si256(reinterpret_cast<intrinsic_t *>(to), value); }
    void store_unaligned(value_t *to) const { _mm256_storeu_si256(reinterpret_cast<intrinsic_t *>(to), value); }

    simd_type() : value(_mm256_setzero_si256()) {}

    simd_type(const value_t v) : value(_mm256_set1_epi32(v)) {}

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

    simd_type operator+(const simd_type &other) const {
#ifdef __AVX2__
      return simd_type{_mm256_add_epi32(value, other.value)};
#else
      const simd_i4 lo{_mm_add_epi32(_mm256_extractf128_si256(value, 0), _mm256_extractf128_si256(other.value, 0))};
      const simd_i4 hi{_mm_add_epi32(_mm256_extractf128_si256(value, 1), _mm256_extractf128_si256(other.value, 1))};
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)};
#endif
    }

    simd_type operator-(const simd_type &other) const {
#ifdef __AVX2__
      return simd_type{_mm256_sub_epi32(value, other.value)};
#else
      const simd_i4 lo{_mm_sub_epi32(_mm256_extractf128_si256(value, 0), _mm256_extractf128_si256(other.value, 0))};
      const simd_i4 hi{_mm_sub_epi32(_mm256_extractf128_si256(value, 1), _mm256_extractf128_si256(other.value, 1))};
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)};
#endif
    }

    simd_type operator*(const simd_type &other) const {
#ifdef __AVX2__
      return simd_type{_mm256_mullo_epi32(value, other.value)};
#else
      const simd_i4 lo{_mm_mullo_epi32(_mm256_extractf128_si256(value, 0), _mm256_extractf128_si256(other.value, 0))};
      const simd_i4 hi{_mm_mullo_epi32(_mm256_extractf128_si256(value, 1), _mm256_extractf128_si256(other.value, 1))};
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)};
#endif
    }

    simd_type operator/(const simd_type &other) const {
#ifdef __AVX512__
      return simd_type{_mm512_cvttpd_epi32(_mm512_div_pd(_mm512_cvtepi32_pd(value), _mm512_cvtepi32_pd(other.value)))};
#else
      const simd_i4 lo = simd_i4{_mm256_extractf128_si256(value, 0)} / simd_i4{_mm256_extractf128_si256(other.value, 0)};
      const simd_i4 hi = simd_i4{_mm256_extractf128_si256(value, 1)} / simd_i4{_mm256_extractf128_si256(other.value, 1)};
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1)};
#endif
    }

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

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<int64_t, 4, abi_tag::AVX> {
    public:
    using intrinsic_t = __m256i;
    using value_t     = int64_t;

    private:
    intrinsic_t value{};

    public:
    explicit simd_type(intrinsic_t v) : value(v) {}

    static constexpr size_t size() { return 4UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); };

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const value_t *from) { value = _mm256_load_si256(reinterpret_cast<const intrinsic_t *>(from)); }
    void load_unaligned(const value_t *from) { value = _mm256_loadu_si256(reinterpret_cast<const intrinsic_t *>(from)); }
    void store(value_t *to) const { _mm256_store_si256(reinterpret_cast<intrinsic_t *>(to), value); }
    void store_unaligned(value_t *to) const { _mm256_storeu_si256(reinterpret_cast<intrinsic_t *>(to), value); }

    simd_type() : value(_mm256_setzero_si256()) {}

    simd_type(const value_t v) : value(_mm256_set1_epi64x(v)) {}

    simd_type(std::initializer_list<long> l) {
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.size() != size()) {
        throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
                                 + " is not equal to size of register: " + std::to_string(size()));
      }
#endif
      load_unaligned(l.begin());
    }

    explicit simd_type(const value_t *v) { load(v); }

    simd_type operator+(const simd_type &other) const {
#ifdef __AVX2__
      return simd_type{_mm256_add_epi64(value, other.value)};
#else
      const simd_l2 lo{_mm_add_epi64(_mm256_extractf128_si256(value, 0), _mm256_extractf128_si256(other.value, 0))};
      const simd_l2 hi{_mm_add_epi64(_mm256_extractf128_si256(value, 1), _mm256_extractf128_si256(other.value, 1))};
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)};
#endif
    }

    simd_type operator-(const simd_type &other) const {
#ifdef __AVX2__
      return simd_type{_mm256_sub_epi64(value, other.value)};
#else
      const simd_l2 lo{_mm_sub_epi64(_mm256_extractf128_si256(value, 0), _mm256_extractf128_si256(other.value, 0))};
      const simd_l2 hi{_mm_sub_epi64(_mm256_extractf128_si256(value, 1), _mm256_extractf128_si256(other.value, 1))};
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)};
#endif
    }

    simd_type operator*(const simd_type &other) const {
#ifdef __AVX2__
      const intrinsic_t upper32_a = _mm256_srli_epi64(value, 32);
      const intrinsic_t upper32_b = _mm256_srli_epi64(other.value, 32);

      // upper * lower
      const intrinsic_t mul1 = _mm256_mul_epu32(upper32_a, other.value);
      const intrinsic_t mul2 = _mm256_mul_epu32(upper32_b, value);
      // Gives us both upper*upper and lower*lower
      const intrinsic_t mul3 = _mm256_mul_epu32(value, other.value);

      const intrinsic_t high = _mm256_slli_epi64(_mm256_add_epi64(mul1, mul2), 32);
      return simd_type{_mm256_add_epi64(high, mul3)};
#else
      const simd_l2 lo_1{_mm256_extractf128_si256(value, 0)};
      const simd_l2 hi_1{_mm256_extractf128_si256(value, 1)};
      const simd_l2 lo_2{_mm256_extractf128_si256(other.value, 0)};
      const simd_l2 hi_2{_mm256_extractf128_si256(other.value, 1)};
      const simd_l2 lo = lo_1 * lo_2;
      const simd_l2 hi = hi_1 * hi_2;
      return simd_type{_mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)};
#endif
    }

    simd_type operator/(const simd_type &other) const {
      alignas(alignment()) std::array<value_t, size()> x{};
      alignas(alignment()) std::array<value_t, size()> y{};
      this->store(x.data());
      other.store(y.data());
      for (int i = 0; i < size(); i++) { x[i] = x[i] / y[i]; }
      return simd_type{x.data()};
    }

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

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<float, 8, abi_tag::AVX> {
    public:
    using intrinsic_t = __m256;
    using value_t     = float;

    private:
    intrinsic_t value{};

    public:
    explicit simd_type(intrinsic_t v) : value(v) {}

    static constexpr size_t size() { return 8UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); };

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const value_t *from) { value = _mm256_load_ps(from); }
    void load_unaligned(const value_t *from) { value = _mm256_loadu_ps(from); }
    void store(value_t *to) const { _mm256_store_ps(to, value); }
    void store_unaligned(value_t *to) const { _mm256_storeu_ps(to, value); }

    simd_type() : value(_mm256_setzero_ps()) {}

    simd_type(const value_t v) : value(_mm256_set1_ps(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm256_add_ps(value, other.value)}; }

    simd_type operator-(const simd_type &other) const { return simd_type{_mm256_sub_ps(value, other.value)}; }

    simd_type operator*(const simd_type &other) const { return simd_type{_mm256_mul_ps(value, other.value)}; }

    simd_type operator/(const simd_type &other) const { return simd_type{_mm256_div_ps(value, other.value)}; }

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
      const intrinsic_t cmp = _mm256_cmp_ps(value, other.value, 0x00);
      return _mm256_movemask_ps(cmp) == 0xFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<double, 4, abi_tag::AVX> {
    public:
    using intrinsic_t = __m256d;
    using value_t     = double;

    private:
    intrinsic_t value{};

    public:
    explicit simd_type(intrinsic_t v) : value(v) {}

    static constexpr size_t size() { return 4UL; }
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const value_t *from) { value = _mm256_load_pd(from); }
    void load_unaligned(const value_t *from) { value = _mm256_loadu_pd(from); }
    void store(value_t *to) const { _mm256_store_pd(to, value); }
    void store_unaligned(value_t *to) const { _mm256_storeu_pd(to, value); }

    simd_type() : value(_mm256_setzero_pd()) {}

    simd_type(const value_t v) : value(_mm256_set1_pd(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm256_add_pd(value, other.value)}; }

    simd_type operator-(const simd_type &other) const { return simd_type{_mm256_sub_pd(value, other.value)}; }

    simd_type operator*(const simd_type &other) const { return simd_type{_mm256_mul_pd(value, other.value)}; }

    simd_type operator/(const simd_type &other) const { return simd_type{_mm256_div_pd(value, other.value)}; }

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
      const intrinsic_t cmp = _mm256_cmp_pd(value, other.value, 0x00);
      return _mm256_movemask_pd(cmp) == 0xF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };

  // c = a * b +  c
  template <>
  class simd_type<std::complex<float>, 4, abi_tag::AVX> {
    public:
    using intrinsic_t = __m256;
    using value_t     = std::complex<float>;
    using complex_t   = float;

    private:
    intrinsic_t value{};

    public:
    explicit simd_type(intrinsic_t v) : value(v) {}

    static constexpr size_t size() { return 4UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const complex_t *from) { value = _mm256_load_ps(from); }
    void load(const value_t *from) { value = _mm256_load_ps(reinterpret_cast<const complex_t *>(from)); }
    void load_unaligned(const complex_t *from) { value = _mm256_loadu_ps(from); }
    void load_unaligned(const value_t *from) { value = _mm256_loadu_ps(reinterpret_cast<const complex_t *>(from)); }

    void store(complex_t *to) const { _mm256_store_ps(to, value); }
    void store(value_t *to) const { _mm256_store_ps(reinterpret_cast<complex_t *>(to), value); }
    void store_unaligned(complex_t *to) const { _mm256_storeu_ps(to, value); }
    void store_unaligned(value_t *to) const { _mm256_storeu_ps(reinterpret_cast<complex_t *>(to), value); }

    simd_type() : value(_mm256_setzero_ps()) {}

    explicit simd_type(const value_t v) : value(_mm256_set_ps(v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real())) {}

    explicit simd_type(const complex_t v) : value(_mm256_set1_ps(v)) {}

    explicit simd_type(const value_t *v) { load(v); }

    explicit simd_type(const complex_t *v) { load(v); }

    simd_type(std::initializer_list<value_t> l) {
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.size() != size()) {
        throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
                                 + " is not equal to size of register: " + std::to_string(size()));
      }
#endif
      load_unaligned(l.begin());
    }

    simd_type(std::initializer_list<complex_t> l) {
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.size() != 2 * size()) {
        throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
                                 + " is not equal to size of register: " + std::to_string(size() * 2));
      }
#endif
      load_unaligned(l.begin());
    }

    simd_type operator+(const simd_type &other) const { return simd_type{_mm256_add_ps(value, other.value)}; }

    simd_type operator-(const simd_type &other) const { return simd_type{_mm256_sub_ps(value, other.value)}; }

    simd_type operator*(const simd_type &other) const {
      /*
       * movehdup = a b c d e f g h -> b b d d f f h h
       * x y z t w k l m -> y x t z k w m l
       * mul = by bx dt dz fk fw hm hl
       *
       */
      const intrinsic_t tmp1 = _mm256_mul_ps(_mm256_movehdup_ps(value), _mm256_permute_ps(other.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2)));
      const intrinsic_t tmp2 = _mm256_moveldup_ps(value);
#ifdef __FMA__
      const intrinsic_t result = _mm256_fmaddsub_ps(tmp2, other.value, tmp1);
#else
      const intrinsic_t result = _mm256_addsub_ps(_mm256_mul_ps(tmp2, other.value), tmp1);
#endif
      return simd_type{result};
    }

    simd_type operator/(const simd_type &other) const {
      // a+bi / c+di = (a+bi) * (c-di) = (ac+bd) (bc-bd)
      const intrinsic_t mask = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
      const simd_type conj   = simd_type{_mm256_xor_ps(other.value, mask)};
      const simd_type upper  = (*this) * conj;
      const intrinsic_t flip = _mm256_permute_ps(other.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
#ifdef __FMA__
      const intrinsic_t lower = _mm256_fmadd_ps(other.value, other.value, _mm256_mul_ps(flip, flip));
#else
      const intrinsic_t lower = _mm256_add_ps(_mm256_mul_ps(other.value, other.value), _mm256_mul_ps(flip, flip));
#endif
      return simd_type{_mm256_div_ps(upper.value, lower)};
    }

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
      const intrinsic_t cmp = _mm256_cmp_ps(value, other.value, 0x00);
      return _mm256_movemask_ps(cmp) == 0xFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<std::complex<double>, 2, abi_tag::AVX> {
    public:
    using intrinsic_t = __m256d;
    using value_t     = std::complex<double>;
    using complex_t   = double;

    private:
    intrinsic_t value;

    public:
    explicit simd_type(intrinsic_t v) : value(v) {}

    static constexpr size_t size() { return 2UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const complex_t *from) { value = _mm256_load_pd(from); }
    void load(const value_t *from) { value = _mm256_load_pd(reinterpret_cast<const complex_t *>(from)); }
    void load_unaligned(const complex_t *from) { value = _mm256_loadu_pd(from); }
    void load_unaligned(const value_t *from) { value = _mm256_loadu_pd(reinterpret_cast<const complex_t *>(from)); }

    void store(complex_t *to) const { _mm256_store_pd(to, value); }
    void store(value_t *to) const { _mm256_store_pd(reinterpret_cast<complex_t *>(to), value); }
    void store_unaligned(complex_t *to) const { _mm256_storeu_pd(to, value); }
    void store_unaligned(value_t *to) const { _mm256_storeu_pd(reinterpret_cast<complex_t *>(to), value); }

    simd_type() : value(_mm256_setzero_pd()) {}

    explicit simd_type(const value_t v) : value(_mm256_set_pd(v.imag(), v.real(), v.imag(), v.real())) {}

    explicit simd_type(const complex_t v) : value(_mm256_set1_pd(v)) {}

    explicit simd_type(const value_t *v) { load(v); }

    explicit simd_type(const complex_t *v) { load(v); }

    simd_type(std::initializer_list<value_t> l) {
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.size() != size()) {
        throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
                                 + " is not equal to size of register: " + std::to_string(size()));
      }
#endif
      load_unaligned(l.begin());
    }

    simd_type(std::initializer_list<complex_t> l) {
#ifdef NDA_ENFORCE_BOUNDCHECK
      if (l.size() != 2 * size()) {
        throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
                                 + " is not equal to size of register: " + std::to_string(size() * 2));
      }
#endif
      load_unaligned(l.begin());
    }

    simd_type operator+(const simd_type &other) const { return simd_type{_mm256_add_pd(value, other.value)}; }
    simd_type operator-(const simd_type &other) const { return simd_type{_mm256_sub_pd(value, other.value)}; }
    simd_type operator*(const simd_type &other) const {
      const intrinsic_t tmp1 = _mm256_mul_pd(_mm256_permute_pd(value, 0xF), _mm256_permute_pd(other.value, 0x5));
      const intrinsic_t tmp2 = _mm256_movedup_pd(value);
#ifdef __FMA__
      const intrinsic_t result = _mm256_fmaddsub_pd(tmp2, other.value, tmp1);
#else
      const intrinsic_t result = _mm256_addsub_pd(_mm256_mul_pd(tmp2, other.value), tmp1);
#endif
      return simd_type{result};
    }
    simd_type operator/(const simd_type &other) const {
      // a+bi / c+di = (a+bi) * (c-di) = (ac+bd) (bc-ad)
      const intrinsic_t mask = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
      const simd_type conj   = simd_type{_mm256_xor_pd(other.value, mask)};
      const simd_type upper  = (*this) * conj;
      const intrinsic_t flip = _mm256_permute_pd(other.value, 0x5);
#ifdef __FMA__
      const intrinsic_t lower = _mm256_fmadd_pd(other.value, other.value, _mm256_mul_pd(flip, flip));
#else
      const intrinsic_t lower = _mm256_add_pd(_mm256_mul_pd(other.value, other.value), _mm256_mul_pd(flip, flip));
#endif
      return simd_type{_mm256_div_pd(upper.value, lower)};
    }

    simd_type &operator+=(const simd_type &other) {
      value = (*this + other).value;
      return *this;
    }

    simd_type &operator-=(const simd_type &other) {
      value = (*this - other).value;
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

    operator intrinsic_t() const { return value; }
  };
} // namespace nda

#endif