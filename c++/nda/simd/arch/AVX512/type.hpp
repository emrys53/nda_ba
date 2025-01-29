#pragma once
#ifdef __AVX512F__

#include <cstddef>
#include <complex>
#include "immintrin.h"
#include "../AVX/type.hpp"
#include "../macros.hpp"

namespace nda {
  template <>
  class simd_type<int32_t, 16, abi_tag::AVX512> {
    using intrinsic_t = __m512i;
    using value_t     = int32_t;

    intrinsic_t value{};

    public:
    explicit simd_type(intrinsic_t v) : value(v) {}

    static constexpr size_t size() { return 16UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); };

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const value_t *from) { value = _mm512_load_epi32(from); }
    void load_unaligned(const value_t *from) { value = _mm512_loadu_epi32(from); }
    void store(value_t *to) const { _mm512_store_epi32(to, value); }
    void store_unaligned(value_t *to) const { _mm512_storeu_epi32(to, value); }

    simd_type() : value(_mm512_setzero_epi32()) {}

    simd_type(const value_t v) : value(_mm512_set1_epi32(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm512_add_epi32(value, other.value)}; }

    simd_type operator-(const simd_type &other) const { return simd_type{_mm512_sub_epi32(value, other.value)}; }

    simd_type operator*(const simd_type &other) const { return simd_type{_mm512_mullo_epi32(value, other.value)}; }

    simd_type operator/(const simd_type &other) const {
      const simd_i8 lo_1(_mm512_extracti64x4_epi64(value, 0));
      const simd_i8 hi_1(_mm512_extracti64x4_epi64(value, 1));
      const simd_i8 lo_2(_mm512_extracti64x4_epi64(other.value, 0));
      const simd_i8 hi_2(_mm512_extracti64x4_epi64(other.value, 1));
      const simd_i8 lo = lo_1 / lo_2;
      const simd_i8 hi = hi_1 / hi_2;
      return simd_type{_mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1)};
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
      const int cmp = _mm512_cmp_epi32_mask(value, other.value, 0x00);
      return cmp == 0xFFFF;
    }
    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<int64_t, 8, abi_tag::AVX512> {
    using intrinsic_t = __m512i;
    using value_t     = int64_t;

    intrinsic_t value{};

    public:
    explicit simd_type(intrinsic_t v) : value(v) {}

    static constexpr size_t size() { return 8UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); };

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const value_t *from) { value = _mm512_load_epi64(from); }
    void load_unaligned(const value_t *from) { value = _mm512_loadu_epi64(from); }
    void store(value_t *to) const { _mm512_store_epi64(to, value); }
    void store_unaligned(value_t *to) const { _mm512_storeu_epi64(to, value); }

    simd_type() : value(_mm512_setzero_si512()) {}

    simd_type(const value_t v) : value(_mm512_set1_epi64(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm512_add_epi64(value, other.value)}; }

    simd_type operator-(const simd_type &other) const { return simd_type{_mm512_sub_epi64(value, other.value)}; }

    simd_type operator*(const simd_type &other) const {
#ifdef __AVX512DQ__
      return simd_type{_mm512_mullo_epi64(value, other.value)};
#else
      return simd_type{_mm512_mullox_epi64(value, other.value)};
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
      const int cmp = _mm512_cmp_epi64_mask(value, other.value, 0x00);
      return cmp == 0xFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<float, 32, abi_tag::AVX512> {
    using intrinsic_t = __m512;
    using value_t     = float;

    intrinsic_t value{};
    explicit simd_type(intrinsic_t v) : value(v) {}

    public:
    static constexpr size_t size() { return 16UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); };

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const value_t *from) { value = _mm512_load_ps(from); }
    void load_unaligned(const value_t *from) { value = _mm512_loadu_ps(from); }
    void store(value_t *to) const { _mm512_store_ps(to, value); }
    void store_unaligned(value_t *to) const { _mm512_storeu_ps(to, value); }

    simd_type() : value(_mm512_setzero_ps()) {}

    simd_type(const value_t v) : value(_mm512_set1_ps(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm512_add_ps(value, other.value)}; }

    simd_type operator-(const simd_type &other) const { return simd_type{_mm512_sub_ps(value, other.value)}; }

    simd_type operator*(const simd_type &other) const { return simd_type{_mm512_mul_ps(value, other.value)}; }

    simd_type operator/(const simd_type &other) const { return simd_type{_mm512_div_ps(value, other.value)}; }

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
      const int cmp = _mm512_cmp_ps_mask(value, other.value, 0x00);
      return cmp == 0xFFFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<double, 8, abi_tag::AVX512> {
    using intrinsic_t = __m512d;
    using value_t     = double;

    intrinsic_t value{};
    explicit simd_type(intrinsic_t v) : value(v) {}

    public:
    static constexpr size_t size() { return 8UL; }
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const value_t *from) { value = _mm512_load_pd(from); }
    void load_unaligned(const value_t *from) { value = _mm512_loadu_pd(from); }
    void store(value_t *to) const { _mm512_store_pd(to, value); }
    void store_unaligned(value_t *to) const { _mm512_storeu_pd(to, value); }

    simd_type() : value(_mm512_setzero_pd()) {}

    simd_type(const value_t v) : value(_mm512_set1_pd(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm512_add_pd(value, other.value)}; }

    simd_type operator-(const simd_type &other) const { return simd_type{_mm512_sub_pd(value, other.value)}; }

    simd_type operator*(const simd_type &other) const { return simd_type{_mm512_mul_pd(value, other.value)}; }

    simd_type operator/(const simd_type &other) const { return simd_type{_mm512_div_pd(value, other.value)}; }

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
      const int cmp = _mm512_cmp_pd_mask(value, other.value, 0x00);
      return cmp == 0xF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<std::complex<float>, 8, abi_tag::AVX512> {
    using intrinsic_t = __m512;
    using value_t     = std::complex<float>;
    using complex_t   = float;

    intrinsic_t value;
    simd_type(intrinsic_t v) : value(v) {}

    public:
    static constexpr size_t size() { return 8UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const complex_t *from) { value = _mm512_load_ps(from); }
    void load(const value_t *from) { value = _mm512_load_ps(from); }
    void load_unaligned(const complex_t *from) { value = _mm512_loadu_ps(from); }
    void load_unaligned(const value_t *from) { value = _mm512_loadu_ps(from); }

    void store(complex_t *to) const { _mm512_store_ps(to, value); }
    void store(value_t *to) const { _mm512_store_ps(to, value); }
    void store_unaligned(complex_t *to) const { _mm512_storeu_ps(to, value); }
    void store_unaligned(value_t *to) const { _mm512_storeu_ps(to, value); }

    simd_type() : value(_mm512_setzero_ps()) {}

    explicit simd_type(const value_t v)
       : value(_mm512_set_ps(v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real(),
                             v.imag(), v.real(), v.imag(), v.real())) {}

    explicit simd_type(const complex_t v) : value(_mm512_set1_ps(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm512_add_ps(value, other.value)}; }
    simd_type operator-(const simd_type &other) const { return simd_type{_mm512_sub_ps(value, other.value)}; }
    simd_type operator*(const simd_type &other) const {
      const intrinsic_t tmp2   = _mm512_mul_ps(_mm512_movehdup_ps(value), _mm512_permute_ps(other.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2)));
      const intrinsic_t result = _mm512_fmaddsub_ps(_mm512_moveldup_ps(value), other.value, tmp2);
      return simd_type{result};
    }

    simd_type operator/(const simd_type &other) const {
      // a+bi / c+di = (a+bi) * (c-di) = (ac+bd) (bc-bd)
      const intrinsic_t mask = _mm512_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
      const simd_type conj   = simd_type{_mm512_xor_ps(other.value, mask)};
      const simd_type upper  = (*this) * conj;
      const intrinsic_t flip = _mm512_permute_ps(other.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
      const intrinsic_t lower = _mm512_fmadd_ps(other.value, other.value, _mm512_mul_ps(flip, flip));
      return simd_type{_mm512_div_ps(upper.value, lower)};
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
      const int cmp = _mm512_cmp_ps_mask(value, other.value, 0x00);
      return cmp == 0xFFFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<std::complex<double>, 4, abi_tag::AVX512> {
    using intrinsic_t = __m512d;
    using value_t     = std::complex<double>;
    using complex_t   = double;

    intrinsic_t value;

    simd_type(intrinsic_t v) : value(v) {}

    public:
    static constexpr size_t size() { return 4UL; };
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const complex_t *from) { value = _mm512_load_pd(from); }
    void load(const value_t *from) { value = _mm512_load_pd(from); }

    void load_unaligned(const complex_t *from) { value = _mm512_loadu_pd(from); }
    void load_unaligned(const value_t *from) { value = _mm512_loadu_pd(from); }

    void store(complex_t *to) const { _mm512_store_pd(to, value); }
    void store(value_t *to) const { _mm512_store_pd(to, value); }

    void store_unaligned(complex_t *to) const { _mm512_storeu_pd(to, value); }
    void store_unaligned(value_t *to) const { _mm512_storeu_pd(to, value); }

    simd_type() : value(_mm512_setzero_pd()) {}

    explicit simd_type(const value_t v) : value(_mm512_set_pd(v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real(), v.imag(), v.real())) {}

    explicit simd_type(const complex_t v) : value(_mm512_set1_pd(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm512_add_pd(value, other.value)}; }
    simd_type operator-(const simd_type &other) const { return simd_type{_mm512_sub_pd(value, other.value)}; }
    simd_type operator*(const simd_type &other) const {
      const intrinsic_t tmp1   = _mm512_shuffle_pd(value, value, 0x0);
      const intrinsic_t tmp2   = _mm512_shuffle_pd(value, value, 0xFF);
      const intrinsic_t tmp3   = _mm512_shuffle_pd(other.value, other.value, 0x55);
      const intrinsic_t odd    = _mm512_mul_pd(tmp2, tmp3);
      const intrinsic_t result = _mm512_fmaddsub_pd(tmp1, other.value, odd);
      return simd_type{result};
    }
    simd_type operator/(const simd_type &other) const {
      // a+bi / c+di = (a+bi) * (c-di) = (ac+bd) (bc-ad)
      const intrinsic_t mask  = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
      const simd_type conj    = simd_type{_mm512_xor_pd(other.value, mask)};
      const simd_type upper   = (*this) * conj;
      const intrinsic_t flip  = _mm512_permute_pd(other.value, 0x5);
      const intrinsic_t lower = _mm512_fmadd_pd(other.value, other.value, _mm512_mul_pd(flip, flip));
      return simd_type{_mm512_div_pd(upper.value, lower)};
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
      const int cmp = _mm512_cmp_pd_mask(value, other.value, 0x00);
      return cmp == 0xF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };
} // namespace nda

#endif
