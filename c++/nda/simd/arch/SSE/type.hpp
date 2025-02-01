#pragma once
#ifdef __SSE2__
#include "../abi.hpp"
#include "../macros.hpp"
#include "../type_forward.hpp"
#include "immintrin.h"

#include <complex>
#include <cstddef>
#include <initializer_list>

namespace nda {
  template <>
  class simd_type<int32_t, 4, abi_tag::SSE> {
    public:
    using intrinsic_t = __m128i;
    using value_t     = int32_t;

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

    void load(const value_t *from) { value = _mm_load_si128(reinterpret_cast<const intrinsic_t *>(from)); }
    void load_unaligned(const value_t *from) { value = _mm_loadu_si128(reinterpret_cast<const intrinsic_t *>(from)); }
    void store(value_t *to) const { _mm_store_si128(reinterpret_cast<intrinsic_t *>(to), value); }
    void store_unaligned(value_t *to) const { _mm_storeu_si128(reinterpret_cast<intrinsic_t *>(to), value); }

    simd_type() : value(_mm_setzero_si128()) {}

    simd_type(const value_t v) : value(_mm_set1_epi32(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm_add_epi32(value, other.value)}; }

    simd_type operator-(const simd_type &other) const { return simd_type{_mm_sub_epi32(value, other.value)}; }

    simd_type operator*(const simd_type &other) const {
#ifdef __SSE4_1__
      return simd_type{_mm_mullo_epi32(value, other.value)};
#else
      // https://stackoverflow.com/questions/17264399/fastest-way-to-multiply-two-vectors-of-32bit-integers-in-c-with-sse
      const intrinsic_t a13    = _mm_shuffle_epi32(value, 0xF5);       // (-,a3,-,a1)
      const intrinsic_t b13    = _mm_shuffle_epi32(other.value, 0xF5); // (-,b3,-,b1)
      const intrinsic_t prod02 = _mm_mul_epu32(value, other.value);    // (-,a2*b2,-,a0*b0)
      const intrinsic_t prod13 = _mm_mul_epu32(a13, b13);              // (-,a3*b3,-,a1*b1)
      const intrinsic_t prod01 = _mm_unpacklo_epi32(prod02, prod13);   // (-,-,a1*b1,a0*b0)
      const intrinsic_t prod23 = _mm_unpackhi_epi32(prod02, prod13);   // (-,-,a3*b3,a2*b2)
      const intrinsic_t prod   = _mm_unpacklo_epi64(prod01, prod23);   // (ab3,ab2,ab1,ab0)
      return simd_type{prod};
#endif
    }

    simd_type operator/(const simd_type &other) const {
#ifdef __AVX__
      return simd_type{_mm256_cvttpd_epi32(_mm256_div_pd(_mm256_cvtepi32_pd(value), _mm256_cvtepi32_pd(other.value)))};
#else
      const intrinsic_t q_lo = _mm_cvttpd_epi32(_mm_div_pd(_mm_cvtepi32_pd(value), _mm_cvtepi32_pd(other.value)));
      const intrinsic_t q_hi = _mm_cvttpd_epi32(_mm_div_pd(_mm_cvtepi32_pd(_mm_shuffle_epi32(value, NDA_SHUFFLE_MASK4(2, 3, 0, 1))),
                                                           _mm_cvtepi32_pd(_mm_shuffle_epi32(other.value, NDA_SHUFFLE_MASK4(2, 3, 0, 1)))));
      return simd_type{_mm_shuffle_epi32(_mm_unpacklo_epi32(q_lo, q_hi), NDA_SHUFFLE_MASK4(2, 1, 0, 3))};
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
      const intrinsic_t cmp = _mm_cmpeq_epi32(value, other.value);
      return _mm_movemask_epi8(cmp) == 0xFFFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); };

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<int64_t, 2, abi_tag::SSE> {
    public:
    using intrinsic_t = __m128i;
    using value_t     = int64_t;

    private:
    intrinsic_t value{};

    public:
    explicit simd_type(intrinsic_t v) : value(v) {}

    static constexpr size_t size() { return 2UL; }
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const value_t *from) { value = _mm_load_si128(reinterpret_cast<const intrinsic_t *>(from)); }
    void load_unaligned(const value_t *from) { value = _mm_loadu_si128(reinterpret_cast<const intrinsic_t *>(from)); }
    void store(value_t *to) const { _mm_store_si128(reinterpret_cast<intrinsic_t *>(to), value); }
    void store_unaligned(value_t *to) const { _mm_storeu_si128(reinterpret_cast<intrinsic_t *>(to), value); }

    simd_type() : value(_mm_setzero_si128()) {}

    simd_type(const value_t v) : value(_mm_set1_epi64x(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm_add_epi64(value, other.value)}; }

    simd_type operator-(const simd_type &other) const { return simd_type{_mm_sub_epi64(value, other.value)}; }

    simd_type operator*(const simd_type &other) const {
#ifdef __AVX512F__
      return simd_type{_mm_mullo_epi64(value, other.value)};
#else

      const intrinsic_t upper32_a = _mm_srli_epi64(value, 32);
      const intrinsic_t upper32_b = _mm_srli_epi64(other.value, 32);

      const intrinsic_t mul1 = _mm_mul_epu32(upper32_a, other.value);
      const intrinsic_t mul2 = _mm_mul_epu32(upper32_b, value);
      const intrinsic_t mul3 = _mm_mul_epu32(value, other.value);

      const intrinsic_t high = _mm_slli_epi64(_mm_add_epi64(mul1, mul2), 32);
      return simd_type{_mm_add_epi64(high, mul3)};
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
      const intrinsic_t cmp = _mm_cmpeq_epi32(value, other.value);
      return _mm_movemask_epi8(cmp) == 0xFFFF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<float, 4, abi_tag::SSE> {
    public:
    using intrinsic_t = __m128;
    using value_t     = float;

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

    void load(const value_t *from) { value = _mm_load_ps(from); }
    void load_unaligned(const value_t *from) { value = _mm_loadu_ps(from); }
    void store(value_t *to) const { _mm_store_ps(to, value); }
    void store_unaligned(value_t *to) const { _mm_storeu_ps(to, value); }

    simd_type() : value(_mm_setzero_ps()) {}

    simd_type(const value_t v) : value(_mm_set1_ps(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm_add_ps(value, other.value)}; }

    simd_type operator-(const simd_type &other) const { return simd_type{_mm_sub_ps(value, other.value)}; }

    simd_type operator*(const simd_type &other) const { return simd_type{_mm_mul_ps(value, other.value)}; }

    simd_type operator/(const simd_type &other) const { return simd_type{_mm_div_ps(value, other.value)}; }

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
      const intrinsic_t cmp = _mm_cmpeq_ps(value, other.value);
      return _mm_movemask_ps(cmp) == 0xF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<double, 2, abi_tag::SSE> {
    public:
    using intrinsic_t = __m128d;
    using value_t     = double;

    private:
    intrinsic_t value{};

    public:
    explicit simd_type(intrinsic_t v) : value(v) {}

    static constexpr size_t size() { return 2UL; }
    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const value_t *from) { value = _mm_load_pd(from); }
    void load_unaligned(const value_t *from) { value = _mm_loadu_pd(from); }
    void store(value_t *to) const { _mm_store_pd(to, value); }
    void store_unaligned(value_t *to) const { _mm_storeu_pd(to, value); }

    simd_type() : value(_mm_setzero_pd()) {}

    simd_type(const value_t v) : value(_mm_set1_pd(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm_add_pd(value, other.value)}; }

    simd_type operator-(const simd_type &other) const { return simd_type{_mm_sub_pd(value, other.value)}; }

    simd_type operator*(const simd_type &other) const { return simd_type{_mm_mul_pd(value, other.value)}; }

    simd_type operator/(const simd_type &other) const { return simd_type{_mm_div_pd(value, other.value)}; }

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
      const intrinsic_t cmp = _mm_cmpeq_pd(value, other.value);
      return _mm_movemask_pd(cmp) == 0x3;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<std::complex<float>, 2, abi_tag::SSE> {
    public:
    using intrinsic_t = __m128;
    using value_t     = std::complex<float>;
    using complex_t   = float;

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

    void load(const complex_t *from) { value = _mm_load_ps(from); }
    void load(const value_t *from) { value = _mm_load_ps(reinterpret_cast<const complex_t *>(from)); }
    void load_unaligned(const complex_t *from) { value = _mm_loadu_ps(from); }
    void load_unaligned(const value_t *from) { value = _mm_loadu_ps(reinterpret_cast<const complex_t *>(from)); }

    void store(complex_t *to) const { _mm_store_ps(to, value); }
    void store(value_t *to) const { _mm_store_ps(reinterpret_cast<complex_t *>(to), value); }
    void store_unaligned(complex_t *to) const { _mm_storeu_ps(to, value); }
    void store_unaligned(value_t *to) const { _mm_storeu_ps(reinterpret_cast<complex_t *>(to), value); }

    simd_type() : value(_mm_setzero_ps()) {}

    explicit simd_type(const value_t v) : value(_mm_set_ps(v.imag(), v.real(), v.imag(), v.real())) {}

    explicit simd_type(const complex_t v) : value(_mm_set1_ps(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm_add_ps(value, other.value)}; }

    simd_type operator-(const simd_type &other) const { return simd_type{_mm_sub_ps(value, other.value)}; }

    simd_type operator*(const simd_type &other) const {
      /*
         * a b c d
         * x y z w
         * (ax-by) (ay+bx) (cz-dw) (cw+dz)
         * movehdup = b b d d
         * shuffle  = y x w z
         * mulps    = by bx dw dz
         * moveldup = a a c c
         * tmp1 = by bx dw dz
         * tmp2 = a a c c
         * fmaddsub = ax ay cz cw -by +bx -dw +dz
         */
#ifdef __SSE3__
      const intrinsic_t tmp1 = _mm_mul_ps(_mm_movehdup_ps(value), _mm_shuffle_ps(other.value, other.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2)));
      const intrinsic_t tmp2 = _mm_moveldup_ps(value);
#else
      const intrinsic_t tmp1 = _mm_mul_ps(_mm_shuffle_ps(value, value, NDA_SHUFFLE_MASK4(1, 1, 3, 3)),
                                          _mm_shuffle_ps(other.value, other.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2)));
      const intrinsic_t tmp2 = _mm_shuffle_ps(value, value, NDA_SHUFFLE_MASK4(0, 0, 2, 2));
#endif
#ifdef __FMA__
      const intrinsic_t result = _mm_fmaddsub_ps(tmp2, other.value, tmp1);
#else
#ifdef __SSE3__
      const intrinsic_t result = _mm_addsub_ps(_mm_mul_ps(tmp2, other.value), tmp1);
#else
      const intrinsic_t mask   = _mm_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f);
      const intrinsic_t result = _mm_add_ps(_mm_mul_ps(tmp2, other.value), _mm_xor_ps(tmp1, mask));
#endif
#endif
      return simd_type{result};
    }

    simd_type operator/(const simd_type &other) const {
      // a+bi / c+di = (a+bi) * (c-di) = (ac+bd) (bc-bd)
      const intrinsic_t mask = _mm_setr_ps(0.0f, -0.0f, 0.0f, -0.0f);
      const simd_type conj   = simd_type{_mm_xor_ps(other.value, mask)};
      const simd_type upper  = (*this) * conj;
      const intrinsic_t flip = _mm_shuffle_ps(other.value, other.value, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
#ifdef __FMA__
      const intrinsic_t lower = _mm_fmadd_ps(other.value, other.value, _mm_mul_ps(flip, flip));
#else
      const intrinsic_t lower = _mm_add_ps(_mm_mul_ps(other.value, other.value), _mm_mul_ps(flip, flip));
#endif
      return simd_type{_mm_div_ps(upper.value, lower)};
    }

    simd_type &operator+=(const simd_type &other) {
      value = _mm_add_ps(value, other.value);
      return *this;
    }

    simd_type &operator-=(const simd_type &other) {
      value = _mm_sub_ps(value, other.value);
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
      const intrinsic_t cmp = _mm_cmpeq_ps(value, other.value);
      return _mm_movemask_ps(cmp) == 0xF;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };

  template <>
  class simd_type<std::complex<double>, 1, abi_tag::SSE> {
    public:
    using intrinsic_t = __m128d;
    using value_t     = std::complex<double>;
    using complex_t   = double;

    private:
    intrinsic_t value;

    public:
    explicit simd_type(intrinsic_t v) : value(v) {}

    static constexpr size_t size() { return 1UL; };

    static constexpr size_t alignment() { return size() * sizeof(value_t); }

    simd_type(const simd_type &other)            = default;
    simd_type &operator=(const simd_type &other) = default;
    simd_type(simd_type &&other)                 = default;
    simd_type &operator=(simd_type &&other)      = default;

    void load(const complex_t *from) { value = _mm_load_pd(from); }
    void load(const value_t *from) { value = _mm_load_pd(reinterpret_cast<const complex_t *>(from)); }
    void load_unaligned(const complex_t *from) { value = _mm_loadu_pd(from); }
    void load_unaligned(const value_t *from) { value = _mm_loadu_pd(reinterpret_cast<const complex_t *>(from)); }

    void store(complex_t *to) const { _mm_store_pd(to, value); }
    void store(value_t *to) const { _mm_store_pd(reinterpret_cast<complex_t *>(to), value); }
    void store_unaligned(complex_t *to) const { _mm_storeu_pd(to, value); }
    void store_unaligned(value_t *to) const { _mm_storeu_pd(reinterpret_cast<complex_t *>(to), value); }

    simd_type() : value(_mm_setzero_pd()) {}

    explicit simd_type(const value_t v) : value(_mm_set_pd(v.imag(), v.real())) {}

    explicit simd_type(const complex_t v) : value(_mm_set1_pd(v)) {}

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

    simd_type operator+(const simd_type &other) const { return simd_type{_mm_add_pd(value, other.value)}; }

    simd_type operator-(const simd_type &other) const { return simd_type{_mm_sub_pd(value, other.value)}; }

    simd_type operator*(const simd_type &other) const {
      const intrinsic_t tmp1 = _mm_mul_pd(_mm_unpackhi_pd(value, value), _mm_shuffle_pd(other.value, other.value, NDA_SHUFFLE_MASK2(1, 0)));
      //(a+bi)*(c+di) = (ac-bd) +(bc+ad)
      //tmp1 = (b b) * (d c) = (bd bc)
#ifdef __SSE3__
      const intrinsic_t tmp2 = _mm_movedup_pd(value); // a a
#else
      const intrinsic_t tmp2 = _mm_unpacklo_pd(value, value);
#endif
#ifdef __FMA__
      const intrinsic_t result = _mm_fmaddsub_pd(tmp2, other.value, tmp1);
#else
#ifdef __SSE3__
      const intrinsic_t result = _mm_addsub_pd(_mm_mul_pd(tmp2, other.value), tmp1);
#else
      const intrinsic_t mask   = _mm_setr_pd(-0.0, 0.0);
      const intrinsic_t result = _mm_add_pd(_mm_mul_pd(tmp2, other.value), _mm_xor_pd(tmp1, mask));
#endif
#endif
      return simd_type{result};
    }

    simd_type operator/(const simd_type &other) const {
      // a+bi / c+di = (a+bi) * (c-di) = (ac+bd) (bc-ad)
      const intrinsic_t mask = _mm_setr_pd(0.0, -0.0);
      const simd_type conj   = simd_type{_mm_xor_pd(other.value, mask)};
      const simd_type upper  = (*this) * conj;
      const intrinsic_t flip = _mm_shuffle_pd(other.value, other.value, 0x1);
#ifdef __FMA__
      const intrinsic_t lower = _mm_fmadd_pd(other.value, other.value, _mm_mul_pd(flip, flip));
#else
      const intrinsic_t lower = _mm_add_pd(_mm_mul_pd(other.value, other.value), _mm_mul_pd(flip, flip));
#endif
      return simd_type{_mm_div_pd(upper.value, lower)};
    }

    simd_type &operator+=(const simd_type &other) {
      value = _mm_add_pd(value, other.value);
      return *this;
    }

    simd_type &operator-=(const simd_type &other) {
      value = _mm_sub_pd(value, other.value);
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
      const intrinsic_t cmp = _mm_cmpeq_pd(value, other.value);
      return _mm_movemask_pd(cmp) == 0x3;
    }

    bool operator!=(const simd_type &other) const { return not(*this == other); }

    operator intrinsic_t() const { return value; }
  };

} // namespace nda
#endif