// #pragma once
// #include "../abi.hpp"
// #if (defined __ARM_NEON) || (defined __ARM_NEON__)
// #include <cstddef>
// #include <complex>
//
// #include <arm_neon.h>
//
// namespace nda {
//   template <>
//   class simd_type<int32_t, 4, abi_tag::NEON> {
//     public:
//     using intrinsic_t = int32x4_t;
//     using value_t     = int32_t;
//
//     private:
//     intrinsic_t value{};
//
//     public:
//     explicit simd_type(intrinsic_t v) : value(v) {}
//
//     static constexpr size_t size() { return 4UL; };
//     static constexpr size_t alignment() { return size() * sizeof(value_t); };
//
//     simd_type(const simd_type &other)            = default;
//     simd_type &operator=(const simd_type &other) = default;
//     simd_type(simd_type &&other)                 = default;
//     simd_type &operator=(simd_type &&other)      = default;
//
//     void load(const value_t *from) { value = vld1q_s32(from); }
//     void load_unaligned(const value_t *from) { value = vld1q_s32(from); }
//     void store(value_t *to) const { vst1q_s32(to, value); }
//     void store_unaligned(value_t *to) const { vst1q_s32(to, value); }
//
//     simd_type() : value(vdupq_n_s32(0)) {}
//
//     simd_type(const value_t v) : value(vdupq_n_s32(v)) {}
//
//     simd_type(std::initializer_list<value_t> l) {
// #ifdef NDA_ENFORCE_BOUNDCHECK
//       if (l.size() != size()) {
//         throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
//                                  + " is not equal to size of register: " + std::to_string(size()));
//       }
// #endif
//       load_unaligned(l.begin());
//     }
//
//     explicit simd_type(const value_t *v) { load(v); }
//
//     simd_type operator+(const simd_type &other) const { return simd_type{vaddq_s32(value, other.value)}; }
//
//     simd_type operator-(const simd_type &other) const { return simd_type{vsubq_s32(value, other.value)}; }
//
//     simd_type operator*(const simd_type &other) const { return simd_type{vmulq_s32(value, other.value)}; }
//
//     simd_type operator/(const simd_type &other) const {
//       alignas(alignment()) std::array<value_t, size()> x{};
//       alignas(alignment()) std::array<value_t, size()> y{};
//       this->store(x.data());
//       other.store(y.data());
//       for (int i = 0; i < size(); i++) { x[i] = x[i] / y[i]; }
//       return simd_type{x.data()};
//     };
//
//     simd_type &operator+=(const simd_type &other) {
//       value = (*this + other).value;
//       return *this;
//     }
//
//     simd_type &operator-=(const simd_type &other) {
//       value = (*this - other).value;
//       return *this;
//     }
//
//     simd_type &operator*=(const simd_type &other) {
//       value = (*this * other).value;
//       return *this;
//     }
//
//     simd_type &operator/=(const simd_type &other) {
//       value = (*this / other).value;
//       return *this;
//     }
//
//     bool operator==(const simd_type &other) const {
//       intrinsic_t cmp = vceqq_s32(value, other.value);
//       return vminvq_u32(cmp) == 0xFFFFFFFF; // Check the integers values as unsigned and if the minimum is full F. Then they are all equal
//     }
//
//     bool operator!=(const simd_type &other) const { return not(*this == other); };
//
//     operator intrinsic_t() const { return value; }
//   };
//
//   template <>
//   class simd_type<int64_t, 2, abi_tag::NEON> {
//     public:
//     using intrinsic_t = int64x2_t;
//     using value_t     = int64_t;
//
//     private:
//     intrinsic_t value{};
//
//     public:
//     explicit simd_type(intrinsic_t v) : value(v) {}
//
//     static constexpr size_t size() { return 2UL; }
//     static constexpr size_t alignment() { return size() * sizeof(value_t); }
//
//     simd_type(const simd_type &other)            = default;
//     simd_type &operator=(const simd_type &other) = default;
//     simd_type(simd_type &&other)                 = default;
//     simd_type &operator=(simd_type &&other)      = default;
//
//     void load(const value_t *from) { value = vld1q_s64(from); }
//     void load_unaligned(const value_t *from) { value = vld1q_s64(from); }
//     void store(value_t *to) const { vst1q_s64((to), value); }
//     void store_unaligned(value_t *to) const { vst1q_s64((to), value); }
//
//     simd_type() : value(vdupq_n_s64(0)) {}
//
//     simd_type(const value_t v) : value(vdupq_n_s64(v)) {}
//
//     simd_type(std::initializer_list<long> l) {
// #ifdef NDA_ENFORCE_BOUNDCHECK
//       if (l.size() != size()) {
//         throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
//                                  + " is not equal to size of register: " + std::to_string(size()));
//       }
// #endif
//       load_unaligned(l.begin());
//     }
//
//     explicit simd_type(const value_t *v) { load(v); }
//
//     simd_type operator+(const simd_type &other) const { return simd_type{vaddq_s64(value, other.value)}; }
//
//     simd_type operator-(const simd_type &other) const { return simd_type{vsubq_s64(value, other.value)}; }
//
//     simd_type operator*(const simd_type &other) const {
//       intrinsic_t tmp = vcombine_s64(vdup_n_s64(vgetq_lane_s64(value, 0) * vgetq_lane_s64(other.value, 0)),
//                                      vdup_n_s64(vgetq_lane_s64(value, 1) * vgetq_lane_s64(other.value, 1)));
//       return simd_type{tmp};
//     }
//
//     simd_type operator/(const simd_type &other) const {
//       alignas(alignment()) std::array<value_t, size()> x{};
//       alignas(alignment()) std::array<value_t, size()> y{};
//       this->store(x.data());
//       other.store(y.data());
//       for (int i = 0; i < size(); i++) { x[i] = x[i] / y[i]; }
//       return simd_type{x.data()};
//     }
//
//     simd_type &operator+=(const simd_type &other) {
//       value = (*this + other).value;
//       return *this;
//     }
//
//     simd_type &operator-=(const simd_type &other) {
//       value = (*this - other).value;
//       return *this;
//     }
//
//     simd_type &operator*=(const simd_type &other) {
//       value = (*this * other).value;
//       return *this;
//     }
//
//     simd_type &operator/=(const simd_type &other) {
//       value = (*this / other).value;
//       return *this;
//     }
//
//     bool operator==(const simd_type &other) const {
//       intrinsic_t cmp = vceqq_u64(value, other.value);
//       return vgetq_lane_u64(cmp, 0) == 0xFFFFFFFFFFFFFFFF and vgetq_lane_u64(cmp, 1) == 0xFFFFFFFFFFFFFFFF;
//     }
//
//     bool operator!=(const simd_type &other) const { return not(*this == other); }
//
//     operator intrinsic_t() const { return value; }
//   };
//
//   template <>
//   class simd_type<float, 4, abi_tag::NEON> {
//     public:
//     using intrinsic_t = float32x4_t;
//     using value_t     = float;
//
//     private:
//     intrinsic_t value{};
//
//     public:
//     explicit simd_type(intrinsic_t v) : value(v) {}
//
//     static constexpr size_t size() { return 4UL; };
//     static constexpr size_t alignment() { return size() * sizeof(value_t); };
//
//     simd_type(const simd_type &other)            = default;
//     simd_type &operator=(const simd_type &other) = default;
//     simd_type(simd_type &&other)                 = default;
//     simd_type &operator=(simd_type &&other)      = default;
//
//     void load(const value_t *from) { value = vld1q_f32(from); }
//     void load_unaligned(const value_t *from) { value = vld1q_f32(from); }
//     void store(value_t *to) const { vst1q_f32(to, value); }
//     void store_unaligned(value_t *to) const { vst1q_f32(to, value); }
//
//     simd_type() : value(vdupq_n_f32(0.0f)) {}
//
//     simd_type(const value_t v) : value(vdupq_n_f32(v)) {}
//
//     simd_type(std::initializer_list<value_t> l) {
// #ifdef NDA_ENFORCE_BOUNDCHECK
//       if (l.size() != size()) {
//         throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
//                                  + " is not equal to size of register: " + std::to_string(size()));
//       }
// #endif
//       load_unaligned(l.begin());
//     }
//
//     explicit simd_type(const value_t *v) { load(v); }
//
//     simd_type operator+(const simd_type &other) const { return simd_type{vaddq_f32(value, other.value)}; }
//
//     simd_type operator-(const simd_type &other) const { return simd_type{vsubq_f32(value, other.value)}; }
//
//     simd_type operator*(const simd_type &other) const { return simd_type{vmulq_f32(value, other.value)}; }
//
//     simd_type operator/(const simd_type &other) const { return simd_type{vdivq_f32(value, other.value)}; }
//
//     simd_type &operator+=(const simd_type &other) {
//       value = (*this + other).value;
//       return *this;
//     }
//
//     simd_type &operator-=(const simd_type &other) {
//       value = (*this - other).value;
//       return *this;
//     }
//
//     simd_type &operator*=(const simd_type &other) {
//       value = (*this * other).value;
//       return *this;
//     }
//
//     simd_type &operator/=(const simd_type &other) {
//       value = (*this / other).value;
//       return *this;
//     }
//
//     bool operator==(const simd_type &other) const {
//       intrinsic_t cmp = vceqq_f32(value, other.value);
//       return vminvq_u32(cmp) == 0xFFFFFFFF;
//     }
//
//     bool operator!=(const simd_type &other) const { return not(*this == other); }
//
//     operator intrinsic_t() const { return value; }
//   };
//
//   template <>
//   class simd_type<double, 2, abi_tag::NEON> {
//     public:
//     using intrinsic_t = float64x2_t;
//     using value_t     = double;
//
//     private:
//     intrinsic_t value{};
//
//     public:
//     explicit simd_type(intrinsic_t v) : value(v) {}
//
//     static constexpr size_t size() { return 2UL; }
//     static constexpr size_t alignment() { return size() * sizeof(value_t); }
//
//     simd_type(const simd_type &other)            = default;
//     simd_type &operator=(const simd_type &other) = default;
//     simd_type(simd_type &&other)                 = default;
//     simd_type &operator=(simd_type &&other)      = default;
//
//     void load(const value_t *from) { value = vld1q_f64(from); }
//     void load_unaligned(const value_t *from) { value = vld1q_f64(from); }
//     void store(value_t *to) const { vst1q_f64(to, value); }
//     void store_unaligned(value_t *to) const { vst1q_f64(to, value); }
//
//     simd_type() : value(vdupq_n_f64(0.0)) {}
//
//     simd_type(const value_t v) : value(vdupq_n_f64(v)) {}
//
//     simd_type(std::initializer_list<value_t> l) {
// #ifdef NDA_ENFORCE_BOUNDCHECK
//       if (l.size() != size()) {
//         throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
//                                  + " is not equal to size of register: " + std::to_string(size()));
//       }
// #endif
//       load_unaligned(l.begin());
//     }
//
//     explicit simd_type(const value_t *v) { load(v); }
//
//     simd_type operator+(const simd_type &other) const { return simd_type{vaddq_f64(value, other.value)}; }
//
//     simd_type operator-(const simd_type &other) const { return simd_type{vsubq_f64(value, other.value)}; }
//
//     simd_type operator*(const simd_type &other) const { return simd_type{vmulq_f64(value, other.value)}; }
//
//     simd_type operator/(const simd_type &other) const { return simd_type{vdivq_f64(value, other.value)}; }
//
//     simd_type &operator+=(const simd_type &other) {
//       value = (*this + other).value;
//       return *this;
//     }
//
//     simd_type &operator-=(const simd_type &other) {
//       value = (*this - other).value;
//       return *this;
//     }
//
//     simd_type &operator*=(const simd_type &other) {
//       value = (*this * other).value;
//       return *this;
//     }
//
//     simd_type &operator/=(const simd_type &other) {
//       value = (*this / other).value;
//       return *this;
//     }
//
//     bool operator==(const simd_type &other) const {
//       intrinsic_t cmp = vceqq_f64(value, other.value);
//       value_t x             = vgetq_lane_s64(cmp, 0);
//       value_t y             = vgetq_lane_s64(cmp, 1);
//       return x == y and y == 0xFFFFFFFFFFFFFFFF;
//     }
//
//     bool operator!=(const simd_type &other) const { return not(*this == other); }
//
//     operator intrinsic_t() const { return value; }
//   };
//
//   template <>
//   class simd_type<std::complex<float>, 2, abi_tag::NEON> {
//     public:
//     using intrinsic_t = float32x4_t;
//     using value_t     = std::complex<float>;
//     using complex_t   = float;
//
//     private:
//     intrinsic_t value;
//
//     public:
//     explicit simd_type(intrinsic_t v) : value(v) {}
//
//     static constexpr size_t size() { return 2UL; };
//     static constexpr size_t alignment() { return size() * sizeof(value_t); }
//
//     simd_type(const simd_type &other)            = default;
//     simd_type &operator=(const simd_type &other) = default;
//     simd_type(simd_type &&other)                 = default;
//     simd_type &operator=(simd_type &&other)      = default;
//
//     void load(const complex_t *from) { value = vld1q_f32(from); }
//     void load(const value_t *from) { value = vld1q_f32(reinterpret_cast<const complex_t *>(from)); }
//     void load_unaligned(const complex_t *from) { value = vld1q_f32(from); }
//     void load_unaligned(const value_t *from) { value = vld1q_f32(reinterpret_cast<const complex_t *>(from)); }
//
//     void store(complex_t *to) const { vst1q_f32(to, value); }
//     void store(value_t *to) const { vst1q_f32(reinterpret_cast<complex_t *>(to), value); }
//     void store_unaligned(complex_t *to) const { vst1q_f32(to, value); }
//     void store_unaligned(value_t *to) const { vst1q_f32(reinterpret_cast<complex_t *>(to), value); }
//
//     simd_type() : value(vdupq_n_f32(0.0f)) {}
//
//     explicit simd_type(const value_t v) {
//       const float32x2_t r64 = vld1_f32(reinterpret_cast<const float *>(&v));
//       value                 = vcombine_f32(r64, r64);
//     }
//
//     explicit simd_type(const complex_t v) : value(vdupq_n_f32(v)) {}
//
//     explicit simd_type(const value_t *v) { load(v); }
//
//     explicit simd_type(const complex_t *v) { load(v); }
//
//     simd_type(std::initializer_list<value_t> l) {
// #ifdef NDA_ENFORCE_BOUNDCHECK
//       if (l.size() != size()) {
//         throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
//                                  + " is not equal to size of register: " + std::to_string(size()));
//       }
// #endif
//       load_unaligned(l.begin());
//     }
//
//     simd_type(std::initializer_list<complex_t> l) {
// #ifdef NDA_ENFORCE_BOUNDCHECK
//       if (l.size() != 2 * size()) {
//         throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
//                                  + " is not equal to size of register: " + std::to_string(size() * 2));
//       }
// #endif
//       load_unaligned(l.begin());
//     }
//
//     simd_type operator+(const simd_type &other) const { return simd_type{vaddq_f32(value, other.value)}; }
//
//     simd_type operator-(const simd_type &other) const { return simd_type{vsubq_f32(value, other.value)}; }
//
//     simd_type operator*(const simd_type &other) const {
// #ifdef __ARM_FEATURE_COMPLEX
//       intrinsic_t zero = vdupq_n_f32(0.0f);
//       intrinsic_t result     = vcmlaq_f32(zero, value, other.value);
//       result                 = vcmlaq_rot90_f32(result, value, other.value);
// #else
//       intrinsic_t v1, v2;
//       uint32x4_t ret = {0x00000000, 0x80000000, 0x00000000, 0x80000000};
//       v1             = vcombine_f32(vdup_lane_f32(vget_low_f32(value), 0), vdup_lane_f32(vget_high_f32(value), 0));
//       // Get the imag values of a | a1_im | a1_im | a2_im | a2_im |
//       v2 = vcombine_f32(vdup_lane_f32(vget_low_f32(value), 1), vdup_lane_f32(vget_high_f32(value), 1));
//       // Multiply the real a with b
//       v1 = vmulq_f32(v1, other.value);
//       // Multiply the imag a with b
//       v2 = vmulq_f32(v2, other.value);
//       // Conjugate v2
//       v2 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v2), ret));
//       // Swap real/imag elements in v2.
//       v2                 = vrev64q_f32(v2);
//       intrinsic_t result = vaddq_f32(v1, v2);
// #endif
//       return simd_type{result};
//     }
//
//     simd_type operator/(const simd_type &other) const {
//       //TODO: Implement
//       return {0};
//     }
//
//     simd_type &operator+=(const simd_type &other) {
//       value = (*this + other).value;
//       return *this;
//     }
//
//     simd_type &operator-=(const simd_type &other) {
//       value = (*this - other).value;
//       return *this;
//     }
//
//     simd_type &operator*=(const simd_type &other) {
//       value = (*this * other).value;
//       return *this;
//     }
//
//     simd_type &operator/=(const simd_type &other) {
//       value = (*this / other).value;
//       return *this;
//     }
//
//     bool operator==(const simd_type &other) const {
//       intrinsic_t cmp = vceqq_f64(value, other.value);
//       value_t x             = vgetq_lane_s64(cmp, 0);
//       value_t y             = vgetq_lane_s64(cmp, 1);
//       return x == y and y == 0xFFFFFFFFFFFFFFFF;
//     }
//
//     bool operator!=(const simd_type &other) const { return not(*this == other); }
//
//     operator intrinsic_t() const { return value; }
//   };
//
//   template <>
//   class simd_type<std::complex<double>, 1, abi_tag::NEON> {
//     public:
//     using intrinsic_t = float64x2_t;
//     using value_t     = std::complex<double>;
//     using complex_t   = double;
//
//     private:
//     intrinsic_t value;
//
//     public:
//     explicit simd_type(intrinsic_t v) : value(v) {}
//
//     static constexpr size_t size() { return 1UL; };
//
//     static constexpr size_t alignment() { return size() * sizeof(value_t); }
//
//     simd_type(const simd_type &other)            = default;
//     simd_type &operator=(const simd_type &other) = default;
//     simd_type(simd_type &&other)                 = default;
//     simd_type &operator=(simd_type &&other)      = default;
//
//     void load(const complex_t *from) { value = vld1q_f64(from); }
//     void load(const value_t *from) { value = vld1q_f64(reinterpret_cast<const complex_t *>(from)); }
//     void load_unaligned(const complex_t *from) { value = vld1q_f64(from); }
//     void load_unaligned(const value_t *from) { value = vld1q_f64(reinterpret_cast<const complex_t *>(from)); }
//
//     void store(complex_t *to) const { vst1q_f64(to, value); }
//     void store(value_t *to) const { vst1q_f64(reinterpret_cast<complex_t *>(to), value); }
//     void store_unaligned(complex_t *to) const { vst1q_f64(to, value); }
//     void store_unaligned(value_t *to) const { vst1q_f64(reinterpret_cast<complex_t *>(to), value); }
//
//     simd_type() : value(vdupq_n_f64(0.0)) {}
//
//     explicit simd_type(const value_t v) { load_unaligned(&v); }
//
//     explicit simd_type(const complex_t v) : value(vdupq_n_f64(v)) {}
//
//     explicit simd_type(const value_t *v) { load(v); }
//
//     explicit simd_type(const complex_t *v) { load(v); }
//
//     simd_type(std::initializer_list<value_t> l) {
// #ifdef NDA_ENFORCE_BOUNDCHECK
//       if (l.size() != size()) {
//         throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
//                                  + " is not equal to size of register: " + std::to_string(size()));
//       }
// #endif
//       load_unaligned(l.begin());
//     }
//
//     simd_type(std::initializer_list<complex_t> l) {
// #ifdef NDA_ENFORCE_BOUNDCHECK
//       if (l.size() != 2 * size()) {
//         throw std::runtime_error("Size of the initializer list: " + std::to_string(l.size())
//                                  + " is not equal to size of register: " + std::to_string(size() * 2));
//       }
// #endif
//       load_unaligned(l.begin());
//     }
//
//     simd_type operator+(const simd_type &other) const { return simd_type{vaddq_f64(value, other.value)}; }
//
//     simd_type operator-(const simd_type &other) const { return simd_type{vsubq_f64(value, other.value)}; }
//
//     simd_type operator*(const simd_type &other) const {
// #ifdef __ARM_FEATURE_COMPLEX
//       intrinsic_t zero   = vdupq_n_f64(0.0);
//       intrinsic_t result = vcmlaq_f64(zero, value, other.value);
//       result             = vcmlaq_rot90_f64(result, value, other.value);
// #else
//       static const uint64_t p2ul_conj_XOR_DATA[] = {0x0, 0x8000000000000000};
//       uint64x2_t ret                             = vld1q_u64(p2ul_conj_XOR_DATA);
//
//       intrinsic_t v1, v2;
//       // Get the real values of a
//       v1 = vdupq_lane_f64(vget_low_f64(value), 0);
//       // Get the imag values of a
//       v2 = vdupq_lane_f64(vget_high_f64(value), 0);
//       // Multiply the real a with b
//       v1 = vmulq_f64(v1, otoher.value);
//       // Multiply the imag a with b
//       v2 = vmulq_f64(v2, other.value);
//       // Conjugate v2
//       v2 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(v2), ret));
//       // Swap real/imag elements in v2.
//       v2                 = vcombine_f64(vget_high_f64(v2), vget_low_f64(v2));
//       intrinsic_t result = vaddq_f64(v1, v2);
// #endif
//       return simd_type{result};
//     }
//
//     simd_type operator/(const simd_type &other) const {
//       //TODO implement this.
//       return {};
//     }
//
//     simd_type &operator+=(const simd_type &other) {
//       value = (*this + other).value;
//       return *this;
//     }
//
//     simd_type &operator-=(const simd_type &other) {
//       value = (*this - other).value;
//       return *this;
//     }
//
//     simd_type &operator*=(const simd_type &other) {
//       value = (*this * other).value;
//       return *this;
//     }
//
//     simd_type &operator/=(const simd_type &other) {
//       value = (*this / other).value;
//       return *this;
//     }
//
//     bool operator==(const simd_type &other) const {
//       intrinsic_t cmp = vceqq_f64(value, other.value);
//       value_t x             = vgetq_lane_s64(cmp, 0);
//       value_t y             = vgetq_lane_s64(cmp, 1);
//       return x == y and y == 0xFFFFFFFFFFFFFFFF;
//     }
//
//     bool operator!=(const simd_type &other) const { return not(*this == other); }
//
//     operator intrinsic_t() const { return value; }
//   };
//
// } // namespace nda
// #endif