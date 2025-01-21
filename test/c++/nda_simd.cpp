#include <gtest/gtest.h>
#include <nda/simd/simd.hpp>
#include <nda/mem/aligned_alloc.hpp>
#include <type_traits>
#include <random>
using namespace nda;

template <typename T>
void check_real_binary_operations() {
  size_t width     = simd<T>::size();
  size_t alignment = width * sizeof(T);
  T *first         = static_cast<T *>(mem::aligned_alloc<mem::Host>(alignment, alignment));
  T *second        = static_cast<T *>(mem::aligned_alloc<mem::Host>(alignment, alignment));
  std::default_random_engine re;

  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> uniform(-10000, 10000);
    for (size_t i = 0; i < width; i++) {
      first[i]  = uniform(re);
      second[i] = uniform(re);
    }
  } else {
    std::uniform_real_distribution<T> uniform(-10000, 10000);
    for (size_t i = 0; i < width; i++) {
      first[i]  = uniform(re);
      second[i] = uniform(re);
    }
  }

  simd<T> x(first);
  simd<T> y(second);
  simd<T> plus     = x + y;
  simd<T> minus    = x - y;
  simd<T> times    = x * y;
  simd<T> division = x / y;
  for (size_t i = 0; i < x.size(); i++) {
    if constexpr (std::is_same_v<T, float>) {
      ASSERT_FLOAT_EQ(plus[i], first[i] + second[i]);
      ASSERT_FLOAT_EQ(minus[i], first[i] - second[i]);
      ASSERT_FLOAT_EQ(times[i], first[i] * second[i]);
      ASSERT_FLOAT_EQ(division[i], first[i] / second[i]);

    } else if constexpr (std::is_same_v<T, double>) {
      ASSERT_DOUBLE_EQ(plus[i], first[i] + second[i]);
      ASSERT_DOUBLE_EQ(minus[i], first[i] - second[i]);
      ASSERT_DOUBLE_EQ(times[i], first[i] * second[i]);
      ASSERT_DOUBLE_EQ(division[i], first[i] / second[i]);

    } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, long long>) {
      ASSERT_EQ(plus[i], first[i] + second[i]);
      ASSERT_EQ(minus[i], first[i] - second[i]);
      ASSERT_EQ(times[i], first[i] * second[i]);
      ASSERT_EQ(division[i], first[i] / second[i]);
    }
  }
  mem::aligned_free<mem::Host>(first);
  mem::aligned_free<mem::Host>(second);
}

template <typename T>
void check_complex_binary_operations() {
  size_t width     = simd<T>::size();
  size_t alignment = width * sizeof(T);
  T *first         = static_cast<T *>(mem::aligned_alloc<mem::Host>(alignment, alignment));
  T *second        = static_cast<T *>(mem::aligned_alloc<mem::Host>(alignment, alignment));
  std::default_random_engine re;
  std::uniform_real_distribution<T> uniform(-10000, 10000);
  for (size_t i = 0; i < width; i++) {
    first[i]  = uniform(re);
    second[i] = uniform(re);
  }
  simd<std::complex<T>> x(first);
  simd<std::complex<T>> y(second);
  simd<std::complex<T>> plus     = x + y;
  simd<std::complex<T>> minus    = x - y;
  simd<std::complex<T>> times    = x * y;
  simd<std::complex<T>> division = x / y;
  for (size_t i = 0; i < x.size(); i++) {
    std::complex<T> z{first[2 * i], first[2 * i + 1]};
    std::complex<T> w{second[2 * i], second[2 * i + 1]};
    std::complex<T> result_plus     = plus[i];
    std::complex<T> result_minus    = minus[i];
    std::complex<T> result_times    = times[i];
    std::complex<T> result_division = division[i];
    if constexpr (std::is_same_v<T, float>) {
      ASSERT_FLOAT_EQ(result_plus.real(), (z + w).real());
      ASSERT_FLOAT_EQ(result_plus.imag(), (z + w).imag());
      ASSERT_FLOAT_EQ(result_minus.real(), (z - w).real());
      ASSERT_FLOAT_EQ(result_minus.imag(), (z - w).imag());
      ASSERT_FLOAT_EQ(result_times.real(), (z * w).real());
      ASSERT_FLOAT_EQ(result_times.imag(), (z * w).imag());
      ASSERT_FLOAT_EQ(result_division.real(), (z / w).real());
      ASSERT_FLOAT_EQ(result_division.imag(), (z / w).imag());
    } else if constexpr (std::is_same_v<T, double>) {
      ASSERT_DOUBLE_EQ(result_plus.real(), (z + w).real());
      ASSERT_DOUBLE_EQ(result_plus.imag(), (z + w).imag());
      ASSERT_DOUBLE_EQ(result_minus.real(), (z - w).real());
      ASSERT_DOUBLE_EQ(result_minus.imag(), (z - w).imag());
      ASSERT_DOUBLE_EQ(result_times.real(), (z * w).real());
      ASSERT_DOUBLE_EQ(result_times.imag(), (z * w).imag());
      ASSERT_DOUBLE_EQ(result_division.real(), (z / w).real());
      ASSERT_DOUBLE_EQ(result_division.imag(), (z / w).imag());
    }
  }

  mem::aligned_free<mem::Host>(first);
  mem::aligned_free<mem::Host>(second);
}

TEST(NDA, RealBinaryOperations) {
  check_real_binary_operations<int>();
  check_real_binary_operations<long long>();
  check_real_binary_operations<float>();
  check_real_binary_operations<double>();
}

TEST(NDA, ComplexBinaryOperations) {
  check_complex_binary_operations<float>();
  check_complex_binary_operations<double>();
}
