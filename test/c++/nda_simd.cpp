#include <gtest/gtest.h>
#include <nda/simd/simd.hpp>
#include <nda/mem/aligned_alloc.hpp>
#include <type_traits>
#include <random>
#include <limits>
#include <cmath>
using namespace nda;

enum { SEED = 1234 };
/// Clang doesnt support experimental simd.
#ifndef __clang__
template <typename T>
void check_real_binary_operations() {
  size_t width     = native_simd<T>::size();
  size_t alignment = width * sizeof(T);
  T *first         = static_cast<T *>(mem::aligned_alloc<mem::Host>(alignment, alignment));
  T *second        = static_cast<T *>(mem::aligned_alloc<mem::Host>(alignment, alignment));
  std::default_random_engine re(SEED);

  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> uniform(-10000, 10000);
    for (size_t i = 0; i < width; i++) {
      first[i]  = uniform(re);
      second[i] = uniform(re);
      while (first[i] == 0) { first[i] = uniform(re); }
      while (second[i] == 0) { second[i] = uniform(re); }
    }
  } else {
    std::uniform_real_distribution<T> uniform(-10000, 10000);
    for (size_t i = 0; i < width; i++) {
      first[i]  = uniform(re);
      second[i] = uniform(re);
      while (first[i] == 0) { first[i] = uniform(re); }
      while (second[i] == 0) { second[i] = uniform(re); }
    }
  }

  native_simd<T> x(first);
  native_simd<T> y(second);
  native_simd<T> plus     = x + y;
  native_simd<T> minus    = x - y;
  native_simd<T> times    = x * y;
  native_simd<T> division = x / y;
  // for (size_t i = 0; i < x.size(); i++) {
  //   if constexpr (std::is_same_v<T, float>) {
  //     ASSERT_FLOAT_EQ(plus[i], first[i] + second[i]);
  //     ASSERT_FLOAT_EQ(minus[i], first[i] - second[i]);
  //     ASSERT_FLOAT_EQ(times[i], first[i] * second[i]);
  //     ASSERT_FLOAT_EQ(division[i], first[i] / second[i]);
  //
  //   } else if constexpr (std::is_same_v<T, double>) {
  //     ASSERT_DOUBLE_EQ(plus[i], first[i] + second[i]);
  //     ASSERT_DOUBLE_EQ(minus[i], first[i] - second[i]);
  //     ASSERT_DOUBLE_EQ(times[i], first[i] * second[i]);
  //     ASSERT_DOUBLE_EQ(division[i], first[i] / second[i]);
  //
  //   } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, long long>) {
  //     ASSERT_EQ(plus[i], first[i] + second[i]);
  //     ASSERT_EQ(minus[i], first[i] - second[i]);
  //     ASSERT_EQ(times[i], first[i] * second[i]);
  //     ASSERT_EQ(division[i], first[i] / second[i]);
  //   }
  // }
  mem::aligned_free<mem::Host>(first);
  mem::aligned_free<mem::Host>(second);
}

template <typename T>
void check_complex_binary_operations() {
  size_t width     = native_simd<T>::size();
  size_t alignment = width * sizeof(T);
  T *first         = static_cast<T *>(mem::aligned_alloc<mem::Host>(alignment, alignment));
  T *second        = static_cast<T *>(mem::aligned_alloc<mem::Host>(alignment, alignment));
  std::default_random_engine re(SEED);
  std::uniform_real_distribution<T> uniform(-10000, 10000);
  for (size_t i = 0; i < width; i++) {
    first[i]  = uniform(re);
    second[i] = uniform(re);
  }
  for (size_t i = 0; i < width / 2; i++) {
    while (first[2 * i] == 0 && first[2 * i + 1] == 0) {
      first[2 * i]     = uniform(re);
      first[2 * i + 1] = uniform(re);
    }
    while (second[2 * i] == 0 && second[2 * i + 1] == 0) {
      second[2 * i]     = uniform(re);
      second[2 * i + 1] = uniform(re);
    }
  }

  native_simd<std::complex<T>> x(first);
  native_simd<std::complex<T>> y(second);
  native_simd<std::complex<T>> plus     = x + y;
  native_simd<std::complex<T>> minus    = x - y;
  native_simd<std::complex<T>> times    = x * y;
  native_simd<std::complex<T>> division = x / y;
  for (size_t i = 0; i < x.size(); i++) {
    std::complex<T> z{first[2 * i], first[2 * i + 1]};
    std::complex<T> w{second[2 * i], second[2 * i + 1]};
    // std::complex<T> result_plus     = plus[i];
    // std::complex<T> result_minus    = minus[i];
    // std::complex<T> result_times    = times[i];
    // std::complex<T> result_division = division[i];
    // if constexpr (std::is_same_v<T, float>) {
    //   ASSERT_FLOAT_EQ(result_plus.real(), (z + w).real());
    //   ASSERT_FLOAT_EQ(result_plus.imag(), (z + w).imag());
    //   ASSERT_FLOAT_EQ(result_minus.real(), (z - w).real());
    //   ASSERT_FLOAT_EQ(result_minus.imag(), (z - w).imag());
    //   ASSERT_FLOAT_EQ(result_times.real(), (z * w).real());
    //   ASSERT_FLOAT_EQ(result_times.imag(), (z * w).imag());
    //   ASSERT_FLOAT_EQ(result_division.real(), (z / w).real());
    //   ASSERT_FLOAT_EQ(result_division.imag(), (z / w).imag());
    // } else if constexpr (std::is_same_v<T, double>) {
    //   ASSERT_DOUBLE_EQ(result_plus.real(), (z + w).real());
    //   ASSERT_DOUBLE_EQ(result_plus.imag(), (z + w).imag());
    //   ASSERT_DOUBLE_EQ(result_minus.real(), (z - w).real());
    //   ASSERT_DOUBLE_EQ(result_minus.imag(), (z - w).imag());
    //   ASSERT_DOUBLE_EQ(result_times.real(), (z * w).real());
    //   ASSERT_DOUBLE_EQ(result_times.imag(), (z * w).imag());
    //   ASSERT_DOUBLE_EQ(result_division.real(), (z / w).real());
    //   ASSERT_DOUBLE_EQ(result_division.imag(), (z / w).imag());
    // }
  }

  mem::aligned_free<mem::Host>(first);
  mem::aligned_free<mem::Host>(second);
}

template <typename T>
void check_limit_cases() {
  size_t width = native_simd<T>::size();
  T min, max, nan, inf;
  min = std::numeric_limits<T>::min();
  max = std::numeric_limits<T>::max();
  if constexpr (std::is_floating_point_v<T>) {
    nan = std::numeric_limits<T>::quiet_NaN();
    inf = std::numeric_limits<T>::infinity();
  }
  native_simd<T> min_simd(min);
  native_simd<T> max_simd(max);
  native_simd<T> nan_simd;
  native_simd<T> inf_simd;
  if constexpr (std::is_floating_point_v<T>) {
    nan_simd = native_simd<T>(nan);
    inf_simd = native_simd<T>(inf);
  }
  T value;
  if constexpr (std::is_integral_v<T>) {
    value = 2;
    native_simd<T> test(value);
    auto plus_max     = max_simd + test;
    auto minus_min    = min_simd - test;
    auto times_max    = max_simd * test;
    auto times_min    = min_simd * test;
    auto division_max = max_simd / test;
    auto division_min = min_simd / test;
    for (size_t i = 0; i < width; i++) {
      // ASSERT_EQ(plus_max[i], max + value);
      // ASSERT_EQ(minus_min[i], min - value);
      // ASSERT_EQ(times_max[i], max * value);
      // ASSERT_EQ(times_min[i], min * value);
      // ASSERT_EQ(division_max[i], max / value);
      // ASSERT_EQ(division_min[i], min / value);
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    value = static_cast<T>(2.5);
    native_simd<T> test(value);
    native_simd<T> minus_test(-value);
    auto plus_max        = max_simd + test;
    auto minus_min       = min_simd - test;
    auto times_max       = max_simd * test;
    auto times_min       = min_simd * test;
    auto division_max    = max_simd / test;
    auto division_min    = min_simd / test;
    auto plus_nan        = nan_simd + test;
    auto minus_nan       = nan_simd - test;
    auto times_nan       = nan_simd * test;
    auto division_nan    = nan_simd / test;
    auto plus_inf        = inf_simd + test;
    auto minus_inf       = inf_simd - test;
    auto times_inf       = inf_simd * (test);
    auto minus_times_inf = inf_simd * minus_test;
    auto division_inf    = inf_simd / test;
    auto divide_by_inf   = test / inf_simd;

    if constexpr (std::is_same_v<T, float>) {
      for (size_t i = 0; i < width; i++) {
        // ASSERT_FLOAT_EQ(plus_max[i], max + value);
        // ASSERT_FLOAT_EQ(minus_min[i], min - value);
        // ASSERT_FLOAT_EQ(times_max[i], max * value);
        // ASSERT_FLOAT_EQ(times_min[i], min * value);
        // ASSERT_FLOAT_EQ(division_max[i], max / value);
        // ASSERT_FLOAT_EQ(division_min[i], min / value);
        // ASSERT_FLOAT_EQ(plus_inf[i], inf + value);
        // ASSERT_FLOAT_EQ(minus_inf[i], inf - value);
        // ASSERT_FLOAT_EQ(times_inf[i], inf * value);
        // ASSERT_FLOAT_EQ(minus_times_inf[i], inf * (-value));
        // ASSERT_FLOAT_EQ(division_inf[i], inf / value);
        // ASSERT_FLOAT_EQ(divide_by_inf[i], value / inf);
        // ASSERT_TRUE(std::isnan(plus_nan[i]));
        // ASSERT_TRUE(std::isnan(minus_nan[i]));
        // ASSERT_TRUE(std::isnan(times_nan[i]));
        // ASSERT_TRUE(std::isnan(division_nan[i]));
      }

    } else if constexpr (std::is_same_v<T, double>) {
      for (size_t i = 0; i < width; i++) {
        // ASSERT_DOUBLE_EQ(plus_max[i], max + value);
        // ASSERT_DOUBLE_EQ(minus_min[i], min - value);
        // ASSERT_DOUBLE_EQ(times_max[i], max * value);
        // ASSERT_DOUBLE_EQ(times_min[i], min * value);
        // ASSERT_DOUBLE_EQ(division_max[i], max / value);
        // ASSERT_DOUBLE_EQ(division_min[i], min / value);
        // ASSERT_DOUBLE_EQ(plus_inf[i], inf + value);
        // ASSERT_DOUBLE_EQ(minus_inf[i], inf - value);
        // ASSERT_DOUBLE_EQ(times_inf[i], inf * value);
        // ASSERT_DOUBLE_EQ(minus_times_inf[i], inf * (-value));
        // ASSERT_DOUBLE_EQ(division_inf[i], inf / value);
        // ASSERT_DOUBLE_EQ(divide_by_inf[i], value / inf);
        // ASSERT_TRUE(std::isnan(plus_nan[i]));
        // ASSERT_TRUE(std::isnan(minus_nan[i]));
        // ASSERT_TRUE(std::isnan(times_nan[i]));
        // ASSERT_TRUE(std::isnan(division_nan[i]));
      }
    }
  }
}


TEST(NDA, RealBinaryOperations) {
  check_real_binary_operations<int>();
  check_real_binary_operations<long>();
  check_real_binary_operations<float>();
  check_real_binary_operations<double>();

}

TEST(NDA, ComplexBinaryOperations) {
  check_complex_binary_operations<float>();
  check_complex_binary_operations<double>();
}

TEST(NDA, LimitCases) {
  check_limit_cases<int>();
  check_limit_cases<long>();
  check_limit_cases<float>();
  check_limit_cases<double>();
}

TEST(NDA, OurSIMD) {
  native_simd<long long int> hop;
  std::cout << hop.size() <<std::endl;
  simd_f8 q{1,2,3,4,5,6,7,8};
  const __m256 mask = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
  q = simd_f8{_mm256_xor_ps(q, mask)};
  alignas(32) std::array<float,8> data;
  simd_cd2 x = {111, 234, 356, -478,5,6,7,8};
  simd_cd2 y = {9,10,-11,12,13,14,15,16};
  alignas(32) std::array<std::complex<double>, x.size()> test;
  auto test2 = x /y;
  test2.store(test.data());
  q.store(data.data());
  for (int i = 0; i < q.size(); i++) { std::cout << data[i] << std::endl; }
}

#endif