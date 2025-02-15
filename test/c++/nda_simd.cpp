#include <gtest/gtest.h>
#include <nda/gtest_tools.hpp>
#include <nda/simd/simd.hpp>
#include <type_traits>
#include <random>
#include <cmath>
#include <array>
#include <complex>
#include <algorithm>
#include <bit>

using namespace nda;

std::mt19937 gen(123);

template <typename T, size_t Width, abi_tag ABI>
void check_simd_array_equal(simd_type<T, Width, ABI> value, std::array<T, Width> correct) {
  alignas(value.alignment()) std::array<T, Width> calculation;
  value.store(calculation.data());
  for (int i = 0; i < Width; i++) {
    if constexpr (std::is_same_v<T, int32_t> or std::is_same_v<T, int64_t>) {
      EXPECT_EQ(calculation[i], correct[i]);
    } else if constexpr (std::is_same_v<T, float>) {
      if (not(std::isnan(calculation[i]) and std::isnan(correct[i]))) { EXPECT_FLOAT_EQ(calculation[i], correct[i]); }
    } else if constexpr (std::is_same_v<T, double>) {
      if (not(std::isnan(calculation[i]) and std::isnan(correct[i]))) { EXPECT_DOUBLE_EQ(calculation[i], correct[i]); }
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      auto max         = std::max(std::abs(calculation[i]), std::abs(correct[i]));
      max              = std::max(max, 1.0f);
      double precision = 1e-6 * max;
      if ((std::isnan(calculation[i].real()) and (std::isnan(correct[i].real())))) {
        if ((std::isnan(calculation[i].imag()) and (std::isnan(correct[i].imag())))) { continue; }
        EXPECT_FLOAT_EQ(calculation[i].imag(), correct[i].imag());
        continue;
      } else {
        if ((std::isnan(calculation[i].imag()) and (std::isnan(correct[i].imag())))) {
          EXPECT_FLOAT_EQ(calculation[i].real(), correct[i].real());
          continue;
        }
        if (std::isnan(std::abs(calculation[i] - correct[i]))) {
          EXPECT_FLOAT_EQ(calculation[i].real(), correct[i].real());
          EXPECT_FLOAT_EQ(calculation[i].imag(), correct[i].imag());
          continue;
        }
      }
      EXPECT_COMPLEX_NEAR(calculation[i], correct[i], precision);
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      auto max         = std::max(std::abs(calculation[i]), std::abs(correct[i]));
      max              = std::max(max, 1.0);
      double precision = 1e-15 * max;
      if ((std::isnan(calculation[i].real()) and (std::isnan(correct[i].real())))) {
        if ((std::isnan(calculation[i].imag()) and (std::isnan(correct[i].imag())))) { continue; }
        EXPECT_DOUBLE_EQ(calculation[i].imag(), correct[i].imag());
        continue;
      } else {
        if ((std::isnan(calculation[i].imag()) and (std::isnan(correct[i].imag())))) {
          EXPECT_DOUBLE_EQ(calculation[i].real(), correct[i].real());
          continue;
        }
      }
      if (std::isnan(std::abs(calculation[i] - correct[i]))) {
        EXPECT_DOUBLE_EQ(calculation[i].real(), correct[i].real());
        EXPECT_DOUBLE_EQ(calculation[i].imag(), correct[i].imag());
        continue;
      }
      EXPECT_COMPLEX_NEAR(calculation[i], correct[i], precision);
    }
  }
}

// Specialization for floating point types (float and double)
template <std::floating_point T, size_t Width>
std::array<T, Width> generate_random_array() {
  std::uniform_real_distribution<T> dist(-10000.0, 10000.0);
  std::array<T, Width> random_array;

  for (size_t i = 0; i < Width; ++i) { random_array[i] = dist(gen); }

  return random_array;
}

template <std::integral T, size_t Width>
std::array<T, Width> generate_random_array() {
  std::uniform_int_distribution<T> dist(-10000, 10000);
  std::array<T, Width> random_array;

  for (size_t i = 0; i < Width; ++i) { random_array[i] = dist(gen); }

  return random_array;
}

template <typename T, size_t Width>
  requires(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
std::array<T, Width> generate_random_array() {
  std::uniform_real_distribution<typename T::value_type> dist_real(-10000.0, 10000.0);
  std::array<T, Width> random_array;

  for (size_t i = 0; i < Width; ++i) {
    random_array[i] = T(dist_real(gen), dist_real(gen)); // Real and imaginary parts
  }

  return random_array;
}

template <typename T, size_t N>
std::array<T, N> generate_initialized_array(const T &value) {
  std::array<T, N> tmp;
  for (size_t i = 0; i < N; ++i) { tmp[i] = value; }
  return tmp;
}

template <typename T, size_t Width, abi_tag ABI>
void simd_type_default_constructor() {
  simd_type<T, Width, ABI> tmp{};

  auto initialized_array = generate_initialized_array<T, Width>(T{0});
  check_simd_array_equal(tmp, initialized_array);
}
template <typename T, size_t Width, abi_tag ABI>
void simd_type_value_constructor(T value) {
  simd_type<T, Width, ABI> tmp(value);

  auto initialized_array = generate_initialized_array<T, Width>(T(value));
  check_simd_array_equal(tmp, initialized_array);
  // Complex types have single value constructor and complex value constructor. Test both of them.
  if constexpr (std::is_same_v<T, std::complex<float>> or std::is_same_v<T, std::complex<double>>) {
    simd_type<T, Width, ABI> complex_tmp(value.real());
    auto complex_initialized_array = generate_initialized_array<T, Width>({value.real(), value.real()});
    check_simd_array_equal(complex_tmp, complex_initialized_array);
  }
}
template <typename T, size_t Width, abi_tag ABI>
void simd_pointer_constructor() {
  alignas(simd_type<T, Width, ABI>::alignment()) auto random_array = generate_random_array<T, Width>();
  simd_type<T, Width, ABI> tmp(random_array.data());
  check_simd_array_equal(tmp, random_array);
}

template <typename T, size_t Width, abi_tag ABI>
void test_simd_size_alignment() {
  simd_type<T, Width, ABI> simd;
  EXPECT_EQ(simd.size(), Width);
  constexpr auto is_abi = [](abi_tag lhs, abi_tag rhs) {
    return static_cast<std::underlying_type_t<abi_tag>>(lhs) == static_cast<std::underlying_type_t<abi_tag>>(rhs);
  };
  if constexpr (is_abi(ABI, abi_tag::Default)) {
    EXPECT_EQ(simd.alignment(), 0);
  } else if constexpr (is_abi(ABI, abi_tag::SSE)) {
    EXPECT_EQ(simd.alignment(), 16);
  } else if constexpr (is_abi(ABI, abi_tag::AVX)) {
    EXPECT_EQ(simd.alignment(), 32);
  } else if constexpr (is_abi(ABI, abi_tag::AVX512)) {
    EXPECT_EQ(simd.alignment(), 64);
  }
}

template <typename T, size_t Width, abi_tag ABI>
void simd_load_store() {
  simd_type<T, Width, ABI> tmp;
  alignas(tmp.alignment()) auto random_array_aligned = generate_random_array<T, Width>();
  auto random_array_unaligned                        = generate_random_array<T, Width>();
  alignas(tmp.alignment()) auto store_aligned        = generate_random_array<T, Width>();
  auto store_unaligned                               = generate_random_array<T, Width>();
  tmp.load(random_array_aligned.data());
  check_simd_array_equal(tmp, random_array_aligned);
  tmp.store(store_aligned.data());
  EXPECT_TRUE(random_array_aligned == store_aligned);
  tmp.load_unaligned(random_array_unaligned.data());
  check_simd_array_equal(tmp, random_array_unaligned);
  tmp.store_unaligned(store_unaligned.data());
  EXPECT_TRUE(random_array_unaligned == store_unaligned);
}
template <typename T, size_t Width, abi_tag ABI>
void simd_compound_binary_operators() {
  simd_type<T, Width, ABI> lhs, rhs, tmp;
  alignas(lhs.alignment()) auto lhs_array = generate_random_array<T, Width>();
  alignas(lhs.alignment()) auto rhs_array = generate_random_array<T, Width>();
  alignas(lhs.alignment()) auto tmp_array = generate_random_array<T, Width>();
  lhs.load(lhs_array.data());
  rhs.load(rhs_array.data());
  tmp = lhs + rhs;
  for (int i = 0; i < Width; ++i) { tmp_array[i] = lhs_array[i] + rhs_array[i]; }
  check_simd_array_equal(tmp, tmp_array);
  tmp = lhs - rhs;
  for (int i = 0; i < Width; ++i) { tmp_array[i] = lhs_array[i] - rhs_array[i]; }
  check_simd_array_equal(tmp, tmp_array);
  tmp = lhs * rhs;
  for (int i = 0; i < Width; ++i) { tmp_array[i] = lhs_array[i] * rhs_array[i]; }
  check_simd_array_equal(tmp, tmp_array);
  tmp = lhs / rhs;
  for (int i = 0; i < Width; ++i) { tmp_array[i] = lhs_array[i] / rhs_array[i]; }
  check_simd_array_equal(tmp, tmp_array);

  tmp += lhs;
  for (int i = 0; i < Width; ++i) { tmp_array[i] += lhs_array[i]; };
  check_simd_array_equal(tmp, tmp_array);
  tmp -= lhs;
  for (int i = 0; i < Width; ++i) { tmp_array[i] -= lhs_array[i]; };
  check_simd_array_equal(tmp, tmp_array);
  tmp *= rhs;
  for (int i = 0; i < Width; ++i) { tmp_array[i] *= rhs_array[i]; }
  tmp /= rhs;
  for (int i = 0; i < Width; ++i) { tmp_array[i] /= rhs_array[i]; }
  check_simd_array_equal(tmp, tmp_array);
}

template <typename T, size_t Width, abi_tag ABI>
void simd_initializer_list_constructor(std::initializer_list<T> l) {
  simd_type<T, Width, ABI> tmp(l);
  std::array<T, Width> tmp_array{};
  std::copy(l.begin(), l.end(), tmp_array.begin());
  check_simd_array_equal(tmp, tmp_array);
}

template <typename T, size_t Width, abi_tag ABI>
void simd_equality_operator() {
  simd_type<T, Width, ABI> lhs, rhs;
  alignas(lhs.alignment()) std::array<T, Width> ones{};
  alignas(lhs.alignment()) std::array<T, Width> ones_alternate{};
  for (int i = 0; i < Width; ++i) {
    ones[i] = 1;
    if (i % 2 == 0) {
      ones_alternate[i] = -1;
    } else {
      ones_alternate[i] = 1;
    }
  }
  lhs.load(ones.data());
  rhs.load(ones.data());
  EXPECT_TRUE(lhs == rhs);
  EXPECT_TRUE(rhs == lhs);
  EXPECT_FALSE(lhs != rhs);
  EXPECT_FALSE(rhs != lhs);
  rhs.load(ones_alternate.data());
  EXPECT_FALSE(lhs == rhs);
  EXPECT_FALSE(rhs == lhs);
  EXPECT_TRUE(lhs != rhs);
  EXPECT_TRUE(rhs != lhs);
  lhs.load(ones_alternate.data());
  EXPECT_TRUE(lhs == rhs);
  EXPECT_TRUE(rhs == lhs);
  EXPECT_FALSE(lhs != rhs);
  EXPECT_FALSE(rhs != lhs);
  rhs.load(ones.data());
  ones[0] = 5;
  lhs.load(ones.data());
  EXPECT_FALSE(lhs == rhs);
  EXPECT_FALSE(rhs == lhs);
  EXPECT_TRUE(lhs != rhs);
  EXPECT_TRUE(rhs != lhs);
}

template <typename T, size_t Width, abi_tag ABI>
void simd_binary_operators_edge_cases() {
  T min = std::numeric_limits<T>::min();
  T max = std::numeric_limits<T>::max();
  simd_type<T, Width, ABI> min_v(min);
  simd_type<T, Width, ABI> max_v(max);
  simd_type<T, Width, ABI> plus_v(T(2));
  simd_type<T, Width, ABI> minus_v(T(-2));
  std::array<T, Width> min_array{};
  std::array<T, Width> max_array{};
  std::array<T, Width> plus_array{};
  std::array<T, Width> minus_array{};
  for (int i = 0; i < Width; ++i) {
    min_array[i]   = min;
    max_array[i]   = max;
    plus_array[i]  = 2;
    minus_array[i] = -2;
  }
  check_simd_array_equal(max_v + plus_v, max_array + plus_array);
  check_simd_array_equal(min_v - plus_v, min_array - plus_array);
  check_simd_array_equal(max_v * plus_v, max_array * plus_array);
  check_simd_array_equal(min_v * plus_v, min_array * plus_array);
  check_simd_array_equal(min_v * minus_v, min_array * minus_array);
  check_simd_array_equal(max_v * minus_v, max_array * minus_array);
  check_simd_array_equal(max_v + min_v, max_array + min_array);
  check_simd_array_equal(max_v + max_v, max_array + max_array);
  check_simd_array_equal(min_v - max_v, min_array - max_array);
  check_simd_array_equal(max_v * max_v, max_array * max_array);
  check_simd_array_equal(min_v * max_v, min_array * max_array);
}

template <typename T, size_t Width, abi_tag ABI>
void simd_check_nan_inf() {
  if constexpr (std::is_floating_point_v<T>) {
    T nan                                        = std::numeric_limits<T>::quiet_NaN();
    T plus_inf                                   = std::numeric_limits<T>::infinity();
    T minus_inf                                  = std::numeric_limits<T>::infinity() * -1;
    constexpr size_t alignment                   = simd_type<T, Width, ABI>::alignment();
    alignas(alignment) std::array<T, Width> mix  = {nan, plus_inf};
    alignas(alignment) std::array<T, Width> mix2 = {minus_inf, 2};
    alignas(alignment) std::array<T, Width> mix3 = {plus_inf, minus_inf};
    alignas(alignment) std::array<T, Width> mix4 = {2, nan};
    for (int i = 2; i < Width; ++i) {
      mix[i]  = 1;
      mix2[i] = 1;
      mix3[i] = 1;
    }
    simd_type<T, Width, ABI> simd_mix(mix.data());
    simd_type<T, Width, ABI> simd_mix2(mix2.data());
    simd_type<T, Width, ABI> simd_mix3(mix3.data());
    simd_type<T, Width, ABI> simd_mix4(mix4.data());
    check_simd_array_equal(simd_mix + simd_mix, mix + mix);
    check_simd_array_equal(simd_mix + simd_mix2, mix + mix2);
    check_simd_array_equal(simd_mix + simd_mix3, mix + mix3);
    check_simd_array_equal(simd_mix + simd_mix4, mix + mix4);
    check_simd_array_equal(simd_mix2 + simd_mix3, mix2 + mix3);
    check_simd_array_equal(simd_mix2 + simd_mix4, mix2 + mix4);
    check_simd_array_equal(simd_mix3 + simd_mix4, mix3 + mix4);

    check_simd_array_equal(simd_mix - simd_mix, mix - mix);
    check_simd_array_equal(simd_mix - simd_mix2, mix - mix2);
    check_simd_array_equal(simd_mix - simd_mix3, mix - mix3);
    check_simd_array_equal(simd_mix - simd_mix4, mix - mix4);
    check_simd_array_equal(simd_mix2 - simd_mix3, mix2 - mix3);
    check_simd_array_equal(simd_mix2 - simd_mix4, mix2 - mix4);
    check_simd_array_equal(simd_mix3 - simd_mix4, mix3 - mix4);

    check_simd_array_equal(simd_mix * simd_mix, mix * mix);
    check_simd_array_equal(simd_mix * simd_mix2, mix * mix2);
    check_simd_array_equal(simd_mix * simd_mix3, mix * mix3);
    check_simd_array_equal(simd_mix * simd_mix4, mix * mix4);
    check_simd_array_equal(simd_mix2 * simd_mix3, mix2 * mix3);
    check_simd_array_equal(simd_mix2 * simd_mix4, mix2 * mix4);
    check_simd_array_equal(simd_mix3 * simd_mix4, mix3 * mix4);
    std::array<T, Width> tmp1{}, tmp2{}, tmp3{}, tmp4{}, tmp5{}, tmp6{}, tmp7{};
    for (int i = 0; i < Width; ++i) {
      tmp1[i] = mix[i] / mix[i];
      tmp2[i] = mix[i] / mix2[i];
      tmp3[i] = mix[i] / mix3[i];
      tmp4[i] = mix[i] / mix4[i];
      tmp5[i] = mix2[i] / mix3[i];
      tmp6[i] = mix2[i] / mix4[i];
      tmp7[i] = mix3[i] / mix4[i];
    }

    check_simd_array_equal(simd_mix / simd_mix, tmp1);
    check_simd_array_equal(simd_mix / simd_mix2, tmp2);
    check_simd_array_equal(simd_mix / simd_mix3, tmp3);
    check_simd_array_equal(simd_mix / simd_mix4, tmp4);
    check_simd_array_equal(simd_mix2 / simd_mix3, tmp5);
    check_simd_array_equal(simd_mix2 / simd_mix4, tmp6);
    check_simd_array_equal(simd_mix3 / simd_mix4, tmp7);
  } else {
    using scalar_t                               = typename simd_type<T, Width, ABI>::scalar_t;
    scalar_t nan                                 = std::numeric_limits<scalar_t>::quiet_NaN();
    scalar_t plus_inf                            = std::numeric_limits<scalar_t>::infinity();
    scalar_t minus_inf                           = std::numeric_limits<scalar_t>::infinity() * -1;
    constexpr size_t alignment                   = simd_type<T, Width, ABI>::alignment();
    alignas(alignment) std::array<T, Width> mix  = {std::complex<scalar_t>{nan, plus_inf}};
    alignas(alignment) std::array<T, Width> mix2 = {std::complex<scalar_t>{minus_inf, 2}};
    alignas(alignment) std::array<T, Width> mix3 = {std::complex<scalar_t>{plus_inf, minus_inf}};
    alignas(alignment) std::array<T, Width> mix4 = {std::complex<scalar_t>{2, nan}};
    for (int i = 1; i < Width; ++i) {
      mix[i]  = {1, 1};
      mix2[i] = {1, 1};
      mix3[i] = {1, 1};
    }
    simd_type<T, Width, ABI> simd_mix(mix.data());
    simd_type<T, Width, ABI> simd_mix2(mix2.data());
    simd_type<T, Width, ABI> simd_mix3(mix3.data());
    simd_type<T, Width, ABI> simd_mix4(mix4.data());
    check_simd_array_equal(simd_mix + simd_mix, mix + mix);
    check_simd_array_equal(simd_mix + simd_mix2, mix + mix2);
    check_simd_array_equal(simd_mix + simd_mix3, mix + mix3);
    check_simd_array_equal(simd_mix + simd_mix4, mix + mix4);
    check_simd_array_equal(simd_mix2 + simd_mix3, mix2 + mix3);
    check_simd_array_equal(simd_mix2 + simd_mix4, mix2 + mix4);
    check_simd_array_equal(simd_mix3 + simd_mix4, mix3 + mix4);

    check_simd_array_equal(simd_mix - simd_mix, mix - mix);
    check_simd_array_equal(simd_mix - simd_mix2, mix - mix2);
    check_simd_array_equal(simd_mix - simd_mix3, mix - mix3);
    check_simd_array_equal(simd_mix - simd_mix4, mix - mix4);
    check_simd_array_equal(simd_mix2 - simd_mix3, mix2 - mix3);
    check_simd_array_equal(simd_mix2 - simd_mix4, mix2 - mix4);
    check_simd_array_equal(simd_mix3 - simd_mix4, mix3 - mix4);

    // std::array<T, Width> tmp1{}, tmp2{}, tmp3{}, tmp4{}, tmp5{}, tmp6{}, tmp7{};
    //TODO: STL is wrong I think.

    // for (int i = 0; i < Width; ++i) {
    //   tmp1[i] = mix[i] * mix[i];
    //   tmp2[i] = mix[i] * mix2[i];
    //   tmp3[i] = mix[i] * mix3[i];
    //   tmp4[i] = mix[i] * mix4[i];
    //   tmp5[i] = mix2[i] * mix3[i];
    //   tmp6[i] = mix2[i] * mix4[i];
    //   tmp7[i] = mix3[i] * mix4[i];
    // }
    //
    // check_simd_array_equal(simd_mix * simd_mix, tmp1);
    // check_simd_array_equal(simd_mix * simd_mix2, tmp2);
    // check_simd_array_equal(simd_mix * simd_mix3, tmp3);
    // check_simd_array_equal(simd_mix * simd_mix4, tmp4);
    // check_simd_array_equal(simd_mix2 * simd_mix3, tmp5);
    // check_simd_array_equal(simd_mix2 * simd_mix4, tmp6);
    // check_simd_array_equal(simd_mix3 * simd_mix4, tmp7);

    //   for (int i = 0; i < Width; ++i) {
    //     tmp1[i] = mix[i] / mix[i];
    //     tmp2[i] = mix[i] / mix2[i];
    //     tmp3[i] = mix[i] / mix3[i];
    //     tmp4[i] = mix[i] / mix4[i];
    //     tmp5[i] = mix2[i] / mix3[i];
    //     tmp6[i] = mix2[i] / mix4[i];
    //     tmp7[i] = mix3[i] / mix4[i];
    //   }
    //
    //   check_simd_array_equal(simd_mix / simd_mix, tmp1);
    //   check_simd_array_equal(simd_mix / simd_mix2, tmp2);
    //   check_simd_array_equal(simd_mix / simd_mix3, tmp3);
    //   check_simd_array_equal(simd_mix / simd_mix4, tmp4);
    //   check_simd_array_equal(simd_mix2 / simd_mix3, tmp5);
    //   check_simd_array_equal(simd_mix2 / simd_mix4, tmp6);
    //   check_simd_array_equal(simd_mix3 / simd_mix4, tmp7);
  }
}

template <typename T, size_t Width, abi_tag ABI>
void simd_function_abs() {
  for (int i = 0; i < 1000; ++i) {
    simd_type<T, Width, ABI> x;
    alignas(x.alignment()) std::array<T, Width> tmp = generate_random_array<T, Width>();
    x.load(tmp.data());
    for (int j = 0; j < Width; ++j) { tmp[j] = std::abs(tmp[j]); }
    check_simd_array_equal(simd::abs(x), tmp);
  }
}

template <typename T, size_t Width, abi_tag ABI>
void simd_function_conj() {
  for (int i = 0; i < 1000; ++i) {
    simd_type<T, Width, ABI> x;
    alignas(x.alignment()) std::array<T, Width> tmp = generate_random_array<T, Width>();
    x.load(tmp.data());
    if constexpr (std::is_same_v<T, std::complex<float>> or std::is_same_v<T, std::complex<double>>) {
      for (int j = 0; j < Width; ++j) { tmp[j] = std::conj(tmp[j]); }
    }

    check_simd_array_equal(simd::conj(x), tmp);
  }
}

template <typename T, size_t Width, abi_tag ABI>
void simd_function_sqrt() {
  for (int i = 0; i < 1000; ++i) {
    simd_type<T, Width, ABI> x;
    alignas(x.alignment()) std::array<T, Width> tmp = generate_random_array<T, Width>();
    x.load(tmp.data());
    for (int j = 0; j < Width; ++j) { tmp[j] = std::sqrt(tmp[j]); }
    check_simd_array_equal(simd::sqrt(x), tmp);
  }
}

template <typename T, size_t Width, abi_tag ABI>
void simd_function_min() {
  for (int i = 0; i < 1000; ++i) {
    simd_type<T, Width, ABI> x, y;
    alignas(x.alignment()) std::array<T, Width> tmp  = generate_random_array<T, Width>();
    alignas(x.alignment()) std::array<T, Width> tmp2 = generate_random_array<T, Width>();
    x.load(tmp.data());
    y.load(tmp2.data());
    for (int j = 0; j < Width; ++j) { tmp[j] = std::min(tmp[j], tmp2[j]); }
    check_simd_array_equal(simd::min(x, y), tmp);
  }
}

template <typename T, size_t Width, abi_tag ABI>
void simd_function_max() {
  for (int i = 0; i < 1000; ++i) {
    simd_type<T, Width, ABI> x, y;
    alignas(x.alignment()) std::array<T, Width> tmp  = generate_random_array<T, Width>();
    alignas(x.alignment()) std::array<T, Width> tmp2 = generate_random_array<T, Width>();
    x.load(tmp.data());
    y.load(tmp2.data());
    for (int j = 0; j < Width; ++j) { tmp[j] = std::max(tmp[j], tmp2[j]); }
    check_simd_array_equal(simd::max(x, y), tmp);
  }
}

template <typename T, size_t Width, abi_tag ABI>
void simd_bitwise_operations() {
  simd_type<T, Width, ABI> first;
  simd_type<T, Width, ABI> second;
  alignas(first.alignment()) std::array<T, Width> first_array   = generate_random_array<T, Width>();
  alignas(second.alignment()) std::array<T, Width> second_array = generate_random_array<T, Width>();
  first.load(first_array.data());
  second.load(second_array.data());
  simd_type<T, Width, ABI> simd_and = first & second;
  simd_type<T, Width, ABI> simd_or  = first | second;
  simd_type<T, Width, ABI> simd_xor = first ^ second;
  alignas(first.alignment()) std::array<T, Width> array_and{}, array_or{}, array_xor{};
  for (int i = 0; i < Width; ++i) {
    if constexpr (std::is_same_v<T, std::complex<float>>) {
      auto and_temp_real = std::bit_cast<float>(std::bit_cast<int32_t>(first_array[i].real()) & std::bit_cast<int32_t>(second_array[i].real()));
      auto and_temp_imag = std::bit_cast<float>(std::bit_cast<int32_t>(first_array[i].imag()) & std::bit_cast<int32_t>(second_array[i].imag()));
      auto or_temp_real  = std::bit_cast<float>(std::bit_cast<int32_t>(first_array[i].real()) | std::bit_cast<int32_t>(second_array[i].real()));
      auto or_temp_imag  = std::bit_cast<float>(std::bit_cast<int32_t>(first_array[i].imag()) | std::bit_cast<int32_t>(second_array[i].imag()));
      auto xor_temp_real = std::bit_cast<float>(std::bit_cast<int32_t>(first_array[i].real()) ^ std::bit_cast<int32_t>(second_array[i].real()));
      auto xor_temp_imag = std::bit_cast<float>(std::bit_cast<int32_t>(first_array[i].imag()) ^ std::bit_cast<int32_t>(second_array[i].imag()));
      array_and[i]       = std::complex<float>{and_temp_real, and_temp_imag};
      array_or[i]        = std::complex<float>{or_temp_real, or_temp_imag};
      array_xor[i]       = std::complex<float>{xor_temp_real, xor_temp_imag};

    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      auto and_temp_real = std::bit_cast<double>(std::bit_cast<int64_t>(first_array[i].real()) & std::bit_cast<int64_t>(second_array[i].real()));
      auto and_temp_imag = std::bit_cast<double>(std::bit_cast<int64_t>(first_array[i].imag()) & std::bit_cast<int64_t>(second_array[i].imag()));
      auto or_temp_real  = std::bit_cast<double>(std::bit_cast<int64_t>(first_array[i].real()) | std::bit_cast<int64_t>(second_array[i].real()));
      auto or_temp_imag  = std::bit_cast<double>(std::bit_cast<int64_t>(first_array[i].imag()) | std::bit_cast<int64_t>(second_array[i].imag()));
      auto xor_temp_real = std::bit_cast<double>(std::bit_cast<int64_t>(first_array[i].real()) ^ std::bit_cast<int64_t>(second_array[i].real()));
      auto xor_temp_imag = std::bit_cast<double>(std::bit_cast<int64_t>(first_array[i].imag()) ^ std::bit_cast<int64_t>(second_array[i].imag()));
      array_and[i]       = std::complex<double>{and_temp_real, and_temp_imag};
      array_or[i]        = std::complex<double>{or_temp_real, or_temp_imag};
      array_xor[i]       = std::complex<double>{xor_temp_real, xor_temp_imag};

    } else {
      if constexpr (sizeof(T) == 4) {
        array_and[i] = std::bit_cast<T>(std::bit_cast<int32_t>(first_array[i]) & std::bit_cast<int32_t>(second_array[i]));
        array_or[i]  = std::bit_cast<T>(std::bit_cast<int32_t>(first_array[i]) | std::bit_cast<int32_t>(second_array[i]));
        array_xor[i] = std::bit_cast<T>(std::bit_cast<int32_t>(first_array[i]) ^ std::bit_cast<int32_t>(second_array[i]));
      } else {
        array_and[i] = std::bit_cast<T>(std::bit_cast<int64_t>(first_array[i]) & std::bit_cast<int64_t>(second_array[i]));
        array_or[i]  = std::bit_cast<T>(std::bit_cast<int64_t>(first_array[i]) | std::bit_cast<int64_t>(second_array[i]));
        array_xor[i] = std::bit_cast<T>(std::bit_cast<int64_t>(first_array[i]) ^ std::bit_cast<int64_t>(second_array[i]));
      }
    }
  }
  check_simd_array_equal(simd_and, array_and);
  check_simd_array_equal(simd_or, array_or);
  check_simd_array_equal(simd_xor, array_xor);
}

TEST(NDA, SimdDefaultConstructor) {
  // Default SIMD types
  simd_type_default_constructor<float, 1, abi_tag::Default>();
  simd_type_default_constructor<double, 1, abi_tag::Default>();
  simd_type_default_constructor<int32_t, 1, abi_tag::Default>();
  simd_type_default_constructor<int64_t, 1, abi_tag::Default>();
  simd_type_default_constructor<std::complex<float>, 1, abi_tag::Default>();
  simd_type_default_constructor<std::complex<double>, 1, abi_tag::Default>();

#ifdef __SSE2__
  // SSE SIMD types
  simd_type_default_constructor<float, 4, abi_tag::SSE>();
  simd_type_default_constructor<double, 2, abi_tag::SSE>();
  simd_type_default_constructor<int32_t, 4, abi_tag::SSE>();
  simd_type_default_constructor<int64_t, 2, abi_tag::SSE>();
  simd_type_default_constructor<std::complex<float>, 2, abi_tag::SSE>();
  simd_type_default_constructor<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  // AVX SIMD types
  simd_type_default_constructor<float, 8, abi_tag::AVX>();
  simd_type_default_constructor<double, 4, abi_tag::AVX>();
  simd_type_default_constructor<int32_t, 8, abi_tag::AVX>();
  simd_type_default_constructor<int64_t, 4, abi_tag::AVX>();
  simd_type_default_constructor<std::complex<float>, 4, abi_tag::AVX>();
  simd_type_default_constructor<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  // AVX512 SIMD types
  simd_type_default_constructor<float, 16, abi_tag::AVX512>();
  simd_type_default_constructor<double, 8, abi_tag::AVX512>();
  simd_type_default_constructor<int32_t, 16, abi_tag::AVX512>();
  simd_type_default_constructor<int64_t, 8, abi_tag::AVX512>();
  simd_type_default_constructor<std::complex<float>, 8, abi_tag::AVX512>();
  simd_type_default_constructor<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, SimdSingleValueConstructor) {
  // Default SIMD types
  simd_type_value_constructor<float, 1, abi_tag::Default>(1.5f);
  simd_type_value_constructor<double, 1, abi_tag::Default>(2.5);
  simd_type_value_constructor<int32_t, 1, abi_tag::Default>(3);
  simd_type_value_constructor<int64_t, 1, abi_tag::Default>(4);
  simd_type_value_constructor<std::complex<float>, 1, abi_tag::Default>(std::complex<float>{5.5f, 35.5f});
  simd_type_value_constructor<std::complex<double>, 1, abi_tag::Default>(std::complex<double>{6.5, 36.5});

#ifdef __SSE2__
  // SSE SIMD types
  simd_type_value_constructor<float, 4, abi_tag::SSE>(7.5f);
  simd_type_value_constructor<double, 2, abi_tag::SSE>(8.5);
  simd_type_value_constructor<int32_t, 4, abi_tag::SSE>(9);
  simd_type_value_constructor<int64_t, 2, abi_tag::SSE>(10);
  simd_type_value_constructor<std::complex<float>, 2, abi_tag::SSE>(std::complex<float>{11.5f, 311.5f});
  simd_type_value_constructor<std::complex<double>, 1, abi_tag::SSE>(std::complex<double>{12.5, 312.5});
#endif

#ifdef __AVX__
  // AVX SIMD types
  simd_type_value_constructor<float, 8, abi_tag::AVX>(13.5f);
  simd_type_value_constructor<double, 4, abi_tag::AVX>(14.5);
  simd_type_value_constructor<int32_t, 8, abi_tag::AVX>(15);
  simd_type_value_constructor<int64_t, 4, abi_tag::AVX>(16);
  simd_type_value_constructor<std::complex<float>, 4, abi_tag::AVX>(std::complex<float>{17.5f, 317.5f});
  simd_type_value_constructor<std::complex<double>, 2, abi_tag::AVX>(std::complex<double>{18.5, 318.5});
#endif

#ifdef __AVX512F__
  // AVX512 SIMD types
  simd_type_value_constructor<float, 16, abi_tag::AVX512>(19.5f);
  simd_type_value_constructor<double, 8, abi_tag::AVX512>(20.5);
  simd_type_value_constructor<int32_t, 16, abi_tag::AVX512>(21);
  simd_type_value_constructor<int64_t, 8, abi_tag::AVX512>(22);
  simd_type_value_constructor<std::complex<float>, 8, abi_tag::AVX512>(std::complex<float>{23.5f, 233.5f});
  simd_type_value_constructor<std::complex<double>, 4, abi_tag::AVX512>(std::complex<double>{24.5, 324.5});
#endif
}

TEST(NDA, SimdPointerConstructor) {
  // Default SIMD types
  simd_pointer_constructor<float, 1, abi_tag::Default>();
  simd_pointer_constructor<double, 1, abi_tag::Default>();
  simd_pointer_constructor<int32_t, 1, abi_tag::Default>();
  simd_pointer_constructor<int64_t, 1, abi_tag::Default>();
  simd_pointer_constructor<std::complex<float>, 1, abi_tag::Default>();
  simd_pointer_constructor<std::complex<double>, 1, abi_tag::Default>();

#ifdef __SSE2__
  // SSE SIMD types
  simd_pointer_constructor<float, 4, abi_tag::SSE>();
  simd_pointer_constructor<double, 2, abi_tag::SSE>();
  simd_pointer_constructor<int32_t, 4, abi_tag::SSE>();
  simd_pointer_constructor<int64_t, 2, abi_tag::SSE>();
  simd_pointer_constructor<std::complex<float>, 2, abi_tag::SSE>();
  simd_pointer_constructor<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  // AVX SIMD types
  simd_pointer_constructor<float, 8, abi_tag::AVX>();
  simd_pointer_constructor<double, 4, abi_tag::AVX>();
  simd_pointer_constructor<int32_t, 8, abi_tag::AVX>();
  simd_pointer_constructor<int64_t, 4, abi_tag::AVX>();
  simd_pointer_constructor<std::complex<float>, 4, abi_tag::AVX>();
  simd_pointer_constructor<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  // AVX512 SIMD types
  simd_pointer_constructor<float, 16, abi_tag::AVX512>();
  simd_pointer_constructor<double, 8, abi_tag::AVX512>();
  simd_pointer_constructor<int32_t, 16, abi_tag::AVX512>();
  simd_pointer_constructor<int64_t, 8, abi_tag::AVX512>();
  simd_pointer_constructor<std::complex<float>, 8, abi_tag::AVX512>();
  simd_pointer_constructor<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, SimdSizeAlignment) {
  // Test for Default ABI types
  test_simd_size_alignment<float, 1, abi_tag::Default>();
  test_simd_size_alignment<double, 1, abi_tag::Default>();
  test_simd_size_alignment<int32_t, 1, abi_tag::Default>();
  test_simd_size_alignment<int64_t, 1, abi_tag::Default>();
  test_simd_size_alignment<std::complex<float>, 1, abi_tag::Default>();
  test_simd_size_alignment<std::complex<double>, 1, abi_tag::Default>();

#ifdef __SSE2__
  // Test for SSE ABI types
  test_simd_size_alignment<float, 4, abi_tag::SSE>();
  test_simd_size_alignment<double, 2, abi_tag::SSE>();
  test_simd_size_alignment<int32_t, 4, abi_tag::SSE>();
  test_simd_size_alignment<int64_t, 2, abi_tag::SSE>();
  test_simd_size_alignment<std::complex<float>, 2, abi_tag::SSE>();
  test_simd_size_alignment<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  // Test for AVX ABI types
  test_simd_size_alignment<float, 8, abi_tag::AVX>();
  test_simd_size_alignment<double, 4, abi_tag::AVX>();
  test_simd_size_alignment<int32_t, 8, abi_tag::AVX>();
  test_simd_size_alignment<int64_t, 4, abi_tag::AVX>();
  test_simd_size_alignment<std::complex<float>, 4, abi_tag::AVX>();
  test_simd_size_alignment<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  // Test for AVX512 ABI types
  test_simd_size_alignment<float, 16, abi_tag::AVX512>();
  test_simd_size_alignment<double, 8, abi_tag::AVX512>();
  test_simd_size_alignment<int32_t, 16, abi_tag::AVX512>();
  test_simd_size_alignment<int64_t, 8, abi_tag::AVX512>();
  test_simd_size_alignment<std::complex<float>, 8, abi_tag::AVX512>();
  test_simd_size_alignment<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, SimdLoadStore) {
  // Test for Default ABI types
  simd_load_store<float, 1, abi_tag::Default>();
  simd_load_store<double, 1, abi_tag::Default>();
  simd_load_store<int32_t, 1, abi_tag::Default>();
  simd_load_store<int64_t, 1, abi_tag::Default>();
  simd_load_store<std::complex<float>, 1, abi_tag::Default>();
  simd_load_store<std::complex<double>, 1, abi_tag::Default>();

#ifdef __SSE2__
  // Test for SSE ABI types
  simd_load_store<float, 4, abi_tag::SSE>();
  simd_load_store<double, 2, abi_tag::SSE>();
  simd_load_store<int32_t, 4, abi_tag::SSE>();
  simd_load_store<int64_t, 2, abi_tag::SSE>();
  simd_load_store<std::complex<float>, 2, abi_tag::SSE>();
  simd_load_store<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  // Test for AVX ABI types
  simd_load_store<float, 8, abi_tag::AVX>();
  simd_load_store<double, 4, abi_tag::AVX>();
  simd_load_store<int32_t, 8, abi_tag::AVX>();
  simd_load_store<int64_t, 4, abi_tag::AVX>();
  simd_load_store<std::complex<float>, 4, abi_tag::AVX>();
  simd_load_store<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  // Test for AVX512 ABI types
  simd_load_store<float, 16, abi_tag::AVX512>();
  simd_load_store<double, 8, abi_tag::AVX512>();
  simd_load_store<int32_t, 16, abi_tag::AVX512>();
  simd_load_store<int64_t, 8, abi_tag::AVX512>();
  simd_load_store<std::complex<float>, 8, abi_tag::AVX512>();
  simd_load_store<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, SimdBinaryCompoundOperators) {
  // Default ABI
  simd_compound_binary_operators<float, 1, abi_tag::Default>();
  simd_compound_binary_operators<double, 1, abi_tag::Default>();
  simd_compound_binary_operators<int32_t, 1, abi_tag::Default>();
  simd_compound_binary_operators<int64_t, 1, abi_tag::Default>();
  simd_compound_binary_operators<std::complex<float>, 1, abi_tag::Default>();
  simd_compound_binary_operators<std::complex<double>, 1, abi_tag::Default>();

#ifdef __SSE2__
  simd_compound_binary_operators<float, 4, abi_tag::SSE>();
  simd_compound_binary_operators<double, 2, abi_tag::SSE>();
  simd_compound_binary_operators<int32_t, 4, abi_tag::SSE>();
  simd_compound_binary_operators<int64_t, 2, abi_tag::SSE>();
  simd_compound_binary_operators<std::complex<float>, 2, abi_tag::SSE>();
  simd_compound_binary_operators<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  simd_compound_binary_operators<float, 8, abi_tag::AVX>();
  simd_compound_binary_operators<double, 4, abi_tag::AVX>();
  simd_compound_binary_operators<int32_t, 8, abi_tag::AVX>();
  simd_compound_binary_operators<int64_t, 4, abi_tag::AVX>();
  simd_compound_binary_operators<std::complex<float>, 4, abi_tag::AVX>();
  simd_compound_binary_operators<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  simd_compound_binary_operators<float, 16, abi_tag::AVX512>();
  simd_compound_binary_operators<double, 8, abi_tag::AVX512>();
  simd_compound_binary_operators<int32_t, 16, abi_tag::AVX512>();
  simd_compound_binary_operators<int64_t, 8, abi_tag::AVX512>();
  simd_compound_binary_operators<std::complex<float>, 8, abi_tag::AVX512>();
  simd_compound_binary_operators<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, SimdInitializerListConstructor) {
  simd_initializer_list_constructor<float, 1, abi_tag::Default>({1.0f});
  simd_initializer_list_constructor<double, 1, abi_tag::Default>({1.0});
  simd_initializer_list_constructor<int32_t, 1, abi_tag::Default>({1});
  simd_initializer_list_constructor<int64_t, 1, abi_tag::Default>({1});
  simd_initializer_list_constructor<std::complex<float>, 1, abi_tag::Default>({{1.0f, 2.0f}});
  simd_initializer_list_constructor<std::complex<double>, 1, abi_tag::Default>({{1.0, 2.0}});

#ifdef __SSE2__
  simd_initializer_list_constructor<float, 4, abi_tag::SSE>({1.0f, 2.0f, 3.0f, 4.0f});
  simd_initializer_list_constructor<double, 2, abi_tag::SSE>({1.0, 2.0});
  simd_initializer_list_constructor<int32_t, 4, abi_tag::SSE>({1, 2, 3, 4});
  simd_initializer_list_constructor<int64_t, 2, abi_tag::SSE>({1, 2});
  simd_initializer_list_constructor<std::complex<float>, 2, abi_tag::SSE>({{1.0f, 2.0f}, {3.0f, 4.0f}});
  simd_initializer_list_constructor<std::complex<double>, 1, abi_tag::SSE>({{1.0, 2.0}});
#endif

#ifdef __AVX__
  simd_initializer_list_constructor<float, 8, abi_tag::AVX>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  simd_initializer_list_constructor<double, 4, abi_tag::AVX>({1.0, 2.0, 3.0, 4.0});
  simd_initializer_list_constructor<int32_t, 8, abi_tag::AVX>({1, 2, 3, 4, 5, 6, 7, 8});
  simd_initializer_list_constructor<int64_t, 4, abi_tag::AVX>({1, 2, 3, 4});
  simd_initializer_list_constructor<std::complex<float>, 4, abi_tag::AVX>({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});
  simd_initializer_list_constructor<std::complex<double>, 2, abi_tag::AVX>({{1.0, 2.0}, {3.0, 4.0}});
#endif

#ifdef __AVX512F__
  simd_initializer_list_constructor<float, 16, abi_tag::AVX512>(
     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
  simd_initializer_list_constructor<double, 8, abi_tag::AVX512>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
  simd_initializer_list_constructor<int32_t, 16, abi_tag::AVX512>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  simd_initializer_list_constructor<int64_t, 8, abi_tag::AVX512>({1, 2, 3, 4, 5, 6, 7, 8});
  simd_initializer_list_constructor<std::complex<float>, 8, abi_tag::AVX512>(
     {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}, {13.0f, 14.0f}, {15.0f, 16.0f}});
  simd_initializer_list_constructor<std::complex<double>, 4, abi_tag::AVX512>({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}});
#endif
  // Now extra argument should cause it to throw an error.
  EXPECT_THROW((simd_type<float, 1, abi_tag::Default>{1.0f, 2.0f}), std::runtime_error);
  EXPECT_THROW((simd_type<double, 1, abi_tag::Default>{1.0, 2.0}), std::runtime_error);
  EXPECT_THROW((simd_type<int32_t, 1, abi_tag::Default>{1, 2}), std::runtime_error);
  EXPECT_THROW((simd_type<int64_t, 1, abi_tag::Default>{1, 2}), std::runtime_error);
  EXPECT_THROW((simd_type<std::complex<float>, 1, abi_tag::Default>{{1.0f, 2.0f}, {3.0f, 4.0f}}), std::runtime_error);
  EXPECT_THROW((simd_type<std::complex<double>, 1, abi_tag::Default>{{1.0, 2.0}, {3.0, 4.0}}), std::runtime_error);

#ifdef __SSE2__
  EXPECT_THROW((simd_type<float, 4, abi_tag::SSE>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}), std::runtime_error);
  EXPECT_THROW((simd_type<double, 2, abi_tag::SSE>{1.0, 2.0, 3.0}), std::runtime_error);
  EXPECT_THROW((simd_type<int32_t, 4, abi_tag::SSE>{1, 2, 3, 4, 5}), std::runtime_error);
  EXPECT_THROW((simd_type<int64_t, 2, abi_tag::SSE>{1, 2, 3}), std::runtime_error);
  EXPECT_THROW((simd_type<std::complex<float>, 2, abi_tag::SSE>{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}), std::runtime_error);
  EXPECT_THROW((simd_type<std::complex<double>, 1, abi_tag::SSE>{{1.0, 2.0}, {3.0, 4.0}}), std::runtime_error);
#endif

#ifdef __AVX__
  EXPECT_THROW((simd_type<float, 8, abi_tag::AVX>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}), std::runtime_error);
  EXPECT_THROW((simd_type<double, 4, abi_tag::AVX>{1.0, 2.0, 3.0, 4.0, 5.0}), std::runtime_error);
  EXPECT_THROW((simd_type<int32_t, 8, abi_tag::AVX>{1, 2, 3, 4, 5, 6, 7, 8, 9}), std::runtime_error);
  EXPECT_THROW((simd_type<int64_t, 4, abi_tag::AVX>{1, 2, 3, 4, 5}), std::runtime_error);
  EXPECT_THROW((simd_type<std::complex<float>, 4, abi_tag::AVX>{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}}),
               std::runtime_error);
  EXPECT_THROW((simd_type<std::complex<double>, 2, abi_tag::AVX>{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}), std::runtime_error);
#endif

#ifdef __AVX512F__
  EXPECT_THROW((simd_type<float, 16, abi_tag::AVX512>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                                      16.0f, 17.0f}),
               std::runtime_error);
  EXPECT_THROW((simd_type<double, 8, abi_tag::AVX512>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}), std::runtime_error);
  EXPECT_THROW((simd_type<int32_t, 16, abi_tag::AVX512>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}), std::runtime_error);
  EXPECT_THROW((simd_type<int64_t, 8, abi_tag::AVX512>{1, 2, 3, 4, 5, 6, 7, 8, 9}), std::runtime_error);
  EXPECT_THROW(
     (simd_type<std::complex<float>, 8, abi_tag::AVX512>{
        {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}, {13.0f, 14.0f}, {15.0f, 16.0f}, {17.0f, 18.0f}}),
     std::runtime_error);
  EXPECT_THROW((simd_type<std::complex<double>, 4, abi_tag::AVX512>{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}, {9.0, 10.0}}),
               std::runtime_error);
#endif
}

TEST(NDA, SimdEqualityOperator) {
  simd_equality_operator<float, 1, abi_tag::Default>();
  simd_equality_operator<double, 1, abi_tag::Default>();
  simd_equality_operator<int32_t, 1, abi_tag::Default>();
  simd_equality_operator<int64_t, 1, abi_tag::Default>();
  simd_equality_operator<std::complex<float>, 1, abi_tag::Default>();
  simd_equality_operator<std::complex<double>, 1, abi_tag::Default>();

#ifdef __SSE2__
  simd_equality_operator<float, 4, abi_tag::SSE>();
  simd_equality_operator<double, 2, abi_tag::SSE>();
  simd_equality_operator<int32_t, 4, abi_tag::SSE>();
  simd_equality_operator<int64_t, 2, abi_tag::SSE>();
  simd_equality_operator<std::complex<float>, 2, abi_tag::SSE>();
  simd_equality_operator<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  simd_equality_operator<float, 8, abi_tag::AVX>();
  simd_equality_operator<double, 4, abi_tag::AVX>();
  simd_equality_operator<int32_t, 8, abi_tag::AVX>();
  simd_equality_operator<int64_t, 4, abi_tag::AVX>();
  simd_equality_operator<std::complex<float>, 4, abi_tag::AVX>();
  simd_equality_operator<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  simd_equality_operator<float, 16, abi_tag::AVX512>();
  simd_equality_operator<double, 8, abi_tag::AVX512>();
  simd_equality_operator<int32_t, 16, abi_tag::AVX512>();
  simd_equality_operator<int64_t, 8, abi_tag::AVX512>();
  simd_equality_operator<std::complex<float>, 8, abi_tag::AVX512>();
  simd_equality_operator<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, SimdEdgeCases) {
  // Default SIMD types
  simd_binary_operators_edge_cases<float, 1, abi_tag::Default>();
  simd_binary_operators_edge_cases<double, 1, abi_tag::Default>();
  simd_binary_operators_edge_cases<int32_t, 1, abi_tag::Default>();
  simd_binary_operators_edge_cases<int64_t, 1, abi_tag::Default>();
  simd_binary_operators_edge_cases<std::complex<float>, 1, abi_tag::Default>();
  simd_binary_operators_edge_cases<std::complex<double>, 1, abi_tag::Default>();

#ifdef __SSE2__
  // SSE SIMD types
  simd_binary_operators_edge_cases<float, 4, abi_tag::SSE>();
  simd_binary_operators_edge_cases<double, 2, abi_tag::SSE>();
  simd_binary_operators_edge_cases<int32_t, 4, abi_tag::SSE>();
  simd_binary_operators_edge_cases<int64_t, 2, abi_tag::SSE>();
  simd_binary_operators_edge_cases<std::complex<float>, 2, abi_tag::SSE>();
  simd_binary_operators_edge_cases<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  // AVX SIMD types
  simd_binary_operators_edge_cases<float, 8, abi_tag::AVX>();
  simd_binary_operators_edge_cases<double, 4, abi_tag::AVX>();
  simd_binary_operators_edge_cases<int32_t, 8, abi_tag::AVX>();
  simd_binary_operators_edge_cases<int64_t, 4, abi_tag::AVX>();
  simd_binary_operators_edge_cases<std::complex<float>, 4, abi_tag::AVX>();
  simd_binary_operators_edge_cases<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  // AVX512 SIMD types
  simd_binary_operators_edge_cases<float, 16, abi_tag::AVX512>();
  simd_binary_operators_edge_cases<double, 8, abi_tag::AVX512>();
  simd_binary_operators_edge_cases<int32_t, 16, abi_tag::AVX512>();
  simd_binary_operators_edge_cases<int64_t, 8, abi_tag::AVX512>();
  simd_binary_operators_edge_cases<std::complex<float>, 8, abi_tag::AVX512>();
  simd_binary_operators_edge_cases<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, SimdNanInf) {
#ifdef __SSE2__
  // SSE SIMD types
  simd_check_nan_inf<float, 4, abi_tag::SSE>();
  simd_check_nan_inf<double, 2, abi_tag::SSE>();
  simd_check_nan_inf<std::complex<float>, 2, abi_tag::SSE>();
  // simd_check_nan_inf<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  // AVX SIMD types
  simd_check_nan_inf<float, 8, abi_tag::AVX>();
  simd_check_nan_inf<double, 4, abi_tag::AVX>();
  simd_check_nan_inf<std::complex<float>, 4, abi_tag::AVX>();
  simd_check_nan_inf<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  // AVX512 SIMD types
  simd_check_nan_inf<float, 16, abi_tag::AVX512>();
  simd_check_nan_inf<double, 8, abi_tag::AVX512>();
  simd_check_nan_inf<std::complex<float>, 8, abi_tag::AVX512>();
  simd_check_nan_inf<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, SimdAbs) {
  simd_function_abs<float, 1, abi_tag::Default>();
  simd_function_abs<double, 1, abi_tag::Default>();
  simd_function_abs<int32_t, 1, abi_tag::Default>();
  simd_function_abs<int64_t, 1, abi_tag::Default>();
  //simd_equality_operator<std::complex<float>, 1, abi_tag::Default>();
  // simd_equality_operator<std::complex<double>, 1, abi_tag::Default>();

#ifdef __SSE2__
  simd_function_abs<float, 4, abi_tag::SSE>();
  simd_function_abs<double, 2, abi_tag::SSE>();
  simd_function_abs<int32_t, 4, abi_tag::SSE>();
  simd_function_abs<int64_t, 2, abi_tag::SSE>();
  // simd_equality_operator<std::complex<float>, 2, abi_tag::SSE>();
  // simd_equality_operator<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  simd_function_abs<float, 8, abi_tag::AVX>();
  simd_function_abs<double, 4, abi_tag::AVX>();
  simd_function_abs<int32_t, 8, abi_tag::AVX>();
  simd_function_abs<int64_t, 4, abi_tag::AVX>();
  //simd_equality_operator<std::complex<float>, 4, abi_tag::AVX>();
  // simd_equality_operator<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  simd_function_abs<float, 16, abi_tag::AVX512>();
  simd_function_abs<double, 8, abi_tag::AVX512>();
  simd_function_abs<int32_t, 16, abi_tag::AVX512>();
  simd_function_abs<int64_t, 8, abi_tag::AVX512>();
  // simd_function_abs<std::complex<float>, 8, abi_tag::AVX512>();
  // simd_function_abs<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, SimdConj) {
  simd_function_conj<float, 1, abi_tag::Default>();
  simd_function_conj<double, 1, abi_tag::Default>();
  simd_function_conj<int32_t, 1, abi_tag::Default>();
  simd_function_conj<int64_t, 1, abi_tag::Default>();
  simd_function_conj<std::complex<float>, 1, abi_tag::Default>();
  simd_function_conj<std::complex<double>, 1, abi_tag::Default>();

#ifdef __SSE2__
  simd_function_conj<float, 4, abi_tag::SSE>();
  simd_function_conj<double, 2, abi_tag::SSE>();
  simd_function_conj<int32_t, 4, abi_tag::SSE>();
  simd_function_conj<int64_t, 2, abi_tag::SSE>();
  simd_function_conj<std::complex<float>, 2, abi_tag::SSE>();
  simd_function_conj<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  simd_function_conj<float, 8, abi_tag::AVX>();
  simd_function_conj<double, 4, abi_tag::AVX>();
  simd_function_conj<int32_t, 8, abi_tag::AVX>();
  simd_function_conj<int64_t, 4, abi_tag::AVX>();
  simd_function_conj<std::complex<float>, 4, abi_tag::AVX>();
  simd_function_conj<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  simd_function_conj<float, 16, abi_tag::AVX512>();
  simd_function_conj<double, 8, abi_tag::AVX512>();
  simd_function_conj<int32_t, 16, abi_tag::AVX512>();
  simd_function_conj<int64_t, 8, abi_tag::AVX512>();
  simd_function_conj<std::complex<float>, 8, abi_tag::AVX512>();
  simd_function_conj<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, SimdSqrt) {
  simd_function_sqrt<float, 1, abi_tag::Default>();
  simd_function_sqrt<double, 1, abi_tag::Default>();
  // simd_function_sqrt<int32_t, 1, abi_tag::Default>();
  // simd_function_sqrt<int64_t, 1, abi_tag::Default>();
  // simd_function_sqrt<std::complex<float>, 1, abi_tag::Default>();
  //simd_function_sqrt<std::complex<double>, 1, abi_tag::Default>();

#ifdef __SSE2__
  simd_function_sqrt<float, 4, abi_tag::SSE>();
  simd_function_sqrt<double, 2, abi_tag::SSE>();
  //simd_function_sqrt<int32_t, 4, abi_tag::SSE>();
  //simd_function_sqrt<int64_t, 2, abi_tag::SSE>();
  //simd_function_sqrt<std::complex<float>, 2, abi_tag::SSE>();
  //simd_function_sqrt<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  simd_function_sqrt<float, 8, abi_tag::AVX>();
  simd_function_sqrt<double, 4, abi_tag::AVX>();
  //simd_function_sqrt<int32_t, 8, abi_tag::AVX>();
  //simd_function_sqrt<int64_t, 4, abi_tag::AVX>();
  //simd_function_sqrt<std::complex<float>, 4, abi_tag::AVX>();
  //simd_function_sqrt<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  simd_function_sqrt<float, 16, abi_tag::AVX512>();
  simd_function_sqrt<double, 8, abi_tag::AVX512>();
  //simd_function_sqrt<int32_t, 16, abi_tag::AVX512>();
  //simd_function_sqrt<int64_t, 8, abi_tag::AVX512>();
  //simd_function_sqrt<std::complex<float>, 8, abi_tag::AVX512>();
  //simd_function_sqrt<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, SimdMin) {
  // Default SIMD types
  simd_function_min<float, 1, abi_tag::Default>();
  simd_function_min<double, 1, abi_tag::Default>();
  simd_function_min<int32_t, 1, abi_tag::Default>();
  simd_function_min<int64_t, 1, abi_tag::Default>();
  // simd_bitwise_operations<std::complex<float>, 1, abi_tag::Default>();
  // simd_bitwise_operations<std::complex<double>, 1, abi_tag::Default>();

#ifdef __SSE2__
  // SSE SIMD types
  simd_function_min<float, 4, abi_tag::SSE>();
  simd_function_min<double, 2, abi_tag::SSE>();
  simd_function_min<int32_t, 4, abi_tag::SSE>();
  simd_function_min<int64_t, 2, abi_tag::SSE>();
  // simd_bitwise_operations<std::complex<float>, 2, abi_tag::SSE>();
  // simd_bitwise_operations<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  // AVX SIMD types
  simd_function_min<float, 8, abi_tag::AVX>();
  simd_function_min<double, 4, abi_tag::AVX>();
  simd_function_min<int32_t, 8, abi_tag::AVX>();
  simd_function_min<int64_t, 4, abi_tag::AVX>();
  // simd_bitwise_operations<std::complex<float>, 4, abi_tag::AVX>();
  // simd_bitwise_operations<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  // AVX512 SIMD types
  simd_function_min<float, 16, abi_tag::AVX512>();
  simd_function_min<double, 8, abi_tag::AVX512>();
  simd_function_min<int32_t, 16, abi_tag::AVX512>();
  simd_function_min<int64_t, 8, abi_tag::AVX512>();
  // simd_bitwise_operations<std::complex<float>, 8, abi_tag::AVX512>();
  // simd_bitwise_operations<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, SimdMax) {
  // Default SIMD types
  simd_function_max<float, 1, abi_tag::Default>();
  simd_function_max<double, 1, abi_tag::Default>();
  simd_function_max<int32_t, 1, abi_tag::Default>();
  simd_function_max<int64_t, 1, abi_tag::Default>();
  // simd_bitwise_operations<std::complex<float>, 1, abi_tag::Default>();
  // simd_bitwise_operations<std::complex<double>, 1, abi_tag::Default>();

#ifdef __SSE2__
  // SSE SIMD types
  simd_function_max<float, 4, abi_tag::SSE>();
  simd_function_max<double, 2, abi_tag::SSE>();
  simd_function_max<int32_t, 4, abi_tag::SSE>();
  simd_function_max<int64_t, 2, abi_tag::SSE>();
  // simd_bitwise_operations<std::complex<float>, 2, abi_tag::SSE>();
  // simd_bitwise_operations<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  // AVX SIMD types
  simd_function_max<float, 8, abi_tag::AVX>();
  simd_function_max<double, 4, abi_tag::AVX>();
  simd_function_max<int32_t, 8, abi_tag::AVX>();
  simd_function_max<int64_t, 4, abi_tag::AVX>();
  // simd_bitwise_operations<std::complex<float>, 4, abi_tag::AVX>();
  // simd_bitwise_operations<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  // AVX512 SIMD types
  simd_function_max<float, 16, abi_tag::AVX512>();
  simd_function_max<double, 8, abi_tag::AVX512>();
  simd_function_max<int32_t, 16, abi_tag::AVX512>();
  simd_function_max<int64_t, 8, abi_tag::AVX512>();
  // simd_bitwise_operations<std::complex<float>, 8, abi_tag::AVX512>();
  // simd_bitwise_operations<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, SimdBitwiseOperations) {

  // Default SIMD types
  simd_bitwise_operations<float, 1, abi_tag::Default>();
  simd_bitwise_operations<double, 1, abi_tag::Default>();
  simd_bitwise_operations<int32_t, 1, abi_tag::Default>();
  simd_bitwise_operations<int64_t, 1, abi_tag::Default>();
  simd_bitwise_operations<std::complex<float>, 1, abi_tag::Default>();
  simd_bitwise_operations<std::complex<double>, 1, abi_tag::Default>();

#ifdef __SSE2__
  // SSE SIMD types
  simd_bitwise_operations<float, 4, abi_tag::SSE>();
  simd_bitwise_operations<double, 2, abi_tag::SSE>();
  simd_bitwise_operations<int32_t, 4, abi_tag::SSE>();
  simd_bitwise_operations<int64_t, 2, abi_tag::SSE>();
  simd_bitwise_operations<std::complex<float>, 2, abi_tag::SSE>();
  simd_bitwise_operations<std::complex<double>, 1, abi_tag::SSE>();
#endif

#ifdef __AVX__
  // AVX SIMD types
  simd_bitwise_operations<float, 8, abi_tag::AVX>();
  simd_bitwise_operations<double, 4, abi_tag::AVX>();
  simd_bitwise_operations<int32_t, 8, abi_tag::AVX>();
  simd_bitwise_operations<int64_t, 4, abi_tag::AVX>();
  simd_bitwise_operations<std::complex<float>, 4, abi_tag::AVX>();
  simd_bitwise_operations<std::complex<double>, 2, abi_tag::AVX>();
#endif

#ifdef __AVX512F__
  // AVX512 SIMD types
  simd_bitwise_operations<float, 16, abi_tag::AVX512>();
  simd_bitwise_operations<double, 8, abi_tag::AVX512>();
  simd_bitwise_operations<int32_t, 16, abi_tag::AVX512>();
  simd_bitwise_operations<int64_t, 8, abi_tag::AVX512>();
  simd_bitwise_operations<std::complex<float>, 8, abi_tag::AVX512>();
  simd_bitwise_operations<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

TEST(NDA, OurSIMD) {
  class add {
    public:
    float operator()(float a, float b) const { return a + b; }

    native_simd<float> load(native_simd<float> a, native_simd<float> b) const { return a + b; }
  };
  const long size1 = 11;
  const long size2 = 22;
  float k          = 1;
  array_aligned<float, 2> s({size1, size2});
  array_aligned<float, 2> x({size1, size2});
  for (int i = 0; i < size1; ++i) {
    for (int j = 0; j < size2; ++j) { s(i, j) = k++; }
  }
  for (int i = 0; i < size1; ++i) {
    for (int j = 0; j < size2; ++j) { x(i, j) = k++; }
  }
  //TODO: this doesnt work check.
  auto test         = nda::map(add{})(s, x);
  auto test2        = nda::map(add{})(test, test);
  array<float, 2> y = test2;
  std::cout << is_simd_enabled_v<float, decltype(test2)> << std::endl;
  std::cout << is_simd_enabled_v<float, decltype(test)> << std::endl;
  for (int i = 0; i < size1; ++i) {
    for (int j = 0; j < size2; ++j) { std::cout << s(i, j) << " "; }
    std::cout << std::endl;
  }
}
