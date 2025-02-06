#include <gtest/gtest.h>
#include <nda/gtest_tools.hpp>
#include <nda/simd/simd.hpp>
#include <type_traits>
#include <random>
#include <cmath>
#include <array>
#include <complex>
#include <algorithm>

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
      EXPECT_FLOAT_EQ(calculation[i], correct[i]);
    } else if constexpr (std::is_same_v<T, double>) {
      EXPECT_DOUBLE_EQ(calculation[i], correct[i]);
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      auto max         = std::max(std::abs(calculation[i]), std::abs(correct[i]));
      max              = std::max(max, 1.0f);
      double precision = 1e-6 * max;
      EXPECT_COMPLEX_NEAR(calculation[i], correct[i], precision);
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
      auto max         = std::max(std::abs(calculation[i]), std::abs(correct[i]));
      max              = std::max(max, 1.0);
      double precision = 1e-15 * max;
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

#ifdef __AVX512__
  // AVX512 SIMD types
  test_simd_type_constructor<float, 16, abi_tag::AVX512>();
  test_simd_type_constructor<double, 8, abi_tag::AVX512>();
  test_simd_type_constructor<int32_t, 16, abi_tag::AVX512>();
  test_simd_type_constructor<int64_t, 8, abi_tag::AVX512>();
  test_simd_type_constructor<std::complex<float>, 8, abi_tag::AVX512>();
  test_simd_type_constructor<std::complex<double>, 4, abi_tag::AVX512>();
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

#ifdef __AVX512__
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

#ifdef __AVX512__
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

#ifdef __AVX512__
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

#ifdef __AVX512__
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

#ifdef __AVX512__
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

#ifdef __AVX512__
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

#ifdef __AVX512__
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

#ifdef __AVX512__
  simd_equality_operator<float, 16, abi_tag::AVX512>();
  simd_equality_operator<double, 8, abi_tag::AVX512>();
  simd_equality_operator<int32_t, 16, abi_tag::AVX512>();
  simd_equality_operator<int64_t, 8, abi_tag::AVX512>();
  simd_equality_operator<std::complex<float>, 8, abi_tag::AVX512>();
  simd_equality_operator<std::complex<double>, 4, abi_tag::AVX512>();
#endif
}

//TODO do overflow/underflow/nan/inf tests to see the behaviour of simd registers.
// Do coverage tests.
TEST(NDA, OurSIMD) {
  // matrix_aligned<double, C_layout> s = matrix_aligned<double, C_layout>::rand({5, 7});
  // for (int i = 0 ; i < 1 ; ++i) {
  //   array_aligned<double , 2> s = array_aligned<double , 2>::rand({2 , 7});
  //   double sum_first = nda::product(s);
  //   double sum_second = nda::product<true>(s);
  //   std::cout << sum_first << std::endl;
  //   std::cout << sum_second << std::endl;
  //   EXPECT_DOUBLE_EQ(sum_first, sum_second);
  // }


  // 1 2 3 0
  // 4 5 6 0
  // 7 8 9 0
  // 10 11 12 0
  // 13 14 15 0
  // Transpose
  // 1 4 7 10 13
  // 2 5  8 11 14
  // 3 6 9 12 15
  // 0 0 0 0 0
  // auto r = transpose(s);
  // std::cout << "EXTent0= " << r.extent(0) << std::endl;
  // std::cout << "EXTent1= " << r.extent(1) << std::endl;
  // std::cout << r.indexmap().get_padding() << std::endl;
  // for (int i = 0; i < 3; ++i) {
  // for (int j = 0; j < 5; ++j) { std::cout << "r(" << i << ", " << j << ")=" << r(i, j) << " Address " << &r(i, j) << std::endl; }
  // }
  // for (int i = 0; i < 20; ++i) { std::cout << r.data()[i] << std::endl; }

  // std::cout << "HOP" << std::endl;
  // std::cout << r.indexmap().capacity() << std::endl;
  // std::cout << r.storage().size() << std::endl;
  // std::cout << s.storage().size() << std::endl;
  // std::cout << blas::get_ld(s) << std::endl;
  // std::cout << blas::get_ld(r) << std::endl;
  // std::cout << blas::has_C_layout<decltype(s)> << std::endl;
  // std::cout << blas::has_C_layout<decltype(r)> << std::endl;
  // std::cout << "ANAN" << std::endl;
  // for (int i = 0; i < 24; ++i) { std::cout << s.storage().data()[i] << std::endl; }
  // for (int i = 0; i < 20; ++i) { std::cout << r.storage().data()[i] << std::endl; }
}
