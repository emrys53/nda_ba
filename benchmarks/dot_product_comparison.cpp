#include "./bench_common.hpp"
#include <nda/simd/simd.hpp>

float dot_product_scalar(const float *a, const float *b, size_t size) {
  float sum = 0.0f;
  for (size_t i = 0; i < size; ++i) sum += a[i] * b[i];
  return sum;
}

float dot_product_sse(const float *a, const float *b, size_t size) {
  __m128 sum = _mm_setzero_ps();
  size_t i   = 0;
  for (; i + 4 <= size; i += 4) {
    __m128 va = _mm_loadu_ps(&a[i]);
    __m128 vb = _mm_loadu_ps(&b[i]);
    sum       = _mm_add_ps(sum, _mm_mul_ps(va, vb));
  }
  float result[4];
  _mm_storeu_ps(result, sum);
  float dot = 0.0f;
  for (int j = 0; j < 4; ++j) dot += result[j];
  for (; i < size; ++i) dot += a[i] * b[i]; // handle tail
  return dot;
}

float dot_product_avx(const float *a, const float *b, size_t size) {
  __m256 sum = _mm256_setzero_ps();
  size_t i   = 0;
  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_loadu_ps(&a[i]);
    __m256 vb = _mm256_loadu_ps(&b[i]);
    sum       = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
  }
  float result[8];
  _mm256_storeu_ps(result, sum);
  float dot = 0.0f;
  for (int j = 0; j < 8; ++j) dot += result[j];
  for (; i < size; ++i) dot += a[i] * b[i]; // handle tail
  return dot;
}

float dot_product_avx512(const float *a, const float *b, size_t size) {
  //  __m512 sum = _mm512_setzero_ps();
  //  size_t i   = 0;
  //  for (; i + 16 <= size; i += 16) {
  //    __m512 va = _mm512_loadu_ps(&a[i]);
  //    __m512 vb = _mm512_loadu_ps(&b[i]);
  //    sum       = _mm512_add_ps(sum, _mm512_mul_ps(va, vb));
  //  }
  //  float result[16];
  //  _mm512_storeu_ps(result, sum);
  //  float dot = 0.0f;
  //  for (int j = 0; j < 16; ++j) dot += result[j];
  //  for (; i < size; ++i) dot += a[i] * b[i]; // handle tail
  //  return dot;
}
float dot_product_generic(const float *a, const float *b, size_t size) {
  using namespace nda;
  using simd_t           = native_simd<float>;
  constexpr size_t width = simd_t::size();
  simd_t acc(simd_zero_initialize_t);
  size_t i = 0;
  for (; i + width <= size; i += width) {
    simd_t va(&a[i], simd_unaligned_memory_t);
    simd_t vb(&b[i], simd_unaligned_memory_t);
    acc = simd::fma_add(va, vb, acc); // Or acc += va * vb;
  }

  float sum = simd::reduce_sum(acc); // Horizontal sum of SIMD register
  for (; i < size; ++i) { sum += a[i] * b[i]; }
  return sum;
}

using value_t   = float;
const long Nmin = 9;
const long Nmax = 1 << 18;

template <typename Array, int Option>
static void dot_product_benchmark(benchmark::State &state) {
  long N = state.range(0);
  auto A = Array::rand({N});
  auto B = Array::rand({N});

  for (auto s : state) {
    if constexpr (Option == 0) {
      const volatile long result = dot_product_scalar(A.data(), B.data(), N);
    } else if constexpr (Option == 1) {
      const volatile long result = dot_product_sse(A.data(), B.data(), N);
    } else if constexpr (Option == 2) {
      const volatile long result = dot_product_avx(A.data(), B.data(), N);
    } else if constexpr (Option == 3) {
      const volatile long result = dot_product_avx512(A.data(), B.data(), N);
    } else if constexpr (Option == 4) {
      const volatile long result = dot_product_generic(A.data(), B.data(), N);
    }
  }
}
BENCHMARK_TEMPLATE(dot_product_benchmark, array<value_t, 1>, 0)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x + 1) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(dot_product_benchmark, array<value_t, 1>, 1)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x + 1) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(dot_product_benchmark, array<value_t, 1>, 2)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x + 1) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);
//BENCHMARK_TEMPLATE(dot_product_benchmark, array<value_t, 1>, 3)
//   ->Apply([](benchmark::internal::Benchmark *b) {
//     for (int x = Nmin; x < Nmax; x = 2 * x + 1) { b->Args({x}); }
//   })
//   ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(dot_product_benchmark, array<value_t, 1>, 4)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x + 1) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);