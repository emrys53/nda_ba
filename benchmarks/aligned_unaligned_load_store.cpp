#include "./bench_common.hpp"
#include <nda/simd/simd.hpp>

template <nda::Vectorizable T> // Vectorizable is a concept where it checks whether native_simd<T>() is constructible
void aligned_store_load(const T *a, const T *b, T *c, size_t size) {
  using namespace nda;
  using simd_t           = native_simd<T>;
  constexpr size_t width = simd_t::size();
  simd_t acc(simd_zero_initialize_t);
  size_t i = 0;
  for (; i + width <= size; i += width) {
    simd_t va(&a[i], simd_aligned_memory_t);
    simd_t vb(&b[i], simd_aligned_memory_t);
    simd_t vc = va + vb;
    vc.store(&c[i]);
  }
}
template <nda::Vectorizable T> // Vectorizable is a concept where it checks whether native_simd<T>() is constructible
void unaligned_store_load(const T *a, const T *b, T *c, size_t size) {
  using namespace nda;
  using simd_t           = native_simd<T>;
  constexpr size_t width = simd_t::size();
  simd_t acc(simd_zero_initialize_t);
  size_t i = 0;
  for (; i + width <= size; i += width) {
    simd_t va(&a[i], simd_unaligned_memory_t);
    simd_t vb(&b[i], simd_unaligned_memory_t);
    simd_t vc = va + vb;
    vc.store_unaligned(&c[i]);
  }
}

using value_t   = float;
const long Nmin = 16;
const long Nmax = 1 << 20;

template <typename Array, int Option>
static void load_store_benchmark(benchmark::State &state) {
  long N = state.range(0);
  auto A = Array::rand({N});
  auto B = Array::rand({N});
  auto C = Array::rand({N});

  for (auto s : state) {
    if constexpr (Option == 0) {
      aligned_store_load(A.data(), B.data(), C.data(), N);
    } else if constexpr (Option == 1) {
      unaligned_store_load(A.data(), B.data(), C.data(), N);
    }
  }
}
BENCHMARK_TEMPLATE(load_store_benchmark, array_aligned<value_t, 1>, 0)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(load_store_benchmark, array<value_t, 1>, 1)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);
