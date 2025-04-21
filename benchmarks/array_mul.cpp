#include "./bench_common.hpp"
#include <nda/nda.hpp>

using value_t   = float;
const long Nmin = 1 << 10;
const long Nmax = 1 << 20;
using namespace nda;

template <Vectorizable T>
struct mul_scalar {
  using simd_t = native_simd<T>;
  T operator()(const T &a, const T &b) const { return a * b; }
};

template <Vectorizable T>
struct mul_simd {
  using simd_t = native_simd<T>;
  T operator()(const T &a, const T &b) const { return a * b; }
  simd_t load(const simd_t &a, const simd_t &b) const { return a * b; }
};



template <typename Array, int Option>
static void nda_algorithms_benchmark(benchmark::State &state) {
  long N        = state.range(0);
  auto A        = Array::rand({N});
  auto B        = Array::rand({N});
  using array_t = get_value_t<decltype(A)>;
  for (auto _ : state) {
    if constexpr (Option == 0) {
      auto ex           = nda::map(mul_scalar<array_t>{})(A, B);
      volatile auto res = nda::make_regular(ex);
    } else if constexpr (Option == 1) {
      auto ex           = nda::map(mul_simd<array_t>{})(A, B);
      volatile auto res = nda::make_regular(ex);
    }
  }
}

BENCHMARK_TEMPLATE(nda_algorithms_benchmark, array<value_t, 1>, 0)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(nda_algorithms_benchmark, array<value_t, 1>, 1)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(nda_algorithms_benchmark, array<std::complex<value_t>, 1>, 0)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(nda_algorithms_benchmark, array<std::complex<value_t>, 1>, 1)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);

