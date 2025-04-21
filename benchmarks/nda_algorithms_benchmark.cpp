#include "./bench_common.hpp"
#include <nda/nda.hpp>

using value_t   = float;
const long Nmin = 1 << 18;
const long Nmax = 1 << 20;
using namespace nda;

template <typename Array, int Option, int Function>
static void nda_algorithms_benchmark(benchmark::State &state) {
  long N        = state.range(0);
  auto A        = Array::rand({N});
  using array_t = get_value_t<decltype(A)>;
  for (auto _ : state) {
    if constexpr (Option == 0) {
      if constexpr (Function == 0) {
        volatile array_t result = nda::sum2(A);
      } else if constexpr (Function == 1) {
        volatile array_t result = nda::max_element2(A);
      }
    } else if constexpr (Option == 1) {
      if constexpr (Function == 0) {
        volatile array_t result = nda::sum(A);
      } else if constexpr (Function == 1) {
        volatile array_t result = nda::max_element(A);
      }
    }
  }
}

BENCHMARK_TEMPLATE(nda_algorithms_benchmark, array<value_t, 1>, 0, 0)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(nda_algorithms_benchmark, array<value_t, 1>, 1, 0)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(nda_algorithms_benchmark, array<value_t, 1>, 0, 1)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(nda_algorithms_benchmark, array<value_t, 1>, 1, 1)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);


