#include "./bench_common.hpp"
#include <nda/simd/simd.hpp>
#include <nda/simd/mock_simd.hpp>
// Scalar-only functor (no SIMD support)
struct log_scalar {
  float operator()(float x) const { return std::log(x); }
};

// Emulated SIMD functor using mock_simd
struct log_emulated : nda::simd::mock_simd<log_emulated, float> {
  float operator()(float x) const { return std::log(x); }
};

using value_t   = float;
const long Nmin = 16;
const long Nmax = 1 << 20;

template <typename Array, int Option>
static void mock_simd_benchmark(benchmark::State &state) {
  long N = state.range(0);
  auto A = Array::rand({N});
  auto B = Array::rand({N});
  auto C = Array::rand({N});
  for (auto _ : state) {

    if constexpr (Option == 0) {
      auto e1            = A + B;
      auto e2            = e1 + A * B * (C + e1);
      auto e3            = e2 * e1;
      auto e4            = e3 * e2;
      auto e5            = e4 + e3;
      auto e6            = e5 + e4;
      auto e7            = e6 + e5;
      auto e8            = e7 + e6;
      auto e9            = e8 + e7;
      auto e10           = nda::map(log_scalar{})(e9);
      volatile value_t f = nda::sum(e10);

    } else if constexpr (Option == 1) {
      auto e1            = A + B;
      auto e2            = e1 + A * B * (C + e1);
      auto e3            = e2 * e1;
      auto e4            = e3 * e2;
      auto e5            = e4 + e3;
      auto e6            = e5 + e4;
      auto e7            = e6 + e5;
      auto e8            = e7 + e6;
      auto e9            = e8 + e7;
      auto e10           = nda::map(log_emulated{})(e9);
      volatile value_t f = nda::sum(e10);
    }
  }
}

BENCHMARK_TEMPLATE(mock_simd_benchmark, array<value_t, 1>, 0)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(mock_simd_benchmark, array<value_t, 1>, 1)
   ->Apply([](benchmark::internal::Benchmark *b) {
     for (int x = Nmin; x < Nmax; x = 2 * x) { b->Args({x}); }
   })
   ->Unit(benchmark::kMicrosecond);
