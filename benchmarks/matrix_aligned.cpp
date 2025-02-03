#include "./bench_common.hpp"
#include <nda/blas.hpp>

using value_t   = double;
const long Nmin = 11;
// const long Nmax = 1 << 10;

template <typename Matrix>
static void GEMM(benchmark::State &state) {
  long N       = state.range(0);
  const long M = N * 2;
  const long K = N / 2;
  auto A       = Matrix::rand({M, N});
  auto B       = Matrix::rand({N, K});
  auto C       = Matrix::zeros({M, K});

  for (auto s : state) { nda::blas::gemm(1.0, A, B, 0.0, C); }

  auto NBytes                = M * K * sizeof(value_t);
  state.counters["bytesize"] = double(NBytes);
  state.counters["Size A"]   = long(M * N);
  state.counters["Size B"]   = long(N * K);
}
BENCHMARK_TEMPLATE(GEMM, matrix<value_t, F_layout, heap<>>)
   ->Args({9})
   ->Args({17})
   ->Args({33})
   ->Args({65})
   ->Args({129})
   ->Args({257})
   ->Args({513})
   ->Args({1025})
   ->Args({2049})
   ->Args({4097})
   ->Args({8193})
   ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(GEMM, matrix_aligned<value_t, F_layout, heap_aligned<>>)
   ->Args({9})
   ->Args({17})
   ->Args({33})
   ->Args({65})
   ->Args({129})
   ->Args({257})
   ->Args({513})
   ->Args({1025})
   ->Args({2049})
   ->Args({4097})
   ->Args({8193})
   ->Unit(benchmark::kMicrosecond);