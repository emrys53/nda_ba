#include "./bench_common.hpp"
#include <nda/blas.hpp>

using value_t   = float;
const long Nmin = 10;
const long Nmax = 1 << 9;

template <typename Matrix>
static void GEMM(benchmark::State &state) {
  long N       = state.range(0);
  const long M = N * 2;
  const long K = N / 2;
  Matrix A({N, N, N});
  Matrix B({N, N, N});
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        A(i, j, k) = 0.001 * j + 0.00001 * k + 0.0002 * i;
        B(i, j, k) = 0.002 * k + 0.00003 * i + 0.00004 * j;
      }
    }
  }
  static const long L = N;
  struct addd {
    __attribute__((optimize("no-tree-vectorize")))  value_t operator()(value_t x, value_t y) const { return x + y; }
    native_simd<value_t> load(const native_simd<value_t> x, native_simd<value_t> y) const {
      return x + y;
      ;
    };
  };
  struct mult {
    __attribute__((optimize("no-tree-vectorize"))) value_t operator()(value_t x, value_t y) const { return x * y; };
    native_simd<value_t> load(native_simd<value_t> x, native_simd<value_t> y) const { return (x * x * x + y) * (x + y); };
  };
  struct add {
    __attribute__((optimize("no-tree-vectorize"))) double operator()(value_t x, value_t y) const { return x + y; };
  };
  struct mul {
    __attribute__((optimize("no-tree-vectorize"))) value_t operator()(value_t x, value_t y) const { return (x * x * x + y) * (x + y); };
  };

  for (auto s : state) {
    if constexpr (not Matrix::is_aligned) {
      auto tmp1 = nda::map(addd{})(A, B);
      auto tmp2 = nda::map(mult{})(tmp1, B);
      auto tmp3 = nda::map(mult{})(tmp2, tmp1);
      auto tmp4 = nda::map(mult{})(tmp3, tmp2);
      auto tmp5 = nda::map(mult{})(tmp4, tmp3);
      auto tmp6 = nda::map(mult{})(tmp5, tmp4);
      volatile Matrix tmp7(tmp6);
      benchmark::DoNotOptimize(tmp7);
      benchmark::ClobberMemory();
    } else {

      auto tmp1 = nda::map(addd{})(A, B);
      auto tmp2 = nda::map(mult{})(tmp1, B);
      auto tmp3 = nda::map(mult{})(tmp2, tmp1);
      auto tmp4 = nda::map(mult{})(tmp3, tmp2);
      auto tmp5 = nda::map(mult{})(tmp4, tmp3);
      auto tmp6 = nda::map(mult{})(tmp5, tmp4);
      volatile Matrix tmp7(tmp6);
      // for (int i = 0; i < N; i++) {
      //   for (int j = 0; j < N; j++) {
      //     if (tmp7(i, j) != tmp77(i, j)) {
      //       std::cout << tmp7(i, j) << std::endl;
      //       std::cout << tmp77(i, j) << std::endl;
      //     }
      //   }
      // array_aligned<double, 3> s1 = nda::map(add{})(A, B);
      // for (int i = 0; i < N; ++i) {
      //   for (int j = 0; j < N; ++j) {
      //     for (int k = 0; k < N; ++k) {
      //       std::cout << s1(i, j, k) << " " << A(i, j, k) + B(i, j, k) << std::endl;
      //     }
      //   }
      // }
      benchmark::DoNotOptimize(tmp7);
      benchmark::ClobberMemory();
    }
  }

  auto NBytes                = M * K * sizeof(value_t);
  state.counters["bytesize"] = double(NBytes);
  state.counters["Size A"]   = long(M * N);
  state.counters["Size B"]   = long(N * K);
}
BENCHMARK_TEMPLATE(GEMM, array<value_t, 3, C_layout>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond); // NOLINT

BENCHMARK_TEMPLATE(GEMM, array_aligned<value_t, 3, C_layout>)->RangeMultiplier(2)->Range(Nmin, Nmax)->Unit(benchmark::kMicrosecond); // NOLINT
