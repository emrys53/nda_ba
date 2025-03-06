//#pragma once
//
//
//namespace nda::simd {
//  template <typename T, size_t Width, abi_tag ABI>
//  inline std::array<simd_type<T, Width, ABI>, Width> transpose(const std::array<simd_type<T, Width, ABI>, Width> &simd_block) {
//    if constexpr (Width == 1) {
//      return simd_block;
//    } else {
//      using dcomplex = std::complex<double>;
//      if constexpr(Width == 2) {
//        static_assert(std::is_same_v<std::remove_cvref_t<T>, dcomplex> and Width == 2, "Other things are not supported yet");
//        using simd_t = simd_type<T, Width, ABI>;
//        simd_t a     = simd_block[0];
//        simd_t b     = simd_block[1];
//        simd_t r0    = simd_t(_mm256_permute2f128_pd(a, b, 0b00100000));
//        simd_t r1    = simd_t(_mm256_permute2f128_pd(a, b, 0b00110001));
//        return {r0, r1};
//      }
//      else {
//        static_assert(std::is_same_v<std::remove_cvref_t<T>, dcomplex> and Width == 4, "Other things are not supported yet"); // AVX512F
//
//      }
//      // a0 a1 a2 a3
//      // b0 b1 b2 b3
//      // Return:
//      // a0 a1 b0 b1
//      // a2 a3 b2 b3
//
//      // a0 a1 a2 a3
//      // b0 b1 b2 b3
//      // c0 c1 c2 c3
//      // d0 d1 d2 d3
//      // Return:
//      // a0 b0 c0 d0
//      // a1 b1 c1 d1
//      // a2 b2 c2 d2
//      // a3 b3 c3 d3
//    }
//  }
//
//} // namespace nda::simd
