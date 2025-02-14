#pragma once
#include "./type.hpp"
#include "../functions_forward.hpp"

#include <cstdlib>
#include <complex>

namespace nda::simd {

  // Absolute value functions:
  template <>
  inline simd_i1 abs(const simd_i1 &x) {
    return simd_i1(std::abs(x));
  }

  template <>
  inline simd_l1 abs(const simd_l1 &x) {
    return simd_l1(std::abs(x));
  }

  template <>
  inline simd_f1 abs(const simd_f1 &x) {
    return simd_f1(std::abs(x));
  }

  template <>
  inline simd_d1 abs(const simd_d1 &x) {
    return simd_d1(std::abs(x));
  }

  // Conj functions
  template <>
  inline simd_i1 conj(const simd_i1 &x) {
    return simd_i1(x);
  }

  template <>
  inline simd_f1 conj(const simd_f1 &x) {
    return simd_f1(x);
  }

  template <>
  inline simd_l1 conj(const simd_l1 &x) {
    return simd_l1(x);
  }

  template <>
  inline simd_d1 conj(const simd_d1 &x) {
    return simd_d1(x);
  }

  template <>
  inline simd_cf1 conj(const simd_cf1 &x) {
    return simd_cf1(std::conj(static_cast<std::complex<float>>(x)));
  }
  template <>
  inline simd_cd1_d conj(const simd_cd1_d &x) {
    return simd_cd1_d(std::conj(static_cast<std::complex<double>>(x)));
  }

  //Sqrt functions

  template <>
  inline simd_f1 sqrt(const simd_f1 &x) {
    return simd_f1(std::sqrt(x));
  }

  template <>
  inline simd_d1 sqrt(const simd_d1 &x) {
    return simd_d1(std::sqrt(x));
  }

} // namespace nda::simd
