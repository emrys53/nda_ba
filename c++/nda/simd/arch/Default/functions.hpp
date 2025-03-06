#pragma once
#include "./type.hpp"
#include "../functions_forward.hpp"

#include <cstdlib>
#include <complex>
#include <algorithm>

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

  // Min functions
  template <>
  inline simd_i1 min(const simd_i1 &x, const simd_i1 &y) {
    return simd_i1(std::min(x, y));
  }

  template <>
  inline simd_l1 min(const simd_l1 &x, const simd_l1 &y) {
    return simd_l1(std::min(x, y));
  }

  template <>
  inline simd_f1 min(const simd_f1 &x, const simd_f1 &y) {
    return simd_f1(std::min(x, y));
  }

  template <>
  inline simd_d1 min(const simd_d1 &x, const simd_d1 &y) {
    return simd_d1(std::min(x, y));
  }

  //Max functions
  template <>
  inline simd_i1 max(const simd_i1 &x, const simd_i1 &y) {
    return simd_i1(std::max(x, y));
  }

  template <>
  inline simd_l1 max(const simd_l1 &x, const simd_l1 &y) {
    return simd_l1(std::max(x, y));
  }

  template <>
  inline simd_f1 max(const simd_f1 &x, const simd_f1 &y) {
    return simd_f1(std::max(x, y));
  }

  template <>
  inline simd_d1 max(const simd_d1 &x, const simd_d1 &y) {
    return simd_d1(std::max(x, y));
  }

  // Reduce Min
  template <>
  inline simd_i1::value_t reduce_min(const simd_i1 &x) {
    return x;
  }

  template <>
  inline simd_l1::value_t reduce_min(const simd_l1 &x) {
    return x;
  }

  template <>
  inline simd_f1::value_t reduce_min(const simd_f1 &x) {
    return x;
  }

  template <>
  inline simd_d1::value_t reduce_min(const simd_d1 &x) {
    return x;
  }

  // Reduce Max
  template <>
  inline simd_i1::value_t reduce_max(const simd_i1 &x) {
    return x;
  }

  template <>
  inline simd_l1::value_t reduce_max(const simd_l1 &x) {
    return x;
  }

  template <>
  inline simd_f1::value_t reduce_max(const simd_f1 &x) {
    return x;
  }

  template <>
  inline simd_d1::value_t reduce_max(const simd_d1 &x) {
    return x;
  }

  //Reduce_sum
  template <>
  inline simd_i1::value_t reduce_sum(const simd_i1 &x) {
    return x;
  }

  template <>
  inline simd_l1::value_t reduce_sum(const simd_l1 &x) {
    return x;
  }

  template <>
  inline simd_f1::value_t reduce_sum(const simd_f1 &x) {
    return x;
  }

  template <>
  inline simd_d1::value_t reduce_sum(const simd_d1 &x) {
    return x;
  }

  template <>
  inline simd_cf1::value_t reduce_sum(const simd_cf1 &x) {
    return x;
  }

  template <>
  inline simd_cd1_d::value_t reduce_sum(const simd_cd1_d &x) {
    return x;
  }

  //Reduce_mul
  template <>
  inline simd_i1::value_t reduce_mul(const simd_i1 &x) {
    return x;
  }

  template <>
  inline simd_l1::value_t reduce_mul(const simd_l1 &x) {
    return x;
  }

  template <>
  inline simd_f1::value_t reduce_mul(const simd_f1 &x) {
    return x;
  }

  template <>
  inline simd_d1::value_t reduce_mul(const simd_d1 &x) {
    return x;
  }

  template <>
  inline simd_cf1::value_t reduce_mul(const simd_cf1 &x) {
    return x;
  }

  template <>
  inline simd_cd1_d::value_t reduce_mul(const simd_cd1_d &x) {
    return x;
  }

  // FMA ADD
  template <>
  inline simd_f1 fma_add(const simd_f1 &x, const simd_f1 &y, const simd_f1 &z) {
    return x * y + z;
  }

  template <>
  inline simd_d1 fma_add(const simd_d1 &x, const simd_d1 &y, const simd_d1 &z) {
    return x * y + z;
  }

  template <>
  inline simd_cf1 fma_add(const simd_cf1 &x, const simd_cf1 &y, const simd_cf1 &z) {
    return x * y + z;
  }

  template <>
  inline simd_cd1_d fma_add(const simd_cd1_d &x, const simd_cd1_d &y, const simd_cd1_d &z) {
    return x * y + z;
  }

  //FMA_SUB
  template <>
  inline simd_f1 fma_sub(const simd_f1 &x, const simd_f1 &y, const simd_f1 &z) {
    return x * y - z;
  }

  template <>
  inline simd_d1 fma_sub(const simd_d1 &x, const simd_d1 &y, const simd_d1 &z) {
    return x * y - z;
  }

  template <>
  inline simd_cf1 fma_sub(const simd_cf1 &x, const simd_cf1 &y, const simd_cf1 &z) {
    return x * y - z;
  }

  template <>
  inline simd_cd1_d fma_sub(const simd_cd1_d &x, const simd_cd1_d &y, const simd_cd1_d &z) {
    return x * y - z;
  }

  //FMA_NADD
  template <>
  inline simd_f1 fma_nadd(const simd_f1 &x, const simd_f1 &y, const simd_f1 &z) {
    return z - (x * y);
  }

  template <>
  inline simd_d1 fma_nadd(const simd_d1 &x, const simd_d1 &y, const simd_d1 &z) {
    return z - (x * y);
  }

  template <>
  inline simd_cf1 fma_nadd(const simd_cf1 &x, const simd_cf1 &y, const simd_cf1 &z) {
    return z - (x * y);
  }

  template <>
  inline simd_cd1_d fma_nadd(const simd_cd1_d &x, const simd_cd1_d &y, const simd_cd1_d &z) {
    return z - (x * y);
  }

  //FMA_NSUB
  template <>
  inline simd_f1 fma_nsub(const simd_f1 &x, const simd_f1 &y, const simd_f1 &z) {
    return -(x * y + z);
  }

  template <>
  inline simd_d1 fma_nsub(const simd_d1 &x, const simd_d1 &y, const simd_d1 &z) {
    return -(x * y + z);
  }

  template <>
  inline simd_cf1 fma_nsub(const simd_cf1 &x, const simd_cf1 &y, const simd_cf1 &z) {
    return -(x * y + z);
  }

  template <>
  inline simd_cd1_d fma_nsub(const simd_cd1_d &x, const simd_cd1_d &y, const simd_cd1_d &z) {
    return -(x * y + z);
  }
  // For int fma simulation
  // FMA ADD
  template <>
  inline simd_i1 fma_add(const simd_i1 &x, const simd_i1 &y, const simd_i1 &z) {
    return x * y + z;
  }

  template <>
  inline simd_l1 fma_add(const simd_l1 &x, const simd_l1 &y, const simd_l1 &z) {
    return x * y + z;
  }

  //FMA_SUB
  template <>
  inline simd_i1 fma_sub(const simd_i1 &x, const simd_i1 &y, const simd_i1 &z) {
    return x * y - z;
  }

  template <>
  inline simd_l1 fma_sub(const simd_l1 &x, const simd_l1 &y, const simd_l1 &z) {
    return x * y - z;
  }

  //FMA_NADD
  template <>
  inline simd_i1 fma_nadd(const simd_i1 &x, const simd_i1 &y, const simd_i1 &z) {
    return z - (x * y);
  }

  template <>
  inline simd_l1 fma_nadd(const simd_l1 &x, const simd_l1 &y, const simd_l1 &z) {
    return z - (x * y);
  }

  //FMA_NSUB
  template <>
  inline simd_i1 fma_nsub(const simd_i1 &x, const simd_i1 &y, const simd_i1 &z) {
    return -(x * y + z);
  }

  template <>
  inline simd_l1 fma_nsub(const simd_l1 &x, const simd_l1 &y, const simd_l1 &z) {
    return -(x * y + z);
  }

  //Gather functions
  template <>
  inline simd_i1 gather(const simd_i1::value_t *from, [[maybe_unused]]const long stride) {
    return simd_i1(from);
  }

  template <>
  inline simd_l1 gather(const simd_l1::value_t *from, [[maybe_unused]]const long stride) {
    return simd_l1(from);
  }

  template <>
  inline simd_f1 gather(const simd_f1::value_t *from,[[maybe_unused]] const long stride) {
    return simd_f1(from);
  }

  template <>
  inline simd_d1 gather(const simd_d1::value_t *from, [[maybe_unused]] const long stride) {
    return simd_d1(from);
  }

  template <>
  inline simd_cf1 gather(const simd_cf1::value_t *from, [[maybe_unused]] const long stride) {
    return simd_cf1(from);
  }

  template <>
  inline simd_cd1_d gather(const simd_cd1_d::value_t *from,[[maybe_unused]] const long stride) {
    return simd_cd1_d(from);
  }

} // namespace nda::simd
