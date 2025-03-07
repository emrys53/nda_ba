#pragma once

#include <array>

namespace nda::simd {
  template <typename T>
  T abs(const T &);

  template <typename T>
  T sqrt(const T &);

  template <typename T>
  T conj(const T &);

  template <typename T>
  T min(const T &, const T &);

  template <typename T>
  T max(const T &, const T &);

  template <typename T>
  typename T::value_t reduce_min(const T &);

  template <typename T>
  typename T::value_t reduce_max(const T &);

  template <typename T>
  typename T::value_t reduce_sum(const T &);

  template <typename T>
  typename T::value_t reduce_mul(const T &);

  template <typename T>
  T fma_add(const T &, const T &, const T &);

  template <typename T>
  T fma_sub(const T &, const T &, const T &);

  template <typename T>
  T fma_nadd(const T &, const T &, const T &);

  template <typename T>
  T fma_nsub(const T &, const T &, const T &);

  template <typename T>
  T gather(const typename T::value_t *, const long);

  template<typename T>
  void scatter(const T&, typename T::value_t*, const long);

} // namespace nda::simd
