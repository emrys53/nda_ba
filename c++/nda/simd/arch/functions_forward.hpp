#pragma once

namespace nda::simd {
  template <typename T>
  T abs(const T &);

  template <typename T>
  T sqrt(const T &);

  template <typename T>
  T conj(const T &);

  template<typename T>
  T logic_and(const T&, const T&);

  template<typename T>
  T logic_xor(const T&, const T&);

  template<typename T>
  T logic_or(const T&, const T&);

  template<typename T>
  T min(const T&, const T&);

  template<typename T>
  T max(const T&, const T&);

  template<typename T>
  typename T::value_t reduce_sum(const T&);

  template<typename T>
  typename T::value_t reduce_mul(const T&);

  template<typename T>
  typename T::value_t reduce_min(const T&);

  template<typename T>
  typename T::value_t reduce_max(const T&);

} // namespace nda::simd
