#pragma once
#include "./concepts.hpp"

#include <iterator>
#include <utility>

namespace nda {
  template <MemoryArray A>
    requires Vectorizable<std::remove_cvref_t<get_value_t<A>>> and (get_layout_info<A>.prop == layout_prop_e::contiguous)
  struct packed {
    A &array;
    packed(A &a) : array(a) {}

    static constexpr bool aligned_and_padded = A::is_aligned and A::is_padded;

    struct packed_iterator {
      using iterator_category = std::random_access_iterator_tag;
      using value_type        = std::remove_cvref_t<get_value_t<A>>;
      using pointer           = value_type *;
      using reference         = value_type &;
      using simd_t            = native_simd<value_type>;

      pointer data;

      size_t index;

      size_t padding = array.indexmap().get_padding();

      packed_iterator(const pointer data_ptr, const std::size_t idx) : data(data_ptr), index(idx) {}

      std::pair<simd_t, pointer> operator*() const {
        // Valid field counts= let j be the fastest dimension: let p be the padding. simd_size
        // (index + simd_size) % lengths[j] (index = 0 lengths[j] = 10,  then valid_fields = 8;, index 8 lengtsah[j] = 10
        // ((lengths[j] - (index + simd_size + (lengths[j] / (index))^-1 * padding) % lengths[j]) % lengths[j])
        // index / (lengths[j] + padding) = row
        // index + simd_size - (lengths[j]) * (row+1) - padding*row // if to check negative
        // 8 + 8 - 10
        // 16 + 8 - 10 * 2 - 6
        if constexpr (aligned_and_padded) {
          return {simd_t(data + index), data + index};
        } else {
          simd_t tmp;
          tmp.load_unaligned(data + index);
          return {tmp, data + index};
        }
      }

      pointer operator->() const { return data + index; }

      packed_iterator &operator++() {
        index += simd_t::size();
        return *this;
      }

      packed_iterator operator++(int) {
        packed_iterator tmp = *this;
        index += simd_t::size();
        return tmp;
      }

      packed_iterator &operator--() {
        index -= simd_t::size();
        return *this;
      }

      packed_iterator operator--(int) {
        packed_iterator tmp = *this;
        index -= simd_t::size();
        return tmp;
      }

      packed_iterator &operator+=(std::ptrdiff_t n) {
        index += n * simd_t::size();
        return *this;
      }

      packed_iterator &operator-=(std::ptrdiff_t n) {
        index -= n * simd_t::size();
        return *this;
      }

      packed_iterator operator+(std::ptrdiff_t n) const { return packed_iterator(data, index + n * simd_t::size()); }
      packed_iterator operator-(std::ptrdiff_t n) const { return packed_iterator(data, index - n * simd_t::size()); }
      std::ptrdiff_t operator-(const packed_iterator &other) const { return (index - other.index) / simd_t::size(); }
      std::pair<simd_t, pointer> operator[](std::ptrdiff_t n) const {
        return {simd_t(data + index + n * simd_t::size()), data + index + n * simd_t::size()};
      }

      bool operator==(const packed_iterator &other) const { return index == other.index; }
      bool operator!=(const packed_iterator &other) const { return !(*this == other); }
      bool operator<(const packed_iterator &other) const { return index < other.index; }
      bool operator>(const packed_iterator &other) const { return index > other.index; }
      bool operator<=(const packed_iterator &other) const { return index <= other.index; }
      bool operator>=(const packed_iterator &other) const { return index >= other.index; }
    };

    packed_iterator begin() { return packed_iterator(array.data(), 0); }
    packed_iterator end() { return packed_iterator(array.data(), array.indexmap().capacity()); }
  };
} // namespace nda
