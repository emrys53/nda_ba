#pragma once
#include "./concepts.hpp"
#include "./mem/alignment.hpp"

#include <iterator>
#include <utility>
#include <tuple>
#include <algorithm>

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

      size_t last_index;

      packed_iterator(const pointer data_ptr, const size_t idx, const size_t last_idx) : data(data_ptr), index(idx), last_index(last_idx) {}

      simd_block<value_type, aligned_and_padded> operator*() const noexcept {
        if constexpr (aligned_and_padded) {
          return {simd_t(data + index), data + index, std::min(simd_t::size(), last_index - index)};
        } else {
          simd_t tmp;
          tmp.load_unaligned(data + index);
          return {tmp, data + index, std::min(simd_t::size(), last_index - index)};
        }
      }

      pointer operator->() const noexcept { return data + index; }

      packed_iterator &operator++() noexcept {
        index += simd_t::size();
        return *this;
      }

      packed_iterator operator++(int) noexcept {
        packed_iterator tmp = *this;
        index += simd_t::size();
        return tmp;
      }

      packed_iterator &operator--() noexcept {
        index -= simd_t::size();
        return *this;
      }

      packed_iterator operator--(int) noexcept {
        packed_iterator tmp = *this;
        index -= simd_t::size();
        return tmp;
      }

      packed_iterator &operator+=(std::ptrdiff_t n) noexcept {
        index += n * simd_t::size();
        return *this;
      }

      packed_iterator &operator-=(std::ptrdiff_t n) noexcept {
        index -= n * simd_t::size();
        return *this;
      }

      packed_iterator operator+(std::ptrdiff_t n) const noexcept { return packed_iterator(data, index + n * simd_t::size()); }
      packed_iterator operator-(std::ptrdiff_t n) const noexcept { return packed_iterator(data, index - n * simd_t::size()); }
      std::ptrdiff_t operator-(const packed_iterator &other) const noexcept { return (index - other.index) / simd_t::size(); }
      simd_block<value_type, aligned_and_padded> operator[](std::ptrdiff_t n) const noexcept {
        return {simd_t(data + index + n * simd_t::size()), data + index + n * simd_t::size(),
                std::min(simd_t::size(), last_index - index + n * simd_t::size())};
      }

      bool operator==(const packed_iterator &other) const noexcept { return index == other.index; }
      bool operator!=(const packed_iterator &other) const noexcept { return !(*this == other); }
      bool operator<(const packed_iterator &other) const noexcept { return index < other.index; }
      bool operator>(const packed_iterator &other) const noexcept { return index > other.index; }
      bool operator<=(const packed_iterator &other) const noexcept { return index <= other.index; }
      bool operator>=(const packed_iterator &other) const noexcept { return index >= other.index; }
    };

    packed_iterator begin() const noexcept { return packed_iterator(array.data(), 0, array.indexmap().capacity()); }
    packed_iterator end() const noexcept {
      return packed_iterator(array.data(), mem::next_multiple(array.indexmap().capacity(), native_simd<std::remove_cvref_t<get_value_t<A>>>::size()),
                             array.indexmap().capacity());
    }
  };
} // namespace nda
