#pragma once

#include "simd_type.hpp"

namespace nda {
    /*
     * Width in bytes.
     */
    template<typename T, int WIDTH = SIMD_WIDTH>
    struct simd {
      static constexpr int number_of_elements = WIDTH / sizeof(T) > 0 ? WIDTH / sizeof(T) : 1; // Special case where double/ll is used but SIMD_WIDTH =4
      typename simd_type<T, number_of_elements>::intrinsic_type value;
    };
} // namespace nda