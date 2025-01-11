#pragma once
#include "simd_type.hpp"
namespace nda {
    template<typename T, int WIDTH = 4 * SIMD_WIDTH>
    struct simd {
    public:
        typename simd_type<T,WIDTH / sizeof(T)>::intrinsic_type value;
    };
}