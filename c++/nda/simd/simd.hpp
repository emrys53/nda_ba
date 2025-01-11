#include "simd_type.hpp"
namespace nda {
    template<typename T, size_t Width = 4 * SIMD_WIDTH>
    struct simd {
    public:
        typename simd_type<T,Width / sizeof(T)>::intrinsic_type value;
    };
}