#pragma once

#include "type_forward.hpp"
#include "abi.hpp"
#ifdef __AVX512F__
#include "AVX512/type.hpp"
#endif
#ifdef __AVX__
#include "AVX/type.hpp"
#endif
#ifdef __SSE2__
#include "SSE/type.hpp"
#endif
#include "Default/type.hpp"

