#pragma once

namespace nda {
  enum class abi_tag {
    Default,
    SSE,
    AVX,
    AVX512,
    NEON
  };

}