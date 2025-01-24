// Copyright (c) 2022-2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Miguel Morales, Nils Wentzell

/**
 * @file
 * @brief Provides functions to calculate alignments
 */

#pragma once
#include "../simd/simd.hpp"
#include "../concepts.hpp"

namespace nda::mem {

  /**
   * @addtogroup mem_utils
   * @{
   */

  /**
   * @brief Round up an integral value to the next multiple of a given factor
   *
   * @tparam T integral type
   * @tparam U integral type
   * @param value Value to be rounded up.
   * @param factor Factor of the next multiple.
   */
  template <typename T, typename U>
  constexpr auto next_multiple(T value, U factor) {
     return ( value + ( factor - ( value % factor ) ) % factor );
  }

  template <typename T>
  struct type_alignment_info {
    static constexpr std::size_t alignment = 0;
    static constexpr std::size_t width     = 0;
  };

  template <Scalar T>
  struct type_alignment_info<T> {
    // Alignment in bytes.
    static constexpr std::size_t alignment = simd<T>::alignment();
    // Maximum of elements I can fit into simd register of type T.
    static constexpr std::size_t width     = alignment / sizeof(T);
  };

} // namespace nda::mem
