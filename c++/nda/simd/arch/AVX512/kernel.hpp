#pragma once
#ifdef __AVX512F__
#include "./type.hpp"
#include "./functions.hpp"
#include "../kernel_forward.hpp"
namespace nda::simd {
  template <>
  inline void kernel_transpose(std::array<simd_i16, 16> &simd_block) {
    // https://gist.github.com/nihui/37d98b705a6a28911d77c502282b4748
    __m512i _tmp0 = _mm512_unpacklo_epi32(simd_block[0], simd_block[1]);
    __m512i _tmp1 = _mm512_unpackhi_epi32(simd_block[0], simd_block[1]);
    __m512i _tmp2 = _mm512_unpacklo_epi32(simd_block[2], simd_block[3]);
    __m512i _tmp3 = _mm512_unpackhi_epi32(simd_block[2], simd_block[3]);
    __m512i _tmp4 = _mm512_unpacklo_epi32(simd_block[4], simd_block[5]);
    __m512i _tmp5 = _mm512_unpackhi_epi32(simd_block[4], simd_block[5]);
    __m512i _tmp6 = _mm512_unpacklo_epi32(simd_block[6], simd_block[7]);
    __m512i _tmp7 = _mm512_unpackhi_epi32(simd_block[6], simd_block[7]);
    __m512i _tmp8 = _mm512_unpacklo_epi32(simd_block[8], simd_block[9]);
    __m512i _tmp9 = _mm512_unpackhi_epi32(simd_block[8], simd_block[9]);
    __m512i _tmpa = _mm512_unpacklo_epi32(simd_block[10], simd_block[11]);
    __m512i _tmpb = _mm512_unpackhi_epi32(simd_block[10], simd_block[11]);
    __m512i _tmpc = _mm512_unpacklo_epi32(simd_block[12], simd_block[13]);
    __m512i _tmpd = _mm512_unpackhi_epi32(simd_block[12], simd_block[13]);
    __m512i _tmpe = _mm512_unpacklo_epi32(simd_block[14], simd_block[15]);
    __m512i _tmpf = _mm512_unpackhi_epi32(simd_block[14], simd_block[15]);

    __m512 _tmpg = _mm512_shuffle_epi32(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmph = _mm512_shuffle_epi32(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpi = _mm512_shuffle_epi32(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpj = _mm512_shuffle_epi32(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpk = _mm512_shuffle_epi32(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpl = _mm512_shuffle_epi32(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpm = _mm512_shuffle_epi32(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpn = _mm512_shuffle_epi32(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpo = _mm512_shuffle_epi32(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpp = _mm512_shuffle_epi32(_tmp8, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpq = _mm512_shuffle_epi32(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpr = _mm512_shuffle_epi32(_tmp9, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmps = _mm512_shuffle_epi32(_tmpc, _tmpe, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpt = _mm512_shuffle_epi32(_tmpc, _tmpe, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpu = _mm512_shuffle_epi32(_tmpd, _tmpf, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpv = _mm512_shuffle_epi32(_tmpd, _tmpf, _MM_SHUFFLE(3, 2, 3, 2));

    _tmp0 = _mm512_shuffle_i32x4(_tmpg, _tmpk, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp1 = _mm512_shuffle_i32x4(_tmpo, _tmps, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp2 = _mm512_shuffle_i32x4(_tmph, _tmpl, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp3 = _mm512_shuffle_i32x4(_tmpp, _tmpt, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp4 = _mm512_shuffle_i32x4(_tmpi, _tmpm, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp5 = _mm512_shuffle_i32x4(_tmpq, _tmpu, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp6 = _mm512_shuffle_i32x4(_tmpj, _tmpn, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp7 = _mm512_shuffle_i32x4(_tmpr, _tmpv, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp8 = _mm512_shuffle_i32x4(_tmpg, _tmpk, _MM_SHUFFLE(3, 1, 3, 1));
    _tmp9 = _mm512_shuffle_i32x4(_tmpo, _tmps, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpa = _mm512_shuffle_i32x4(_tmph, _tmpl, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpb = _mm512_shuffle_i32x4(_tmpp, _tmpt, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpc = _mm512_shuffle_i32x4(_tmpi, _tmpm, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpd = _mm512_shuffle_i32x4(_tmpq, _tmpu, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpe = _mm512_shuffle_i32x4(_tmpj, _tmpn, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpf = _mm512_shuffle_i32x4(_tmpr, _tmpv, _MM_SHUFFLE(3, 1, 3, 1));

    simd_block[0]  = simd_i16(_mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[1]  = simd_i16(_mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[2]  = simd_i16(_mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[3]  = simd_i16(_mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[4]  = simd_i16(_mm512_shuffle_i32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[5]  = simd_i16(_mm512_shuffle_i32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[6]  = simd_i16(_mm512_shuffle_i32x4(_tmpc, _tmpd, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[7]  = simd_i16(_mm512_shuffle_i32x4(_tmpe, _tmpf, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[8]  = simd_i16(_mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[9]  = simd_i16(_mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[10] = simd_i16(_mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[11] = simd_i16(_mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[12] = simd_i16(_mm512_shuffle_i32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[13] = simd_i16(_mm512_shuffle_i32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[14] = simd_i16(_mm512_shuffle_i32x4(_tmpc, _tmpd, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[15] = simd_i16(_mm512_shuffle_i32x4(_tmpe, _tmpf, _MM_SHUFFLE(3, 1, 3, 1)));
  }

  template <>
  inline void kernel_transpose(std::array<simd_l8, 8> &simd_block) {
    __m512i a0b0a2b2a4b4a6b6 = _mm512_unpacklo_epi64(simd_block[0], simd_block[1]);
    __m512i a1b1a3b3a5b5a7b7 = _mm512_unpackhi_epi64(simd_block[0], simd_block[1]);
    __m512i c0d0c2d2c4d4c6d6 = _mm512_unpacklo_epi64(simd_block[2], simd_block[3]);
    __m512i c1d1c3d3c5d5c7d7 = _mm512_unpackhi_epi64(simd_block[2], simd_block[3]);
    __m512i e0f0e2f2e4f4e6f6 = _mm512_unpacklo_epi64(simd_block[4], simd_block[5]);
    __m512i e1f1e3f3e5f5e5f7 = _mm512_unpackhi_epi64(simd_block[4], simd_block[5]);
    __m512i g0h0g2h2g4h4g6h6 = _mm512_unpacklo_epi64(simd_block[6], simd_block[7]);
    __m512i g1h1g3h3g5h5g7h7 = _mm512_unpackhi_epi64(simd_block[6], simd_block[7]);

    __m512i a2b2a0b0a6b6a4b4 = _mm512_shuffle_i64x2(a0b0a2b2a4b4a6b6, a0b0a2b2a4b4a6b6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512i a3b3a1b1a7b7a5b5 = _mm512_shuffle_i64x2(a1b1a3b3a5b5a7b7, a1b1a3b3a5b5a7b7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512i c2d2c0d0c6d6c4d4 = _mm512_shuffle_i64x2(c0d0c2d2c4d4c6d6, c0d0c2d2c4d4c6d6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512i c3d3c1d1c7d7c5d5 = _mm512_shuffle_i64x2(c1d1c3d3c5d5c7d7, c1d1c3d3c5d5c7d7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512i e2f2e0f0e6f6e4f4 = _mm512_shuffle_i64x2(e0f0e2f2e4f4e6f6, e0f0e2f2e4f4e6f6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512i e3f3e1f1e7f7e5f5 = _mm512_shuffle_i64x2(e1f1e3f3e5f5e5f7, e1f1e3f3e5f5e5f7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512i g2h2g0h0g6h6g4h4 = _mm512_shuffle_i64x2(g0h0g2h2g4h4g6h6, g0h0g2h2g4h4g6h6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512i g3h3g1h1g7h7g5h5 = _mm512_shuffle_i64x2(g1h1g3h3g5h5g7h7, g1h1g3h3g5h5g7h7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));

    __m512i a2b2c2d2a6b6c6d6 = _mm512_mask_blend_epi64(0b11001100, a2b2a0b0a6b6a4b4, c0d0c2d2c4d4c6d6);
    __m512i a3b3c3d3a7b7c7d7 = _mm512_mask_blend_epi64(0b11001100, a3b3a1b1a7b7a5b5, c1d1c3d3c5d5c7d7);
    __m512i a0b0c0d0a4b4c4d4 = _mm512_mask_blend_epi64(0b11001100, a0b0a2b2a4b4a6b6, c2d2c0d0c6d6c4d4);
    __m512i a1b1c1d1a5b5c5d5 = _mm512_mask_blend_epi64(0b11001100, a1b1a3b3a5b5a7b7, c3d3c1d1c7d7c5d5);
    __m512i e2f2g2h2e6f6g6h6 = _mm512_mask_blend_epi64(0b11001100, e2f2e0f0e6f6e4f4, g0h0g2h2g4h4g6h6);
    __m512i e3f3g3h3e7f7g7h7 = _mm512_mask_blend_epi64(0b11001100, e3f3e1f1e7f7e5f5, g1h1g3h3g5h5g7h7);
    __m512i e0f0g0h0e4f4g4h4 = _mm512_mask_blend_epi64(0b11001100, e0f0e2f2e4f4e6f6, g2h2g0h0g6h6g4h4);
    __m512i e1f1g1h1e5f5g5h5 = _mm512_mask_blend_epi64(0b11001100, e1f1e3f3e5f5e5f7, g3h3g1h1g7h7g5h5);

    __m512i e0f0g0h0e0f0g0h0 = _mm512_shuffle_i64x2(e0f0g0h0e4f4g4h4, e0f0g0h0e4f4g4h4, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512i e1f1g1h1e1f1g1h1 = _mm512_shuffle_i64x2(e1f1g1h1e5f5g5h5, e1f1g1h1e5f5g5h5, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512i e2f2g2h2e2f2g2h2 = _mm512_shuffle_i64x2(e2f2g2h2e6f6g6h6, e2f2g2h2e6f6g6h6, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512i e3f3g3h3e3f3g3h3 = _mm512_shuffle_i64x2(e3f3g3h3e7f7g7h7, e3f3g3h3e7f7g7h7, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512i a4b4c4d4a4b4c4d4 = _mm512_shuffle_i64x2(a0b0c0d0a4b4c4d4, a0b0c0d0a4b4c4d4, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512i a5b5c5d5a5b5c5d5 = _mm512_shuffle_i64x2(a1b1c1d1a5b5c5d5, a1b1c1d1a5b5c5d5, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512i a6b6c6d6a6b6c6d6 = _mm512_shuffle_i64x2(a2b2c2d2a6b6c6d6, a2b2c2d2a6b6c6d6, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512i a7b7c7d7a7b7c7d7 = _mm512_shuffle_i64x2(a3b3c3d3a7b7c7d7, a3b3c3d3a7b7c7d7, NDA_SHUFFLE_MASK4(2, 3, 2, 3));

    simd_block[0] = simd_l8(_mm512_mask_blend_epi64(0b11110000, a0b0c0d0a4b4c4d4, e0f0g0h0e0f0g0h0));
    simd_block[1] = simd_l8(_mm512_mask_blend_epi64(0b11110000, a1b1c1d1a5b5c5d5, e1f1g1h1e1f1g1h1));
    simd_block[2] = simd_l8(_mm512_mask_blend_epi64(0b11110000, a2b2c2d2a6b6c6d6, e2f2g2h2e2f2g2h2));
    simd_block[3] = simd_l8(_mm512_mask_blend_epi64(0b11110000, a3b3c3d3a7b7c7d7, e3f3g3h3e3f3g3h3));
    simd_block[4] = simd_l8(_mm512_mask_blend_epi64(0b11110000, a4b4c4d4a4b4c4d4, e0f0g0h0e4f4g4h4));
    simd_block[5] = simd_l8(_mm512_mask_blend_epi64(0b11110000, a5b5c5d5a5b5c5d5, e1f1g1h1e5f5g5h5));
    simd_block[6] = simd_l8(_mm512_mask_blend_epi64(0b11110000, a6b6c6d6a6b6c6d6, e2f2g2h2e6f6g6h6));
    simd_block[7] = simd_l8(_mm512_mask_blend_epi64(0b11110000, a7b7c7d7a7b7c7d7, e3f3g3h3e7f7g7h7));
  }

  template <>
  inline void kernel_transpose(std::array<simd_f16, 16> &simd_block) {
    // https://gist.github.com/nihui/37d98b705a6a28911d77c502282b4748
    __m512 _tmp0 = _mm512_unpacklo_ps(simd_block[0], simd_block[1]);
    __m512 _tmp1 = _mm512_unpackhi_ps(simd_block[0], simd_block[1]);
    __m512 _tmp2 = _mm512_unpacklo_ps(simd_block[2], simd_block[3]);
    __m512 _tmp3 = _mm512_unpackhi_ps(simd_block[2], simd_block[3]);
    __m512 _tmp4 = _mm512_unpacklo_ps(simd_block[4], simd_block[5]);
    __m512 _tmp5 = _mm512_unpackhi_ps(simd_block[4], simd_block[5]);
    __m512 _tmp6 = _mm512_unpacklo_ps(simd_block[6], simd_block[7]);
    __m512 _tmp7 = _mm512_unpackhi_ps(simd_block[6], simd_block[7]);
    __m512 _tmp8 = _mm512_unpacklo_ps(simd_block[8], simd_block[9]);
    __m512 _tmp9 = _mm512_unpackhi_ps(simd_block[8], simd_block[9]);
    __m512 _tmpa = _mm512_unpacklo_ps(simd_block[10], simd_block[11]);
    __m512 _tmpb = _mm512_unpackhi_ps(simd_block[10], simd_block[11]);
    __m512 _tmpc = _mm512_unpacklo_ps(simd_block[12], simd_block[13]);
    __m512 _tmpd = _mm512_unpackhi_ps(simd_block[12], simd_block[13]);
    __m512 _tmpe = _mm512_unpacklo_ps(simd_block[14], simd_block[15]);
    __m512 _tmpf = _mm512_unpackhi_ps(simd_block[14], simd_block[15]);

    __m512 _tmpg = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmph = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpi = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpj = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpk = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpl = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpm = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpn = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpo = _mm512_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpp = _mm512_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpq = _mm512_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpr = _mm512_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmps = _mm512_shuffle_ps(_tmpc, _tmpe, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpt = _mm512_shuffle_ps(_tmpc, _tmpe, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 _tmpu = _mm512_shuffle_ps(_tmpd, _tmpf, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 _tmpv = _mm512_shuffle_ps(_tmpd, _tmpf, _MM_SHUFFLE(3, 2, 3, 2));

    _tmp0 = _mm512_shuffle_f32x4(_tmpg, _tmpk, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp1 = _mm512_shuffle_f32x4(_tmpo, _tmps, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp2 = _mm512_shuffle_f32x4(_tmph, _tmpl, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp3 = _mm512_shuffle_f32x4(_tmpp, _tmpt, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp4 = _mm512_shuffle_f32x4(_tmpi, _tmpm, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp5 = _mm512_shuffle_f32x4(_tmpq, _tmpu, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp6 = _mm512_shuffle_f32x4(_tmpj, _tmpn, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp7 = _mm512_shuffle_f32x4(_tmpr, _tmpv, _MM_SHUFFLE(2, 0, 2, 0));
    _tmp8 = _mm512_shuffle_f32x4(_tmpg, _tmpk, _MM_SHUFFLE(3, 1, 3, 1));
    _tmp9 = _mm512_shuffle_f32x4(_tmpo, _tmps, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpa = _mm512_shuffle_f32x4(_tmph, _tmpl, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpb = _mm512_shuffle_f32x4(_tmpp, _tmpt, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpc = _mm512_shuffle_f32x4(_tmpi, _tmpm, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpd = _mm512_shuffle_f32x4(_tmpq, _tmpu, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpe = _mm512_shuffle_f32x4(_tmpj, _tmpn, _MM_SHUFFLE(3, 1, 3, 1));
    _tmpf = _mm512_shuffle_f32x4(_tmpr, _tmpv, _MM_SHUFFLE(3, 1, 3, 1));

    simd_block[0]  = simd_f16(_mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[1]  = simd_f16(_mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[2]  = simd_f16(_mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[3]  = simd_f16(_mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[4]  = simd_f16(_mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[5]  = simd_f16(_mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[6]  = simd_f16(_mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[7]  = simd_f16(_mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(2, 0, 2, 0)));
    simd_block[8]  = simd_f16(_mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[9]  = simd_f16(_mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[10] = simd_f16(_mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[11] = simd_f16(_mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[12] = simd_f16(_mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[13] = simd_f16(_mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[14] = simd_f16(_mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(3, 1, 3, 1)));
    simd_block[15] = simd_f16(_mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(3, 1, 3, 1)));
  }

  template <>
  inline void kernel_transpose(std::array<simd_d8, 8> &simd_block) {
    __m512d a0b0a2b2a4b4a6b6 = _mm512_unpacklo_pd(simd_block[0], simd_block[1]);
    __m512d a1b1a3b3a5b5a7b7 = _mm512_unpackhi_pd(simd_block[0], simd_block[1]);
    __m512d c0d0c2d2c4d4c6d6 = _mm512_unpacklo_pd(simd_block[2], simd_block[3]);
    __m512d c1d1c3d3c5d5c7d7 = _mm512_unpackhi_pd(simd_block[2], simd_block[3]);
    __m512d e0f0e2f2e4f4e6f6 = _mm512_unpacklo_pd(simd_block[4], simd_block[5]);
    __m512d e1f1e3f3e5f5e5f7 = _mm512_unpackhi_pd(simd_block[4], simd_block[5]);
    __m512d g0h0g2h2g4h4g6h6 = _mm512_unpacklo_pd(simd_block[6], simd_block[7]);
    __m512d g1h1g3h3g5h5g7h7 = _mm512_unpackhi_pd(simd_block[6], simd_block[7]);

    __m512d a2b2a0b0a6b6a4b4 = _mm512_shuffle_f64x2(a0b0a2b2a4b4a6b6, a0b0a2b2a4b4a6b6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d a3b3a1b1a7b7a5b5 = _mm512_shuffle_f64x2(a1b1a3b3a5b5a7b7, a1b1a3b3a5b5a7b7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d c2d2c0d0c6d6c4d4 = _mm512_shuffle_f64x2(c0d0c2d2c4d4c6d6, c0d0c2d2c4d4c6d6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d c3d3c1d1c7d7c5d5 = _mm512_shuffle_f64x2(c1d1c3d3c5d5c7d7, c1d1c3d3c5d5c7d7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d e2f2e0f0e6f6e4f4 = _mm512_shuffle_f64x2(e0f0e2f2e4f4e6f6, e0f0e2f2e4f4e6f6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d e3f3e1f1e7f7e5f5 = _mm512_shuffle_f64x2(e1f1e3f3e5f5e5f7, e1f1e3f3e5f5e5f7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d g2h2g0h0g6h6g4h4 = _mm512_shuffle_f64x2(g0h0g2h2g4h4g6h6, g0h0g2h2g4h4g6h6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d g3h3g1h1g7h7g5h5 = _mm512_shuffle_f64x2(g1h1g3h3g5h5g7h7, g1h1g3h3g5h5g7h7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));

    __m512d a2b2c2d2a6b6c6d6 = _mm512_mask_blend_pd(0b11001100, a2b2a0b0a6b6a4b4, c0d0c2d2c4d4c6d6);
    __m512d a3b3c3d3a7b7c7d7 = _mm512_mask_blend_pd(0b11001100, a3b3a1b1a7b7a5b5, c1d1c3d3c5d5c7d7);
    __m512d a0b0c0d0a4b4c4d4 = _mm512_mask_blend_pd(0b11001100, a0b0a2b2a4b4a6b6, c2d2c0d0c6d6c4d4);
    __m512d a1b1c1d1a5b5c5d5 = _mm512_mask_blend_pd(0b11001100, a1b1a3b3a5b5a7b7, c3d3c1d1c7d7c5d5);
    __m512d e2f2g2h2e6f6g6h6 = _mm512_mask_blend_pd(0b11001100, e2f2e0f0e6f6e4f4, g0h0g2h2g4h4g6h6);
    __m512d e3f3g3h3e7f7g7h7 = _mm512_mask_blend_pd(0b11001100, e3f3e1f1e7f7e5f5, g1h1g3h3g5h5g7h7);
    __m512d e0f0g0h0e4f4g4h4 = _mm512_mask_blend_pd(0b11001100, e0f0e2f2e4f4e6f6, g2h2g0h0g6h6g4h4);
    __m512d e1f1g1h1e5f5g5h5 = _mm512_mask_blend_pd(0b11001100, e1f1e3f3e5f5e5f7, g3h3g1h1g7h7g5h5);

    __m512d e0f0g0h0e0f0g0h0 = _mm512_shuffle_f64x2(e0f0g0h0e4f4g4h4, e0f0g0h0e4f4g4h4, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d e1f1g1h1e1f1g1h1 = _mm512_shuffle_f64x2(e1f1g1h1e5f5g5h5, e1f1g1h1e5f5g5h5, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d e2f2g2h2e2f2g2h2 = _mm512_shuffle_f64x2(e2f2g2h2e6f6g6h6, e2f2g2h2e6f6g6h6, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d e3f3g3h3e3f3g3h3 = _mm512_shuffle_f64x2(e3f3g3h3e7f7g7h7, e3f3g3h3e7f7g7h7, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d a4b4c4d4a4b4c4d4 = _mm512_shuffle_f64x2(a0b0c0d0a4b4c4d4, a0b0c0d0a4b4c4d4, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512d a5b5c5d5a5b5c5d5 = _mm512_shuffle_f64x2(a1b1c1d1a5b5c5d5, a1b1c1d1a5b5c5d5, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512d a6b6c6d6a6b6c6d6 = _mm512_shuffle_f64x2(a2b2c2d2a6b6c6d6, a2b2c2d2a6b6c6d6, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512d a7b7c7d7a7b7c7d7 = _mm512_shuffle_f64x2(a3b3c3d3a7b7c7d7, a3b3c3d3a7b7c7d7, NDA_SHUFFLE_MASK4(2, 3, 2, 3));

    simd_block[0] = simd_d8(_mm512_mask_blend_pd(0b11110000, a0b0c0d0a4b4c4d4, e0f0g0h0e0f0g0h0));
    simd_block[1] = simd_d8(_mm512_mask_blend_pd(0b11110000, a1b1c1d1a5b5c5d5, e1f1g1h1e1f1g1h1));
    simd_block[2] = simd_d8(_mm512_mask_blend_pd(0b11110000, a2b2c2d2a6b6c6d6, e2f2g2h2e2f2g2h2));
    simd_block[3] = simd_d8(_mm512_mask_blend_pd(0b11110000, a3b3c3d3a7b7c7d7, e3f3g3h3e3f3g3h3));
    simd_block[4] = simd_d8(_mm512_mask_blend_pd(0b11110000, a4b4c4d4a4b4c4d4, e0f0g0h0e4f4g4h4));
    simd_block[5] = simd_d8(_mm512_mask_blend_pd(0b11110000, a5b5c5d5a5b5c5d5, e1f1g1h1e5f5g5h5));
    simd_block[6] = simd_d8(_mm512_mask_blend_pd(0b11110000, a6b6c6d6a6b6c6d6, e2f2g2h2e6f6g6h6));
    simd_block[7] = simd_d8(_mm512_mask_blend_pd(0b11110000, a7b7c7d7a7b7c7d7, e3f3g3h3e7f7g7h7));
  }

  template <>
  inline void kernel_transpose(std::array<simd_cf8, 8> &simd_block) {
    __m512d a0b0a2b2a4b4a6b6 = _mm512_unpacklo_pd(_mm512_castps_pd(simd_block[0]), _mm512_castps_pd(simd_block[1]));
    __m512d a1b1a3b3a5b5a7b7 = _mm512_unpackhi_pd(_mm512_castps_pd(simd_block[0]), _mm512_castps_pd(simd_block[1]));
    __m512d c0d0c2d2c4d4c6d6 = _mm512_unpacklo_pd(_mm512_castps_pd(simd_block[2]), _mm512_castps_pd(simd_block[3]));
    __m512d c1d1c3d3c5d5c7d7 = _mm512_unpackhi_pd(_mm512_castps_pd(simd_block[2]), _mm512_castps_pd(simd_block[3]));
    __m512d e0f0e2f2e4f4e6f6 = _mm512_unpacklo_pd(_mm512_castps_pd(simd_block[4]), _mm512_castps_pd(simd_block[5]));
    __m512d e1f1e3f3e5f5e5f7 = _mm512_unpackhi_pd(_mm512_castps_pd(simd_block[4]), _mm512_castps_pd(simd_block[5]));
    __m512d g0h0g2h2g4h4g6h6 = _mm512_unpacklo_pd(_mm512_castps_pd(simd_block[6]), _mm512_castps_pd(simd_block[7]));
    __m512d g1h1g3h3g5h5g7h7 = _mm512_unpackhi_pd(_mm512_castps_pd(simd_block[6]), _mm512_castps_pd(simd_block[7]));

    __m512d a2b2a0b0a6b6a4b4 = _mm512_shuffle_f64x2(a0b0a2b2a4b4a6b6, a0b0a2b2a4b4a6b6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d a3b3a1b1a7b7a5b5 = _mm512_shuffle_f64x2(a1b1a3b3a5b5a7b7, a1b1a3b3a5b5a7b7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d c2d2c0d0c6d6c4d4 = _mm512_shuffle_f64x2(c0d0c2d2c4d4c6d6, c0d0c2d2c4d4c6d6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d c3d3c1d1c7d7c5d5 = _mm512_shuffle_f64x2(c1d1c3d3c5d5c7d7, c1d1c3d3c5d5c7d7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d e2f2e0f0e6f6e4f4 = _mm512_shuffle_f64x2(e0f0e2f2e4f4e6f6, e0f0e2f2e4f4e6f6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d e3f3e1f1e7f7e5f5 = _mm512_shuffle_f64x2(e1f1e3f3e5f5e5f7, e1f1e3f3e5f5e5f7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d g2h2g0h0g6h6g4h4 = _mm512_shuffle_f64x2(g0h0g2h2g4h4g6h6, g0h0g2h2g4h4g6h6, NDA_SHUFFLE_MASK4(1, 0, 3, 2));
    __m512d g3h3g1h1g7h7g5h5 = _mm512_shuffle_f64x2(g1h1g3h3g5h5g7h7, g1h1g3h3g5h5g7h7, NDA_SHUFFLE_MASK4(1, 0, 3, 2));

    __m512d a2b2c2d2a6b6c6d6 = _mm512_mask_blend_pd(0b11001100, a2b2a0b0a6b6a4b4, c0d0c2d2c4d4c6d6);
    __m512d a3b3c3d3a7b7c7d7 = _mm512_mask_blend_pd(0b11001100, a3b3a1b1a7b7a5b5, c1d1c3d3c5d5c7d7);
    __m512d a0b0c0d0a4b4c4d4 = _mm512_mask_blend_pd(0b11001100, a0b0a2b2a4b4a6b6, c2d2c0d0c6d6c4d4);
    __m512d a1b1c1d1a5b5c5d5 = _mm512_mask_blend_pd(0b11001100, a1b1a3b3a5b5a7b7, c3d3c1d1c7d7c5d5);
    __m512d e2f2g2h2e6f6g6h6 = _mm512_mask_blend_pd(0b11001100, e2f2e0f0e6f6e4f4, g0h0g2h2g4h4g6h6);
    __m512d e3f3g3h3e7f7g7h7 = _mm512_mask_blend_pd(0b11001100, e3f3e1f1e7f7e5f5, g1h1g3h3g5h5g7h7);
    __m512d e0f0g0h0e4f4g4h4 = _mm512_mask_blend_pd(0b11001100, e0f0e2f2e4f4e6f6, g2h2g0h0g6h6g4h4);
    __m512d e1f1g1h1e5f5g5h5 = _mm512_mask_blend_pd(0b11001100, e1f1e3f3e5f5e5f7, g3h3g1h1g7h7g5h5);

    __m512d e0f0g0h0e0f0g0h0 = _mm512_shuffle_f64x2(e0f0g0h0e4f4g4h4, e0f0g0h0e4f4g4h4, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d e1f1g1h1e1f1g1h1 = _mm512_shuffle_f64x2(e1f1g1h1e5f5g5h5, e1f1g1h1e5f5g5h5, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d e2f2g2h2e2f2g2h2 = _mm512_shuffle_f64x2(e2f2g2h2e6f6g6h6, e2f2g2h2e6f6g6h6, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d e3f3g3h3e3f3g3h3 = _mm512_shuffle_f64x2(e3f3g3h3e7f7g7h7, e3f3g3h3e7f7g7h7, NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d a4b4c4d4a4b4c4d4 = _mm512_shuffle_f64x2(a0b0c0d0a4b4c4d4, a0b0c0d0a4b4c4d4, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512d a5b5c5d5a5b5c5d5 = _mm512_shuffle_f64x2(a1b1c1d1a5b5c5d5, a1b1c1d1a5b5c5d5, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512d a6b6c6d6a6b6c6d6 = _mm512_shuffle_f64x2(a2b2c2d2a6b6c6d6, a2b2c2d2a6b6c6d6, NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512d a7b7c7d7a7b7c7d7 = _mm512_shuffle_f64x2(a3b3c3d3a7b7c7d7, a3b3c3d3a7b7c7d7, NDA_SHUFFLE_MASK4(2, 3, 2, 3));

    simd_block[0] = simd_cf8(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a0b0c0d0a4b4c4d4, e0f0g0h0e0f0g0h0)));
    simd_block[1] = simd_cf8(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a1b1c1d1a5b5c5d5, e1f1g1h1e1f1g1h1)));
    simd_block[2] = simd_cf8(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a2b2c2d2a6b6c6d6, e2f2g2h2e2f2g2h2)));
    simd_block[3] = simd_cf8(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a3b3c3d3a7b7c7d7, e3f3g3h3e3f3g3h3)));
    simd_block[4] = simd_cf8(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a4b4c4d4a4b4c4d4, e0f0g0h0e4f4g4h4)));
    simd_block[5] = simd_cf8(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a5b5c5d5a5b5c5d5, e1f1g1h1e5f5g5h5)));
    simd_block[6] = simd_cf8(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a6b6c6d6a6b6c6d6, e2f2g2h2e6f6g6h6)));
    simd_block[7] = simd_cf8(_mm512_castpd_ps(_mm512_mask_blend_pd(0b11110000, a7b7c7d7a7b7c7d7, e3f3g3h3e7f7g7h7)));
  }

  template <>
  inline void kernel_transpose(std::array<simd_cd4, 4> &simd_block) {
    __m512d a0a1b0b1 = _mm512_shuffle_f64x2(simd_block[0], simd_block[1], NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d a2a3b2b3 = _mm512_shuffle_f64x2(simd_block[0], simd_block[1], NDA_SHUFFLE_MASK4(2, 3, 2, 3));
    __m512d c0c1d0d1 = _mm512_shuffle_f64x2(simd_block[2], simd_block[3], NDA_SHUFFLE_MASK4(0, 1, 0, 1));
    __m512d c2c3d2d3 = _mm512_shuffle_f64x2(simd_block[2], simd_block[3], NDA_SHUFFLE_MASK4(2, 3, 2, 3));

    simd_block[0] = simd_cd4(_mm512_shuffle_f64x2(a0a1b0b1, c0c1d0d1, NDA_SHUFFLE_MASK4(0, 2, 0, 2)));
    simd_block[1] = simd_cd4(_mm512_shuffle_f64x2(a0a1b0b1, c0c1d0d1, NDA_SHUFFLE_MASK4(1, 3, 1, 3)));
    simd_block[2] = simd_cd4(_mm512_shuffle_f64x2(a2a3b2b3, c2c3d2d3, NDA_SHUFFLE_MASK4(0, 2, 0, 2)));
    simd_block[3] = simd_cd4(_mm512_shuffle_f64x2(a2a3b2b3, c2c3d2d3, NDA_SHUFFLE_MASK4(1, 3, 1, 3)));
  }
} // namespace nda::simd

#endif