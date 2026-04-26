#pragma once

#include <humming/datatype/base_conversion.cuh>
#include <humming/datatype/dtypes.cuh>
#include <humming/utils/all.cuh>

CUDA_INLINE uint2 dequant_single_for_mxfp4_fp8(const uint32_t qb, const uint32_t exp_offset) {
  uint32_t qb_org = qb;
  uint32_t qb_ls4 = qb << 4;
  uint32_t qb_rs4 = qb >> 4;

  uint32_t res[2];
  uint32_t signs[2] = {qb_ls4, qb};
  uint32_t others[2] = {qb & 0x07070707, qb_rs4 & 0x07070707};

  uint32_t exp_offset_buffer1 = (exp_offset * 0x08080800) + (exp_offset ? -0x00000400 : 0);
  uint32_t exp_offset_buffer2 = exp_offset * 0x08080808;

  uint32_t exp_offsets[2] = {
    __byte_perm(exp_offset_buffer1, exp_offset_buffer2, qb),
    __byte_perm(exp_offset_buffer1, exp_offset_buffer2, qb >> 16)
  };

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 2; i++) {
    uint32_t val = lop3_and_or(signs[i], 0x80808080, others[i] << 2);
    val = val + __byte_perm(exp_offsets[0], exp_offsets[1], 0x6420 + 0x1111 * i);
    res[i] = val;
  }

  return *reinterpret_cast<uint2*>(res);
}

template <uint32_t kCount, bool kUseWgmma>
CUDA_INLINE void fused_dequant_for_mxfp4_fp8(const uint32_t *qb_ptrs, uint32_t *res_ptrs, uint32_t scales) {
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kCount * 2; i++) {
    uint32_t exp_offset = reinterpret_cast<uint8_t*>(&scales)[i];
    uint2 res = dequant_single_for_mxfp4_fp8(qb_ptrs[i], exp_offset);
    res_ptrs[i * 2] = res.x;
    res_ptrs[i * 2 + 1] = res.y;
  }

  if constexpr (kUseWgmma) {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kCount; i++) {
      uint32_t tmp = res_ptrs[i * 4 + 1];
      res_ptrs[i * 4 + 1] = res_ptrs[i * 4 + 2];
      res_ptrs[i * 4 + 2] = tmp;
    }
  }
}
