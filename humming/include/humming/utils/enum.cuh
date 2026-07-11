#pragma once


enum class WeightScaleType : uint32_t {
  GROUP,
  BLOCK,
  CHANNEL,
  TENSOR,
  GROUP_TENSOR,
};


enum class MmaType : uint32_t {
  MMA,
  WGMMA,
  // Blackwell tcgen05.mma -- accumulator in TMEM, both operands in SMEM.
  // See humming/mma/tcgen05_mma.cuh.
  TCGEN05,
};


enum class GemmType : uint32_t {
  DENSE,
  INDEXED,
  GROUPED_CONTIGUOUS,
  GROUPED_MASKED,
};
