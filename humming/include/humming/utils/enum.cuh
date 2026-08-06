#pragma once


enum class WeightScaleType : uint32_t {
  GROUP,
  BLOCK,
  CHANNEL,
  TENSOR,
};


enum class WeightScale2Type : uint32_t {
  NONE,
  CHANNEL,
  TENSOR,
};


enum class MmaType : uint32_t {
  MMA,
  WGMMA,
  // Blackwell tcgen05.mma (UMMA) -- accumulator in TMEM.
  // See humming/mma/tcgen05_mma.cuh (SS) and tcgen05_ts_mma.cuh (TS).
  TCGEN05,
  MXMMA
};


enum class GemmType : uint32_t {
  DENSE,
  INDEXED,
  GROUPED_CONTIGUOUS,
  GROUPED_MASKED,
};
