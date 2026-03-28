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
  WGMMA
};
