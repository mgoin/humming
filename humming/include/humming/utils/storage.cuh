#pragma once

#include <humming/utils/base.cuh>


template <
    class MmaOpClass,
    class BlockShape, class WarpShape,
    class ElementA, class ElementB, class ElementBS,
    class LayerConfig, class ComputeConfig, class TuningConfig>
struct SharedStorage {
private:
  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsChannelInputScale = kHasInputScale && LayerConfig::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupInputScale = kHasInputScale && LayerConfig::kInputScaleGroupSize > 0;
  static constexpr bool kIsChannelWeightScale = LayerConfig::kIsChannelWeightScale;
  static constexpr bool kIsGroupWeightScale = LayerConfig::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = LayerConfig::kIsBlockWeightScale;
  static constexpr bool kIsGroupOrBlockWeightScale = kIsGroupWeightScale || kIsBlockWeightScale;
  static constexpr bool kHasZeroPoint = LayerConfig::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = LayerConfig::kIsFpZeroPoint;

public:
  static constexpr uint32_t kNumExperts = LayerConfig::kNumExperts;
  static constexpr uint32_t kNumStages = TuningConfig::kNumStages;
  static constexpr uint32_t kNumWriteSplits = TuningConfig::kNumWriteSplits;
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kNumWarpsDimK = BlockShape::K / WarpShape::K;
  static constexpr uint32_t kMmaCTypeBits = MmaOpClass::kCTypeBits;
  static constexpr uint32_t M_WARPS = (BlockShape::M / WarpShape::M);
  static constexpr uint32_t kWarpReduceSize = M_WARPS * 16 * BlockShape::N * kMmaCTypeBits / 128 * (kNumWarpsDimK / 2);
  static constexpr uint32_t kBlockOutputSize = BlockShape::M * BlockShape::N / 2 / 4 / kNumWriteSplits;
  static constexpr uint32_t kNumZPBits = kIsFpZeroPoint ? 16 : MAX(4, static_next_power_of_2(ElementB::kBits));

  static constexpr uint32_t kSmemStrideA = BlockShape::K * ElementA::kBits / 32 / 4;
  static constexpr uint32_t kSmemStrideB = BlockShape::N * kPartMmaShapeK * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kSmemStrideBS = BlockShape::N * ElementBS::kBits / 32 / 4;
  static constexpr uint32_t kSmemStrideBZP = BlockShape::N * kNumZPBits / 32 / 4;
  static constexpr uint32_t kSmemStrideBias = BlockShape::N * 16 / 32 / 4;

  static constexpr uint32_t kGroupSzieA = LayerConfig::kInputScaleGroupSize;
  static constexpr uint32_t kGroupSzieB = LayerConfig::kWeightScaleGroupSize;
  static constexpr uint32_t kNumGroupsA = kIsGroupInputScale ? CEIL_DIV(BlockShape::K, kGroupSzieA) : 0;
  static constexpr uint32_t kNumGroupsB = kIsGroupOrBlockWeightScale ? CEIL_DIV(BlockShape::K, kGroupSzieB) : 0;

  static constexpr uint32_t kStageSizeA = BlockShape::M * kSmemStrideA;
  static constexpr uint32_t kStageSizeB = BlockShape::K / kPartMmaShapeK * kSmemStrideB;
  static constexpr uint32_t kStageSizeAS = kNumGroupsA * BlockShape::M / 4;
  static constexpr uint32_t kStageSizeBS = kNumGroupsB * kSmemStrideBS;
  static constexpr uint32_t kStageSizeBZP = kNumGroupsB * kSmemStrideBZP;

  static constexpr uint32_t kChannelSizeAS = kIsChannelInputScale ? BlockShape::M / 4 : 0;
  static constexpr uint32_t kChannelSizeBS = kIsChannelWeightScale ? kSmemStrideBS : 0;
  static constexpr uint32_t kChannelSizeBZP = (kIsChannelWeightScale && kHasZeroPoint) ? kSmemStrideBZP : 0;
  static constexpr uint32_t kBiasSize = LayerConfig::kHasBias ? kSmemStrideBias : 0;

  static constexpr bool kUseWarpSpec = TuningConfig::kUseWarpSpec;
  static constexpr bool kUseMBarrier = TuningConfig::kUseMBarrier;
  static constexpr bool kIsIndexedGemm = ComputeConfig::kGemmType == GemmType::INDEXED;
  static constexpr bool kIsGroupedGemm = ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS || ComputeConfig::kGemmType == GemmType::GROUPED_MASKED;

  union alignas(128) {
    struct {
      int4 a[kNumStages][kStageSizeA];
      int4 b[kNumStages][kStageSizeB];
      int4 as[kNumStages][kStageSizeAS];
      int4 bs[kNumStages][kStageSizeBS];
      int4 bzp[kIsChannelWeightScale ? 1 : kNumStages][kIsChannelWeightScale ? kChannelSizeBZP : kStageSizeBZP];
      int4 bs_c[kChannelSizeBS];
      int4 bias[kBiasSize];
      int4 as_c[kChannelSizeAS];
    };
    int4 reduce[MAX(kWarpReduceSize, kBlockOutputSize)];
  };

  uint32_t rd_row_index[kIsIndexedGemm ? BlockShape::M : 0];
  uint32_t wr_row_index[kIsIndexedGemm ? BlockShape::M : 0];

  CUtensorMap tensor_map_buffer[kIsGroupedGemm ? 1 : 0];
  uint32_t expert_offset[ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS ? (kNumExperts + 1) : 0];
  uint32_t expert_tokens[kIsGroupedGemm ? kNumExperts : 0];
  uint32_t total_m_blocks[kIsGroupedGemm ? 1: 0];

  alignas(128) uint64_t load_mbar[kUseMBarrier ? (kNumStages + 2) : 0];
  uint64_t math_mbar[kUseWarpSpec ? (kNumStages + 1) : 0];
};
