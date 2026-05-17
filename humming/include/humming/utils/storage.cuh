#pragma once

#include <humming/utils/base.cuh>

// Conditional member macros: when the condition is false, the member is completely eliminated.

#if HUMMING_HAS_INPUT_SCALE && HUMMING_INPUT_SCALE_GROUP_SIZE > 0
#define IF_HAS_STAGE_INPUT_SCALE(x) x
#else
#define IF_HAS_STAGE_INPUT_SCALE(x)
#endif

#if HUMMING_WEIGHT_SCALE_GROUP_SIZE > 0
#define IF_HAS_STAGE_WEIGHT_SCALE(x) x
#else
#define IF_HAS_STAGE_WEIGHT_SCALE(x)
#endif

#if HUMMING_HAS_ZERO_POINT
#define IF_HAS_ZERO_POINT(x) x
#else
#define IF_HAS_ZERO_POINT(x)
#endif

#if HUMMING_IS_CHANNEL_WEIGHT_SCALE
#define IF_HAS_CHANNEL_WEIGHT_SCALE(x) x
#else
#define IF_HAS_CHANNEL_WEIGHT_SCALE(x)
#endif

#if HUMMING_HAS_BIAS
#define IF_HAS_BIAS(x) x
#else
#define IF_HAS_BIAS(x)
#endif

#if HUMMING_HAS_INPUT_SCALE && HUMMING_INPUT_SCALE_GROUP_SIZE == 0
#define IF_HAS_CHANNEL_INPUT_SCALE(x) x
#else
#define IF_HAS_CHANNEL_INPUT_SCALE(x)
#endif

#if HUMMING_IS_INDEXED_GEMM
#define IF_IS_INDEXED_GEMM(x) x
#else
#define IF_IS_INDEXED_GEMM(x)
#endif

#if HUMMING_IS_GROUPED_GEMM
#define IF_IS_GROUPED_GEMM(x) x
#else
#define IF_IS_GROUPED_GEMM(x)
#endif

#if HUMMING_IS_GROUPED_CONTIGUOUS_GEMM
#define IF_IS_GROUPED_CONTIGUOUS_GEMM(x) x
#else
#define IF_IS_GROUPED_CONTIGUOUS_GEMM(x)
#endif

#if HUMMING_USE_MBARRIER
#define IF_USE_MBARRIER(x) x
#else
#define IF_USE_MBARRIER(x)
#endif

#if HUMMING_USE_WARP_SPEC
#define IF_USE_WARP_SPEC(x) x
#else
#define IF_USE_WARP_SPEC(x)
#endif

#if HUMMING_USE_TCGEN05
#define IF_USE_TCGEN05(x) x
#else
#define IF_USE_TCGEN05(x)
#endif


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

  static constexpr uint32_t kStageBytesA = kStageSizeA * sizeof(int4);
  static constexpr uint32_t kStageBytesB = kStageSizeB * sizeof(int4);
  static constexpr uint32_t kStageBytesAS = kStageSizeAS * sizeof(int4);
  static constexpr uint32_t kStageBytesBS = kStageSizeBS * sizeof(int4);
  static constexpr uint32_t kStageBytesBZP = kStageSizeBZP * sizeof(int4);
  static constexpr uint32_t kChannelBytesAS = kChannelSizeAS * sizeof(int4);
  static constexpr uint32_t kChannelBytesBS = kChannelSizeBS * sizeof(int4);
  static constexpr uint32_t kBiasBytes = kBiasSize * sizeof(int4);

  // Staging buffer for the tcgen05 path: dequantised B operand in bf16/fp16
  // staged from registers back to SMEM before the tcgen05.mma reads it.
  // Sized as one full (BlockN x BlockK) tile per pipeline stage. The
  // element type matches ElementA (= MMA operand type) so the descriptor
  // construction in TCGEN05::run() uses the same swizzle as A.
  static constexpr uint32_t kSmemStrideBDequant = BlockShape::N * BlockShape::K * ElementA::kBits / 32 / 4;
  static constexpr uint32_t kStageSizeBDequant = kSmemStrideBDequant;
  static constexpr uint32_t kStageBytesBDequant = kStageSizeBDequant * sizeof(int4);

  static constexpr bool kUseWarpSpec = TuningConfig::kUseWarpSpec;
  static constexpr bool kUseMBarrier = TuningConfig::kUseMBarrier;
  static constexpr bool kIsIndexedGemm = ComputeConfig::kGemmType == GemmType::INDEXED;
  static constexpr bool kIsGroupedGemm = ComputeConfig::kGemmType == GemmType::GROUPED_CONTIGUOUS || ComputeConfig::kGemmType == GemmType::GROUPED_MASKED;

  union alignas(128) {
    struct {
      int4 a[kNumStages][kStageSizeA];
      int4 b[kNumStages][kStageSizeB];
      IF_HAS_STAGE_INPUT_SCALE(int4 as[kNumStages][kStageSizeAS];)
      IF_HAS_STAGE_WEIGHT_SCALE(int4 bs[kNumStages][kStageSizeBS];)
      IF_HAS_ZERO_POINT(int4 bzp[kIsChannelWeightScale ? 1 : kNumStages][kIsChannelWeightScale ? kChannelSizeBZP : kStageSizeBZP];)
      IF_HAS_CHANNEL_WEIGHT_SCALE(int4 bs_c[kChannelSizeBS];)
      IF_HAS_BIAS(int4 bias[kBiasSize];)
      IF_HAS_CHANNEL_INPUT_SCALE(int4 as_c[kChannelSizeAS];)
      // Dequantised bf16 B staging for the tcgen05 path. Only emitted
      // when `use_tcgen05` is set in the TuningConfig; otherwise the
      // mma.sync / wgmma paths dequant directly into RMEM and need no
      // SMEM here.
      IF_USE_TCGEN05(int4 b_dequant[kNumStages][kStageSizeBDequant];)
    };
    int4 reduce[MAX(kWarpReduceSize, kBlockOutputSize)];
  };

  IF_IS_INDEXED_GEMM(uint32_t rd_row_index[BlockShape::M];)
  IF_IS_INDEXED_GEMM(uint32_t wr_row_index[BlockShape::M];)

  IF_IS_GROUPED_GEMM(CUtensorMap tensor_map_buffer[1];)
  IF_IS_GROUPED_GEMM(uint32_t expert_tokens[kNumExperts];)
  IF_IS_GROUPED_GEMM(uint32_t total_m_blocks[1];)
  IF_IS_GROUPED_CONTIGUOUS_GEMM(uint32_t expert_offset[kNumExperts + 1];)

  IF_USE_MBARRIER(alignas(128) uint64_t load_mbar[kNumStages + 2];)
  IF_USE_WARP_SPEC(uint64_t math_mbar[kNumStages + 1];)

  // Per-CTA TMEM column index returned by `tcgen05.alloc`. The issuing
  // thread writes this; all warps in the CTA read it after the alloc
  // sync.  Only present when use_tcgen05 is set.
  IF_USE_TCGEN05(alignas(16) uint32_t tcgen05_tmem_col;)
  IF_USE_TCGEN05(alignas(8) uint64_t tcgen05_mbar;)
};
