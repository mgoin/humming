#pragma once

#include <humming/memory/s2r_loader/loader_a.cuh>
#include <humming/memory/s2r_loader/loader_as.cuh>
#include <humming/memory/s2r_loader/loader_b.cuh>
#include <humming/memory/s2r_loader/loader_bias.cuh>
#include <humming/memory/s2r_loader/loader_bs.cuh>
#include <humming/memory/s2r_loader/loader_bzp.cuh>
#include <humming/utils/all.cuh>

template <class Ctx, class MMA, class Epilogue>
class S2RMemoryPipeline {
private:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;

  static constexpr bool kUseWgmma = Ctx::kUseWgmma;
  // tcgen05.mma reads A directly from SMEM via the SS descriptor, so the
  // s2r loader_a into RMEM is dead work for it.
  static constexpr bool kUseTcgen05 = Ctx::kMmaType == MmaType::TCGEN05;
  // TS mode: codes/scale/zp are packed in the TS register-layout
  // CONTRACT order (lane = weight row) -- bypass loader_b's fragment
  // gathers and loader_bs/bzp's fragment ownership entirely.
  static constexpr bool kUseTcgen05Ts =
      kUseTcgen05 && Ctx::TuningConfig::kUseTcgen05Ts;
  static constexpr uint32_t kPartMmaShapeK = Ctx::kPartMmaShapeK;
  static constexpr uint32_t kNumStages = Ctx::TuningConfig::kNumStages;

  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsChannelInputScale = kHasInputScale && Ctx::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupInputScale = kHasInputScale && Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kIsChannelWeightScale = Ctx::kIsChannelWeightScale;
  static constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = Ctx::kIsBlockWeightScale;
  static constexpr bool kIsGroupOrBlockWeightScale = kIsGroupWeightScale || kIsBlockWeightScale;

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kHasBias = Ctx::kHasBias;

  using LoaderA = S2RMemoryLoaderA<Ctx>;
  using LoaderB = S2RMemoryLoaderB<Ctx>;
  using LoaderAS = S2RMemoryLoaderAS<Ctx>;
  using LoaderBS = S2RMemoryLoaderBS<Ctx>;
  using LoaderBZP = S2RMemoryLoaderBZP<Ctx>;
  using LoaderBias = S2RMemoryLoaderBias<Ctx>;

public:
  Ctx &ctx;
  MMA &mma;
  Epilogue &epilogue;
  LoaderA loader_a;
  LoaderB loader_b;
  LoaderAS loader_as;
  LoaderBS loader_bs;
  LoaderBZP loader_bzp;
  LoaderBias loader_bias;

  CUDA_INLINE
  S2RMemoryPipeline(Ctx &ctx, MMA &mma, Epilogue &epilogue)
      : ctx(ctx), mma(mma), epilogue(epilogue),
        loader_a(ctx), loader_b(ctx), loader_as(ctx),
        loader_bs(ctx), loader_bzp(ctx), loader_bias(ctx) {
  }

  template <bool kIsFirst = false>
  CUDA_INLINE void load_stage_iter(uint32_t stage_id, uint32_t iter_id) {
    stage_id = (stage_id + iter_id / Ctx::kWarpIters) % kNumStages;
    iter_id = iter_id % Ctx::kWarpIters;
    uint32_t buffer_id = iter_id % 2;
    auto &smem = ctx.smem;

    if constexpr (kUseTcgen05Ts) {
      load_stage_iter_ts(stage_id, iter_id, buffer_id);
      return;
    }

    loader_b.load(smem.stages[stage_id].b, mma.regs_qb_as_ptr(buffer_id), iter_id);
    if constexpr (!kUseWgmma && !kUseTcgen05)
      loader_a.load(smem.stages[stage_id].a, mma.regs_a_as_ptr(buffer_id), iter_id, stage_id);
    if constexpr (kIsGroupInputScale)
      loader_as.load(smem.stages[stage_id].as, mma.arith.regs_as_as_ptr(buffer_id), iter_id);
    if constexpr (kIsGroupOrBlockWeightScale)
      loader_bs.load(smem.stages[stage_id].bs, mma.arith.regs_bs_as_ptr(buffer_id), iter_id);
    if constexpr (kHasZeroPoint && (kIsGroupOrBlockWeightScale || kIsFirst)) {
      if constexpr (kIsChannelWeightScale)
        loader_bzp.load(smem.bzp_c, mma.arith.regs_zp_as_ptr(buffer_id), iter_id);
      else
        loader_bzp.load(smem.stages[stage_id].bzp, mma.arith.regs_zp_as_ptr(buffer_id), iter_id);
    }
  }

  // TS-mode contract loads. Thread (math warp w, lane l) owns weight
  // row n = 32*(w%4) + l of the 128-row MMA-M tile:
  //   * codes: the production slot-paired layout (docs/
  //     tcgen05_ts_packing.md, humming_pack_weight with kUseTcgen05Ts)
  //     is byte-compatible with loader_b's WarpN==32 half-group gather,
  //     which delivers row n's uint2 (16 uint4 codes, lop3
  //     pre-interleaved) with zero loader changes.
  //   * scale: bf16 at smem.bs[n] (identity N order, one group per
  //     stage since group_size >= BlockK), broadcast to bf16x2.
  //   * zp: uint4 nibble n of smem.bzp's group row, folded into the
  //     dequant bias bf16x2(128 + zp) = 0x4300 | zp per half.
  CUDA_INLINE void load_stage_iter_ts(uint32_t stage_id, uint32_t iter_id,
                                      uint32_t buffer_id) {
    auto &smem = ctx.smem;
    uint32_t warp = ctx.warp_id() % 4u;
    uint32_t lane = ctx.lane_id();
    uint32_t n = warp * 32u + lane;

    loader_b.load(smem.stages[stage_id].b, mma.regs_qb_as_ptr(buffer_id),
                  iter_id);

    // Sub-stage group index within the stage. gs >= BlockK: one group
    // per stage (bs_group == 0). gs < BlockK (a multiple of the 16-K
    // iter): this iter lies entirely in group (iter*kPartMmaShapeK)/gs.
    // Scale AND zp share this granularity, so the same index drives both.
    uint32_t bs_group = 0;
    if constexpr (kIsGroupWeightScale &&
                  Ctx::kWeightScaleGroupSize < BlockShape::K) {
      bs_group = (iter_id * kPartMmaShapeK) / Ctx::kWeightScaleGroupSize;
    }

    if constexpr (kIsGroupWeightScale) {
      // g2s stages kNumGroups scale rows contiguously at bf16 stride
      // BlockN, so row n of group g is bs16[g * BlockN + n].
      const uint16_t *bs16 =
          reinterpret_cast<const uint16_t *>(smem.stages[stage_id].bs);
      uint32_t s = bs16[bs_group * BlockShape::N + n];
      mma.regs_bs2_ts[buffer_id] = (s << 16) | s;
    }
    // The zp operand format is per weight-dtype (transform_b's
    // ts_dequant_b_pair consumes it accordingly):
    //   * kBits <= 4 (uint2/4, fp4): 4-bit zp nibble folded into the bf16
    //     subtrahend bf16(128 + zp) == 0x4300 | zp; no-zp uses the
    //     symmetric midpoint 2^(kBits-1) (matches the reference's
    //     quanted - 2^(bits-1), utils/test.generate_random_weight).
    //   * kBits == 8 (uint8/fp8): the RAW integer zp byte (normalized_
    //     uint_to_fp broadcasts it); fp8 has no zp.
    constexpr uint32_t kBBits = Ctx::ElementB::kBits;
    if constexpr (kBBits <= 4) {
      // Dequant base per ElementA: bf16 128.0 == 0x4300, fp16 1024.0 ==
      // 0x6400 (both exactly hold 2^(kBits-1)+zp in the low mantissa, so
      // uint_to_f16 subtracts (base|zp) to emit code - zp). Folded into
      // the per-lane subtrahend the transform passes to uint_to_f16.
      constexpr uint32_t kBiasBase =
          std::is_same<ElementA, Float16>::value ? 0x64006400u : 0x43004300u;
      uint32_t zp = 1u << (kBBits - 1u);
      if constexpr (kHasZeroPoint) {
        // Group zp is per-stage (stages[].bzp); channelwise zp is
        // K-invariant, staged once in bzp_c by the channel g2s load.
        // Both use the identical nibble packing (row n -> byte n/2,
        // nibble n%2), so only the base pointer differs.
        // Nibble zp group stride = BlockN * kNumZPBits(=4) / 8 = BlockN/2
        // bytes. Sub-stage groups pick group bs_group, matching the scale.
        const uint8_t *bzp8;
        uint32_t byte = n >> 1;
        if constexpr (kIsChannelWeightScale)
          bzp8 = reinterpret_cast<const uint8_t *>(smem.bzp_c);
        else {
          bzp8 = reinterpret_cast<const uint8_t *>(smem.stages[stage_id].bzp);
          byte += bs_group * (BlockShape::N / 2u);
        }
        zp = (bzp8[byte] >> ((n & 1u) * 4u)) & 0xFu;
      }
      mma.regs_bias2_ts[buffer_id] = kBiasBase | (zp << 16) | zp;
    } else {
      // uint8 byte zp: group stride = BlockN * kNumZPBits(=8) / 8 = BlockN.
      uint32_t zp = 0u;
      if constexpr (kHasZeroPoint) {
        const uint8_t *bzp8 =
            reinterpret_cast<const uint8_t *>(smem.stages[stage_id].bzp);
        zp = bzp8[bs_group * BlockShape::N + n];
      }
      mma.regs_bias2_ts[buffer_id] = zp;
    }
  }

  CUDA_INLINE void load_channel(uint32_t slice_id) {
    auto &smem = ctx.smem;
    if constexpr (kIsChannelInputScale) loader_as.load(smem.as_c, epilogue.arith.regs_as_as_ptr(), -1);
    if constexpr (kIsChannelWeightScale) loader_bs.load(smem.bs_c, epilogue.arith.regs_bs_as_ptr(), -1);
    if constexpr (kHasBias) loader_bias.load(smem.bias, epilogue.arith.regs_bias_as_ptr(), slice_id == 0);
  }
};
