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
  // tcgen05.mma reads A straight from SMEM through its descriptor, so the
  // s2r loader_a into RMEM is dead work for it. TS mode goes further and
  // takes codes/scale/zp in the TS contract order (lane = weight row),
  // bypassing loader_b's fragment gather and loader_bs/bzp's ownership.
  static constexpr bool kUseTcgen05 = Ctx::kMmaType == MmaType::TCGEN05;
  static constexpr bool kUseTcgen05Ts = kUseTcgen05 && Ctx::TuningConfig::kUseTcgen05Ts;
  static constexpr bool kUseMxmma = Ctx::kUseMxmma;
  static constexpr uint32_t kPartMmaShapeK = Ctx::kPartMmaShapeK;
  static constexpr uint32_t kNumStages = Ctx::TuningConfig::kNumStages;

  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsChannelInputScale = kHasInputScale && Ctx::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupInputScale = kHasInputScale && Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kIsChannelWeightScale = Ctx::kIsChannelWeightScale;
  static constexpr bool kIsChannelWeightScale2 = Ctx::kIsChannelWeightScale2;
  static constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = Ctx::kIsBlockWeightScale;
  static constexpr bool kIsGroupOrBlockWeightScale = kIsGroupWeightScale || kIsBlockWeightScale;

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kHasBias = Ctx::kHasBias;

  // The TS group indexing below derives the scale/zp group straight from
  // iter_id, which is the packed-K k_iter_id convention only when the packed
  // layout is off (bf16 activations cannot select it either way).
  static_assert(!kUseTcgen05Ts || !Ctx::kUsePackedKLayout,
                "tcgen05 TS mode does not support the packed-K layout");

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
    uint32_t k_iter_id = Ctx::kUsePackedKLayout ? 0 : iter_id;
    auto &smem = ctx.smem;

    if constexpr (kUseTcgen05Ts) {
      load_stage_iter_ts(stage_id, k_iter_id, buffer_id);
      return;
    }

    loader_b.load(smem.stages[stage_id].b, mma.regs_qb_as_ptr(buffer_id), iter_id);
    if constexpr (!kUseWgmma && !kUseTcgen05)
      loader_a.load(smem.stages[stage_id].a, mma.regs_a_as_ptr(buffer_id), iter_id, stage_id);
    if constexpr (kUseMxmma) {
      if constexpr (kIsGroupInputScale)
        loader_as.load_sf(smem.stages[stage_id].as, mma.regs_sfa_as_ptr(buffer_id), k_iter_id);
      if constexpr (kIsGroupOrBlockWeightScale)
        loader_bs.load_sf(smem.stages[stage_id].bs, mma.regs_sfb_as_ptr(buffer_id), k_iter_id);
    } else {
      if constexpr (kIsGroupInputScale)
        loader_as.load(smem.stages[stage_id].as, mma.arith.regs_as_as_ptr(buffer_id), k_iter_id);
      if constexpr (kIsGroupOrBlockWeightScale)
        loader_bs.load(smem.stages[stage_id].bs, mma.arith.regs_bs_as_ptr(buffer_id), k_iter_id);
    }
    if constexpr (kHasZeroPoint && (kIsGroupOrBlockWeightScale || kIsFirst)) {
      if constexpr (kIsChannelWeightScale)
        loader_bzp.load(smem.bzp_c, mma.arith.regs_zp_as_ptr(buffer_id), k_iter_id);
      else
        loader_bzp.load(smem.stages[stage_id].bzp, mma.arith.regs_zp_as_ptr(buffer_id), k_iter_id);
    }
  }

  // TS-mode contract loads. Thread (math warp w, lane l) owns weight row
  // n = 32 * (w % 4) + l of the 128-row MMA-M tile:
  //   * codes: the slot-paired layout of docs/tcgen05_ts_packing.md is
  //     byte-compatible with loader_b's WarpN == 32 half-group gather, which
  //     delivers row n's codes with no loader change;
  //   * scale: bf16 at smem.bs[n] (identity N order), broadcast to bf16x2;
  //   * zp: folded into the per-lane dequant bias, format per weight dtype.
  CUDA_INLINE void load_stage_iter_ts(uint32_t stage_id, uint32_t iter_id, uint32_t buffer_id) {
    auto &smem = ctx.smem;
    uint32_t n = (ctx.warp_id() % 4u) * 32u + ctx.lane_id();

    loader_b.load(smem.stages[stage_id].b, mma.regs_qb_as_ptr(buffer_id), iter_id);

    // Sub-stage group index within the stage. gs >= BlockK gives one group per
    // stage; gs < BlockK is a multiple of the 16-K iter, so this iter lies
    // entirely inside group (iter * kPartMmaShapeK) / gs. Scale and zp share
    // the granularity, so one index drives both.
    uint32_t bs_group = 0;
    if constexpr (kIsGroupWeightScale && Ctx::kWeightScaleGroupSize < BlockShape::K) {
      bs_group = (iter_id * kPartMmaShapeK) / Ctx::kWeightScaleGroupSize;
    }

    if constexpr (kIsGroupWeightScale) {
      // g2s stages the group's scale rows contiguously at bf16 stride BlockN,
      // so row n of group g is bs16[g * BlockN + n].
      const uint16_t *bs16 = reinterpret_cast<const uint16_t *>(smem.stages[stage_id].bs);
      uint32_t s = bs16[bs_group * BlockShape::N + n];
      mma.regs_bs2_ts[buffer_id] = (s << 16) | s;
    }

    // The zp operand format is per weight dtype; transform_b's
    // ts_dequant_b_pair consumes it accordingly.
    constexpr uint32_t kBBits = Ctx::ElementB::kBits;
    // Dequant base per ElementA: bf16 128.0 == 0x4300, fp16 1024.0 == 0x6400.
    // Both hold 2^(kBits - 1) + zp exactly in the low mantissa, so uint_to_f16
    // subtracting (base | zp) emits code - zp.
    constexpr uint32_t kBiasBase = std::is_same<ElementA, Float16>::value ? 0x64006400u : 0x43004300u;
    if constexpr (kIsFpZeroPoint) {
      // A per-lane bf16 in the same [K / gs, N] stream layout as the scale.
      // transform_b subtracts it from the raw code before the scale, so the
      // bias register only carries the raw-code dequant base.
      mma.regs_bias2_ts[buffer_id] = kBiasBase;
      uint32_t z = 0u;
      if constexpr (kHasZeroPoint) z = zp_elem<uint16_t>(stage_id, bs_group * BlockShape::N + n);
      mma.regs_zpfp2_ts[buffer_id] = (z << 16) | z;
    } else if constexpr (kBBits <= 4) {
      // 4-bit zp nibble folded into the bf16 subtrahend; no-zp uses the
      // symmetric midpoint 2^(kBits - 1), matching the reference dequant.
      // Row n lives in byte n / 2, nibble n % 2, so the group stride halves.
      uint32_t zp = 1u << (kBBits - 1u);
      if constexpr (kHasZeroPoint) {
        uint32_t byte = zp_elem<uint8_t>(stage_id, bs_group * (BlockShape::N / 2u) + (n >> 1));
        zp = (byte >> ((n & 1u) * 4u)) & 0xFu;
      }
      mma.regs_bias2_ts[buffer_id] = kBiasBase | (zp << 16) | zp;
    } else {
      // uint8 byte zp: one byte per row.
      uint32_t zp = 0u;
      if constexpr (kHasZeroPoint) zp = zp_elem<uint8_t>(stage_id, bs_group * BlockShape::N + n);
      mma.regs_bias2_ts[buffer_id] = zp;
    }
  }

  // One zp element of the packed stream. Group zp is staged per stage;
  // channelwise zp is K-invariant, staged once in bzp_c, and always has
  // bs_group == 0, so only the base pointer differs.
  template <class T>
  CUDA_INLINE uint32_t zp_elem(uint32_t stage_id, uint32_t offset) {
    auto &smem = ctx.smem;
    if constexpr (kIsChannelWeightScale) {
      return reinterpret_cast<const T *>(smem.bzp_c)[offset];
    } else {
      return reinterpret_cast<const T *>(smem.stages[stage_id].bzp)[offset];
    }
  }

  CUDA_INLINE void load_channel(uint32_t slice_id) {
    auto &smem = ctx.smem;
    if constexpr (kIsChannelInputScale) loader_as.load(smem.as_c, epilogue.arith.regs_as_as_ptr(), -1);
    if constexpr (kIsChannelWeightScale) loader_bs.load(smem.bs_c, epilogue.arith.regs_bs_as_ptr(), -1);
    if constexpr (kIsChannelWeightScale2) loader_bias.load(smem.bs2_c, epilogue.arith.regs_bs2_as_ptr(), 1);
    if constexpr (kHasBias) loader_bias.load(smem.bias, epilogue.arith.regs_bias_as_ptr(), slice_id == 0);
  }
};
