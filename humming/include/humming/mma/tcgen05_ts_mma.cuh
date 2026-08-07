#pragma once

// TS-mode tcgen05.mma: dequantised weights staged in TMEM via tcgen05.st, MMA-A
// = weights (MmaM = BlockN), MMA-B = activations from SMEM, TMEM D transposed.
// TMEM column map, S = kNumTsSlots staging slots:
//   base + 8j .. 8(j+1)        staging slot j (16 K ElementA, 2 per cell)
//   base + 8S .. 8S + BlockM   D accumulator (f32)

#include <humming/arith/exp_offset.cuh>
#include <humming/datatype/dequant_single.cuh>
#include <humming/epilogue/tmem_ts_drain.cuh>
#include <humming/utils/all.cuh>
#include <humming/utils/ptx/barrier.cuh>
#include <humming/utils/ptx/shared.cuh>
#include <humming/utils/ptx/tcgen05.cuh>


// Dequant one code pair in TS contract order: `shifted` has the target pair at
// the low kBits of each 16-bit half. Returns the ElementA pair, pre-scale.
template <class EB, class EA, bool kHasZeroPoint, bool kIsFpZeroPoint = false>
CUDA_INLINE uint32_t ts_dequant_b_pair(uint32_t shifted, uint32_t bias2) {
  using Scalar2 = typename F16Conversion<EA>::scalar_t2;
  if constexpr (EB::kIsIntegerType && EB::kBits <= EA::kMantissaBits) {
    // uint{2,4}: bias2 is the folded ElementA (base + zp) subtrahend so
    // uint_to_f16 emits code - zp; the fp-zp arm returns the raw code. kOff == 0.
    return uint_to_f16<EB, EA, /*kHasZeroPoint=*/true, kIsFpZeroPoint>(
        shifted, bias2);
  } else if constexpr (EB::kIsIntegerType) {
    // uint8 on bf16 A exceeds bf16's mantissa, so route to normalized_uint_to_fp;
    // the result is subnormal ~(code - zp) * 2^-133 and 2^kOff corrects it.
    // ts_mul_pow2 runs before the fp-zp subtract, so unlike the generic mainloop
    // the zp needs no pre-scale.
    static_assert(EB::kBits == 8, "TS integer dtypes: {uint2, uint4, uint8}");
    constexpr uint32_t kOff =
        get_dtype_dequant_exp_offset<EA, EB, kHasZeroPoint>();
    uint32_t v = normalized_uint_to_fp<EB, EA, kHasZeroPoint, kIsFpZeroPoint>(
        shifted, bias2);
    Scalar2 t = ts_mul_pow2<kOff, EA>(*reinterpret_cast<Scalar2 *>(&v));
    return *reinterpret_cast<uint32_t *>(&t);
  } else if constexpr (EB::kIsFloatingPointType && EB::kBits <= 8u) {
    // Software fp -> EA: relocate the code to the top of its 16-bit half, decode
    // with fp_to_fp (exponent copied, not rebiased), then correct the bias by
    // 2^kOff; always one exact step, so no epilogue offset is left over.
    constexpr uint32_t kOff = get_dtype_dequant_exp_offset<EA, EB>();
    uint32_t v = fp_to_fp<EB, EA>(shifted << (EA::kBits - EB::kBits));
    Scalar2 t = ts_mul_pow2<kOff, EA>(*reinterpret_cast<Scalar2 *>(&v));
    return *reinterpret_cast<uint32_t *>(&t);
  } else {
    static_assert(EB::kBits == 0,
                  "TCGEN05TS transform_b: weight dtype not wired -- add a "
                  "ts_dequant_b_pair branch and extend the allowlist");
    return 0u;
  }
}


template <class Ctx, class ArithClass>
struct TCGEN05TS {
public:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using MmaShape = typename Ctx::MmaShape;
  using SharedStorage = typename Ctx::SharedStorage;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;
  using ElementC = typename Ctx::ElementC;
  using CRegistersType = typename MmaOpClass::CRegisters;
  // Never used on this path; any well-formed shape satisfies the epilogue.
  using CRegistersArrayType = CRegistersType[1][1];

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;
  static constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
  static constexpr bool kIsChannelWeightScale = Ctx::kIsChannelWeightScale;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;

  // Weight-dtype geometry, all compile-time. A thread holds one row of a 16-K
  // chunk, so kWordsPerKChunk * kRegsPerWord == 8 ElementA pairs for any width.
  static constexpr uint32_t kBBits = ElementB::kBits;
  static constexpr uint32_t kWordsPerKChunk = 16u * kBBits / 32u;
  static constexpr uint32_t kCodesPerWord = 32u / kBBits;
  static constexpr uint32_t kRegsPerWord = kCodesPerWord / 2u;

  // One 128-row weight tile, 4 warps x 32 lanes.
  static constexpr uint32_t kMmaM = 128;
  static constexpr uint32_t kTsSlotCols = kPartMmaShapeK * 16u / 32u;  // 8
  // One slot per 16-K iter, so a whole BlockK stage is staged before the batch
  // that consumes it and the handshake is paid once per stage, not once per iter.
  static constexpr uint32_t kTsSlotsPerStage = Ctx::kWarpIters;
  static constexpr uint32_t kNumTsGroups = SharedStorage::kTcgen05TsGroups;
  static constexpr uint32_t kNumTsSlots = kNumTsGroups * kTsSlotsPerStage;
  static constexpr uint32_t kDColOffset = kNumTsSlots * kTsSlotCols;
  // WAR wait site. With one group it must precede the first overwrite (iter 0);
  // with more, the batch reads a different group, so it slides to the last
  // transform_b before the mainloop's consumer.arrive.
  static constexpr uint32_t kTsWaitIter =
      kNumTsGroups > 1 ? kTsSlotsPerStage - 2 : 0;

  static_assert(kNumTsSlots == SharedStorage::kTcgen05TsSlots &&
                    kDColOffset + BlockShape::M <=
                        SharedStorage::kTcgen05TmemCols,
                "TCGEN05TS: staging depth must fit the TMEM reservation");
  // The mainloop stages one iter ahead -- index i calls transform_b(i + 1) --
  // so kTsWaitIter = kWarpIters - 2 is reached at mainloop index kWarpIters - 3,
  // which must exist and precede the consumer.arrive at index kWarpIters - 2.
  static_assert(kNumTsGroups == 1 || kTsSlotsPerStage >= 3,
                "TCGEN05TS: multi-group staging needs the WAR wait one warp "
                "iter ahead of the mainloop's consumer.arrive");
  static_assert(MmaOpClass::kCtaGroup == 1,
                "TCGEN05TS: only cta_group::1 is wired up");
  static_assert(BlockShape::N == 128,
                "TCGEN05TS: BlockN must be 128 (exactly one 128-row "
                "MMA-M tile; multi-tile BlockN=256 is not wired up)");
  static_assert(WarpShape::N == 32,
                "TCGEN05TS: WarpN must be 32 (contract: warp w owns "
                "rows (w%4)*32 + lane; a warp can only tcgen05.st its "
                "own TMEM sub-partition)");
  static_assert(WarpShape::M == BlockShape::M,
                "TCGEN05TS: M_WARPS must be 1 (TMEM D is CTA-level; "
                "multiple M-warps would need sequential m-passes)");
  static_assert(WarpShape::K == BlockShape::K,
                "TCGEN05TS: K_WARPS must be 1 (K accumulates in TMEM D)");
  static_assert(BlockShape::K == 64,
                "TCGEN05TS: BlockK must be 64 ElementA (single 64-K "
                "section; BlockK > 64 needs section-major staging)");
  static_assert(BlockShape::M == 32 || BlockShape::M == 64 ||
                    BlockShape::M == 128,
                "TCGEN05TS: BlockM (= MMA-N) must be 32, 64 or 128 "
                "(M=128 atom requires N % 16 == 0, N <= 256; the drain "
                "loops kBlockM/32 so BlockM must be a multiple of 32)");
  static_assert(std::is_same<ElementA, BFloat16>::value ||
                    std::is_same<ElementA, Float16>::value,
                "TCGEN05TS: ElementA must be BFloat16 or Float16 (both issue "
                "kind::f16; the dequant base is chosen per ElementA -- bf16 "
                "0x4300 vs fp16 0x6400)");
  // TS weight-dtype allowlist. Extending it requires, in lockstep: a
  // ts_dequant_b_pair branch, a TCGEN05_TS_B_DTYPES entry (config/config.py),
  // and a reference-packer width (tests/kernels/humming/_ts_packing_ref.py).
  static constexpr bool kTsBDtypeSupported =
      std::is_same<ElementB, UInt4>::value ||
      std::is_same<ElementB, UInt2>::value ||
      std::is_same<ElementB, UInt8>::value ||
      std::is_same<ElementB, Float4E2M1>::value ||
      std::is_same<ElementB, Float4E3M0>::value ||
      std::is_same<ElementB, Float8E4M3>::value ||
      std::is_same<ElementB, Float8E5M2>::value ||
      std::is_same<ElementB, Float8E1M6>::value;
  static_assert(kTsBDtypeSupported,
                "TCGEN05TS: ElementB not in the TS weight-dtype allowlist "
                "(currently {uint2, uint4, uint8, float4e2m1, float4e3m0, "
                "float8e1m6, float8e4m3, float8e5m2})");
  static_assert(!kIsFpZeroPoint ||
                    (ElementB::kIsIntegerType && !ElementB::kIsSigned),
                "TCGEN05TS fp zero-point: unsigned-integer weight dtypes only "
                "(mirrors MainloopArithmetic); both dequant arms return the "
                "raw code at full magnitude, so transform_b's post-dequant "
                "subtract is exact for either");
  static_assert(Ctx::kIsGroupWeightScale || Ctx::kIsChannelWeightScale,
                "TCGEN05TS: group or channelwise weight scale (block/mx "
                "unsupported)");
  static_assert(!Ctx::kIsGroupWeightScale ||
                    Ctx::kWeightScaleGroupSize >= BlockShape::K ||
                    (BlockShape::K % Ctx::kWeightScaleGroupSize == 0 &&
                     Ctx::kWeightScaleGroupSize % kPartMmaShapeK == 0),
                "TCGEN05TS group scale: gs >= BlockK (one group per stage), "
                "OR gs divides BlockK and is a multiple of the 16-K iter so "
                "each iter stays within a single group (no intra-iter split)");
  static_assert(!Ctx::kReduceOverlapLastStageOnly,
                "TCGEN05TS: reduce_overlap_last_stage_only unsupported");

  Ctx &ctx;
  SharedStorage &smem;
  ArithClass &arith;

  // Interface parity: never written.
  alignas(16) int4 regs_a[1];
  // Per-thread quantised codes, double-buffered. Written by the TS branch of
  // s2r_pipeline. alignas(16): loader_b vectorizes the u8 gather as int4.
  alignas(16) uint32_t regs_qb[2][kWordsPerKChunk];
  // Per-lane ElementA broadcast scale (s, s) and dequant bias
  // (base + zp, base + zp), also filled by the s2r TS branch.
  uint32_t regs_bs2[2];
  uint32_t regs_bias2[2];
  // Per-lane ElementA fp zero-point (kIsFpZeroPoint only), subtracted post-dequant
  // and pre-scale. Never written on the integer-zp path.
  uint32_t regs_zpfp2[2];

  CUDA_INLINE
  TCGEN05TS(Ctx &ctx_, ArithClass &arith_)
      : ctx(ctx_), smem(ctx_.smem), arith(arith_) {}

  CUDA_INLINE
  void zero_accum() {
    // First MMA of the tile overwrites D via scale_d. The slot counters must NOT
    // reset here; mbar phases persist across tiles.
    first_issue_ = true;
  }

  // Dequant (contract order) + WAR-gated r2t into slot `iter_id`. `buffer_id`
  // selects the s2r register double-buffer only.
  CUDA_INLINE
  void transform_b(uint32_t buffer_id, uint32_t iter_id) {
    using Scalar2 = typename F16Conversion<ElementA>::scalar_t2;
    uint32_t out[8];
    uint32_t bias2 = regs_bias2[buffer_id];
    const Scalar2 scale =
        *reinterpret_cast<const Scalar2 *>(&regs_bs2[buffer_id]);
    PRAGMA_UNROLL
    for (uint32_t w = 0; w < kWordsPerKChunk; w++) {
      uint32_t q = regs_qb[buffer_id][w];
      PRAGMA_UNROLL
      for (uint32_t r = 0; r < kRegsPerWord; r++) {
        // Extract the (r, r + kRegsPerWord) codes into the lo/hi halves; the
        // pack pre-compensates the slot order to yield reg = (K = 2*idx,
        // K = 2*idx + 1).
        uint32_t v =
            ts_dequant_b_pair<ElementB, ElementA, kHasZeroPoint, kIsFpZeroPoint>(
                q >> (r * kBBits), bias2);
        Scalar2 t = *reinterpret_cast<Scalar2 *>(&v);
        // ts_dequant returned the raw code, so the fp zp is subtracted here,
        // before the scale, matching the SS order (code - zp_fp) * scale.
        if constexpr (kIsFpZeroPoint) {
          const Scalar2 zpfp =
              *reinterpret_cast<const Scalar2 *>(&regs_zpfp2[buffer_id]);
          t = __hsub2(t, zpfp);
        }
        // Channelwise scale is K-invariant and commutes with the K-sum, so it is
        // deferred to the drain; only the group scale folds in here.
        if constexpr (kIsGroupWeightScale) t = __hmul2(t, scale);
        out[w * kRegsPerWord + r] = *reinterpret_cast<uint32_t *>(&t);
      }
    }

    // WAR gate, once per stage: wait out the previous stage's batch, which
    // transitively retires every earlier one. Its commit was the n-th arrival
    // on mbarrier (T - 1) % kNumTsGroups, so the wait phase is (n - 1) & 1.
    if (iter_id == kTsWaitIter && stage_ctr_ > 0u) {
      uint32_t prev = stage_ctr_ - 1u;
      mbarrier_wait(&smem.tcgen05_ts_mbar[prev % kNumTsGroups],
                    (prev / kNumTsGroups) & 1u);
    }

    // r2t: warp w writes rows 32w..32w+31 (its own sub-partition).
    uint32_t slot = (stage_ctr_ % kNumTsGroups) * kTsSlotsPerStage + iter_id;
    uint32_t warp = threadIdx.x / 32u;
    uint32_t addr = (smem.tcgen05_tmem_col + slot * kTsSlotCols)
                    | ((warp % 4u) * 32u << 16);
    tcgen05_st_32x32b_x8(addr, out);
    // wait::st retires every outstanding store of this thread, so the whole
    // stage is published by the last slot's wait alone.
    if (iter_id == kTsSlotsPerStage - 1u) {
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
    }
  }

  // Issue the stage's whole MMA batch, once the last slot has been staged.
  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    if (iter_id != kTsSlotsPerStage - 1u) return;
    // Cross-warp st -> mma ordering needs all three links: before_thread_sync
    // (end of transform_b), a real thread sync, then after_thread_sync.
    ctx.sync_math_threads();
    tcgen05_fence_after_thread_sync();

    // A<->B swap: idesc M = weight rows (128), N = activation MmaN. Both operand
    // formats follow ElementA, so the kind stays f16 for fp16 and bf16 alike.
    uint32_t idesc =
        tcgen05_instr_desc_f16fam<MmaOpClass::kInstrDescAFormat,
                                  MmaOpClass::kInstrDescBFormat,
                                  MmaOpClass::kInstrDescCFormat>(
            kMmaM, BlockShape::M);

    uint32_t group = stage_ctr_ % kNumTsGroups;
    uint32_t tmem_base = smem.tcgen05_tmem_col;
    if (threadIdx.x < 32 && tcgen05_elect_one_sync()) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kTsSlotsPerStage; j++) {
        // Same Swizzle<3,4,3> K-major layout as the SS kernel's A; BlockK == 64,
        // so the advance is j * 2 uint128.
        int4 *act_ptr = &smem.stages[stage_id].a[0] + j * 2u;
        uint64_t b_desc = tcgen05_smem_desc<128, BlockShape::K>(act_ptr);
        tcgen05_mma_ts_bf16(
            tmem_base + kDColOffset,
            tmem_base + (group * kTsSlotsPerStage + j) * kTsSlotCols,
            b_desc, idesc, !first_issue_ || j > 0);
      }
      // Commit the batch (all MMAs so far); the arrival transitively proves
      // every slot's reader retired.
      tcgen05_commit_to_mbarrier(
          cast_smem_ptr_to_uint(&smem.tcgen05_ts_mbar[group]));
    }
    first_issue_ = false;
    stage_ctr_++;
  }

  // Drain TMEM D (lane = weight row n, col = activation m) into smem.reduce in
  // gmem_writer's layout, bypassing smem_writer. Returns nullptr as the sentinel.
  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    if (threadIdx.x < 32 && tcgen05_elect_one_sync()) {
      tcgen05_commit_to_mbarrier(cast_smem_ptr_to_uint(&smem.tcgen05_mbar));
    }
    mbarrier_wait(&smem.tcgen05_mbar, mbar_phase_);
    mbar_phase_ ^= 1u;
    tcgen05_fence_after_thread_sync();

    uint32_t warp = threadIdx.x / 32u;
    uint32_t lane = threadIdx.x % 32u;
    uint32_t n = (warp % 4u) * 32u + lane;  // this thread's weight row

    float bias_val = 0.0f;
    if constexpr (Ctx::kHasBias) {
      using ScalarC = typename F16Conversion<ElementC>::scalar_t;
      const ScalarC *smem_bias =
          reinterpret_cast<const ScalarC *>(&smem.bias[0]);
      bias_val = F16Conversion<ElementC>::num22float2(
                     F16Conversion<ElementC>::num2num2(smem_bias[n]))
                     .x;
    }

    // Channelwise weight scale: one ElementBS per output row n in smem.bs_c. It
    // commutes with the K-sum, so it folds here rather than in transform_b. Read
    // in natural n order, matching the bias read above.
    float scale_val = 1.0f;
    if constexpr (kIsChannelWeightScale) {
      using ScalarBS = typename F16Conversion<
          typename Ctx::ElementBS>::scalar_t;
      const ScalarBS *smem_bs =
          reinterpret_cast<const ScalarBS *>(&smem.bs_c[0]);
      scale_val = F16Conversion<typename Ctx::ElementBS>::num22float2(
                      F16Conversion<typename Ctx::ElementBS>::num2num2(
                          smem_bs[n]))
                      .x;
    }

    uint32_t smem_reduce_base = offsetof(SharedStorage, reduce) / 128u % 8u;
    uint32_t d_base = smem.tcgen05_tmem_col + kDColOffset;
    tmem_ts_drain_transposed<BlockShape::M, ElementC, kIsChannelWeightScale>(
        d_base, n, smem.reduce, smem_reduce_base, bias_val, scale_val);
    ctx.sync_math_threads();
    return nullptr;
  }

  // s2r pipeline interface parity.
  template <class T = uint32_t>
  CUDA_INLINE T *regs_a_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(regs_a);
  }

  template <class T = uint32_t>
  CUDA_INLINE T *regs_qb_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(regs_qb[buffer_id]);
  }

  template <class T = uint32_t>
  CUDA_INLINE T *regs_b_as_ptr() {
    return reinterpret_cast<T *>(regs_qb);
  }

private:
  bool first_issue_ = true;
  uint32_t mbar_phase_ = 0;
  // Stages issued so far; picks the staging group and the WAR mbarrier phase.
  // Never reset: mbarrier phases persist across output tiles.
  uint32_t stage_ctr_ = 0;
};
