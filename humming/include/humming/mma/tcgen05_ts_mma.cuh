#pragma once
//
// TCGEN05 TS-mode MMA class for Blackwell sm_100+. Selected instead of
// the SS-mode TCGEN05 class when TuningConfig::kUseTcgen05Ts is set.
//
// Data flow (vs the SS class in tcgen05_mma.cuh):
//   * Dequantised weights are staged in TMEM via tcgen05.st (r2t), not
//     scattered to smem.b_dequant (r2s). The per-K-iter 2052-cycle SMEM
//     scatter and its 256-thread bar.sync disappear; a 128-thread
//     bar.sync + tcgen05 fences remain for st->mma visibility.
//   * The MMA is A<->B swapped: MMA-A = weights from TMEM (MmaM =
//     BlockN = 128 rows = TMEM lanes), MMA-B = activations from SMEM
//     via the same Swizzle<3,4,3> descriptor the SS kernel used for its
//     A operand (MmaN = BlockM). TMEM D is transposed: D[lane = weight
//     row n][col = activation m].
//   * Weights arrive in the TS register-layout CONTRACT order (produced
//     by humming/utils/ts_packing.py; spec in
//     docs/tcgen05_ts_packing.md): thread (warp w, lane l) owns weight
//     row n = 32w + l; per K-iter it holds the row's 16-K chunk as
//     16 * kBits / 32 uint32 of pre-interleaved codes; the dequant below
//     emits reg r = ElementA pair (K=2r, K=2r+1), the only TMEM-A layout
//     TS-mode accepts. Scale/zp are per-lane (lane = row ownership),
//     loaded by the TS branch in s2r_pipeline.cuh. Humming's
//     fragment-ownership dequant/arith path is bypassed entirely.
//
// TMEM column map (single alloc of SharedStorage::kTcgen05TmemCols), with
// S = kNumTsSlots = kNumTsGroups x kWarpIters staging slots, i.e. one group of
// kWarpIters slots per resident BlockK stage:
//   base + 8j .. 8(j+1)         W staging slot j (16 K ElementA, 2/cell)
//   base + 8S .. 8S + BlockM    D accumulator (f32, MmaN = BlockM cols)
//
// Staging cadence: transform_b() fills its group's slot iter_id; run() is a
// no-op until the last iter, where it publishes the group with one bar.sync
// and issues the stage's kWarpIters UMMAs back to back. The per-iter WAR gate,
// wait::st and bar.sync of a ping-pong schedule dominated the math warps' time
// (~36%); staging depth alone does not help, only the handshake cadence does.
// A second group then lets a stage's dequant run against the previous stage's
// in-flight batch rather than waiting it out.
//
// Transform2Mma handshake (WAR on the staging slots): one mbarrier per group;
// run() commits its batch to the group's mbarrier, and transform_b() waits the
// previous stage's commit (which transitively retires every earlier batch) at
// kTsWaitIter. All math threads keep a consistent stage counter because they
// execute run()/transform_b() in identical order.

#include <humming/arith/exp_offset.cuh>
#include <humming/datatype/dequant_single.cuh>
#include <humming/epilogue/tmem_ts_drain.cuh>
#include <humming/utils/all.cuh>
#include <humming/utils/ptx/barrier.cuh>
#include <humming/utils/ptx/shared.cuh>
#include <humming/utils/ptx/tcgen05.cuh>


// Per-dtype dequant of one ElementA code pair in TS contract order: given a
// code word right-shifted so the target pair sits at the low kBits of each
// 16-bit half, return the ElementA pair (before the per-lane group scale,
// which the caller applies). Dispatch is compile-time on ElementB; a new
// weight dtype adds a branch here rather than re-writing transform_b's
// inline lop3. Per-branch dequant/zp/exp-offset details are documented at
// each case below.


// ts_mul_pow2 (the exact 2^kOff multiply, one or two steps against the
// ElementA exponent ceiling) lives in arith/exp_offset.cuh: the TS s2r scale
// decode needs it too.

// Dequant one 16-bit-format code pair to ElementA (bf16 or fp16). EA drives
// the dequant magic (bf16 0x4300 vs fp16 0x6400 base, via uint_to_f16 /
// fp_to_fp / normalized_uint_to_fp) and the exp-offset element type. The
// mantissa width picks the integer arm, so uint8 splits: normalized on bf16,
// uint_to_f16 on fp16.
template <class EB, class EA, bool kHasZeroPoint, bool kIsFpZeroPoint = false>
CUDA_INLINE uint32_t ts_dequant_b_pair(uint32_t shifted, uint32_t bias2) {
  using Scalar2 = typename F16Conversion<EA>::scalar_t2;
  if constexpr (EB::kIsIntegerType && EB::kBits <= EA::kMantissaBits) {
    // uint{2,4}: with an integer zp, bias2 is the folded EA(2^(k-1) + zp)
    // subtrahend so uint_to_f16 emits (code - zp) (no-zp midpoint baked
    // in by s2r). With an fp zp, uint_to_f16 subtracts only the base and
    // returns the RAW code as EA; the caller subtracts the per-lane EA
    // zp post-dequant, pre-scale. No exp offset (kOff==0) either way.
    return uint_to_f16<EB, EA, /*kHasZeroPoint=*/true, kIsFpZeroPoint>(
        shifted, bias2);
  } else if constexpr (EB::kIsIntegerType) {
    // uint8 on bf16 A: 8 > bf16's mantissa, so the 0x4300 trick breaks; route
    // to normalized_uint_to_fp with the RAW integer zp (bias2, broadcast
    // internally; its no-zp branch applies the symmetric midpoint). The
    // result is a subnormal ~(code-zp)*2^-133, corrected by 2^kOff (133).
    // With an fp zp, normalized_uint_to_fp returns the raw code (also
    // scaled by 2^-133) and the caller subtracts the per-lane ElementA zp.
    // ts_mul_pow2 runs BEFORE that subtract, so unlike the generic mainloop
    // (mainloop_arith.cuh, which caps its offset at 127 and pre-scales the zp
    // by 2^(kExpOffset.x - 133)) the zp needs no rescale here: the value is
    // already back at full magnitude when the subtract happens.
    static_assert(EB::kBits == 8, "TS integer dtypes: {uint2, uint4, uint8}");
    constexpr uint32_t kOff =
        get_dtype_dequant_exp_offset<EA, EB, kHasZeroPoint>();
    uint32_t v = normalized_uint_to_fp<EB, EA, kHasZeroPoint, kIsFpZeroPoint>(
        shifted, bias2);
    Scalar2 t = ts_mul_pow2<kOff, EA>(*reinterpret_cast<Scalar2 *>(&v));
    return *reinterpret_cast<uint32_t *>(&t);
  } else if constexpr (EB::kIsFloatingPointType && EB::kBits <= 8u) {
    // Software fp -> EA: relocate each code to the top of its 16-bit
    // half, decode via fp_to_fp (exponent bits copied, NOT rebiased),
    // then multiply by 2^kOff to correct the bias. kOff is
    // 2^(EA_exp-1) - 2^(EB_exp-1): bf16 96..127, fp16 0..15, always one
    // ts_mul_pow2 step and always an exact reconstruction of the source
    // value, so the caller's per-lane group-scale hmul2 needs NO epilogue
    // exp-offset plumbing (cf SS kEpilogueExpOffset).
    constexpr uint32_t kOff = get_dtype_dequant_exp_offset<EA, EB>();
    uint32_t v = fp_to_fp<EB, EA>(shifted << (EA::kBits - EB::kBits));
    Scalar2 t = ts_mul_pow2<kOff, EA>(*reinterpret_cast<Scalar2 *>(&v));
    return *reinterpret_cast<uint32_t *>(&t);
  } else {
    static_assert(EB::kBits == 0,
                  "TCGEN05_TS transform_b: weight dtype not wired -- add a "
                  "ts_dequant_b_pair branch and extend the allowlist");
    return 0u;
  }
}


template <class Ctx, class ArithClass>
struct TCGEN05_TS {
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
  // Never used on this path (K_WARPS == 1 so smem_reducer is dead and
  // smem_writer is bypassed); any well-formed shape works.
  using CRegistersArrayType = CRegistersType[1][1];

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;
  static constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
  static constexpr bool kIsChannelWeightScale = Ctx::kIsChannelWeightScale;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;

  // Weight-dtype geometry, all compile-time (runtime-indexed member arrays
  // demote the object from registers to a local-memory stack frame).
  static constexpr uint32_t kBBits = ElementB::kBits;
  // Words per row per 16-K chunk (u2:1, u4:2, u8:4).
  static constexpr uint32_t kWpr = 16u * kBBits / 32u;
  // Codes per packed word (u2:16, u4:8, u8:4).
  static constexpr uint32_t kVpw = 32u / kBBits;
  // ElementA pairs produced per word (= kVpw / 2). kWpr * kRegsPerWord == 8.
  static constexpr uint32_t kRegsPerWord = kVpw / 2u;

  // The MMA-M tile: exactly one 128-row weight tile covered by 4 warps
  // of 32 lanes.
  static constexpr uint32_t kMmaM = 128;
  static constexpr uint32_t kTsSlotCols = kPartMmaShapeK * 16u / 32u;  // 8
  // One slot per 16-K warp iter: a whole BlockK stage is staged before the
  // single MMA batch that consumes it, so the WAR gate, the wait::st and the
  // 128-thread bar.sync are paid once per stage instead of once per iter.
  // kNumTsGroups such stages are resident, so a stage's dequant overlaps the
  // previous stage's in-flight batch instead of waiting it out.
  static constexpr uint32_t kTsSlotsPerStage = Ctx::kWarpIters;
  static constexpr uint32_t kNumTsGroups = SharedStorage::kTcgen05TsGroups;
  static constexpr uint32_t kNumTsSlots = kNumTsGroups * kTsSlotsPerStage;
  static constexpr uint32_t kDColOffset = kNumTsSlots * kTsSlotCols;
  // Where the WAR wait for the previous stage's batch sits. One group: it must
  // precede the first overwrite, so iter 0. More: that batch reads a different
  // group, so the wait slides to the last transform_b before the mainloop's
  // consumer.arrive -- which is what releases the stage the batch reads, and
  // hence the latest safe point.
  static constexpr uint32_t kTsWaitIter =
      kNumTsGroups > 1 ? kTsSlotsPerStage - 2 : 0;

  static_assert(kNumTsSlots == SharedStorage::kTcgen05TsSlots &&
                    kDColOffset + BlockShape::M <=
                        SharedStorage::kTcgen05TmemCols,
                "TCGEN05_TS: staging depth must fit the TMEM reservation");
  // kTsWaitIter is reached at mainloop index kWarpIters - 3, which must exist
  // and precede the consumer.arrive at index kWarpIters - 2.
  static_assert(kNumTsGroups == 1 || kTsSlotsPerStage >= 3,
                "TCGEN05_TS: multi-group staging needs the WAR wait one warp "
                "iter ahead of the mainloop's consumer.arrive");
  static_assert(MmaOpClass::kCtaGroup == 1,
                "TCGEN05_TS: only cta_group::1 is wired up");
  static_assert(BlockShape::N == 128,
                "TCGEN05_TS: BlockN must be 128 (exactly one 128-row "
                "MMA-M tile; multi-tile BlockN=256 is not wired up)");
  static_assert(WarpShape::N == 32,
                "TCGEN05_TS: WarpN must be 32 (contract: warp w owns "
                "rows (w%4)*32 + lane; a warp can only tcgen05.st its "
                "own TMEM sub-partition)");
  static_assert(WarpShape::M == BlockShape::M,
                "TCGEN05_TS: M_WARPS must be 1 (TMEM D is CTA-level; "
                "multiple M-warps would need sequential m-passes)");
  static_assert(WarpShape::K == BlockShape::K,
                "TCGEN05_TS: K_WARPS must be 1 (K accumulates in TMEM D)");
  static_assert(BlockShape::K == 64,
                "TCGEN05_TS: BlockK must be 64 ElementA (single 64-K "
                "section; BlockK > 64 needs section-major staging)");
  static_assert(BlockShape::M == 32 || BlockShape::M == 64 ||
                    BlockShape::M == 128,
                "TCGEN05_TS: BlockM (= MMA-N) must be 32, 64 or 128 "
                "(M=128 atom requires N % 16 == 0, N <= 256; the drain "
                "loops kBlockM/32 so BlockM must be a multiple of 32)");
  static_assert(std::is_same<ElementA, BFloat16>::value ||
                    std::is_same<ElementA, Float16>::value,
                "TCGEN05_TS: ElementA must be BFloat16 or Float16 (both issue "
                "kind::f16; the dequant base is chosen per ElementA -- bf16 "
                "0x4300 vs fp16 0x6400)");
  // TS weight-dtype allowlist. Extending it requires, in lockstep: a
  // ts_dequant_b_pair branch, a ts_packing.py guard, and a
  // supports_tcgen05_ts clause.
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
                "TCGEN05_TS: ElementB not in the TS weight-dtype allowlist "
                "(currently {uint2, uint4, uint8, float4e2m1, float4e3m0, "
                "float8e1m6, float8e4m3, float8e5m2})");
  static_assert(!kIsFpZeroPoint ||
                    (ElementB::kIsIntegerType && !ElementB::kIsSigned),
                "TCGEN05_TS fp zero-point: unsigned-integer weight dtypes only "
                "(mirrors MainloopArithmetic); both dequant arms return the "
                "raw code at full magnitude, so transform_b's post-dequant "
                "subtract is exact for either");
  static_assert(Ctx::kIsGroupWeightScale || Ctx::kIsChannelWeightScale,
                "TCGEN05_TS: group or channelwise weight scale (block/mx "
                "unsupported)");
  static_assert(!Ctx::kIsGroupWeightScale ||
                    Ctx::kWeightScaleGroupSize >= BlockShape::K ||
                    (BlockShape::K % Ctx::kWeightScaleGroupSize == 0 &&
                     Ctx::kWeightScaleGroupSize % kPartMmaShapeK == 0),
                "TCGEN05_TS group scale: gs >= BlockK (one group per stage), "
                "OR gs divides BlockK and is a multiple of the 16-K iter so "
                "each iter stays within a single group (no intra-iter split)");
  static_assert(!Ctx::kReduceOverlapLastStageOnly,
                "TCGEN05_TS: reduce_overlap_last_stage_only unsupported");

  Ctx &ctx;
  SharedStorage &smem;
  ArithClass &arith;

  // Interface parity: never written (activations are read from SMEM by
  // the MMA descriptor; s2r skips loader_a for TCGEN05).
  alignas(16) int4 regs_a[1];
  // Per-thread quantised codes: one row x 16 K, packed into kWpr uint32
  // (u2:1, u4:2, u8:4), double-buffered. Written by the TS branch of
  // s2r_pipeline. alignas(16): loader_b vectorizes the u8 gather as int4.
  alignas(16) uint32_t regs_qb[2][kWpr];
  // Per-lane ElementA broadcast scale (s, s) and dequant bias
  // (base + zp, base + zp), also filled by the s2r TS branch.
  uint32_t regs_bs2_ts[2];
  uint32_t regs_bias2_ts[2];
  // Per-lane ElementA broadcast fp zero-point (kIsFpZeroPoint only): a real
  // ElementA subtracted post-dequant / pre-scale to give (code - zp_fp) * scale.
  // Filled by the s2r TS branch from the [K/gs, N] zp stream. Unused
  // (never written) on the integer-zp path.
  uint32_t regs_zpfp2_ts[2];

  CUDA_INLINE
  TCGEN05_TS(Ctx &ctx_, ArithClass &arith_)
      : ctx(ctx_), smem(ctx_.smem), arith(arith_) {}

  CUDA_INLINE
  void zero_accum() {
    // First MMA of the tile overwrites D via scale_d = false. The
    // slot counters must NOT reset here -- mbar phases persist across
    // tiles.
    first_issue_ = true;
  }

  // Dequant (contract order) + WAR-gated r2t into slot `iter_id`. `buffer_id`
  // selects the s2r register double-buffer only; the per-lane scale/zp
  // registers are refreshed per K-iter by the s2r TS branch.
  CUDA_INLINE
  void transform_b(uint32_t buffer_id, uint32_t iter_id) {
    using Scalar2 = typename F16Conversion<ElementA>::scalar_t2;
    uint32_t out[8];
    uint32_t bias2 = regs_bias2_ts[buffer_id];
    const Scalar2 scale =
        *reinterpret_cast<const Scalar2 *>(&regs_bs2_ts[buffer_id]);
    PRAGMA_UNROLL
    for (uint32_t w = 0; w < kWpr; w++) {
      uint32_t q = regs_qb[buffer_id][w];
      PRAGMA_UNROLL
      for (uint32_t r = 0; r < kRegsPerWord; r++) {
        // Extract the (r, r + kVpw/2) codes of q into the lo/hi ElementA
        // halves and dequant per ElementB; the pack pre-compensates the
        // slot order so this yields reg = (K = 2*idx, K = 2*idx + 1).
        uint32_t v =
            ts_dequant_b_pair<ElementB, ElementA, kHasZeroPoint, kIsFpZeroPoint>(
                q >> (r * kBBits), bias2);
        Scalar2 t = *reinterpret_cast<Scalar2 *>(&v);
        // FP zero point: ts_dequant returned the raw code as EA, so the
        // zp is a real ElementA subtracted here, BEFORE the scale, matching
        // the SS order (code - zp_fp) * scale. Integer zp is already
        // folded into the code by uint_to_f16 (nothing to do here).
        if constexpr (kIsFpZeroPoint) {
          const Scalar2 zpfp =
              *reinterpret_cast<const Scalar2 *>(&regs_zpfp2_ts[buffer_id]);
          t = __hsub2(t, zpfp);
        }
        // Group scale folds here (per-lane, per-stage). Channelwise scale
        // is K-invariant and commutes with the K-sum, so it is deferred
        // to the drain (epilogue-fold) and NOT applied per code here.
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
    // Publish all 4 warps' tcgen05.st to the MMA-issuing thread:
    // before_thread_sync (end of transform_b) -> REAL thread sync ->
    // after_thread_sync. All three links are required for cross-warp
    // ordering.
    ctx.sync_math_threads();
    tcgen05_fence_after_thread_sync();

    // A<->B swap: idesc M = weight rows (128), N = activation MmaN. Weights
    // are dequanted to ElementA, so both operand formats follow ElementA and
    // the MMA kind stays f16 for both fp16 and bf16.
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
        // Activation descriptor: same canonical Swizzle<3,4,3> K-major
        // layout + 16-K-per-issue advance the SS kernel uses for A.
        // BlockK == 64 -> single section, advance is iter * 2 uint128.
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

  // Drain TMEM D (transposed: lane = weight row n, col = activation m)
  // directly into smem.reduce in gmem_writer's sectioned XOR-swizzled
  // layout, bypassing smem_writer (EpiloguePipeline already skips it
  // for kMmaType == TCGEN05). Returns nullptr as the sentinel.
  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    if (threadIdx.x < 32 && tcgen05_elect_one_sync()) {
      tcgen05_commit_to_mbarrier(cast_smem_ptr_to_uint(&smem.tcgen05_mbar));
    }
    mbarrier_wait(&smem.tcgen05_mbar, mbar_phase_);
    mbar_phase_ ^= 1u;
    tcgen05_fence_view_async_tmem_store();

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

    // Channelwise weight scale: one ElementBS scalar per output row n, staged
    // in smem.bs_c by the channel g2s load. It commutes with the K-sum,
    // so we fold it here (the group path folds its scale in transform_b
    // instead). Read in natural n order -- the g2s copies the CTA's
    // 128-row N slice contiguously, matching the bias read above.
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
    // Vectorized drain (tmem_ts_drain.cuh): 8x8 register transpose +
    // four swizzled 128-bit stores per lane per 32-m chunk, replacing
    // the scalar 2-byte scatter.
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

  template <class T = uint32_t>
  CUDA_INLINE T *regs_c_as_ptr(uint32_t buffer_id = 0) {
    // No RMEM accumulator on this path; the accumulator lives in TMEM
    // until final_regs_c_as_ptr drains it.
    return reinterpret_cast<T *>(regs_a);
  }

private:
  bool first_issue_ = true;
  uint32_t mbar_phase_ = 0;
  // Stages issued so far; picks the staging group and the WAR mbarrier phase.
  // Consistent across all math threads by construction, and never reset --
  // mbarrier phases persist across output tiles.
  uint32_t stage_ctr_ = 0;
};
