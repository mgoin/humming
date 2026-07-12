#pragma once
//
// TCGEN05 TS-mode MMA class for Blackwell sm_100+ ("method 2 proper",
// track b-ts-staging prototype). Selected instead of the SS-mode
// TCGEN05 class when TuningConfig::kUseTcgen05Ts is set.
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
//   * Weights arrive in the TS register-layout CONTRACT order (see
//     tests/ts_contract_pack.py): thread (warp w, lane l) owns weight
//     row n = 32w + l; per K-iter it holds the row's 16-K chunk as
//     2 uint32 of pre-interleaved uint4 codes; the lop3 dequant below
//     emits reg r = bf16 pair (K=2r, K=2r+1), the only TMEM-A layout
//     TS-mode accepts. Scale/zp are per-lane (lane = row ownership),
//     loaded by the TS branch in s2r_pipeline.cuh. Humming's
//     fragment-ownership dequant/arith path is bypassed entirely.
//
// TMEM column map (single alloc of SharedStorage::kTcgen05TmemCols):
//   base + 0  .. 8            W staging slot 0 (16 K bf16, 2/cell)
//   base + 8  .. 16           W staging slot 1
//   base + 16 .. 16 + BlockM  D accumulator (f32, MmaN = BlockM cols)
//
// Transform2Mma handshake (the WAR fix Jinzhen's skeleton lacks):
// per-slot mbarriers smem.tcgen05_ts_mbar[2]; run() commits the MMA
// batch to its slot's mbar; transform_b() waits on the slot's mbar
// before re-storing when arrivals_ > waits_. All math threads keep
// consistent per-slot counters because they execute run()/transform_b()
// in identical order.

#include <humming/utils/all.cuh>
#include <humming/utils/ptx/barrier.cuh>
#include <humming/utils/ptx/shared.cuh>
#include <humming/utils/ptx/tcgen05.cuh>


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
  using CRegistersType = typename MmaOpClass::CRegisters;
  // Never used on this path (K_WARPS == 1 so smem_reducer is dead and
  // smem_writer is bypassed); any well-formed shape works.
  using CRegistersArrayType = CRegistersType[1][1];

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;

  // The MMA-M tile: min(BlockN, 128) weight rows. Prototype pins it to
  // exactly one 128-row tile covered by 4 warps of 32 lanes.
  static constexpr uint32_t kMmaM = 128;
  static constexpr uint32_t kTsSlotCols = kPartMmaShapeK * 16u / 32u;  // 8
  static constexpr uint32_t kNumTsSlots = 2;
  static constexpr uint32_t kDColOffset = kNumTsSlots * kTsSlotCols;   // 16

  // Prototype constraint set (one correct config beats six broken ones):
  static_assert(BlockShape::N == 128,
                "TCGEN05_TS prototype: BlockN must be 128 (exactly one "
                "128-row MMA-M tile; multi-tile BlockN=256 is future work)");
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
                "TCGEN05_TS prototype: BlockK must be 64 bf16 (single "
                "64-K section; BlockK > 64 needs section-major staging)");
  static_assert(BlockShape::M == 64 || BlockShape::M == 128,
                "TCGEN05_TS: BlockM (= MMA-N) must be 64 or 128 "
                "(M=128 atom requires N % 16 == 0, N <= 256)");
  static_assert(std::is_same<ElementA, BFloat16>::value,
                "TCGEN05_TS: ElementA must be BFloat16 (kind::f16 idesc "
                "and the 0x4300 dequant trick are bf16-specific)");
  static_assert(ElementB::kBits == 4 && !kIsFpZeroPoint,
                "TCGEN05_TS prototype: ElementB must be uint4 with "
                "integer (or no) zero point");
  static_assert(Ctx::kIsGroupWeightScale,
                "TCGEN05_TS prototype: group weight scale only");
  static_assert(Ctx::kWeightScaleGroupSize >= BlockShape::K,
                "TCGEN05_TS prototype: one scale group per stage "
                "(group_size >= BlockK)");
  static_assert(!Ctx::kReduceOverlapLastStageOnly,
                "TCGEN05_TS: reduce_overlap_last_stage_only unsupported");

  Ctx &ctx;
  SharedStorage &smem;
  ArithClass &arith;

  // Interface parity: never written (activations are read from SMEM by
  // the MMA descriptor; s2r skips loader_a for TCGEN05).
  alignas(16) int4 regs_a[1];
  // Per-thread quantised codes: one row x 16 K x 4 b = 2 uint32,
  // double-buffered. Written by the TS branch of s2r_pipeline.
  alignas(8) uint32_t regs_qb[2][2];
  // Per-lane bf16x2 broadcast scale (s, s) and dequant bias
  // bf16x2(128 + zp, 128 + zp), also filled by the s2r TS branch.
  uint32_t regs_bs2_ts[2];
  uint32_t regs_bias2_ts[2];

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

  // Dequant (contract order) + WAR-gated r2t into the slot.
  CUDA_INLINE
  void transform_b(uint32_t buffer_id) {
    uint32_t out[8];
    uint32_t bias2 = regs_bias2_ts[buffer_id];
    uint32_t scale2 = regs_bs2_ts[buffer_id];
    PRAGMA_UNROLL
    for (uint32_t w = 0; w < 2u; w++) {
      uint32_t q = regs_qb[buffer_id][w];
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < 4u; j++) {
        // Extract nibbles (j, j+4) of q into the lo/hi bf16 halves and
        // OR in the 0x4300 exponent: bf16(128 + code), exact for
        // code in [0, 128). Pack order is pre-compensated so this
        // yields reg r = (K = 2r, K = 2r + 1).
        uint32_t v;
        asm volatile("lop3.b32 %0, %1, %2, %3, 0xea;\n"
                     : "=r"(v)
                     : "r"(q >> (4u * j)), "n"(0x000f000f), "n"(0x43004300));
        __nv_bfloat162 t = *reinterpret_cast<__nv_bfloat162 *>(&v);
        t = __hsub2(t, *reinterpret_cast<const __nv_bfloat162 *>(&bias2));
        t = __hmul2(t, *reinterpret_cast<const __nv_bfloat162 *>(&scale2));
        out[w * 4u + j] = *reinterpret_cast<uint32_t *>(&t);
      }
    }

    // WAR gate: the slot may still be read by an in-flight MMA from
    // two iterations ago. arrivals_/waits_ differ by at most 1.
    uint32_t slot = buffer_id;
    if (arrivals_[slot] > waits_[slot]) {
      mbarrier_wait(&smem.tcgen05_ts_mbar[slot], waits_[slot] & 1u);
      waits_[slot]++;
    }

    // r2t: warp w writes rows 32w..32w+31 (its own sub-partition).
    uint32_t warp = threadIdx.x / 32u;
    uint32_t addr = (smem.tcgen05_tmem_col + slot * kTsSlotCols)
                    | ((warp % 4u) * 32u << 16);
    tcgen05_st_32x32b_x8(addr, out);
    tcgen05_wait_st();
    tcgen05_fence_before_thread_sync();
  }

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    // Publish all 4 warps' tcgen05.st to the MMA-issuing thread:
    // before_thread_sync (end of transform_b) -> REAL thread sync ->
    // after_thread_sync. This is the cross-warp ordering Jinzhen's
    // skeleton was missing.
    ctx.sync_math_threads();
    tcgen05_fence_after_thread_sync();

    uint32_t slot = iter_id % 2u;
    // Activation descriptor: same canonical Swizzle<3,4,3> K-major
    // layout + 16-K-per-issue advance the SS kernel uses for A.
    // BlockK == 64 -> single section, advance is iter_id * 2 uint128.
    int4 *act_ptr = &smem.stages[stage_id].a[0] + iter_id * 2u;
    uint64_t b_desc = tcgen05_smem_desc<128, BlockShape::K>(act_ptr);
    // A<->B swap: idesc M = weight rows (128), N = activation MmaN.
    uint32_t idesc = tcgen05_instr_desc_bf16_bf16_f32(kMmaM, BlockShape::M);

    bool scale_d = !first_issue_;
    first_issue_ = false;

    uint32_t tmem_base = smem.tcgen05_tmem_col;
    if (threadIdx.x < 32 && tcgen05_elect_one_sync()) {
      tcgen05_mma_ts_bf16(tmem_base + kDColOffset,
                          tmem_base + slot * kTsSlotCols,
                          b_desc, idesc, scale_d);
      // Commit the batch (all MMAs so far) to this slot's mbar; the
      // arrival transitively proves the slot's reader retired.
      tcgen05_commit_to_mbarrier(
          cast_smem_ptr_to_uint(&smem.tcgen05_ts_mbar[slot]));
    }
    arrivals_[slot]++;
  }

  // Drain TMEM D (transposed: lane = weight row n, col = activation m)
  // directly into smem.reduce in gmem_writer's sectioned XOR-swizzled
  // layout, bypassing smem_writer (EpiloguePipeline already skips it
  // for kMmaType == TCGEN05). Returns nullptr as the sentinel.
  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    if (threadIdx.x < 32 && tcgen05_elect_one_sync()) {
      tcgen05_commit_to_mbarrier(cast_smem_ptr_to_uint(&smem.tcgen05_mbar[0]));
    }
    mbarrier_wait(&smem.tcgen05_mbar[0], mbar_phase_);
    mbar_phase_ ^= 1u;
    tcgen05_fence_view_async_tmem_store();

    uint32_t warp = threadIdx.x / 32u;
    uint32_t lane = threadIdx.x % 32u;
    uint32_t n = (warp % 4u) * 32u + lane;  // this thread's weight row

    float bias_val = 0.0f;
    if constexpr (Ctx::kHasBias) {
      const __nv_bfloat16 *smem_bias =
          reinterpret_cast<const __nv_bfloat16 *>(&smem.bias[0]);
      bias_val = __bfloat162float(smem_bias[n]);
    }

    uint16_t *red16 = reinterpret_cast<uint16_t *>(smem.reduce);
    uint32_t smem_reduce_base = offsetof(SharedStorage, reduce) / 128u % 8u;
    uint32_t d_base = smem.tcgen05_tmem_col + kDColOffset;
    uint32_t section_row_base = (n / 64u) * BlockShape::M;
    uint32_t section_col = (n / 8u) % 8u;

    PRAGMA_UNROLL
    for (uint32_t chunk = 0; chunk < BlockShape::M / 32u; chunk++) {
      uint32_t tmp[32];
      tcgen05_ld_32x32b_x32(d_base + chunk * 32u, tmp);
      tcgen05_wait_ld();
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 32u; i++) {
        uint32_t m = chunk * 32u + i;
        float f = *reinterpret_cast<float *>(&tmp[i]) + bias_val;
        // gmem_writer layout: smem_row = (n/64)*BlockM + m; the int4
        // col (n/8)%8 is XOR-swizzled by the row phase; the bf16 sits
        // at sub-index n%8 of that int4. Scalar 2-byte stores --
        // correctness first, track e-epilogue-tmem owns the real one.
        uint32_t smem_row = section_row_base + m;
        uint32_t col = section_col ^ ((smem_row + smem_reduce_base) % 8u);
        __nv_bfloat16 fb = __float2bfloat16(f);
        red16[(smem_row * 8u + col) * 8u + (n % 8u)] =
            *reinterpret_cast<uint16_t *>(&fb);
      }
    }
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
  // Per-slot Transform2Mma handshake counters (consistent across all
  // math threads by construction).
  uint32_t arrivals_[kNumTsSlots] = {0, 0};
  uint32_t waits_[kNumTsSlots] = {0, 0};
};
