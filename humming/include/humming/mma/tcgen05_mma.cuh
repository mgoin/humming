#pragma once
//
// TCGEN05 MMA class for Blackwell sm_100+.
//
// Mirrors the WMMA / WGMMA interface (`zero_accum`, `transform_b`,
// `run`, `final_regs_c_as_ptr`) so the existing `kernel/humming.cuh`
// mainloop can drive it via the three-way dispatch at humming.cuh:71.
//
// What differs from WMMA on the data flow:
//   * A operand comes from SMEM (smem.a[stage]) via an SMEM descriptor,
//     NOT from registers. The s2r_pipe still pulls A into regs_a but
//     we ignore that copy here (workbook decision: profile first).
//   * B operand: the s2r_pipe loads the *quantised* int4 codes into
//     regs_qb, transform_b() dequantises them into RMEM bf16 and writes
//     them to smem.b_dequant[buffer_id] via the helper r2s in this file.
//     tcgen05.mma then reads that SMEM staging buffer.
//   * Accumulator lives in TMEM (column allocated by the kernel entry).
//     `run()` issues a single tcgen05.mma per K-block; no warp-level
//     subdivision since the instruction shape covers the full BlockM/N.
//   * `final_regs_c_as_ptr` does the t2r dance (tcgen05.fence +
//     tcgen05.ld_32x32b_x32) and exposes a plain RMEM `float*` to the
//     epilogue.
//
// Phase 0 simplifications (not final perf):
//   * 1-CTA only (cta_group::1). No clustering, no use_2cta.
//   * No mbarrier-based commit: callers wrap the issue in a fence +
//     __syncthreads. This costs a barrier per K-iter; revisited once
//     the pipeline plumbing lands an mbarrier slot.
//   * No accumulator double-buffering -- one TMEM region per CTA.

#include <humming/utils/all.cuh>
#include <humming/utils/ptx/barrier.cuh>
#include <humming/utils/ptx/tcgen05.cuh>


// fence_proxy.async.shared::cta -- ensures prior r2s of dequantised B
// is observable by subsequent tcgen05.mma SMEM reads. Same primitive
// wgmma uses; defined here so this file is self-contained.
CUDA_INLINE void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}


template <
    class MmaOpClass_, class SharedStorage, class ArithClass,
    class BlockShape, class WarpShape,
    class ElementA, class ElementB,
    class LayerConfig>
struct TCGEN05 {
public:
  using MmaOpClass = MmaOpClass_;
  using MmaShape = typename MmaOpClass::MmaShape;

  static constexpr bool kHasZeroPoint = LayerConfig::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = LayerConfig::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = LayerConfig::kUseFusedE8m0Scale;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kNumWarpShapeNSplits = WarpShape::N == ElementA::kBits * 2 ? 2 : 1;

  // SMEM descriptor swizzle for A and B. Picking B2 from the workbook:
  // both A and B use 128B swizzle so the descriptor math is shared with
  // humming's existing TMA/cp.async A loader. (The r2s in transform_b
  // must produce data in the swizzled layout for this to be correct;
  // see workbook §"Phase B.6.3" for the lane-mapping spec.)
  static constexpr uint32_t kSwizzleBytesA = 128;
  static constexpr uint32_t kSwizzleBytesB = 128;

  // Each tcgen05.mma.kind::f16 consumes 16 bf16 of K per issue. The
  // 128B-swizzle atom is 8 bf16 K-wide, so 16 bf16 = 2 atoms = 2 uint128_t.
  // Per-K-iter SMEM pointer offset (in int4 / uint128_t units).
  static constexpr uint32_t kKChunkUint128 = 2;

  SharedStorage &smem;
  ArithClass &arith;

  // We KEEP the same regs_a layout as WMMA even though tcgen05.mma reads
  // A from SMEM, because humming's existing s2r_pipe unconditionally
  // loads A into RMEM via mma.regs_a_as_ptr(). Wastes registers but
  // avoids forking the s2r path for Phase 0. Profile to confirm it's
  // tolerable; if not, branch s2r_pipe on kUseTcgen05.
  // The number of MmaShape entries here mirrors WMMA's: warp tile divided
  // by the per-warp MmaShape.
  static constexpr uint32_t kFakeMmaShapeM = 16;
  static constexpr uint32_t kFakeMmaShapeK = 16;
  uint32_t regs_a[2][MAX(1u, WarpShape::M / kFakeMmaShapeM)]
                    [MAX(1u, kPartMmaShapeK / kFakeMmaShapeK)];

  // Quantised B codes loaded by s2r_pipe (same layout as WMMA).
  uint32_t regs_qb[2][ElementB::kBits * (16 / ElementA::kBits)];
  // Dequantised B in RMEM, pre-r2s. Sized to the per-thread slice of the
  // BlockN x BlockK B tile (matches the post-dequant footprint of WMMA's
  // regs_b for the same warp shape).
  uint32_t regs_b_tmp[2][WarpShape::N * kPartMmaShapeK * ElementA::kBits / 32 / 32];
  // Final post-t2r RMEM accumulator the epilogue reads. Single buffer
  // (no double-buffer like WMMA needs for register/scale gating).
  // MmaOpClass::CRegisters comes from Tcgen05OpClassImpl codegen and is
  // sized to one warp's slice = warp_M * warp_N / 32 lanes per thread.
  typename MmaOpClass::CRegisters regs_c;

  CUDA_INLINE
  TCGEN05(SharedStorage &smem_, ArithClass &arith_)
      : smem(smem_), arith(arith_) {}

  CUDA_INLINE
  void zero_accum() {
    // tcgen05.mma's `scale_d` predicate handles "first issue overwrites,
    // subsequent issues accumulate". We model that by carrying a runtime
    // flag through `run()`, and zero the RMEM-side regs_c so the
    // epilogue sees zeros if no K-iters fired (unusual but safe).
    uint32_t *p = reinterpret_cast<uint32_t *>(regs_c);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs_c) / 4; i++) p[i] = 0;
    first_issue_ = true;
  }

  // Dequant int4 (from regs_qb) -> bf16 (RMEM) -> SMEM b_dequant staging.
  CUDA_INLINE
  void transform_b(uint32_t buffer_id) {
    // For dtypes where ElementA == ElementB we'd skip dequant; tcgen05
    // bf16xbf16 isn't our target so just emit the int4 path inline.
    static_assert(!std::is_same<ElementA, ElementB>::value,
                  "TCGEN05 path is only wired up for narrow-B (int4) today");

    uint32_t *regs_b_ptr = regs_b_tmp[buffer_id];
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < WarpShape::N / 16; i++) {
      uint4 zp_vals = arith.prepare_zp_for_dequant(buffer_id, i);
      uint32_t *zp_vals_ptr = reinterpret_cast<uint32_t *>(&zp_vals);
      dequant<ElementB, ElementA, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(
          regs_qb[buffer_id], regs_b_ptr, i, zp_vals_ptr);
      arith.may_apply_bs_and_zp_on_b(regs_b_ptr, i, buffer_id);
    }

    // r2s: copy the per-thread dequantised tile back to smem.b_dequant.
    //
    // TODO(sm100): the lane->SMEM mapping below is a placeholder that
    // assigns each thread a contiguous int4 (16 B) slot. The resulting
    // SMEM layout is NOT swizzle-correct for tcgen05.mma -- the kernel
    // will compile and run, but produce wrong output. Phase B.4's job is
    // to either (a) match the tcgen05.mma SMEM swizzle here, or
    // (b) build a custom swizzle descriptor that consumes this layout.
    // Tracked in workbook.md.
    constexpr uint32_t kBytesPerThread = sizeof(regs_b_tmp[0]);
    static_assert(kBytesPerThread >= 4, "r2s tile must be non-empty");
    if constexpr (kBytesPerThread >= 16) {
      // 16B-aligned -- emit int4 stores.
      int4 *smem_b_dst = reinterpret_cast<int4 *>(
          &smem.b_dequant[buffer_id][0]) + threadIdx.x * (kBytesPerThread / 16);
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kBytesPerThread / 16; i++) {
        smem_b_dst[i] = reinterpret_cast<int4 *>(regs_b_ptr)[i];
      }
    } else {
      // Tiny per-thread tile -- fall back to uint32 stores.
      uint32_t *smem_b_dst = reinterpret_cast<uint32_t *>(
          &smem.b_dequant[buffer_id][0]) + threadIdx.x * (kBytesPerThread / 4);
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kBytesPerThread / 4; i++) {
        smem_b_dst[i] = regs_b_ptr[i];
      }
    }

    // Fence so subsequent SMEM reads on the tcgen05 path observe the
    // r2s; the issuing warp will pair this with a __syncthreads().
    fence_proxy_async_shared_cta();
  }

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    uint32_t buffer_id = iter_id % 2;

    // A descriptor reads from smem.a[stage_id]; advance the pointer by
    // `iter_id * kKChunkUint128` so this MMA processes K-chunk `iter_id`.
    // (tcgen05.mma.kind::f16 only sees 16 bf16 of K per issue; the
    // outer mainloop's K-loop drives `iter_id` over the BlockK range.)
    int4 *a_ptr = &smem.a[stage_id][0] + iter_id * kKChunkUint128;
    int4 *b_ptr = &smem.b_dequant[buffer_id][0] + iter_id * kKChunkUint128;

    uint64_t a_desc = tcgen05_smem_desc<kSwizzleBytesA, BlockShape::K>(a_ptr);
    uint64_t b_desc = tcgen05_smem_desc<kSwizzleBytesB, BlockShape::K>(b_ptr);

    uint32_t idesc =
        tcgen05_instr_desc_bf16_bf16_f32(BlockShape::M, BlockShape::N);

    // First issue of a tile: overwrite D (scale_d=false).
    // Subsequent K-iters: accumulate (scale_d=true).
    bool scale_d = !first_issue_;
    first_issue_ = false;

    // tcgen05.mma is .sync.aligned per the PTX spec -- it has to be
    // executed warp-uniformly. Issue from all 32 threads of warp 0; the
    // hardware treats this as one CTA-group issue.
    if (threadIdx.x < 32) {
      tcgen05_mma_ss_bf16(smem.tcgen05_tmem_col, a_desc, b_desc, idesc, scale_d);
    }
  }

  // Run the t2r and expose a plain RMEM accumulator pointer for the
  // epilogue. Called once per output tile, AFTER the K-loop completes.
  //
  // Drains all in-flight tcgen05.mma issues via a single commit; waits
  // on the per-CTA mbarrier; then issues N TMEM->RMEM loads to cover
  // this warp's (warp_M x warp_N) slice of the accumulator tile.
  //
  // Each `tcgen05_ld_32x32b_x32` call reads 32 rows x 32 fp32-cols per
  // warp (= 4 KB). A (warp_M x warp_N) slice needs
  //   (warp_M / 32) x (warp_N / 32) calls per warp.
  // TMEM address encoding: col index in low 16 bits, row index in
  // high 16 bits. The warp's base offset within the per-CTA TMEM column
  // allocation is derived from (m_warp_id, n_warp_id).
  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    // tcgen05.commit is .sync.aligned -- warp-uniform.
    if (threadIdx.x < 32) {
      uint32_t mbar_addr = cast_smem_ptr_to_uint(&smem.tcgen05_mbar);
      tcgen05_commit_to_mbarrier(mbar_addr);
    }
    // All threads wait on the same mbarrier; phase flips after each
    // tile's wait so the next tile's commit re-arms cleanly.
    mbarrier_wait(&smem.tcgen05_mbar, mbar_phase_);
    mbar_phase_ ^= 1u;

    // The mbarrier_wait guarantees all tcgen05.mma stores to TMEM are
    // visible; an explicit fence isn't required by the spec, but keeping
    // it is cheap and defensive while we shake the layout out.
    tcgen05_fence_view_async_tmem_store();

    // Per-warp slice of the (BlockM x BlockN) TMEM accumulator tile.
    // Number of warps that cooperate on the t2r along each dim:
    static constexpr uint32_t kMWarps = MAX(BlockShape::M / WarpShape::M, 1u);
    static constexpr uint32_t kNWarps = MAX(BlockShape::N / WarpShape::N, 1u);
    // Each tcgen05.ld.32x32b.x32 call reads 32 rows x 32 fp32 cols per
    // warp; tile this within the per-warp slice.
    static constexpr uint32_t kCallsM = MAX(WarpShape::M / 32u, 1u);
    static constexpr uint32_t kCallsN = MAX(WarpShape::N / 32u, 1u);
    static_assert(
        sizeof(regs_c) == kCallsM * kCallsN * 32u * sizeof(uint32_t),
        "regs_c footprint mismatch: codegen warp_shape must match "
        "(kCallsM * kCallsN * 32) uint32 per thread per warp");

    uint32_t warp_id = threadIdx.x / 32u;
    uint32_t m_warp_id = warp_id % kMWarps;
    uint32_t n_warp_id = (warp_id / kMWarps) % kNWarps;

    // TMEM addr encoding: low 16 bits = column index (units of 32 bits),
    // high 16 bits = row index. Base is the per-CTA column allocation
    // start written by `tcgen05_alloc`.
    uint32_t base_addr = smem.tcgen05_tmem_col
                       + ((m_warp_id * WarpShape::M) << 16)
                       + (n_warp_id * WarpShape::N);

    uint32_t *regs = reinterpret_cast<uint32_t *>(regs_c);
    PRAGMA_UNROLL
    for (uint32_t im = 0; im < kCallsM; im++) {
      PRAGMA_UNROLL
      for (uint32_t in = 0; in < kCallsN; in++) {
        uint32_t addr = base_addr + ((im * 32u) << 16) + (in * 32u);
        tcgen05_ld_32x32b_x32(addr, regs + (im * kCallsN + in) * 32u);
      }
    }
    __syncthreads();
    return reinterpret_cast<T *>(regs_c);
  }

  // The s2r_pipe reads these accessors (same shape as WMMA's interface).
  template <class T = uint32_t>
  CUDA_INLINE T *regs_a_as_ptr(uint32_t buffer_id) {
    // s2r_pipe expects this. We don't read it back -- tcgen05 reads A
    // from SMEM via descriptor -- but the s2r-side load fills these
    // registers harmlessly.
    return reinterpret_cast<T *>(regs_a[buffer_id]);
  }

  template <class T = uint32_t>
  CUDA_INLINE T *regs_qb_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(regs_qb[buffer_id]);
  }

  template <class T = uint32_t>
  CUDA_INLINE T *regs_b_as_ptr() {
    // TCGEN05 doesn't keep dequantised B in RMEM beyond the transform
    // step; expose the temporary tile so any debug/arith path that
    // touches it during transform still works.
    return reinterpret_cast<T *>(regs_b_tmp);
  }

  template <class T = uint32_t>
  CUDA_INLINE T *regs_c_as_ptr(uint32_t buffer_id = 0) {
    return reinterpret_cast<T *>(regs_c);
  }

private:
  // True until the first tcgen05.mma issue lands, used to drive scale_d.
  bool first_issue_ = true;
  // mbarrier phase parity bit. Flips after each tile's commit/wait pair.
  uint32_t mbar_phase_ = 0;
};


