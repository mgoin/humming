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

    // We defer the r2s of dequantised bf16 to TCGEN05::run(iter_id),
    // because the r2s needs the K-iter index to compute SMEM offsets
    // (one K-chunk of 16 bf16 per K-iter) and `transform_b` only
    // receives buffer_id from humming's mainloop. The dequant results
    // stay in `regs_b_tmp[buffer_id]` until run() consumes them.
    fence_proxy_async_shared_cta();
  }

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    uint32_t buffer_id = iter_id % 2;

    // ---- r2s of the just-dequantised B tile to swizzled SMEM ----
    // transform_b left bf16 in `regs_b_tmp[buffer_id]` in mma.sync
    // m16n8k16 B-fragment layout. We now scatter to smem.b_dequant
    // at logical (n, k) positions matching the tcgen05.mma descriptor
    // (128B-swizzled, K-major).
    //
    // Per-thread fragment layout (from CUTLASS mma_traits_sm80.hpp:89,
    // BLayout = `((4,8),(2,2)):((16,1),(8,64))`):
    //   v=0: (n = t%4 + frag_n_base + 0, k = t/4 + 0)
    //   v=1: (n = t%4 + frag_n_base + 0, k = t/4 + 8)
    //   v=2: (n = t%4 + frag_n_base + 4, k = t/4 + 0)
    //   v=3: (n = t%4 + frag_n_base + 4, k = t/4 + 8)
    // humming's `dequant<>` call with j=i covers TWO consecutive
    // n-frags of 8N each (= 16 N) starting at n = i*16. So per dequant
    // call the 8 outputs span:
    //   bf16[0..3]: frag_n_base = i*16     (first 8 N)
    //   bf16[4..7]: frag_n_base = i*16+8   (next 8 N)
    {
      __nv_bfloat16 *smem_b_bf16 =
          reinterpret_cast<__nv_bfloat16 *>(&smem.b_dequant[buffer_id][0]);
      __nv_bfloat16 *regs_b_bf16 =
          reinterpret_cast<__nv_bfloat16 *>(regs_b_tmp[buffer_id]);
      // Row stride of the (BlockN, BlockK) bf16 tile, in bytes.
      constexpr uint32_t kRowBytes = BlockShape::K * sizeof(__nv_bfloat16);
      uint32_t t = threadIdx.x % 32u;
      uint32_t k_base = iter_id * kPartMmaShapeK;  // global K-base for this iter
      constexpr uint32_t kBf16PerCall = 8;
      constexpr uint32_t kCalls = WarpShape::N / 16u;
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kCalls; i++) {
        PRAGMA_UNROLL
        for (uint32_t v = 0; v < kBf16PerCall; v++) {
          uint32_t frag_id = v / 4u;
          uint32_t v_in_frag = v % 4u;
          uint32_t n =
              i * 16u + frag_id * 8u + (t % 4u) + (v_in_frag / 2u) * 4u;
          uint32_t k = k_base + (t / 4u) + (v_in_frag % 2u) * 8u;
          uint32_t linear_bytes = n * kRowBytes + k * sizeof(__nv_bfloat16);
          // Swizzle<3,4,3>: XOR bits [7,10) of the linear address into
          // the atom-within-row position (bits [4,7)).
          uint32_t xor_mask = ((linear_bytes >> 7) & 0x7u) << 4;
          uint32_t swizzled = linear_bytes ^ xor_mask;
          uint32_t reg_index = i * kBf16PerCall + v;
          smem_b_bf16[swizzled / sizeof(__nv_bfloat16)] = regs_b_bf16[reg_index];
        }
      }
    }
    fence_proxy_async_shared_cta();
    __syncthreads();

    // ---- now build descriptors + issue tcgen05.mma ----
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

    // tcgen05.mma per CUTLASS pattern (cute/arch/mma_sm100_umma.hpp:65):
    // ONE elected thread of warp 0 issues. Other threads wait at branch
    // reconvergence. tcgen05.mma is NOT .sync.aligned (unlike alloc/
    // dealloc which require warp-uniform participation), so this is
    // safe and matches CUTLASS exactly.
    if (threadIdx.x < 32 && tcgen05_elect_one_sync()) {
      tcgen05_mma_ss_bf16(smem.tcgen05_tmem_col, a_desc, b_desc, idesc, scale_d);
    }
  }

  // Run the t2r and expose a plain RMEM accumulator pointer for the
  // epilogue. Called once per output tile, AFTER the K-loop completes.
  //
  // Step 1: drain MMAs via tcgen05.commit + mbarrier wait.
  // Step 2: tcgen05.ld.32x32b.x32 delivers TMEM -> RMEM, BUT in a
  //   "row-per-thread" layout that DOES NOT match what humming's
  //   existing smem_writer (mma.sync epilogue) expects.  After the
  //   t2r, thread t holds row t of each 32x32 quarter-tile.
  // Step 3: Round-trip through smem.b_dequant (no longer needed for
  //   B staging post-mainloop, and conveniently sized at exactly
  //   BlockM*BlockN*sizeof(fp32) bytes for the common shapes) to
  //   convert the layout into the m16n8 C-fragment-per-thread layout
  //   the smem_writer reads. This keeps the rest of the epilogue
  //   (smem_writer + gmem_writer) untouched.
  //
  // m16n8 C-fragment layout the smem_writer expects: thread t holds
  // 4 fp32 per (m_8x8, n_8x8) sub-tile at positions
  //   (m = 8*m_8x8 + t/4, n = 4*n_8x8 + (t%4)*2 + {0,1})
  //   plus the second-row pair which the writer derives separately.
  // Concretely the writer reads `regs_c` as a flat float2 array
  // indexed by `n_8x8 * 8 + m_8x8`, with each float2 holding the two
  // consecutive-N values at that thread's (row, col).
  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    // tcgen05.commit uses elect_one_sync pattern (one thread issues).
    if (threadIdx.x < 32 && tcgen05_elect_one_sync()) {
      uint32_t mbar_addr = cast_smem_ptr_to_uint(&smem.tcgen05_mbar);
      tcgen05_commit_to_mbarrier(mbar_addr);
    }
    // All threads wait on the mbar; phase flips after each tile.
    mbarrier_wait(&smem.tcgen05_mbar, mbar_phase_);
    mbar_phase_ ^= 1u;
    tcgen05_fence_view_async_tmem_store();

    // Per-warp slice of the (BlockM x BlockN) TMEM accumulator tile.
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

    // Load TMEM into a per-thread scratch buffer (still in row-per-thread
    // layout). Cannot land directly in regs_c, which we'll repopulate
    // in m16n8 layout after the SMEM round-trip.
    uint32_t scratch[kCallsM * kCallsN * 32u];
    PRAGMA_UNROLL
    for (uint32_t im = 0; im < kCallsM; im++) {
      PRAGMA_UNROLL
      for (uint32_t in = 0; in < kCallsN; in++) {
        uint32_t addr = base_addr + ((im * 32u) << 16) + (in * 32u);
        tcgen05_ld_32x32b_x32(addr, scratch + (im * kCallsN + in) * 32u);
      }
    }
    tcgen05_fence_view_async_tmem_store();

    // Reinterpret smem.b_dequant as a flat fp32 [BlockM * BlockN] scratch.
    // Layout: smem_fp32[m * BlockN + n].
    constexpr uint32_t kFp32PerInt4 = 16u / 4u;  // 4 fp32 per int4
    static_assert(
        BlockShape::M * BlockShape::N * sizeof(float) <=
        SharedStorage::kStageSizeBDequant * sizeof(int4) * SharedStorage::kNumStages,
        "TCGEN05 epilogue layout-convert needs at least BlockM*BlockN fp32 "
        "of scratch SMEM; smem.b_dequant is undersized for this shape");
    float *smem_fp32 = reinterpret_cast<float *>(&smem.b_dequant[0][0]);

    // Step 3a: write each thread's row-per-thread fp32 to smem in
    // pure row-major (BlockM, BlockN) layout.
    uint32_t laneid = threadIdx.x % 32u;
    constexpr uint32_t kRowsPerQuarter = 32u;
    constexpr uint32_t kColsPerQuarter = 32u;
    constexpr uint32_t kBlockN = BlockShape::N;
    PRAGMA_UNROLL
    for (uint32_t im = 0; im < kCallsM; im++) {
      uint32_t row_base_in_warp = im * kRowsPerQuarter;
      uint32_t m_full = (m_warp_id * WarpShape::M) + row_base_in_warp + laneid;
      PRAGMA_UNROLL
      for (uint32_t in = 0; in < kCallsN; in++) {
        uint32_t col_base_in_warp = in * kColsPerQuarter;
        uint32_t *src = scratch + (im * kCallsN + in) * 32u;
        PRAGMA_UNROLL
        for (uint32_t c = 0; c < kColsPerQuarter; c++) {
          uint32_t n_full = (n_warp_id * WarpShape::N) + col_base_in_warp + c;
          smem_fp32[m_full * kBlockN + n_full] = *reinterpret_cast<float *>(&src[c]);
        }
      }
    }
    __syncthreads();

    // Step 3b: read back into regs_c in the m16n8 C-fragment layout
    // that the existing smem_writer expects.
    //
    // The writer reads `regs[0][0]` as a float2[64] indexed by
    // `n_8x8 * 8 + m_8x8`. Each float2 holds (output[row, col],
    // output[row, col+1]) for this thread, where:
    //   row = 8 * m_8x8 + (laneid / 4)
    //   col = 4 * n_8x8 + (laneid % 4) * 2
    constexpr uint32_t kInnerM = 8u;  // = MmaShape::M / 8 = BlockM / 8
    constexpr uint32_t kInnerN = 8u;  // 64 float2 / kInnerM
    static_assert(BlockShape::M / 8u == kInnerM,
                  "TCGEN05 layout-convert assumes BlockM=64; "
                  "generalise the 8x8 sub-tile indexing for other shapes");
    static_assert(BlockShape::N / 4u / 2u == kInnerN,
                  "TCGEN05 layout-convert assumes BlockN=64; "
                  "generalise the 4-col / 2-pair indexing for other shapes");
    float *regs_fp32 = reinterpret_cast<float *>(regs_c);
    PRAGMA_UNROLL
    for (uint32_t n_8x8 = 0; n_8x8 < kInnerN; n_8x8++) {
      PRAGMA_UNROLL
      for (uint32_t m_8x8 = 0; m_8x8 < kInnerM; m_8x8++) {
        uint32_t m_full = 8u * m_8x8 + (laneid / 4u);
        uint32_t n_full = 4u * n_8x8 + (laneid % 4u) * 2u;
        uint32_t reg_idx = (n_8x8 * kInnerM + m_8x8) * 2u;
        regs_fp32[reg_idx + 0] = smem_fp32[m_full * kBlockN + n_full + 0];
        regs_fp32[reg_idx + 1] = smem_fp32[m_full * kBlockN + n_full + 1];
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


