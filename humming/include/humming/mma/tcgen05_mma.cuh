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

// Debug switches (set to non-zero to enable):
//   TCGEN05_DEBUG_CONST_B: bulk-fill smem.b_dequant with bf16(1.0),
//     bypassing dequant + scatter. Output should be N-independent
//     = sum_k A[m, k]. Used to isolate MMA+t2r+epilogue from dequant.
//   TCGEN05_DEBUG_SKIP_TMEM: skip the t2r and fill scratch with
//     (lane * 1000 + idx). Output reveals (lane, scratch_idx) ->
//     (m, n) layout directly.
//     (Validated 2026-05-18: SMEM-write + gmem_writer chain is
//      correct; bf16 precision around 1000 masks individual values.)
// Both off by default -- only re-enable when investigating regressions.
// #define TCGEN05_DEBUG_CONST_B 1
// #define TCGEN05_DEBUG_SKIP_TMEM 1
// #define TCGEN05_DEBUG_SCATTER_SENTINEL 1
// #define TCGEN05_DEBUG_REGS_B_SENTINEL 1
// #define TCGEN05_DEBUG_NO_SCATTER 1
// #define TCGEN05_DEBUG_TMEM_DUMP 1
// (the regs_qb alignas(16) fix above is the actual production change)


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

  // SMEM descriptor swizzle for A and B.
  //
  // KNOWN MISMATCH (workbook §"Phase B.9"): humming's loader_a writes
  // Swizzle<2,4,3> (col XOR by row & 3) but with a 128-byte row stride.
  // CUTE's canonical 64B (Swizzle<2,4,3>) layout expects a 64-byte row
  // stride (4 uint128_t/row), and the canonical 128B (Swizzle<3,4,3>)
  // expects col-XOR by row & 7. NEITHER matches humming's actual data,
  // so the tcgen05.mma descriptor reads the WRONG A bytes for half the
  // rows. The next step is either to fix humming's loader_a swizzle to
  // match Swizzle<3,4,3>, or to restage A into a tcgen05-private buffer
  // with the canonical layout. Keeping 128 here as the closest fit
  // until that restage lands.
  static constexpr uint32_t kSwizzleBytesA = 128;
  static constexpr uint32_t kSwizzleBytesB = 128;

  // Each tcgen05.mma.kind::f16 consumes 16 bf16 of K per issue. The
  // 128B-swizzle atom is 8 bf16 K-wide, so 16 bf16 = 2 atoms = 2 uint128_t.
  // Per-K-iter SMEM pointer offset (in int4 / uint128_t units).
  static constexpr uint32_t kKChunkUint128 = 2;

  SharedStorage &smem;
  ArithClass &arith;

  // Phase B.20: `s2r_pipeline.cuh` now skips `loader_a.load` for the
  // TCGEN05 path (tcgen05.mma reads A from SMEM via descriptor), so
  // this storage is never written. Keep a single dummy int4 so the
  // `regs_a_as_ptr()` accessor below has somewhere to point -- the
  // s2r pipe takes the pointer unconditionally even when it doesn't
  // dereference. alignas(16) is defensive.
  alignas(16) int4 regs_a[1];

  // Quantised B codes loaded by s2r_pipe (same layout as WMMA).
  alignas(16) uint32_t regs_qb[2][ElementB::kBits * (16 / ElementA::kBits)];
  // Dequantised B in RMEM, pre-r2s. Sized to the per-thread slice of the
  // BlockN x BlockK B tile (matches the post-dequant footprint of WMMA's
  // regs_b for the same warp shape).
  alignas(16) uint32_t regs_b_tmp[2][WarpShape::N * kPartMmaShapeK * ElementA::kBits / 32 / 32];
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

  // Phase B.18 known-good config space (verified by
  // tests/test_tcgen05.py):
  //   * BlockShape::M in {64, 128}
  //   * BlockShape::N in {64, 128, 256}
  //   * BlockShape::K == 64    (one 128B swizzle atom per A-row)
  //   * WarpShape::M == BlockShape::M / 4   (4 M-warps, one per TMEM
  //                                          sub-partition; M=64 atom
  //                                          has 16 valid M per sub-
  //                                          part, M=128 atom has 32)
  //   * WarpShape::N == 64     (loader_b's kIsWarpHalfGroup path
  //                             unmodelled at WarpN < 64)
  //   * kNumStages in {2, 3, 4}
  //   * has_zero_point in {True, False}
  //   * has_bias in {True, False}
  static_assert(BlockShape::M == 64 || BlockShape::M == 128,
                "TCGEN05 path Phase B.18: BlockM must be 64 or 128");
  static_assert(BlockShape::N == 64 || BlockShape::N == 128
                || BlockShape::N == 256,
                "TCGEN05 path Phase B.18: BlockN must be 64, 128, or 256");
  static_assert(WarpShape::N == 64,
                "TCGEN05 path Phase B.18: WarpN<64 hits loader_b "
                "half-group path (unmodelled in scatter)");
  static_assert(BlockShape::K == 64 || BlockShape::K == 128
                || BlockShape::K == 256,
                "TCGEN05 path Phase B.18: BlockK must be 64, 128, or 256");
  static_assert(WarpShape::M * 4 == BlockShape::M,
                "TCGEN05 path Phase B.18: must have exactly 4 M-warps "
                "so each warp owns one TMEM sub-partition's worth of M");
  static_assert(WarpShape::K == BlockShape::K,
                "TCGEN05 path Phase B.18: K-warps not supported -- "
                "tcgen05.mma covers the full BlockK by issuing one MMA "
                "per 16-K-bf16 atom from a single warp.");

  // Dequant int4 (from regs_qb) -> bf16 (RMEM) -> SMEM b_dequant staging.
  CUDA_INLINE
  void transform_b(uint32_t buffer_id) {
    // For dtypes where ElementA == ElementB we'd skip dequant; tcgen05
    // bf16xbf16 isn't our target so just emit the int4 path inline.
    static_assert(!std::is_same<ElementA, ElementB>::value,
                  "TCGEN05 path is only wired up for narrow-B (int4) today");

    // dequant_b1248 writes 4 uint32 starting at the passed pointer per
    // call. Each outer-i call corresponds to a DIFFERENT m16n8 fragment
    // pair in regs_b_tmp, so advance the destination by 4 uint32 per
    // iter (same pattern as wmma.cuh:60 -- previously this was reusing
    // the base pointer and overwriting on every call).
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < WarpShape::N / 16; i++) {
      uint32_t *regs_b_ptr = &regs_b_tmp[buffer_id][i * 4u];
      uint4 zp_vals = arith.prepare_zp_for_dequant(buffer_id, i);
      uint32_t *zp_vals_ptr = reinterpret_cast<uint32_t *>(&zp_vals);
      dequant<ElementB, ElementA, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(
          regs_qb[buffer_id], regs_b_ptr, i, zp_vals_ptr);
      arith.may_apply_bs_and_zp_on_b(regs_b_ptr, i, buffer_id);
    }
#ifdef TCGEN05_DEBUG_REGS_B_SENTINEL
    // Overwrite regs_b_tmp with a per-(reg_index)-derived sentinel so
    // the scatter writes bf16(my_n+1) at (my_n, my_k). If the (n,k)
    // mapping in run() matches the PTX m16n8k16 fragment layout that
    // mma.sync uses (and that my scatter assumes), the production test
    // with this on should show effective_n[col] == col.
    {
      uint32_t t = threadIdx.x % 32u;
      __nv_bfloat16 *regs_b_bf16 =
          reinterpret_cast<__nv_bfloat16 *>(regs_b_tmp[buffer_id]);
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < WarpShape::N / 16u; i++) {
        PRAGMA_UNROLL
        for (uint32_t v = 0; v < 8u; v++) {
          uint32_t frag_id = v / 4u;
          uint32_t v_in_frag = v % 4u;
          uint32_t my_n = i * 16u + 8u * frag_id + (t / 4u);
          regs_b_bf16[i * 8u + v] = __float2bfloat16(float(my_n) + 1.0f);
        }
      }
    }
#endif

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

#ifdef TCGEN05_DEBUG_CONST_B
    // Debug: ALL threads bulk-fill smem.b_dequant[buffer_id] with bf16(1.0),
    // covering the FULL buffer (BlockN * BlockK bf16). Swizzle is
    // irrelevant since the fill is constant. Output should equal
    //   out[m, n] = sum_k A[m, k] * 1 = sum_k A[m, k]
    // N-independent. Lets us isolate MMA + t2r + epilogue from dequant.
    {
      __nv_bfloat16 one = __float2bfloat16(1.0f);
      __nv_bfloat162 one2 = __halves2bfloat162(one, one);
      uint32_t one2_uint = *reinterpret_cast<uint32_t *>(&one2);
      uint32_t *smem_b_u32 = reinterpret_cast<uint32_t *>(
          &smem.b_dequant[buffer_id][0]);
      constexpr uint32_t kTotalU32 =
          BlockShape::N * BlockShape::K / 2;  // bf16 elems / 2 per uint32
      uint32_t t = threadIdx.x;
      PRAGMA_UNROLL
      for (uint32_t i = t; i < kTotalU32; i += blockDim.x) {
        smem_b_u32[i] = one2_uint;
      }
    }
#else
    // ---- r2s of the just-dequantised B tile to swizzled SMEM ----
    //
    // Per PTX ISA 7.0 Table 32 (mma.m16n8k16.f16, B-matrix layout),
    // thread t's 4 b16 of B are at:
    //   v=0: (k = 2*(t%4) + 0, n = t/4)   — first b32 lo
    //   v=1: (k = 2*(t%4) + 1, n = t/4)   — first b32 hi
    //   v=2: (k = 2*(t%4) + 8, n = t/4)   — second b32 lo
    //   v=3: (k = 2*(t%4) + 9, n = t/4)   — second b32 hi
    //
    // humming's `dequant<>` call with j=i fills regs_b_tmp[i*4..i*4+3]
    // (= 4 b32 = 8 b16) which mma.sync interprets as TWO m16n8 instances:
    //   instance 2i (n_base = i*16 + 0): {b0=res[0], b1=res[1]}
    //   instance 2i+1 (n_base = i*16 + 8): {b0=res[2], b1=res[3]}
    //
    // Final (reg_index = i*8 + v in [0, 32)) -> (n, k) mapping:
    //   frag_id   = v / 4                       (0 or 1)
    //   v_in_frag = v % 4
    //   n = i*16 + 8*frag_id + (t / 4)
    //   k = k_base + 2*(t%4) + (v_in_frag & 1) + 8 * (v_in_frag >> 1)
    {
      __nv_bfloat16 *smem_b_bf16 =
          reinterpret_cast<__nv_bfloat16 *>(&smem.b_dequant[buffer_id][0]);
      __nv_bfloat16 *regs_b_bf16 =
          reinterpret_cast<__nv_bfloat16 *>(regs_b_tmp[buffer_id]);
      // For BlockK > 64 we section-major-ise B in SMEM (same as A in
      // loader_a -- each section holds 64 K-bf16 of all N) so the
      // descriptor's `((8, n), 2):((8, SBO_uint128=64), 1)` matches
      // the SMEM layout regardless of total BlockK. Each section's
      // row stride is 128 B (= 64 K-bf16 × 2 B), and we step the
      // descriptor's start address by `section_size = BlockN * 128 B`
      // when crossing section boundaries.
      constexpr uint32_t kKPerSectionB =
          BlockShape::K < 64u ? BlockShape::K : 64u;
      constexpr uint32_t kRowBytes = kKPerSectionB * sizeof(__nv_bfloat16);
      constexpr uint32_t kBSectionSizeBytes = BlockShape::N * kRowBytes;
      uint32_t t = threadIdx.x % 32u;
      uint32_t k_base = iter_id * kPartMmaShapeK;
      constexpr uint32_t kBf16PerCall = 8;
      constexpr uint32_t kCalls = WarpShape::N / 16u;
      // Per-warp N-slice base. With BlockN == WarpShape::N (kNWarps==1)
      // the only warp_id_scatter is 0 and n_base degenerates to 0;
      // this matches the original working single-N-warp scatter. The
      // BlockN > WarpShape::N path is gated by the static_assert
      // above -- the scatter math here is N-fastest (matches loader_b
      // and arith), but t2r and TMEM tile geometry need additional
      // work for multi-N-warp configs (workbook: "Phase B.15 - N>64
      // descriptor + scatter mismatch") and that mode is rejected at
      // build time until that lands.
      constexpr uint32_t kNWarps = MAX(BlockShape::N / WarpShape::N, 1u);
      uint32_t warp_id_local = threadIdx.x / 32u;
      uint32_t n_warp_id_scatter = warp_id_local % kNWarps;
      uint32_t n_base = n_warp_id_scatter * WarpShape::N;
      // Hardware Swizzle<3,4,3> applies to the absolute byte address:
      // the descriptor encodes (smem_base >> 4) in its start_address,
      // and the HW XOR'ing of bits [4, 7) uses bits [7, 10) of the
      // *full* abs byte, so smem_base/128 contributes to the XOR
      // amount and must be included here.
      uint32_t smem_base_div_128 =
          cast_smem_ptr_to_uint(smem_b_bf16) >> 7;
      // Pack 2 adjacent bf16 into one uint32 store: per PTX Table 32
      // (mma.m16n8k16.f16 B fragment), v=2p and v=2p+1 share the same
      // n and k_lo and have k_hi = k_lo + 1 -- so the 2 bf16 land at
      // 2 adjacent SMEM bytes inside the same swizzle column. The XOR
      // phase depends on bits [4..7) of the byte address and 2
      // adjacent bytes differ only in bit 0, so both bf16's of a pair
      // get the same `xor_shift` -- one uint32 store does both. This
      // halves the per-K-iter SMEM store count vs the prior per-bf16
      // loop.
      uint32_t *regs_b_u32_buf =
          reinterpret_cast<uint32_t *>(regs_b_tmp[buffer_id]);
      uint32_t *smem_b_u32 =
          reinterpret_cast<uint32_t *>(&smem.b_dequant[buffer_id][0]);
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kCalls; i++) {
        PRAGMA_UNROLL
        for (uint32_t frag_id = 0; frag_id < 2u; frag_id++) {
          uint32_t n = n_base + i * 16u + 8u * frag_id + (t / 4u);
          PRAGMA_UNROLL
          for (uint32_t pair_idx = 0; pair_idx < 2u; pair_idx++) {
            // pair (v_lo=4*frag_id+2*pair_idx, v_hi=v_lo+1) writes
            // the bf16 pair at (n, k_lo) and (n, k_lo + 1).
            uint32_t k_lo = k_base + 2u * (t % 4u) + 8u * pair_idx;
            uint32_t v_lo = frag_id * 4u + pair_idx * 2u;
            uint32_t reg_index_pair = (i * kBf16PerCall + v_lo) / 2u;
            uint32_t k_section = k_lo / kKPerSectionB;
            uint32_t k_in_section = k_lo % kKPerSectionB;
            uint32_t section_offset_bytes = k_section * kBSectionSizeBytes;
            uint32_t linear_in_section =
                n * kRowBytes + k_in_section * sizeof(__nv_bfloat16);
            uint32_t linear_bytes = section_offset_bytes + linear_in_section;
            uint32_t xor_shift =
                (smem_base_div_128 + (linear_in_section >> 7)) & 7u;
            uint32_t swizzled = linear_bytes ^ (xor_shift << 4);
#ifdef TCGEN05_DEBUG_SCATTER_SENTINEL
            __nv_bfloat16 lo_bf16 = __float2bfloat16(float(n) + 1.0f);
            __nv_bfloat16 hi_bf16 = __float2bfloat16(float(n) + 1.0f);
            uint32_t packed =
                (static_cast<uint32_t>(
                     *reinterpret_cast<uint16_t *>(&hi_bf16)) << 16) |
                static_cast<uint32_t>(
                    *reinterpret_cast<uint16_t *>(&lo_bf16));
            smem_b_u32[swizzled / sizeof(uint32_t)] = packed;
#else
            smem_b_u32[swizzled / sizeof(uint32_t)] = regs_b_u32_buf[reg_index_pair];
#endif
          }
        }
      }
    }
#endif
    // The scatter above uses regular SMEM stores (non-async), so the
    // implicit __threadfence_block from __syncthreads is sufficient
    // to make them visible to subsequent tcgen05.mma SMEM reads. The
    // earlier explicit `fence_proxy_async_shared_cta()` was a holdover
    // from when we expected to use async copies for the scatter.
    __syncthreads();

    // ---- now build descriptors + issue tcgen05.mma ----
    // A descriptor reads from smem.a[stage_id]; advance the pointer by
    // `iter_id * kKChunkUint128` so this MMA processes K-chunk `iter_id`.
    // (tcgen05.mma.kind::f16 only sees 16 bf16 of K per issue; the
    // outer mainloop's K-loop drives `iter_id` over the BlockK range.)
    //
    // For BlockK > 64, humming's loader_a sectionises A into chunks of
    // 64 K-bf16 each (loader_a.cuh:110: `gmem_col = smem_row /
    // BlockM * 8 + smem_col` -- rows 0..BlockM-1 hold K=0..63,
    // BlockM..2*BlockM-1 hold K=64..127, ...). The descriptor's
    // 8-M-row group stride is 1024 B = 64 uint128 (= 64 K-bf16 worth)
    // *within* a section regardless of BlockK, so SBO for A is always
    // `kKPerSection` = MIN(BlockK, 64). To advance the descriptor
    // start across section boundaries we jump by the section size
    // (`BlockM * 128 B = BlockM * 8` uint128) instead of by atoms.
    constexpr uint32_t kKPerSection = BlockShape::K < 64u ? BlockShape::K : 64u;
    constexpr uint32_t kKItersPerSection = kKPerSection / 16u;
    constexpr uint32_t kSectionSizeUint128 = BlockShape::M * 8u;
    uint32_t section_idx = iter_id / kKItersPerSection;
    uint32_t iter_in_section = iter_id % kKItersPerSection;
    int4 *a_ptr = &smem.a[stage_id][0]
                  + section_idx * kSectionSizeUint128
                  + iter_in_section * kKChunkUint128;
    // B is now sectionised the same way A is (since Phase B.19): the
    // scatter above writes section-major, with each section holding
    // 64 K-bf16 of all N. So B's descriptor SBO is also fixed at 64
    // K-bf16, and the iter advance crosses sections via `section_idx
    // * kBSectionSizeUint128` (where the B section size in uint128
    // is `BlockN * 8`).
    constexpr uint32_t kBSectionSizeUint128 = BlockShape::N * 8u;
    int4 *b_ptr = &smem.b_dequant[buffer_id][0]
                  + section_idx * kBSectionSizeUint128
                  + iter_in_section * kKChunkUint128;

    uint64_t a_desc = tcgen05_smem_desc<kSwizzleBytesA, kKPerSection>(a_ptr);
    uint64_t b_desc = tcgen05_smem_desc<kSwizzleBytesB, kKPerSection>(b_ptr);

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

  // Run the t2r and write the result directly into `smem.reduce` in
  // the layout that humming's `gmem_writer::write_legacy` expects,
  // bypassing the existing `smem_writer` (which assumes ≥2 N-warps
  // covering BlockN -- a constraint TCGEN05 violates by design).
  //
  // gmem_writer.cuh:103 reads smem.reduce as a row-major `int4 tile
  // [BlockM][BlockN / 8]` with this XOR swizzle on the int4 col:
  //   swizzled_int4_col = int4_col ^ ((row + smem_base) % 8)
  // Each int4 holds 8 bf16 (one 8-wide N-strip of a row).
  //
  // The caller (EpiloguePipeline::call) must skip `smem_writer.write`
  // for TCGEN05 -- the SMEM is already filled by the time we return.
  // We return `nullptr` as a sentinel so the dispatcher can assert.
  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    // ---- 1. Drain MMAs via commit + mbarrier wait ----
    if (threadIdx.x < 32 && tcgen05_elect_one_sync()) {
      uint32_t mbar_addr = cast_smem_ptr_to_uint(&smem.tcgen05_mbar);
      tcgen05_commit_to_mbarrier(mbar_addr);
    }
    mbarrier_wait(&smem.tcgen05_mbar, mbar_phase_);
    mbar_phase_ ^= 1u;
    tcgen05_fence_view_async_tmem_store();

    // ---- 2. tcgen05.ld -> per-thread scratch (row-per-thread) ----
    //
    // M=64 cta_group::1 TMEM atom (per mma_traits_sm100.hpp:507):
    //   Shape ((16, 4), N_MMA), Stride ((1, 32), 128)
    // Valid M values are at DPs {0..15, 32..47, 64..79, 96..111}, spanning
    // 4 TMEM sub-partitions of 32 DPs each. Per PTX spec, a warp can only
    // access DPs in its own sub-partition, so we need 4 warps -- each
    // reading its sub-partition's first 16 DPs (= 16 valid M values).
    // Lanes 16..31 of each warp see garbage at warp-local DPs 16..31 and
    // skip the write.
    static constexpr uint32_t kMWarps = MAX(BlockShape::M / WarpShape::M, 1u);
    static constexpr uint32_t kNWarps = MAX(BlockShape::N / WarpShape::N, 1u);
    static constexpr uint32_t kCallsN = MAX(WarpShape::N / 32u, 1u);
    static_assert(WarpShape::M == 16 || WarpShape::M == 32,
                  "TCGEN05 path requires WarpShape::M to be 16 (M=64 "
                  "atom, 16 valid M per sub-partition) or 32 (M=128 "
                  "atom, 32 valid M per sub-partition).");

    uint32_t warp_id = threadIdx.x / 32u;
    // TMEM access is sub-partition-bound: warp `w` can ONLY read
    // DPs (w % 4). The TMEM atom places M=0..15 in sub-part 0,
    // M=16..31 in sub-part 1, M=32..47 in sub-part 2, M=48..63
    // in sub-part 3 (per CUTE mma_traits_sm100.hpp:507). So the
    // M dim MUST be fastest in warp_id, regardless of how the s2r
    // loader_b assigns N -- the loader and t2r operate on different
    // SMEM buffers (smem.b for s2r, TMEM for t2r), so they can use
    // different warp layouts.
    uint32_t m_warp_id = warp_id % kMWarps;
    uint32_t n_warp_id = (warp_id / kMWarps) % kNWarps;
    uint32_t laneid = threadIdx.x % 32u;

    // Per-warp implicit sub-partition base (lane->DP binding is HW-fixed).
    // The taddr's DP field is warp-local: DP=0 = the warp's first DP.
    uint32_t base_addr = smem.tcgen05_tmem_col + (n_warp_id * WarpShape::N);

    // ---- 3. Per-warp t2r + pack + SMEM write ----
    int4 *smem_reduce = smem.reduce;
    uint32_t smem_reduce_base = cast_smem_ptr_to_uint(smem_reduce) / 128u;
    constexpr uint32_t kBlockN = BlockShape::N;
    constexpr uint32_t kInt4ColsPerRow = kBlockN / 8u;
    PRAGMA_UNROLL
    for (uint32_t ni = 0; ni < kCallsN; ni++) {
      uint32_t tmp[32];
#ifdef TCGEN05_DEBUG_SKIP_TMEM
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 32u; i++) {
        float val = float(threadIdx.x) * 1000.0f + float(ni * 32u + i);
        tmp[i] = *reinterpret_cast<uint32_t *>(&val);
      }
#else
      uint32_t addr = base_addr + ni * 32u;
      tcgen05_ld_32x32b_x32(addr, tmp);
      tcgen05_fence_view_async_tmem_store();
#ifdef TCGEN05_DEBUG_TMEM_DUMP
      // Print tmp[0..3] (= C[M=laneid, N=n_base+ni*32 + 0..3]) for lane
      // 0 of each warp. With A=delta(k=0) + SCATTER_SENTINEL, C[M=lane,
      // N=n] should equal sentinel(n) = n+1. If TMEM has wrong values
      // for n_warp_id==1, MMA's B-read is wrong; if TMEM is right but
      // smem.reduce ends up wrong, the t2r-to-smem mapping is wrong.
      if (laneid == 0 && blockIdx.x == 0 && blockIdx.y == 0
          && m_warp_id == 0) {
        uint32_t n0 = n_warp_id * WarpShape::N + ni * 32u;
        float *t_f = reinterpret_cast<float *>(tmp);
        printf("t2r warp=%u n_warp=%u ni=%u n_base=%u "
               "tmp[0..3]=%.1f,%.1f,%.1f,%.1f tmp[28..31]=%.1f,%.1f,%.1f,%.1f\n",
               warp_id, n_warp_id, ni, n0,
               t_f[0], t_f[1], t_f[2], t_f[3],
               t_f[28], t_f[29], t_f[30], t_f[31]);
      }
#endif
#endif
      // For M=64 atom, only the FIRST 16 DPs per sub-partition hold
      // valid M values (DPs 16..31 are uninitialised); for M=128 atom
      // ALL 32 DPs are valid. Gate the SMEM write by `laneid <
      // WarpShape::M` so the same code handles both cases.
      if (laneid < WarpShape::M) {
        uint32_t m_full = (m_warp_id * WarpShape::M) + laneid;
        uint32_t col_int4_base = (n_warp_id * WarpShape::N + ni * 32u) / 8u;
        PRAGMA_UNROLL
        for (uint32_t int4_in_quarter = 0; int4_in_quarter < 4u;
             int4_in_quarter++) {
          int4 packed;
          uint32_t *packed_u32 = reinterpret_cast<uint32_t *>(&packed);
          float *src_fp32 =
              reinterpret_cast<float *>(tmp + int4_in_quarter * 8u);
          // Per-N base for the 8 bf16 we're about to pack into one
          // int4. With LayerConfig::kHasBias, smem.bias holds
          // BlockN bf16 values laid out linearly (bias[n] = bias for
          // output column n); add it to C before the f32->bf16 cast.
          uint32_t int4_col_global_for_n =
              (n_warp_id * WarpShape::N + ni * 32u) / 8u
              + int4_in_quarter;
          uint32_t n_base_pack = int4_col_global_for_n * 8u;
          PRAGMA_UNROLL
          for (uint32_t pair = 0; pair < 4u; pair++) {
            float f0 = src_fp32[pair * 2u + 0u];
            float f1 = src_fp32[pair * 2u + 1u];
            if constexpr (LayerConfig::kHasBias) {
              const __nv_bfloat16 *smem_bias_bf16 =
                  reinterpret_cast<const __nv_bfloat16 *>(&smem.bias[0]);
              uint32_t n0 = n_base_pack + pair * 2u;
              f0 += __bfloat162float(smem_bias_bf16[n0]);
              f1 += __bfloat162float(smem_bias_bf16[n0 + 1u]);
            }
            __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
            packed_u32[pair] = *reinterpret_cast<uint32_t *>(&v);
          }
          // gmem_writer.cuh:100-108 treats smem.reduce as 8-int4-wide
          // rows -- the "smem_row" coord is `gmem_row + (gmem_col / 8) *
          // BlockM`, and the "smem_col" is `gmem_col % 8`. For BlockN<=
          // 64 the high-section is empty and `m_full * kInt4ColsPerRow +
          // int4_col` happened to collapse to the same offset; for
          // BlockN > 64 we MUST split high-N int4 cols (8..15, 16..23,
          // ...) into separate "rows" at offset (section_idx * BlockM
          // + m_full) * 8 + section_col. Apply the gmem_writer XOR
          // swizzle on the (smem_row, smem_col) coord, not on int4_col.
          uint32_t int4_col_global = col_int4_base + int4_in_quarter;
          uint32_t section_idx = int4_col_global / 8u;        // gmem_col / 8
          uint32_t section_col = int4_col_global % 8u;        // gmem_col % 8
          uint32_t smem_row = section_idx * BlockShape::M + m_full;
          uint32_t row_xor = (smem_row + smem_reduce_base) % 8u;
          uint32_t swizzled_col = section_col ^ row_xor;
          smem_reduce[smem_row * 8u + swizzled_col] = packed;
        }
      }
    }
    __syncthreads();
    return nullptr;
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


