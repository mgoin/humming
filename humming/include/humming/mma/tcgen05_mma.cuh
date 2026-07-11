#pragma once
//
// TCGEN05 MMA class for Blackwell sm_100+ (1-CTA, SS-mode mainloop).
//
// Mirrors the WMMA / WGMMA interface (`zero_accum`, `transform_b`,
// `run`, `final_regs_c_as_ptr`) so the kernel mainloop drives it via
// the `MmaSelector<MmaType::TCGEN05, ...>` specialization in
// `mma/all.cuh`.
//
// Data flow (differences vs WMMA):
//   * A operand comes from SMEM (`smem.stages[stage].a`) via an SMEM
//     descriptor, not RMEM. `s2r_pipe` skips the A load for the
//     TCGEN05 path (see `s2r_pipeline.cuh`).
//   * B operand: `s2r_pipe` loads the *quantised* int4 codes into
//     `regs_qb`, `transform_b()` dequantises into RMEM bf16, and the
//     scatter inside `run()` writes them to `smem.b_dequant[buf_id]`.
//     tcgen05.mma reads that SMEM staging buffer.
//   * Accumulator lives in TMEM (one CTA-private allocation of 128
//     cols, made once at kernel entry).
//   * `final_regs_c_as_ptr` does the t2r dance (tcgen05.fence +
//     tcgen05.ld_32x32b_x32) and writes the result directly into
//     `smem.reduce` in the layout the `gmem_writer` expects, bypassing
//     `EpilogueSmemWriter` (which assumes >= 2 N-warps).
//
// Limitations of this mainloop (see `docs/tcgen05_ts_mode_path2.md`
// for the planned next-gen TS-mode mainloop that lifts these):
//   * 1-CTA only (`cta_group::1`). No clustering, no 2-CTA mma.
//   * No accumulator double-buffering -- one TMEM region per CTA.
//   * Per-K-iter `bar.sync 1, kNumMathThreads` is required to publish
//     the scatter to the tcgen05.mma issuer. Empirically the bar
//     itself is cheap (~4 cyc) but the underlying SMEM scatter is
//     the per-K-iter bottleneck (~2000 cyc).

#include <humming/arith/exp_offset.cuh>
#include <humming/utils/all.cuh>
#include <humming/utils/ptx/barrier.cuh>
#include <humming/utils/ptx/shared.cuh>
#include <humming/utils/ptx/tcgen05.cuh>

// Debug switches (set to non-zero to enable):
//   TCGEN05_DEBUG_CONST_B: bulk-fill smem.b_dequant with bf16(1.0),
//     bypassing dequant + scatter. Output should be N-independent
//     = sum_k A[m, k]. Used to isolate MMA + t2r + epilogue from
//     dequant.
//   TCGEN05_DEBUG_SKIP_TMEM: skip the t2r and fill scratch with
//     (lane * 1000 + idx). Reveals the (lane, scratch_idx) ->
//     (m, n) layout directly.
//   TCGEN05_DEBUG_SCATTER_SENTINEL / _REGS_B_SENTINEL: write
//     position-encoded sentinel values from the scatter / dequant
//     so the gmem output can be inverted to (n, k) coordinates.
//   TCGEN05_DEBUG_NO_SCATTER / _TMEM_DUMP: variants that turn off
//     the dequant scatter or surface TMEM raw values respectively.
// All off by default -- enable only when investigating regressions.
// #define TCGEN05_DEBUG_CONST_B 1
// #define TCGEN05_DEBUG_SKIP_TMEM 1
// #define TCGEN05_DEBUG_SCATTER_SENTINEL 1
// #define TCGEN05_DEBUG_REGS_B_SENTINEL 1
// #define TCGEN05_DEBUG_NO_SCATTER 1
// #define TCGEN05_DEBUG_TMEM_DUMP 1


// fence_proxy.async.shared::cta -- ensures prior r2s of dequantised B
// is observable by subsequent tcgen05.mma SMEM reads. Same primitive
// wgmma uses; defined here so this file is self-contained.
CUDA_INLINE void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

#ifndef HUMMING_TCGEN05_ACC_STAGES
#define HUMMING_TCGEN05_ACC_STAGES 1
#endif


template <class Ctx, class ArithClass>
struct TCGEN05 {
public:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using MmaShape = typename Ctx::MmaShape;
  using SharedStorage = typename Ctx::SharedStorage;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;
  using CRegistersType = typename MmaOpClass::CRegisters;
  // Instantiated by EpilogueSmemReducer / EpilogueSmemWriter, but never
  // used on the tcgen05 path (K_WARPS == 1 so reduce() is dead, and the
  // smem_writer is bypassed via the nullptr sentinel below). Any
  // well-formed shape works.
  using CRegistersArrayType = CRegistersType[1][1];

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kNumWarpShapeNSplits = WarpShape::N == ElementA::kBits * 2 ? 2 : 1;

  // SMEM descriptor swizzle for A and B. Both use Swizzle<3,4,3>
  // (128-byte swizzle, col XOR by row & 7). `loader_a` emits this
  // layout when `kUseTcgen05` is set (see `loader_a.cuh`); the
  // scatter in `run()` writes B in the same swizzle below.
  static constexpr uint32_t kSwizzleBytesA = 128;
  static constexpr uint32_t kSwizzleBytesB = 128;

  // Each tcgen05.mma.kind::f16 consumes 16 bf16 of K per issue. The
  // 128B-swizzle atom is 8 bf16 K-wide, so 16 bf16 = 2 atoms = 2 uint128_t.
  // Per-K-iter SMEM pointer offset (in int4 / uint128_t units).
  static constexpr uint32_t kKChunkUint128 = 2;

  Ctx &ctx;
  SharedStorage &smem;
  ArithClass &arith;

  // `s2r_pipeline.cuh` skips `loader_a.load` for the TCGEN05 path
  // (tcgen05.mma reads A from SMEM via the descriptor), so this
  // storage is never written. Keep a single dummy int4 so the
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
  TCGEN05(Ctx &ctx_, ArithClass &arith_)
      : ctx(ctx_), smem(ctx_.smem), arith(arith_) {}

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

  // TMEM accumulator multi-staging (kAccStages == 2): the WS kernel
  // rotates the active accumulator buffer per output tile so tile i's
  // drain overlaps tile i+1's producer loads / MMA tail. Buffer b's
  // accumulator columns start at `tmem_col + b * BlockN` (B.33-repro
  // measured per-tile column alternation at <= 1.2% cost vs static).
  static constexpr uint32_t kAccStages = HUMMING_TCGEN05_ACC_STAGES;
  static_assert(kAccStages == 1 || kAccStages == 2,
                "TCGEN05: only 1 or 2 TMEM accumulator stages supported");
  static_assert(kAccStages == 1 || BlockShape::N <= 128,
                "TCGEN05: 2 accumulator stages need 2 * BlockN <= 256 "
                "TMEM columns (alloc<256> at kernel entry)");
  static_assert(kAccStages == 1 || !Ctx::kHasBias,
                "TCGEN05: deferred drain reads smem.bias, which the "
                "producer may have already overwritten with the next "
                "tile's bias -- bias unsupported with kAccStages > 1");
  // load_channel() refreshes the epilogue's channel-scale registers per
  // tile, and indexed/grouped gemms refresh smem rd/wr_row_index per
  // tile -- a deferred drain of tile i would consume tile i+1's values.
  static_assert(kAccStages == 1 || Ctx::kIsDenseGemm,
                "TCGEN05: kAccStages > 1 requires dense gemm (deferred "
                "drain would read the next tile's smem row indices / "
                "expert state)");
  static_assert(kAccStages == 1 || !Ctx::kIsChannelWeightScale,
                "TCGEN05: kAccStages > 1 incompatible with channel "
                "weight scale (per-tile epilogue registers)");
  static_assert(kAccStages == 1 || !(Ctx::ElementA::kBits != 16 &&
                                     Ctx::kInputScaleGroupSize == 0),
                "TCGEN05: kAccStages > 1 incompatible with channel "
                "input scale (per-tile epilogue registers)");

  CUDA_INLINE
  void set_accum_buf(uint32_t buf) {
    if constexpr (kAccStages > 1) acc_buf_ = buf;
  }

  // Compile-time-zero accessors for the single-stage build. Keeping
  // every member access scalar (NO runtime-indexed member arrays --
  // see mbar_phase_bits_) is what keeps the whole TCGEN05 object
  // SSA-promotable; a dynamically indexed member array demotes the
  // object to a local-memory stack frame (measured: STACK 16 -> 1104 B
  // and a 4x kernel slowdown, 2781 -> 11210 us Llama70B-down M=2048).
  CUDA_INLINE uint32_t accum_buf() const {
    return kAccStages > 1 ? acc_buf_ : 0u;
  }
  CUDA_INLINE uint32_t accum_col_off() const {
    return kAccStages > 1 ? acc_buf_ * BlockShape::N : 0u;
  }

  // Supported config space (verified by tests/test_tcgen05.py and
  // tests/test_tcgen05_dtypes.py):
  //   * BlockShape::M in {64, 128}
  //   * BlockShape::N in {64, 128, 256}
  //   * BlockShape::K in {64, 128, 256}  (B is section-major-ised
  //                                       across 64-K-bf16 sections)
  //   * WarpShape::M == BlockShape::M / 4   (4 M-warps, one per TMEM
  //                                          sub-partition)
  //   * WarpShape::N == 64    (smaller WarpN hits loader_b's
  //                            half-group path, unmodelled here)
  //   * WarpShape::K == BlockShape::K   (no K-warps; tcgen05.mma
  //                                      covers full BlockK by
  //                                      issuing one MMA per 16-K
  //                                      atom from a single warp)
  //   * kNumStages in {2, 3, 4}
  //   * has_zero_point, has_bias both in {True, False}
  //   * ElementA == BFloat16 only (fp16 A would need a parallel
  //                                instruction-descriptor + scatter)
  static_assert(BlockShape::M == 64 || BlockShape::M == 128,
                "TCGEN05: BlockM must be 64 or 128");
  static_assert(BlockShape::N == 64 || BlockShape::N == 128
                || BlockShape::N == 256,
                "TCGEN05: BlockN must be 64, 128, or 256");
  static_assert(WarpShape::N == 64,
                "TCGEN05: WarpN < 64 hits loader_b's half-group path "
                "(unmodelled in the dequant scatter)");
  static_assert(BlockShape::K == 64 || BlockShape::K == 128
                || BlockShape::K == 256,
                "TCGEN05: BlockK must be 64, 128, or 256");
  static_assert(WarpShape::M * 4 == BlockShape::M,
                "TCGEN05: must have exactly 4 M-warps so each warp "
                "owns one TMEM sub-partition's worth of M");
  static_assert(WarpShape::K == BlockShape::K,
                "TCGEN05: K-warps not supported -- tcgen05.mma "
                "covers the full BlockK by issuing one MMA per "
                "16-K-bf16 atom from a single warp");
  // `tcgen05.mma.kind::f16` is issued via `tcgen05_mma_ss_bf16` with
  // `tcgen05_instr_desc_bf16_bf16_f32` -- both hardcoded to bf16.
  // fp16 A requires a parallel instruction-descriptor + scatter (the
  // scatter must match fp16's SMEM-bit semantics, not bf16's). Without
  // that, the bf16-shaped instruction would reinterpret fp16 bit
  // patterns and produce garbage (error magnitudes ~1e16).
  static_assert(std::is_same<ElementA, BFloat16>::value,
                "TCGEN05: ElementA must be BFloat16. fp16 A requires "
                "a parallel instruction-descriptor + scatter path "
                "that is not wired up.");
  // With reduce_overlap_last_stage_only, `smem.reduce` overlays the last
  // stage AND everything after it -- including `b_dequant`. The t2r in
  // final_regs_c_as_ptr would then race the producer refill. Untested
  // combination; the heuristic never sets it, so forbid it outright.
  static_assert(!Ctx::kReduceOverlapLastStageOnly,
                "TCGEN05: reduce_overlap_last_stage_only is not "
                "supported (untested interaction with the b_dequant "
                "staging buffer in the reduce union).");

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
      // Per-warp N-slice base. kNWarps ∈ {1, 2, 4} for BlockN ∈
      // {64, 128, 256}; warps with the same `n_warp_id_scatter`
      // write redundantly (the 4 M-warps that share an N-slice all
      // emit the same bytes). The HW serialises the resulting
      // 4-way bank conflict cheaper than divergent gating -- a
      // 1-warp-per-N-slice variant was measured 10% slower.
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
    // implicit __threadfence_block from the bar.sync/__syncthreads is
    // sufficient to make them visible to subsequent tcgen05.mma SMEM
    // reads. ctx.sync_math_threads() becomes:
    //   * __syncthreads() when kNumMathThreads == kNumThreads
    //     (non-warp-spec path)
    //   * bar.sync 1, kNumMathThreads under warp-spec (producer
    //     threads must NOT be awaited here -- they're busy doing
    //     gmem->smem loads).
    ctx.sync_math_threads();

    // ---- now build descriptors + issue tcgen05.mma ----
    // A descriptor reads from smem.stages[stage_id].a; advance the pointer by
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
    int4 *a_ptr = &smem.stages[stage_id].a[0]
                  + section_idx * kSectionSizeUint128
                  + iter_in_section * kKChunkUint128;
    // B is sectionised the same way A is: the scatter above writes
    // section-major, with each section holding 64 K-bf16 of all N.
    // So B's descriptor SBO is also fixed at 64 K-bf16, and the iter
    // advance crosses sections via `section_idx * kBSectionSizeUint128`
    // (where the B section size in uint128 is `BlockN * 8`).
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
      tcgen05_mma_ss_bf16(smem.tcgen05_tmem_col + accum_col_off(),
                          a_desc, b_desc, idesc, scale_d);
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
  // Close the current accumulator buffer's MMA batch: all prior
  // tcgen05.mma issues from this CTA arrive on mbar[acc_buf_] when they
  // retire. Elect-one; safe to call right after the last K-iter.
  CUDA_INLINE void commit_accum() {
    if (threadIdx.x < 32 && tcgen05_elect_one_sync()) {
      uint32_t mbar_addr =
          cast_smem_ptr_to_uint(&smem.tcgen05_mbar[accum_buf()]);
      tcgen05_commit_to_mbarrier(mbar_addr);
    }
  }

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    commit_accum();
    wait_accum();
    return drain_accum<T>();
  }

  // Wait for the accumulator buffer's committed MMA batch to retire.
  // Every math thread must call this. In the multi-stage flow this runs
  // at the END of the tile that issued the MMAs -- it is what makes
  // releasing the producer safe (in-flight tcgen05.mma still reads the
  // stage SMEM through the async proxy until the batch retires; the
  // producer's next-tile loads would corrupt the last K-iters
  // otherwise -- observed as K-dependent scattered output errors).
  CUDA_INLINE void wait_accum() {
    mbarrier_wait(&smem.tcgen05_mbar[accum_buf()],
                  (mbar_phase_bits_ >> accum_buf()) & 1u);
    mbar_phase_bits_ ^= 1u << accum_buf();
    tcgen05_fence_view_async_tmem_store();
  }

  // t2r the accumulator buffer and write bf16 into `smem.reduce` in
  // gmem_writer layout. Caller must have already commit_accum() +
  // wait_accum() the SAME buffer (set_accum_buf first in the deferred
  // flow). With kAccStages > 1 the WS kernel defers this by one tile.
  template <class T = uint32_t>
  CUDA_INLINE T *drain_accum() {

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
    uint32_t base_addr =
        smem.tcgen05_tmem_col + accum_col_off() + (n_warp_id * WarpShape::N);

    // ---- 3. Per-warp t2r + pack + SMEM write ----
    // Compile-time swizzle base -- must match gmem_writer's
    // `offsetof(SharedStorage, reduce) / 128 % 8` read-side phase
    // (valid because the smem union is alignas(1024)).
    int4 *smem_reduce = smem.reduce;
    uint32_t smem_reduce_base = offsetof(SharedStorage, reduce) / 128u % 8u;
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
      // No per-ni `tcgen05_fence_view_async_tmem_store()` -- the
      // outer commit+mbar_wait that drained the K-loop MMA chain
      // (above) already ordered TMEM writes vs these reads, and
      // consecutive `tcgen05.ld` calls within the same warp don't
      // race against each other.
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
          // int4. With Ctx::kHasBias, smem.bias holds
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
            if constexpr (Ctx::kHasBias) {
              const __nv_bfloat16 *smem_bias_bf16 =
                  reinterpret_cast<const __nv_bfloat16 *>(&smem.bias[0]);
              uint32_t n0 = n_base_pack + pair * 2u;
              f0 += __bfloat162float(smem_bias_bf16[n0]);
              f1 += __bfloat162float(smem_bias_bf16[n0 + 1u]);
            }
            __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
            // Epilogue residual exponent rescale (mirrors
            // `EpilogueArithmetic::may_apply_on_smem_write`'s
            // `apply_exp_offset()`): when the total dequant
            // exp_offset exceeds the mainloop's max_allowed_offset,
            // the leftover lives in `kEpilogueExpOffset.x` and must
            // be applied here -- WMMA/WGMMA pick it up via the
            // smem_writer, but TCGEN05 bypasses smem_writer so we
            // replicate the multiply inline.
            // `!kIsTensorWeightScale` mirrors the smem_writer guard:
            // tensor_weight_scale folds the rescale into `gs` via
            // `may_process_on_smem_write`.
            if constexpr (ArithClass::kEpilogueExpOffset.x
                          && !Ctx::kIsTensorWeightScale) {
              __nv_bfloat162 scale =
                  prepare_exp_scale_factor<__nv_bfloat162,
                                           ArithClass::kEpilogueExpOffset.x>();
              v = __hmul2(v, scale);
            }
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
    // Math-only barrier (same reasoning as the post-scatter one above):
    // collapses to __syncthreads when not in warp-spec mode.
    ctx.sync_math_threads();
    return nullptr;
  }

  // The s2r_pipe reads these accessors (same shape as WMMA's interface).
  template <class T = uint32_t>
  CUDA_INLINE T *regs_a_as_ptr(uint32_t buffer_id) {
    // Interface parity with WMMA/WGMMA only. The s2r pipeline gates out
    // `loader_a.load` for the tcgen05 path (A is read from SMEM via the
    // descriptor), so this pointer is never written or dereferenced.
    return reinterpret_cast<T *>(regs_a);
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
  // Per-accumulator-buffer mbarrier phase parity BITS (bit b = buffer
  // b), flipped by that buffer's commit/wait pair. A scalar bitfield,
  // deliberately NOT an array: dynamic indexing of a member array
  // demotes the object to local memory (see comment on accum_buf()).
  uint32_t mbar_phase_bits_ = 0;
  // Active accumulator buffer (only meaningful for kAccStages > 1).
  uint32_t acc_buf_ = 0;
};


