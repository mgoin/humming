#pragma once

// SS-mode tcgen05.mma (UMMA) mainloop: A from SMEM, dequantised B scattered to
// smem.b_dequant, accumulator in TMEM. cta_group::1 only.

#include <humming/arith/exp_offset.cuh>
#include <humming/utils/all.cuh>
#include <humming/utils/ptx/barrier.cuh>
#include <humming/utils/ptx/shared.cuh>
#include <humming/utils/ptx/tcgen05.cuh>


// Publishes r2s stores to the async proxy tcgen05.mma reads B through.
CUDA_INLINE void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}


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
  // Never used on this path; any well-formed shape satisfies the epilogue.
  using CRegistersArrayType = CRegistersType[1][1];

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kNumWarpShapeNSplits = WarpShape::N == ElementA::kBits * 2 ? 2 : 1;

  // A and B both use Swizzle<3,4,3>: loader_a emits it for A, the scatter in
  // run() writes B in the same layout.
  static constexpr uint32_t kSwizzleBytesA = 128;
  static constexpr uint32_t kSwizzleBytesB = 128;

  // tcgen05.mma.kind::f16 consumes 16 bf16 of K per issue = 2 128B-swizzle atoms.
  static constexpr uint32_t kKChunkUint128 = 2;

  // B staging is section-major for BlockK > 64 (64 K-bf16 of all N per section),
  // mirroring loader_a.
  static constexpr uint32_t kKPerSectionB =
      BlockShape::K < 64u ? BlockShape::K : 64u;
  static constexpr uint32_t kRowBytesB = kKPerSectionB * 2u;
  static constexpr uint32_t kBSectionSizeBytes = BlockShape::N * kRowBytesB;
  static constexpr uint32_t kNScatterWarps =
      MAX(BlockShape::N / WarpShape::N, 1u);

  // Closed-form scatter addressing: every additive term of the element-wise
  // offset occupies a disjoint bit range, so + == ^ among them and the swizzle
  // XOR phase depends only on n's low 3 bits. scatter_closed_form_matches()
  // below checks the two agree over the whole index space.
  static constexpr uint32_t scatter_ref_offset(
      uint32_t t, uint32_t n_base, uint32_t iter, uint32_t i,
      uint32_t frag, uint32_t pair, uint32_t base_div128) {
    uint32_t n = n_base + i * 16u + 8u * frag + t / 4u;
    uint32_t k_lo = iter * kPartMmaShapeK + 2u * (t % 4u) + 8u * pair;
    uint32_t k_section = k_lo / kKPerSectionB;
    uint32_t k_in_section = k_lo % kKPerSectionB;
    uint32_t linear_in_section = n * kRowBytesB + k_in_section * 2u;
    uint32_t linear = k_section * kBSectionSizeBytes + linear_in_section;
    uint32_t xor_shift = (base_div128 + (linear_in_section >> 7)) & 7u;
    return linear ^ (xor_shift << 4);
  }

  static constexpr uint32_t kKItersPerSectionB =
      kKPerSectionB / kPartMmaShapeK;

  static constexpr uint32_t scatter_closed_base0(
      uint32_t t, uint32_t n_base, uint32_t iter, uint32_t base_div128) {
    uint32_t n0 = n_base + t / 4u;
    uint32_t pre = n0 * kRowBytesB + (t % 4u) * 4u;
    uint32_t mask = ((base_div128 + n0) & 7u) << 4;
    return ((pre ^ mask) ^ ((iter % kKItersPerSectionB) * kPartMmaShapeK * 2u))
           + (iter / kKItersPerSectionB) * kBSectionSizeBytes;
  }

  static constexpr uint32_t scatter_closed_offset(
      uint32_t t, uint32_t n_base, uint32_t iter, uint32_t i,
      uint32_t frag, uint32_t pair, uint32_t base_div128) {
    return (scatter_closed_base0(t, n_base, iter, base_div128)
            ^ (pair * 16u))
           + i * 16u * kRowBytesB + frag * 8u * kRowBytesB;
  }

  static constexpr bool scatter_closed_form_matches() {
    // Base phases are sampled: the base enters both formulas identically under &7.
    constexpr uint32_t bases[3] = {0u, 3u, 7u};
    for (uint32_t t = 0; t < 32u; t++)
      for (uint32_t w = 0; w < kNScatterWarps; w++)
        for (uint32_t iter = 0; iter < BlockShape::K / kPartMmaShapeK; iter++)
          for (uint32_t b = 0; b < 3u; b++)
            for (uint32_t i = 0; i < WarpShape::N / 16u; i++)
              for (uint32_t frag = 0; frag < 2u; frag++)
                for (uint32_t pair = 0; pair < 2u; pair++)
                  if (scatter_ref_offset(t, w * WarpShape::N, iter, i,
                                         frag, pair, bases[b])
                      != scatter_closed_offset(t, w * WarpShape::N, iter,
                                               i, frag, pair, bases[b]))
                    return false;
    return true;
  }

  static_assert(scatter_closed_form_matches(),
                "TCGEN05: closed-form scatter offsets diverge from the "
                "reference element-wise swizzle formula");

  Ctx &ctx;
  SharedStorage &smem;
  ArithClass &arith;

  // Interface parity only: the s2r pipe takes this pointer unconditionally but
  // never dereferences it on the tcgen05 path.
  alignas(16) int4 regs_a[1];

  // Quantised B codes loaded by s2r_pipe (same layout as WMMA).
  alignas(16) uint32_t regs_qb[2][ElementB::kBits * (16 / ElementA::kBits)];
  // Dequantised B in RMEM, pre-r2s: the per-thread slice of the BlockN x BlockK tile.
  alignas(16) uint32_t regs_b_tmp[2][WarpShape::N * kPartMmaShapeK * ElementA::kBits / 32 / 32];
  // Post-t2r RMEM accumulator read by the epilogue.
  typename MmaOpClass::CRegisters regs_c;

  CUDA_INLINE
  TCGEN05(Ctx &ctx_, ArithClass &arith_)
      : ctx(ctx_), smem(ctx_.smem), arith(arith_) {}

  CUDA_INLINE
  void zero_accum() {
    // scale_d drives overwrite-vs-accumulate on the TMEM side; zero the RMEM
    // regs so the epilogue is defined if no K-iter fires.
    uint32_t *p = reinterpret_cast<uint32_t *>(regs_c);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs_c) / 4; i++) p[i] = 0;
    first_issue_ = true;
  }

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
  static_assert(MmaOpClass::kCtaGroup == 1,
                "TCGEN05: only cta_group::1 is wired up");
  static_assert(std::is_same<ElementA, BFloat16>::value,
                "TCGEN05: ElementA must be BFloat16. fp16 A requires "
                "a parallel instruction-descriptor + scatter path "
                "that is not wired up.");
  static_assert(!Ctx::kReduceOverlapLastStageOnly,
                "TCGEN05: reduce_overlap_last_stage_only is not "
                "supported (untested interaction with the b_dequant "
                "staging buffer in the reduce union).");

  // Dequant int4 -> ElementA in RMEM; the r2s happens in run(), which knows the K index.
  CUDA_INLINE
  void transform_b(uint32_t buffer_id, uint32_t iter_id) {
    static_assert(!std::is_same<ElementA, ElementB>::value,
                  "TCGEN05 path is only wired up for narrow-B (int4) today");

    // Each i is a different m16n8 fragment pair, so advance by 4 uint32 (as in wmma.cuh).
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < WarpShape::N / 16; i++) {
      uint32_t *regs_b_ptr = &regs_b_tmp[buffer_id][i * 4u];
      uint4 zp_vals = arith.prepare_zp_for_dequant(buffer_id, i);
      uint32_t *zp_vals_ptr = reinterpret_cast<uint32_t *>(&zp_vals);
      dequant<ElementB, ElementA, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(
          regs_qb[buffer_id], regs_b_ptr, i, zp_vals_ptr);
      arith.may_apply_bs_and_zp_on_b(regs_b_ptr, i, buffer_id);
    }

    // The r2s is deferred to run(), which knows the K index. Nothing to fence
    // here: this writes registers only.
  }

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    uint32_t buffer_id = iter_id % 2;

    // Thread t's 4 b16 of B (PTX ISA Table 32, mma.m16n8k16) with reg_index
    // = i*8 + v in [0, 32):
    //   n = i*16 + 8*(v / 4) + t / 4
    //   k = k_base + 2*(t%4) + (v & 1) + 8 * ((v % 4) >> 1)
    {
      uint32_t t = threadIdx.x % 32u;
      constexpr uint32_t kCalls = WarpShape::N / 16u;
      // The 4 M-warps sharing an N-slice write the same bytes; the redundancy
      // costs wavefronts, not conflicts, and beats a 1-warp variant's divergence.
      uint32_t warp_id_local = threadIdx.x / 32u;
      uint32_t n_warp_id_scatter = warp_id_local % kNScatterWarps;
      uint32_t n_base = n_warp_id_scatter * WarpShape::N;
      // Swizzle<3,4,3> XORs bits [4,7) using bits [7,10) of the ABSOLUTE byte
      // address, so smem_base/128 enters the phase.
      uint32_t smem_base_div_128 =
          cast_smem_ptr_to_uint(&smem.b_dequant[buffer_id][0]) >> 7;
      // Each uint32 store covers the (k, k+1) bf16 pair of one row.
      uint32_t *regs_b_u32_buf =
          reinterpret_cast<uint32_t *>(regs_b_tmp[buffer_id]);
      char *smem_b_bytes =
          reinterpret_cast<char *>(&smem.b_dequant[buffer_id][0]);
      uint32_t base0 = scatter_closed_base0(t, n_base, iter_id, smem_base_div_128);
      uint32_t base1 = base0 ^ 16u;
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kCalls; i++) {
        PRAGMA_UNROLL
        for (uint32_t frag_id = 0; frag_id < 2u; frag_id++) {
          uint32_t imm = i * 16u * kRowBytesB + frag_id * 8u * kRowBytesB;
          PRAGMA_UNROLL
          for (uint32_t pair_idx = 0; pair_idx < 2u; pair_idx++) {
            uint32_t reg_index_pair = i * 4u + frag_id * 2u + pair_idx;
            uint32_t addr = (pair_idx ? base1 : base0) + imm;
            *reinterpret_cast<uint32_t *>(smem_b_bytes + addr) =
                regs_b_u32_buf[reg_index_pair];
          }
        }
      }
    }
    // Cross-proxy ordering: bar.sync alone gives only generic-proxy ordering.
    // The fence must sit here rather than at the end of transform_b; there the
    // dequant is register-only, so ptxas sinks it past the UMMA and each issue
    // formally reads unpublished data.
    fence_proxy_async_shared_cta();
    // bar.sync over math threads only; producers are mid gmem->smem load.
    ctx.sync_math_threads();

    // For BlockK > 64, loader_a sectionises A into 64-K-bf16 chunks, so the
    // descriptor's SBO is always MIN(BlockK, 64) and crossing a section means
    // jumping by the section size rather than by atoms.
    constexpr uint32_t kKPerSection = BlockShape::K < 64u ? BlockShape::K : 64u;
    constexpr uint32_t kKItersPerSection = kKPerSection / 16u;
    constexpr uint32_t kSectionSizeUint128 = BlockShape::M * 8u;
    uint32_t section_idx = iter_id / kKItersPerSection;
    uint32_t iter_in_section = iter_id % kKItersPerSection;
    int4 *a_ptr = &smem.stages[stage_id].a[0]
                  + section_idx * kSectionSizeUint128
                  + iter_in_section * kKChunkUint128;
    // B is sectionised the same way, at BlockN * 8 uint128 per section.
    constexpr uint32_t kBSectionSizeUint128 = BlockShape::N * 8u;
    int4 *b_ptr = &smem.b_dequant[buffer_id][0]
                  + section_idx * kBSectionSizeUint128
                  + iter_in_section * kKChunkUint128;

    uint64_t a_desc = tcgen05_smem_desc<kSwizzleBytesA, kKPerSection>(a_ptr);
    uint64_t b_desc = tcgen05_smem_desc<kSwizzleBytesB, kKPerSection>(b_ptr);

    uint32_t idesc =
        tcgen05_instr_desc_f16fam<MmaOpClass::kInstrDescAFormat,
                                  MmaOpClass::kInstrDescBFormat,
                                  MmaOpClass::kInstrDescCFormat>(
            BlockShape::M, BlockShape::N);

    bool scale_d = !first_issue_;
    first_issue_ = false;

    // One elected thread issues. tcgen05.mma is not .sync.aligned (unlike
    // alloc/dealloc), so warp-uniform participation is not required.
    if (threadIdx.x < 32 && tcgen05_elect_one_sync()) {
      tcgen05_mma_ss_bf16(smem.tcgen05_tmem_col, a_desc, b_desc, idesc,
                          scale_d);
    }
  }

  // Close the tile's MMA batch; arrivals land on the mbarrier as issues retire.
  CUDA_INLINE void commit_accum() {
    if (threadIdx.x < 32 && tcgen05_elect_one_sync()) {
      tcgen05_commit_to_mbarrier(cast_smem_ptr_to_uint(&smem.tcgen05_mbar));
    }
  }

  // Must precede any producer release: in-flight tcgen05.mma reads stage SMEM
  // through the async proxy until the batch retires, so the producer's next-tile
  // loads would corrupt the last K-iters.
  CUDA_INLINE void wait_accum() {
    mbarrier_wait(&smem.tcgen05_mbar, mbar_phase_);
    mbar_phase_ ^= 1u;
    tcgen05_fence_after_thread_sync();
  }

  // t2r straight into smem.reduce in gmem_writer's layout, bypassing smem_writer
  // (which assumes >= 2 N-warps). gmem_writer reads smem.reduce as row-major
  // int4 [BlockM][BlockN / 8] with
  //   swizzled_int4_col = int4_col ^ ((row + smem_base) % 8)
  // Returns nullptr as the sentinel.
  template <class T = uint32_t>
  CUDA_INLINE T *drain_accum() {
    // M=64 cta_group::1 TMEM atom: valid M sits at DPs {0..15, 32..47, 64..79,
    // 96..111}, i.e. the first 16 DPs of each of 4 sub-partitions. A warp can
    // only access its own sub-partition, hence 4 warps; lanes 16..31 see garbage.
    static constexpr uint32_t kMWarps = MAX(BlockShape::M / WarpShape::M, 1u);
    static constexpr uint32_t kNWarps = MAX(BlockShape::N / WarpShape::N, 1u);
    static constexpr uint32_t kCallsN = MAX(WarpShape::N / 32u, 1u);
    static_assert(WarpShape::M == 16 || WarpShape::M == 32,
                  "TCGEN05 path requires WarpShape::M to be 16 (M=64 "
                  "atom, 16 valid M per sub-partition) or 32 (M=128 "
                  "atom, 32 valid M per sub-partition).");

    uint32_t warp_id = threadIdx.x / 32u;
    // M must be fastest in warp_id: the atom binds M=0..15 to sub-part 0, etc.
    // loader_b's N assignment is independent (different buffers).
    uint32_t m_warp_id = warp_id % kMWarps;
    uint32_t n_warp_id = (warp_id / kMWarps) % kNWarps;
    uint32_t laneid = threadIdx.x % 32u;

    // The taddr DP field is warp-local; the lane -> DP binding is HW-fixed.
    uint32_t base_addr = smem.tcgen05_tmem_col + (n_warp_id * WarpShape::N);

    // Must match gmem_writer's read-side phase
    // offsetof(SharedStorage, reduce) / 128 % 8 (valid because the union is alignas(1024)).
    int4 *smem_reduce = smem.reduce;
    uint32_t smem_reduce_base = offsetof(SharedStorage, reduce) / 128u % 8u;
    constexpr uint32_t kBlockN = BlockShape::N;
    constexpr uint32_t kInt4ColsPerRow = kBlockN / 8u;
    PRAGMA_UNROLL
    for (uint32_t ni = 0; ni < kCallsN; ni++) {
      uint32_t tmp[32];
      uint32_t addr = base_addr + ni * 32u;
      tcgen05_ld_32x32b_x32(addr, tmp);
      // tcgen05.ld is async; `tmp` is undefined until wait::ld.
      tcgen05_wait_ld();
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
          // smem.bias is BlockN ElementC laid out linearly; add before the convert.
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
            // Residual dequant exp offset: WMMA/WGMMA apply it in smem_writer,
            // which TCGEN05 bypasses. The !kHasTensorWeightScale guard mirrors
            // smem_writer's; tensor_weight_scale folds the rescale into `gs`.
            if constexpr (ArithClass::kEpilogueExpOffset.x
                          && !Ctx::kHasTensorWeightScale) {
              __nv_bfloat162 scale =
                  prepare_exp_scale_factor<__nv_bfloat162,
                                           ArithClass::kEpilogueExpOffset.x>();
              v = __hmul2(v, scale);
            }
            packed_u32[pair] = *reinterpret_cast<uint32_t *>(&v);
          }
          // gmem_writer's smem_row is gmem_row + (gmem_col / 8) * BlockM and its
          // smem_col is gmem_col % 8, so high-N int4 cols must split into
          // separate rows. The XOR swizzle applies to (smem_row, smem_col), not
          // to int4_col.
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
    ctx.sync_math_threads();
    return nullptr;
  }

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    commit_accum();
    wait_accum();
    return drain_accum<T>();
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
    return reinterpret_cast<T *>(regs_b_tmp);
  }

  template <class T = uint32_t>
  CUDA_INLINE T *regs_c_as_ptr(uint32_t buffer_id = 0) {
    return reinterpret_cast<T *>(regs_c);
  }

private:
  // True until the first tcgen05.mma issue lands, used to drive scale_d.
  bool first_issue_ = true;
  // Every member must stay scalar: a runtime-indexed member array demotes the
  // whole object from registers to a local stack frame.
  uint32_t mbar_phase_ = 0;
};
