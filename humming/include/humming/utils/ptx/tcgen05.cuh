#pragma once
//
// PTX wrappers for Blackwell tcgen05.mma (UMMA).
//
// Targets sm_100a / sm_103a. The wrappers exposed here cover the minimum
// surface needed to drive a W4A16-style mainloop:
//
//   * tcgen05_alloc / dealloc / relinquish_alloc_permit
//     -- per-CTA TMEM column allocation (must precede any tcgen05.mma)
//
//   * tcgen05_smem_desc()
//     -- construct an A or B SMEM operand descriptor (same swizzle scheme
//        as wgmma's matrix descriptor; details in `make_smem_desc()`)
//
//   * tcgen05_instr_desc()
//     -- pack the instruction descriptor required by the {sparse,dense}
//        variants of tcgen05.mma (shape, dtypes, transpose, etc.)
//
//   * tcgen05_mma_ss<KIND>(d_tmem, a_desc, b_desc, idesc, scale_d)
//     -- issue one tcgen05.mma instruction (SS = both operands SMEM)
//
//   * tcgen05_commit / tcgen05_wait / tcgen05_fence
//     -- group-based completion: commit() closes the issue group,
//        wait() blocks until all preceding groups retire.
//
//   * tcgen05_ld_*  -- TMEM→register loads for the epilogue (the .x32
//        and .x128 variants cover the 32/128-lane patterns needed by
//        16x256-wide accumulator tiles).
//
// These are *thin* wrappers. The descriptor construction is at runtime
// (kernel-time), not at codegen-time, because the instruction descriptor
// encodes things like SMEM swizzle bits that depend on stage layout.
//
// References:
//   - PTX ISA 8.7, §9.7.16 (tcgen05 operations)
//   - PTX ISA 8.7, §9.7.16.6 (Instruction Descriptor format)
//   - CUTLASS include/cute/arch/mma_sm100_desc.hpp for canonical bit packing.

#include <humming/utils/base.cuh>


// ============================================================================
// TMEM allocation
// ============================================================================
//
// Each CTA allocates a contiguous column range in TMEM via
// `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32`, which reserves
// `num_cols` columns and writes the starting column index to a SMEM uint32
// at the supplied address. The "shared::cta.b32" form means the column
// pointer is delivered through SMEM (so a single thread issues the alloc
// and all warps in the CTA observe the column index after a sync).
//
// `num_cols` must be a power of 2 between 32 and 512.
//
// Pair with `tcgen05_dealloc()` before kernel exit. `relinquish_alloc_permit`
// is required by the spec on the issuing warp before dealloc.

template <uint32_t NumColumns>
CUDA_INLINE void tcgen05_alloc(uint32_t smem_addr_for_col_index) {
  static_assert(NumColumns == 32 || NumColumns == 64 || NumColumns == 128 ||
                NumColumns == 256 || NumColumns == 512,
                "tcgen05_alloc<N>: N must be 32 / 64 / 128 / 256 / 512");
  // Issuing thread must be the first thread of a single warp in the CTA.
  if constexpr (NumColumns == 32) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n"
                 :: "r"(smem_addr_for_col_index) : "memory");
  } else if constexpr (NumColumns == 64) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;\n"
                 :: "r"(smem_addr_for_col_index) : "memory");
  } else if constexpr (NumColumns == 128) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;\n"
                 :: "r"(smem_addr_for_col_index) : "memory");
  } else if constexpr (NumColumns == 256) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;\n"
                 :: "r"(smem_addr_for_col_index) : "memory");
  } else if constexpr (NumColumns == 512) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;\n"
                 :: "r"(smem_addr_for_col_index) : "memory");
  }
}

CUDA_INLINE void tcgen05_relinquish_alloc_permit() {
  asm volatile(
      "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n"
      ::: "memory");
}

template <uint32_t NumColumns>
CUDA_INLINE void tcgen05_dealloc(uint32_t tmem_col_index) {
  static_assert(NumColumns == 32 || NumColumns == 64 || NumColumns == 128 ||
                NumColumns == 256 || NumColumns == 512,
                "tcgen05_dealloc<N>: N must be 32 / 64 / 128 / 256 / 512");
  if constexpr (NumColumns == 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"
                 :: "r"(tmem_col_index) : "memory");
  } else if constexpr (NumColumns == 64) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;\n"
                 :: "r"(tmem_col_index) : "memory");
  } else if constexpr (NumColumns == 128) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\n"
                 :: "r"(tmem_col_index) : "memory");
  } else if constexpr (NumColumns == 256) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;\n"
                 :: "r"(tmem_col_index) : "memory");
  } else if constexpr (NumColumns == 512) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;\n"
                 :: "r"(tmem_col_index) : "memory");
  }
}


// ============================================================================
// SMEM operand descriptor (matches wgmma's layout; same builder used).
// ============================================================================
//
// tcgen05.mma reads each operand from SMEM via a 64-bit descriptor that
// encodes:
//   - bits  0..13: SMEM start address (>> 4)
//   - bits 14..15: reserved
//   - bits 16..29: leading dimension byte offset (>> 4)
//   - bits 30..31: reserved
//   - bits 32..45: stride dimension byte offset (>> 4)
//   - bits 46..48: reserved
//   - bits 49..51: matrix base offset (>> 4)
//   - bits 52..61: reserved
//   - bits 62..63: swizzle mode (0=none, 1=128B, 2=64B, 3=32B)
//
// This is the same layout wgmma uses; humming already has the helper at
// mma/wgmma.cuh:7-20. We expose it under a more neutral name here so the
// tcgen05 path can build descriptors without depending on the wgmma header.

template <uint32_t SwizzleBytes = 128>
CUDA_INLINE uint64_t tcgen05_smem_desc(const void *smem_ptr) {
  static_assert(SwizzleBytes == 128 || SwizzleBytes == 64 ||
                SwizzleBytes == 32 || SwizzleBytes == 0,
                "tcgen05_smem_desc: SwizzleBytes must be 128/64/32/0");

  // Encode swizzle in bits 62-63.
  constexpr uint64_t swizzle_type =
      SwizzleBytes == 128 ? 1ULL :
      SwizzleBytes == 64  ? 2ULL :
      SwizzleBytes == 32  ? 3ULL :
                            0ULL;
  // Leading-dim offset (defaults: matches a 128-wide stage, override
  // per-config when needed). Keep stride sentinel 0 for now; caller fills.
  constexpr uint64_t stride_imm = (SwizzleBytes * 8) >> 4;
  constexpr uint64_t desc_base = (swizzle_type << 62) | (stride_imm << 32);

  uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
  uint64_t desc = desc_base;
  reinterpret_cast<uint32_t *>(&desc)[0] = (smem_addr >> 4);
  return desc;
}


// ============================================================================
// Instruction descriptor
// ============================================================================
//
// The 32-bit instruction descriptor encodes the shape/dtype/transpose flags
// for one tcgen05.mma issue. Layout (per PTX 8.7 §9.7.16.6) for the dense
// .kind::f16 / .kind::bf16 / .kind::tf32 family:
//
//   bits  0.. 5  sparsity (0 for dense)
//   bits  6.. 8  saturation / negate (we leave 0)
//   bits  9..11  d_type (00=f16, 01=bf16, 10=tf32, 11=f32)
//   bits 12..14  a_type / b_type
//   bits 15..16  negate A / B
//   bit  17      A is transposed
//   bit  18      B is transposed
//   bits 19..23  M-shape (units of 32, so M=64 -> 2)
//   bits 24..28  N-shape (units of 8, so N=128 -> 16)
//   bits 29..30  cta_group (00=1, 01=2)
//   bit  31      reserved
//
// For our W4A16 path the inputs are A=bf16, B=bf16 (after dequant), D=f32,
// cta_group=1, no negate, no transpose, dense, M variable, N variable.

namespace tcgen05_dtype_codes {
  // Codes for the A/B/D type fields. Match PTX 8.7 Table 38.
  constexpr uint32_t F16 = 0;
  constexpr uint32_t BF16 = 1;
  constexpr uint32_t TF32 = 2;
  constexpr uint32_t F32 = 3;
}

CUDA_INLINE uint32_t tcgen05_instr_desc_bf16_bf16_f32(uint32_t shape_m,
                                                     uint32_t shape_n) {
  // shape_m in {64, 128, 256}; shape_n in {32, 64, ..., 256, step 32}.
  uint32_t idesc = 0;
  idesc |= (tcgen05_dtype_codes::F32  & 0x7) << 9;   // d_type
  idesc |= (tcgen05_dtype_codes::BF16 & 0x7) << 12;  // a_type (and shared b_type slot)
  // M-shape in units of 32:
  idesc |= ((shape_m >> 5) & 0x1f) << 19;
  // N-shape in units of 8:
  idesc |= ((shape_n >> 3) & 0x1f) << 24;
  // cta_group::1 -> bits 29-30 = 00.
  return idesc;
}


// ============================================================================
// tcgen05.mma issue (SS = both operands from SMEM)
// ============================================================================
//
// Form: tcgen05.mma.cta_group::1.kind::f16.collector::a::fill
//         [d_tmem], a_desc, b_desc, idesc, scale_d;
//
// d_tmem      : TMEM destination address (column index in low 16 bits).
// a_desc      : 64-bit SMEM matrix descriptor for A.
// b_desc      : 64-bit SMEM matrix descriptor for B.
// idesc       : 32-bit instruction descriptor (see above).
// scale_d     : 0 = overwrite D, 1 = accumulate into D.
//
// The .kind::f16 family covers f16/bf16 inputs with f32 acc; we use the
// bf16 variant.

CUDA_INLINE void tcgen05_mma_ss_bf16(uint32_t d_tmem,
                                     uint64_t a_desc,
                                     uint64_t b_desc,
                                     uint32_t idesc,
                                     bool scale_d) {
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %4, 0;\n"
      "  tcgen05.mma.cta_group::1.kind::f16.collector::a::fill "
      "    [%0], %1, %2, %3, p;\n"
      "}\n"
      :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
         "r"((uint32_t)scale_d)
      : "memory");
}


// ============================================================================
// Group commit / wait
// ============================================================================
//
// NOTE: Unlike wgmma's `commit_group / wait_group` pair, tcgen05.mma signals
// completion through an *mbarrier*. The canonical idiom is:
//
//   tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbar_smem_addr];
//   ...
//   mbarrier.try_wait.parity.shared.b64 ...
//
// That means the call site must thread an mbarrier SMEM address through to
// the commit, and the kernel must initialise that mbarrier in shared
// storage. This is meaningful surgery to humming's `g2s_pipeline` /
// `consumer_pipeline` -- they already manage mbarriers for cp.async/TMA,
// and the tcgen05 path needs its own mbarrier alongside the existing ones.
//
// Sketched out below but commented until the pipeline plumbing in
// `kernel/humming.cuh` and `epilogue/pipeline.cuh` allocates the mbarrier
// and threads its SMEM address down. Until then, callers must use a
// blocking `__syncthreads()` after the issue and accept the perf hit (it's
// only the prototype path).

#if 0  // TODO(sm100): land alongside the mbarrier wiring in humming.cuh.
CUDA_INLINE void tcgen05_commit_to_mbarrier(uint32_t mbar_smem_addr) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
      :: "r"(mbar_smem_addr) : "memory");
}
#endif

CUDA_INLINE void tcgen05_fence_view_async_tmem_store() {
  // Ensures prior TMEM stores from this CTA are visible before subsequent
  // SMEM/RMEM-side reads. Cheap; emit unconditionally.
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}


// ============================================================================
// TMEM→register load (t2r) for the epilogue
// ============================================================================
//
// Pattern: tcgen05.ld.sync.aligned.32x32b.x{N}.b32 {d0..dN-1}, [tmem_addr];
//
// The shapes are dictated by hardware; for our prototype we expose the
// .x32 form which loads 32 lanes × 32 bits = 128 B per warp, matching the
// per-warp accumulator slice for an M=64 tile.

CUDA_INLINE void tcgen05_ld_32x32b_x32(uint32_t tmem_addr, uint32_t *dst) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
      : "=r"(dst[0]),  "=r"(dst[1]),  "=r"(dst[2]),  "=r"(dst[3]),
        "=r"(dst[4]),  "=r"(dst[5]),  "=r"(dst[6]),  "=r"(dst[7]),
        "=r"(dst[8]),  "=r"(dst[9]),  "=r"(dst[10]), "=r"(dst[11]),
        "=r"(dst[12]), "=r"(dst[13]), "=r"(dst[14]), "=r"(dst[15]),
        "=r"(dst[16]), "=r"(dst[17]), "=r"(dst[18]), "=r"(dst[19]),
        "=r"(dst[20]), "=r"(dst[21]), "=r"(dst[22]), "=r"(dst[23]),
        "=r"(dst[24]), "=r"(dst[25]), "=r"(dst[26]), "=r"(dst[27]),
        "=r"(dst[28]), "=r"(dst[29]), "=r"(dst[30]), "=r"(dst[31])
      : "r"(tmem_addr)
      : "memory");
}
