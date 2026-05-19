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

// cta_group::2 variant: leader CTA in a 2x cluster issues this; the
// allocator returns a column index in shared::cluster (visible from
// both CTAs' TMEM views). The peer CTA must NOT also call alloc; it
// reads the result from the leader's smem_addr_for_col_index after
// the cluster barrier.
template <uint32_t NumColumns>
CUDA_INLINE void tcgen05_alloc_2cta(uint32_t smem_addr_for_col_index) {
  static_assert(NumColumns == 32 || NumColumns == 64 || NumColumns == 128 ||
                NumColumns == 256 || NumColumns == 512,
                "tcgen05_alloc_2cta<N>: N must be 32/64/128/256/512");
  if constexpr (NumColumns == 32) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 32;\n"
                 :: "r"(smem_addr_for_col_index) : "memory");
  } else if constexpr (NumColumns == 64) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 64;\n"
                 :: "r"(smem_addr_for_col_index) : "memory");
  } else if constexpr (NumColumns == 128) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 128;\n"
                 :: "r"(smem_addr_for_col_index) : "memory");
  } else if constexpr (NumColumns == 256) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 256;\n"
                 :: "r"(smem_addr_for_col_index) : "memory");
  } else if constexpr (NumColumns == 512) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;\n"
                 :: "r"(smem_addr_for_col_index) : "memory");
  }
}

CUDA_INLINE void tcgen05_relinquish_alloc_permit() {
  asm volatile(
      "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n"
      ::: "memory");
}

CUDA_INLINE void tcgen05_relinquish_alloc_permit_2cta() {
  asm volatile(
      "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n"
      ::: "memory");
}

// One-thread-elect-out-of-warp helper, used to gate `.sync.aligned`
// tcgen05 instructions per CUTLASS's pattern
// (`cute/arch/cluster_sm90.hpp:elect_one_sync`).  Returns true on the
// elected lane (typically lane 0), false on the others. The implicit
// branch reconvergence after the if-block lets the elected thread
// issue the asm while the other 31 wait, satisfying `.sync.aligned`.
CUDA_INLINE bool tcgen05_elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
      "{\n"
      "  .reg .b32 %rx;\n"
      "  .reg .pred %px;\n"
      "  elect.sync %rx | %px, 0xFFFFFFFF;\n"
      "  @%px mov.s32 %1, 1;\n"
      "  mov.s32 %0, %%laneid;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred));
  return pred != 0;
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

template <uint32_t NumColumns>
CUDA_INLINE void tcgen05_dealloc_2cta(uint32_t tmem_col_index) {
  static_assert(NumColumns == 32 || NumColumns == 64 || NumColumns == 128 ||
                NumColumns == 256 || NumColumns == 512,
                "tcgen05_dealloc_2cta<N>: N must be 32/64/128/256/512");
  if constexpr (NumColumns == 32) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 32;\n"
                 :: "r"(tmem_col_index) : "memory");
  } else if constexpr (NumColumns == 64) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 64;\n"
                 :: "r"(tmem_col_index) : "memory");
  } else if constexpr (NumColumns == 128) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 128;\n"
                 :: "r"(tmem_col_index) : "memory");
  } else if constexpr (NumColumns == 256) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 256;\n"
                 :: "r"(tmem_col_index) : "memory");
  } else if constexpr (NumColumns == 512) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;\n"
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

// SM100 SmemDescriptor layout (per CUTLASS
// `include/cute/arch/mma_sm100_desc.hpp:98`):
//
//   bits [0, 14)  start_address (>> 4)
//   bits [16,30)  leading_byte_offset (>> 4)
//   bits [32,46)  stride_byte_offset (>> 4)
//   bits [46,48)  version  (we set to 1 for blackwell)
//   bits [49,52)  base_offset
//   bit  52       lbo_mode (0 = legacy)
//   bits [61,64)  layout_type:
//       0 = NONE
//       1 = 128B_BASE32B
//       2 = 128B           (3-bit: 010)
//       4 = 64B            (3-bit: 100)
//       6 = 32B            (3-bit: 110)
//
// For our K-major bf16 tiles with 128B swizzle the canonical UMMA-K
// layout is `Swizzle<3,4,3> o ((8,n),2):((8,SBO),1)` (uint128_t units).
// Mapping to descriptor fields:
//   SBO = stride between successive 8-N-row blocks (in uint128_t units)
//       = 8_rows * row_byte_stride / 16
//       = 8 * BlockK * sizeof(bf16) / 16
//       = BlockK  (bf16 elements)
//   LBO = stride between the two K-chunks of the swizzle atom = 1
//
// Humming's existing `make_wgmma_smem_desc` hardcodes SBO = 64
// (correct only for BlockK <= 64 bf16); we parameterise on BlockK here.

template <uint32_t SwizzleBytes, uint32_t BlockKElems>
CUDA_INLINE uint64_t tcgen05_smem_desc(const void *smem_ptr) {
  static_assert(SwizzleBytes == 128 || SwizzleBytes == 64,
                "tcgen05_smem_desc: only 128/64 B swizzle wired up so far");
  static_assert(BlockKElems > 0 && (BlockKElems % 8) == 0,
                "BlockK must be a positive multiple of 8 bf16 elements");

  // Layout type (3 bits at [61,63]).
  constexpr uint64_t layout_type =
      SwizzleBytes == 128 ? 2ULL :
      SwizzleBytes == 64  ? 4ULL :
                            0ULL;

  // SBO in uint128_t units. For K-major bf16 with the canonical
  // `((8,n),2):((8,SBO),1)` layout:
  //   SBO_uint128 = BlockKElems (bf16) / 8 * 8 = BlockKElems
  // Wait, derivation: 8 N-rows * BlockKElems bf16/row * 2 bytes/bf16
  //                   / 16 bytes/uint128_t = BlockKElems
  constexpr uint64_t sbo = BlockKElems;
  // LBO = stride between the two K-chunks of the swizzle atom.
  // In K-major, the second K-chunk is one uint128_t (= 8 bf16) away.
  constexpr uint64_t lbo = 1;

  uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);

  uint64_t desc = 0;
  desc |= ((uint64_t)(smem_addr >> 4)) & 0x3FFFULL;        // [0,14)
  desc |= (lbo & 0x3FFFULL) << 16;                          // [16,30)
  desc |= (sbo & 0x3FFFULL) << 32;                          // [32,46)
  desc |= (1ULL << 46);                                     // version=1
  desc |= (layout_type & 0x7ULL) << 61;                     // [61,64)
  return desc;
}


// ============================================================================
// Instruction descriptor
// ============================================================================
//
// 32-bit "instruction descriptor" passed to every tcgen05.mma issue. The
// layout follows CUTLASS's `UMMA::InstrDescriptor` exactly
// (`include/cute/arch/mma_sm100_desc.hpp:412`); reproduce it as a union so
// we don't have to maintain hand-rolled shift arithmetic. PTX ISA 8.7
// §9.7.16.5.1 is the spec.
//
//   sparse_id2  : 2  [ 0, 2)  -- meta id for sparse
//   sparse_flag : 1  [ 2, 3)  -- dense=0, sparse=1
//   saturate    : 1  [ 3, 4)  -- int8 saturate; 0 for f16/bf16
//   c_format    : 2  [ 4, 6)  -- 0=F16, 1=F32, 2=S32
//   (reserved)  : 1  [ 6, 7)
//   a_format    : 3  [ 7,10)  -- F16=0, BF16=1, TF32=2 (kind::f16 family)
//   b_format    : 3  [10,13)
//   a_negate    : 1  [13,14)
//   b_negate    : 1  [14,15)
//   a_major     : 1  [15,16)  -- 0 = K-major (standard), 1 = MN-major
//   b_major     : 1  [16,17)
//   n_dim       : 6  [17,23)  -- N >> 3.  N=32 -> 4, ..., N=256 -> 32
//   (reserved)  : 1  [23,24)
//   m_dim       : 5  [24,29)  -- M >> 4.  M=64  -> 4,  M=128 -> 8, M=256 -> 16
//   (reserved)  : 1  [29,30)
//   max_shift   : 2  [30,32)

union Tcgen05InstrDescriptor {
  uint32_t desc;
  struct {
    uint16_t sparse_id2  : 2,
             sparse_flag : 1,
             saturate    : 1,
             c_format    : 2,
             _r0         : 1,
             a_format    : 3,
             b_format    : 3,
             a_negate    : 1,
             b_negate    : 1,
             a_major     : 1;
    uint16_t b_major     : 1,
             n_dim       : 6,
             _r1         : 1,
             m_dim       : 5,
             _r2         : 1,
             max_shift   : 2;
  };
};

namespace tcgen05_fmt {
  constexpr uint32_t F16  = 0;
  constexpr uint32_t BF16 = 1;
  constexpr uint32_t TF32 = 2;
}
namespace tcgen05_cfmt {
  constexpr uint32_t F16 = 0;
  constexpr uint32_t F32 = 1;
  constexpr uint32_t S32 = 2;
}

CUDA_INLINE uint32_t tcgen05_instr_desc_bf16_bf16_f32(uint32_t shape_m,
                                                     uint32_t shape_n) {
  Tcgen05InstrDescriptor d{};
  d.c_format = tcgen05_cfmt::F32;
  d.a_format = tcgen05_fmt::BF16;
  d.b_format = tcgen05_fmt::BF16;
  d.n_dim    = (shape_n >> 3);   // N=128 -> 16
  d.m_dim    = (shape_m >> 4);   // M=64  -> 4, M=128 -> 8
  return d.desc;
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
  // Real PTX syntax per CUTLASS
  // `cute/arch/mma_sm100_umma.hpp:111` (SM100_MMA_F16BF16_SS::fma):
  //
  //   tcgen05.mma.cta_group::1.kind::f16
  //       [tmem_c], desc_a, desc_b, idesc, {mask0, mask1, mask2, mask3}, p
  //
  // The `{m0..m3}` operand is a 128-bit sparsity/disable mask -- all-zero
  // means "no masking". Without this operand the instruction parses to a
  // different variant and hangs / never retires. Caused the bulk of
  // Phase B.6 debug pain; caught via `compute-sanitizer --tool synccheck`
  // which fingered the trailing `mbarrier_wait` as a "Missing wait".
  uint32_t mask[4] = {0u, 0u, 0u, 0u};
  asm volatile(
      "{\n\t"
      "  .reg .pred p;\n\t"
      "  setp.ne.b32 p, %4, 0;\n\t"
      "  tcgen05.mma.cta_group::1.kind::f16 "
      "    [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
      "}\n"
      :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
         "r"((uint32_t)scale_d),
         "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
      : "memory");
}

// cta_group::2 variant: issued by leader CTA in a 2x cluster. The
// effective MMA shape is (2 * BlockM, BlockN); the leader writes
// M=0..BlockM-1 to its TMEM, the peer CTA gets M=BlockM..2*BlockM-1
// in its own TMEM (cluster-aligned). The peer MUST be at a cluster
// barrier when this issues -- it does not call this PTX itself.
CUDA_INLINE void tcgen05_mma_ss_bf16_2cta(uint32_t d_tmem,
                                          uint64_t a_desc,
                                          uint64_t b_desc,
                                          uint32_t idesc,
                                          bool scale_d) {
  uint32_t mask[4] = {0u, 0u, 0u, 0u};
  asm volatile(
      "{\n\t"
      "  .reg .pred p;\n\t"
      "  setp.ne.b32 p, %4, 0;\n\t"
      "  tcgen05.mma.cta_group::2.kind::f16 "
      "    [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
      "}\n"
      :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
         "r"((uint32_t)scale_d),
         "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
      : "memory");
}


// ============================================================================
// Group commit / wait
// ============================================================================
//
// Unlike wgmma's `commit_group / wait_group` pair, tcgen05.mma signals
// completion through an *mbarrier*. The canonical idiom is:
//
//   tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbar_smem_addr];
//   ...
//   mbarrier.try_wait.parity.shared::cta.b64 P, [mbar_smem_addr], phase;
//
// `tcgen05.commit` packages ALL prior tcgen05.mma issues from this CTA
// into a single batch and arrives on `mbar` exactly once when they all
// retire. The reader busy-waits on the mbarrier with a phase parity bit
// that flips on every use of the same mbar.
//
// Caller obligations:
//   * Init the mbar at kernel start: `__mbarrier_init(&mbar, 1)` -- one
//     expected arrival because each tcgen05.commit arrives exactly once.
//   * Only one thread issues `tcgen05_commit_to_mbarrier()` per use.
//   * All threads waiting on the result call `mbarrier_wait(mbar, phase)`
//     and the caller toggles `phase` after each wait.

CUDA_INLINE void tcgen05_commit_to_mbarrier(uint32_t mbar_smem_addr) {
  // The `.shared::cluster` qualifier matters: per CUTLASS
  // `cutlass/arch/barrier.h:770`, tcgen05.commit treats the mbarrier
  // address as cluster-shared. Without this qualifier ptxas accepts
  // the asm but the hardware never arrives on the mbar, so any
  // subsequent mbarrier.try_wait spins forever.
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
      :: "r"(mbar_smem_addr) : "memory");
}

CUDA_INLINE void tcgen05_commit_to_mbarrier_2cta(uint32_t mbar_smem_addr) {
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
      :: "r"(mbar_smem_addr) : "memory");
}

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
