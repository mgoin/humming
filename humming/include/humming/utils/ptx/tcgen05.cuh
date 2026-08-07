#pragma once

// PTX wrappers for Blackwell tcgen05.mma (UMMA), sm_100a / sm_103a. Bit layouts
// follow PTX ISA 8.7 sec 9.7.16 and CUTLASS cute/arch/mma_sm100_desc.hpp.

#include <humming/utils/base.cuh>


// `num_cols` must be a power of two in [32, 512]. Pair with tcgen05_dealloc;
// relinquish_alloc_permit is required on the issuing warp before dealloc.
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

// Elect one lane per warp (CUTLASS elect_one_sync). Branch reconvergence after
// the if-block satisfies .sync.aligned for the other lanes.
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


// SM100 SMEM descriptor (CUTLASS cute/arch/mma_sm100_desc.hpp):
//   [0,14)  start_address >> 4      [16,30) leading_byte_offset >> 4
//   [32,46) stride_byte_offset >> 4 [46,48) version (1 = Blackwell)
//   [49,52) base_offset             52      lbo_mode
//   [61,64) layout_type: 0 = none, 2 = 128B, 4 = 64B, 6 = 32B
// For K-major tiles at 128B swizzle the UMMA-K layout is
// Swizzle<3,4,3> o ((8,n),2):((8,SBO),1) in uint128_t units, so
// SBO = 8 rows * BlockK * 2 B / 16 B = BlockK and LBO = 1.
template <uint32_t SwizzleBytes, uint32_t BlockKElems>
CUDA_INLINE uint64_t tcgen05_smem_desc(const void *smem_ptr) {
  static_assert(SwizzleBytes == 128 || SwizzleBytes == 64,
                "tcgen05_smem_desc: only 128/64 B swizzle wired up so far");
  static_assert(BlockKElems > 0 && (BlockKElems % 8) == 0,
                "BlockK must be a positive multiple of 8 bf16 elements");

  constexpr uint64_t layout_type =
      SwizzleBytes == 128 ? 2ULL :
      SwizzleBytes == 64  ? 4ULL :
                            0ULL;

  constexpr uint64_t sbo = BlockKElems;
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


// 32-bit instruction descriptor, one per tcgen05.mma issue:
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

// kind::f16 descriptor. shape_m/shape_n are the MMA-operand extents, which
// differ from the block tile on the TS path (A and B are swapped).
template <uint32_t kAFmt, uint32_t kBFmt, uint32_t kCFmt>
CUDA_INLINE uint32_t tcgen05_instr_desc_f16fam(uint32_t shape_m,
                                               uint32_t shape_n) {
  Tcgen05InstrDescriptor d{};
  d.c_format = kCFmt;
  d.a_format = kAFmt;
  d.b_format = kBFmt;
  d.n_dim    = (shape_n >> 3);   // N=128 -> 16
  d.m_dim    = (shape_m >> 4);   // M=64  -> 4, M=128 -> 8
  return d.desc;
}


// SS = both operands from SMEM.
CUDA_INLINE void tcgen05_mma_ss_bf16(uint32_t d_tmem,
                                     uint64_t a_desc,
                                     uint64_t b_desc,
                                     uint32_t idesc,
                                     bool scale_d) {
  // The {m0..m3} operand is a 128-bit disable mask; all-zero means no masking.
  // Omitting it parses to a different variant that never retires, and the hang
  // surfaces at the trailing mbarrier_wait rather than at the mma.
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

// TS mode: A from TMEM (bracketed), B from SMEM. a_tmem's lane field must be 0
// (the MMA reads all M lanes) and the column field selects the staging slot;
// A must be K-major in TMEM (2 ElementA per cell), the only layout TS accepts.
CUDA_INLINE void tcgen05_mma_ts_bf16(uint32_t d_tmem,
                                     uint32_t a_tmem,
                                     uint64_t b_desc,
                                     uint32_t idesc,
                                     bool scale_d) {
  uint32_t mask[4] = {0u, 0u, 0u, 0u};
  asm volatile(
      "{\n\t"
      "  .reg .pred p;\n\t"
      "  setp.ne.b32 p, %4, 0;\n\t"
      "  tcgen05.mma.cta_group::1.kind::f16 "
      "    [%0], [%1], %2, %3, {%5, %6, %7, %8}, p;\n\t"
      "}\n"
      :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(idesc),
         "r"((uint32_t)scale_d),
         "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
      : "memory");
}


// tcgen05.mma signals completion through an mbarrier, not a wait_group:
// tcgen05.commit batches ALL prior issues from this CTA and arrives once when
// they retire. Init the mbar with expected_count = 1, one thread commits per
// use, and readers toggle the phase parity after each wait.
CUDA_INLINE void tcgen05_commit_to_mbarrier(uint32_t mbar_smem_addr) {
  // The .shared::cluster qualifier is required: without it ptxas accepts the asm
  // but the hardware never arrives on the mbar and try_wait spins forever.
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
      :: "r"(mbar_smem_addr) : "memory");
}

CUDA_INLINE void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}

CUDA_INLINE void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}

// tcgen05.wait::st: blocks until this warp's prior tcgen05.st ops complete.
CUDA_INLINE void tcgen05_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
}

// tcgen05.ld is async: the destination registers are undefined until this wait
// retires, and reading them earlier is UB that SASS scheduling can mask.
CUDA_INLINE void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}

// r2t: store 8 consecutive TMEM columns (one 32-bit cell per lane per column)
// into the issuing warp's own sub-partition. reg r -> column (taddr.col + r).
CUDA_INLINE void tcgen05_st_32x32b_x8(uint32_t tmem_addr,
                                      const uint32_t (&r)[8]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], "
      "{%1, %2, %3, %4, %5, %6, %7, %8};\n"
      :: "r"(tmem_addr),
         "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]),
         "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7])
      : "memory");
}

// t2r: 32 lanes x 32 bits per warp.
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
