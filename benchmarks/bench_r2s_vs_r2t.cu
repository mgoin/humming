// Microbench: r2s + bar.sync (humming Path 1) vs r2t + tcgen05.fence (Path 2)
//
// Per K-iter both paths move 4 KiB of dequant'd bf16 from registers to a
// shared memory or TMEM staging area, then synchronize before the consuming
// tcgen05.mma issues. This measures the cost of just the move + sync, in
// isolation, on a single CTA.
//
// Build:
//   /usr/local/cuda/bin/nvcc \
//       -gencode=arch=compute_103a,code=sm_103a \
//       -O3 bench_r2s_vs_r2t.cu -o bench_r2s_vs_r2t
//
// Run (use a quiet B300 SM):
//   CUDA_VISIBLE_DEVICES=5 ./bench_r2s_vs_r2t
//
// Headline result on B300 (sm_103a, May 2026, n_iters=10000):
//   Path 1  (r2s + bar.sync, 16 stores/thread):  2052 cyc/iter
//   Path 1' (r2s NO bar, store cost only):       2048 cyc/iter
//   Path 2a (r2t 8-warp + fence):                  43 cyc/iter
//   Path 2b (r2t 2-warp + fence + bar.sync):       54 cyc/iter
//
// Takeaway: the per-K-iter SMEM scatter is ~50x more expensive than the
// r2t (tcgen05.st) variant. The bar.sync is essentially free in isolation
// (4 cyc). The NCU "barrier" stall in the production humming WS path
// doesn't come from the bar itself -- it comes from warps idling at the
// bar because the scatter is the bottleneck and they all arrive at once.
// Moving to a TS-mode mainloop where dequant'd B lives in TMEM eliminates
// the per-K-iter scatter entirely.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdint.h>

#define PRAGMA_UNROLL _Pragma("unroll")

// ----- helpers --------------------------------------------------------------
__device__ __forceinline__ uint32_t cast_smem_ptr_to_uint(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// TMEM alloc / dealloc (cta_group::1).
template <uint32_t kNumCols>
__device__ __forceinline__ void tcgen05_alloc(uint32_t smem_dst_col) {
  static_assert(kNumCols == 32 || kNumCols == 64 || kNumCols == 128 ||
                kNumCols == 256 || kNumCols == 512, "");
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
      :
      : "r"(smem_dst_col), "n"(kNumCols)
      : "memory");
}

template <uint32_t kNumCols>
__device__ __forceinline__ void tcgen05_dealloc(uint32_t tmem_col) {
  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
      :
      : "r"(tmem_col), "n"(kNumCols)
      : "memory");
}

__device__ __forceinline__ void tcgen05_relinquish_alloc_permit() {
  asm volatile(
      "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n"
      ::: "memory");
}

// tcgen05.st.sync.aligned.32x32b.x32.b32 -- per-warp r2t writing 32 b32
// per lane = 128 B/lane × 32 lanes = 4 KiB into a contiguous TMEM region.
__device__ __forceinline__ void tcgen05_st_32x32b_x32(
    uint32_t tmem_addr, uint32_t const *src) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 "
      "[%0], "
      "{%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "
      " %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};\n"
      :
      : "r"(tmem_addr),
        "r"(src[0]),  "r"(src[1]),  "r"(src[2]),  "r"(src[3]),
        "r"(src[4]),  "r"(src[5]),  "r"(src[6]),  "r"(src[7]),
        "r"(src[8]),  "r"(src[9]),  "r"(src[10]), "r"(src[11]),
        "r"(src[12]), "r"(src[13]), "r"(src[14]), "r"(src[15]),
        "r"(src[16]), "r"(src[17]), "r"(src[18]), "r"(src[19]),
        "r"(src[20]), "r"(src[21]), "r"(src[22]), "r"(src[23]),
        "r"(src[24]), "r"(src[25]), "r"(src[26]), "r"(src[27]),
        "r"(src[28]), "r"(src[29]), "r"(src[30]), "r"(src[31])
      : "memory");
}

// tcgen05.fence::before_thread_sync -- ordering primitive for tcgen05.st
// writes vs subsequent tcgen05.mma issued from the same warp.
__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}

// ----- constants ------------------------------------------------------------
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kKChunk = 16;          // bf16 K per tcgen05.mma issue
constexpr int kKItersPerTile = 8;    // 128 / 16
constexpr int kMathThreads = 256;    // matches warp-spec config
constexpr int kMathWarps = kMathThreads / 32;

// Per K-iter we move kBlockN × kKChunk bf16 = 128 × 16 × 2 = 4096 B.
// 256 threads × 16 stores × 4 B = 16384 B = 4× over (4-way redundant scatter
// as in humming today).
constexpr int kBytesPerKIter = kBlockN * kKChunk * 2;  // 4096
constexpr int kStoresPerThreadR2S = 16;                // matches humming scatter

// r2t per K-iter: write kBytesPerKIter into TMEM via tcgen05.st. Each warp's
// .32x32b.x32 call writes 32 lanes × 32 b32 = 4 KiB. So ONE call per warp
// per K-iter covers it -- but we need to do it for ALL math warps (to fully
// land kBytesPerKIter into TMEM), giving us 8 warps × 4 KiB = 32 KiB per K-iter
// of work. To make this an apples-to-apples comparison with the r2s scatter
// (which also has all 8 math warps writing 4-way redundantly), the r2t side
// also has all 8 math warps writing -- but since tcgen05.st has no
// redundancy semantics, they write to NON-OVERLAPPING TMEM cols. (For a real
// TS-mode mainloop, only ONE M-warp would write per N-warp, with 2 N-warps
// at BlockN=128. We measure BOTH variants below: "8-warp" and "2-warp".)

// ----- kernels --------------------------------------------------------------

// PATH 1: r2s scatter pattern matching humming today.
__global__ void kernel_r2s(int n_iters, uint64_t *out_cycles, uint32_t *sink) {
  __shared__ uint32_t smem_buf[kBlockN * kKChunk / 2];  // 4 KiB
  uint32_t tid = threadIdx.x;
  uint32_t regs[16];
  PRAGMA_UNROLL
  for (int i = 0; i < 16; i++) regs[i] = tid * 16u + i + sink[tid];  // taint

  __syncthreads();

  uint64_t t0;
  PRAGMA_UNROLL
  for (int rep = 0; rep < 2; rep++) {
    if (rep == 1) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    for (int it = 0; it < n_iters; it++) {
      // Match humming's scatter: 16 stores per thread per K-iter.
      PRAGMA_UNROLL
      for (int s = 0; s < kStoresPerThreadR2S; s++) {
        uint32_t off = (tid * 16u + s + (it * 7u)) & (kBlockN * kKChunk / 2 - 1u);
        smem_buf[off] = regs[s] + it;  // make per-iter side effect visible
      }
      asm volatile("bar.sync 1, %0;" :: "n"(kMathThreads));
    }
  }
  uint64_t t1;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

  // Sink read so the stores can't be DCE'd.
  uint32_t acc = 0;
  PRAGMA_UNROLL
  for (int s = 0; s < 16; s++) acc += smem_buf[(tid * 16 + s) & 1023];
  sink[tid] = acc;
  if (tid == 0) out_cycles[0] = t1 - t0;
}

// PATH 1 NO-BAR: same as Path 1 but without bar.sync. Isolates the store
// cost.
__global__ void kernel_r2s_nobar(int n_iters, uint64_t *out_cycles, uint32_t *sink) {
  __shared__ uint32_t smem_buf[kBlockN * kKChunk / 2];
  uint32_t tid = threadIdx.x;
  uint32_t regs[16];
  PRAGMA_UNROLL
  for (int i = 0; i < 16; i++) regs[i] = tid * 16u + i + sink[tid];
  __syncthreads();

  uint64_t t0;
  PRAGMA_UNROLL
  for (int rep = 0; rep < 2; rep++) {
    if (rep == 1) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    for (int it = 0; it < n_iters; it++) {
      PRAGMA_UNROLL
      for (int s = 0; s < kStoresPerThreadR2S; s++) {
        uint32_t off = (tid * 16u + s + (it * 7u)) & (kBlockN * kKChunk / 2 - 1u);
        smem_buf[off] = regs[s] + it;
      }
      // No bar.sync
    }
  }
  uint64_t t1;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

  uint32_t acc = 0;
  PRAGMA_UNROLL
  for (int s = 0; s < 16; s++) acc += smem_buf[(tid * 16 + s) & 1023];
  sink[tid] = acc;
  if (tid == 0) out_cycles[0] = t1 - t0;
}

// PATH 1p: r2s with humming's PRODUCTION per-element addressing (the real
// swizzle formula from tcgen05_mma.cuh::run pre-track-f). Unlike Path 1's
// synthetic `(tid*16+s+it*7) & 1023` pattern -- which strides 16 words
// between adjacent lanes and is therefore ~16-way bank conflicted, nothing
// like the production kernel -- this reproduces the real (n, k) -> swizzled
// offset math and the real conflict-free access pattern.
__global__ void kernel_r2s_prod_addr(int n_iters, uint64_t *out_cycles,
                                     uint32_t *sink) {
  // One full b_dequant slot: BlockN=128 rows x 64 K-bf16 sections x 2 B.
  __shared__ uint32_t smem_buf[kBlockN * 64 / 2];  // 16 KiB
  uint32_t tid = threadIdx.x;
  uint32_t regs[16];
  PRAGMA_UNROLL
  for (int i = 0; i < 16; i++) regs[i] = tid * 16u + i + sink[tid];
  __syncthreads();

  uint32_t t = tid % 32u;
  uint32_t n_base = (tid / 32u % 2u) * 64u;  // kNWarps=2 at BlockN=128
  uint32_t smem_base_div_128 = cast_smem_ptr_to_uint(smem_buf) >> 7;

  uint64_t t0;
  PRAGMA_UNROLL
  for (int rep = 0; rep < 2; rep++) {
    if (rep == 1) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    for (int it = 0; it < n_iters; it++) {
      uint32_t k_base = (it % 4) * kKChunk;  // stay inside one section
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 4u; i++) {
        PRAGMA_UNROLL
        for (uint32_t frag = 0; frag < 2u; frag++) {
          uint32_t n = n_base + i * 16u + 8u * frag + t / 4u;
          PRAGMA_UNROLL
          for (uint32_t pair = 0; pair < 2u; pair++) {
            uint32_t k_lo = k_base + 2u * (t % 4u) + 8u * pair;
            uint32_t lin = n * 128u + (k_lo % 64u) * 2u;
            uint32_t xs = (smem_base_div_128 + (lin >> 7)) & 7u;
            uint32_t sw = lin ^ (xs << 4);
            smem_buf[sw / 4u] = regs[i * 4u + frag * 2u + pair] + it;
          }
        }
      }
      asm volatile("bar.sync 1, %0;" :: "n"(kMathThreads));
    }
  }
  uint64_t t1;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

  uint32_t acc = 0;
  PRAGMA_UNROLL
  for (int s = 0; s < 16; s++) acc += smem_buf[(tid * 16 + s) & 4095];
  sink[tid] = acc;
  if (tid == 0) out_cycles[0] = t1 - t0;
}

// PATH 1c: r2s with the track-f CLOSED-FORM addressing -- swizzle XOR phase
// hoisted to a per-thread constant, all per-store offsets folded into STS
// immediates off two base registers.
__global__ void kernel_r2s_closed_form(int n_iters, uint64_t *out_cycles,
                                       uint32_t *sink) {
  __shared__ uint32_t smem_buf[kBlockN * 64 / 2];  // 16 KiB
  uint32_t tid = threadIdx.x;
  uint32_t regs[16];
  PRAGMA_UNROLL
  for (int i = 0; i < 16; i++) regs[i] = tid * 16u + i + sink[tid];
  __syncthreads();

  uint32_t t = tid % 32u;
  uint32_t n_base = (tid / 32u % 2u) * 64u;
  uint32_t n0 = n_base + t / 4u;
  uint32_t pre = n0 * 128u + (t % 4u) * 4u;
  uint32_t mask = (((cast_smem_ptr_to_uint(smem_buf) >> 7) + n0) & 7u) << 4;
  uint32_t base_thread = pre ^ mask;
  char *smem_bytes = reinterpret_cast<char *>(smem_buf);

  uint64_t t0;
  PRAGMA_UNROLL
  for (int rep = 0; rep < 2; rep++) {
    if (rep == 1) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    for (int it = 0; it < n_iters; it++) {
      uint32_t base0 = base_thread ^ ((it % 4) * 32u);
      uint32_t base1 = base0 ^ 16u;
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 4u; i++) {
        PRAGMA_UNROLL
        for (uint32_t frag = 0; frag < 2u; frag++) {
          uint32_t imm = i * 2048u + frag * 1024u;
          PRAGMA_UNROLL
          for (uint32_t pair = 0; pair < 2u; pair++) {
            uint32_t addr = (pair ? base1 : base0) + imm;
            *reinterpret_cast<uint32_t *>(smem_bytes + addr) =
                regs[i * 4u + frag * 2u + pair] + it;
          }
        }
      }
      asm volatile("bar.sync 1, %0;" :: "n"(kMathThreads));
    }
  }
  uint64_t t1;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

  uint32_t acc = 0;
  PRAGMA_UNROLL
  for (int s = 0; s < 16; s++) acc += smem_buf[(tid * 16 + s) & 4095];
  sink[tid] = acc;
  if (tid == 0) out_cycles[0] = t1 - t0;
}

// PATH 2 (8-warp): all 8 math warps issue tcgen05.st per K-iter.
__global__ void kernel_r2t_8warp(int n_iters, uint64_t *out_cycles, uint32_t *sink) {
  __shared__ uint32_t smem_tmem_col;
  uint32_t tid = threadIdx.x;

  if (tid < 32) tcgen05_alloc<256>(cast_smem_ptr_to_uint(&smem_tmem_col));
  __syncthreads();
  uint32_t tmem_base = smem_tmem_col;
  uint32_t warp_id = tid / 32u;

  uint32_t regs[32];
  PRAGMA_UNROLL
  for (int i = 0; i < 32; i++) regs[i] = tid * 32u + i + sink[tid];

  uint32_t my_tmem = tmem_base + warp_id * 32u;

  uint64_t t0;
  PRAGMA_UNROLL
  for (int rep = 0; rep < 2; rep++) {
    if (rep == 1) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    for (int it = 0; it < n_iters; it++) {
      // Mutate regs per-iter to avoid DCE.
      regs[0] += it;
      tcgen05_st_32x32b_x32(my_tmem, regs);
      tcgen05_fence_before_thread_sync();
    }
  }
  uint64_t t1;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

  // Drain: read back from TMEM so the writes can't be DCE'd.
  uint32_t dst[32];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      " %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : "=r"(dst[0]),"=r"(dst[1]),"=r"(dst[2]),"=r"(dst[3]),
        "=r"(dst[4]),"=r"(dst[5]),"=r"(dst[6]),"=r"(dst[7]),
        "=r"(dst[8]),"=r"(dst[9]),"=r"(dst[10]),"=r"(dst[11]),
        "=r"(dst[12]),"=r"(dst[13]),"=r"(dst[14]),"=r"(dst[15]),
        "=r"(dst[16]),"=r"(dst[17]),"=r"(dst[18]),"=r"(dst[19]),
        "=r"(dst[20]),"=r"(dst[21]),"=r"(dst[22]),"=r"(dst[23]),
        "=r"(dst[24]),"=r"(dst[25]),"=r"(dst[26]),"=r"(dst[27]),
        "=r"(dst[28]),"=r"(dst[29]),"=r"(dst[30]),"=r"(dst[31])
      : "r"(my_tmem) : "memory");
  uint32_t acc = 0;
  PRAGMA_UNROLL
  for (int s = 0; s < 32; s++) acc += dst[s];
  sink[tid] = acc;

  __syncthreads();
  if (tid < 32) {
    tcgen05_relinquish_alloc_permit();
    tcgen05_dealloc<256>(tmem_base);
  }

  if (tid == 0) out_cycles[0] = t1 - t0;
}

// PATH 2 (2-warp): only 2 warps issue tcgen05.st per K-iter (one per N-warp
// in a hypothetical TS-mode mainloop where the M-warp redundancy goes away).
// In a real TS mainloop the issuer doesn't need a 256-thread bar.sync, only
// a tcgen05.fence (warp-local) + an mbarrier the issuer waits on. This bench
// keeps a bar.sync to model the cross-warp ordering -- in a final TS mainloop
// the bar can be replaced by an mbar arrive/wait pattern.
__global__ void kernel_r2t_2warp(int n_iters, uint64_t *out_cycles, uint32_t *sink) {
  __shared__ uint32_t smem_tmem_col;
  uint32_t tid = threadIdx.x;

  if (tid < 32) tcgen05_alloc<128>(cast_smem_ptr_to_uint(&smem_tmem_col));
  __syncthreads();
  uint32_t tmem_base = smem_tmem_col;
  uint32_t warp_id = tid / 32u;
  bool is_scatter_warp = (warp_id < 2);

  uint32_t regs[32];
  PRAGMA_UNROLL
  for (int i = 0; i < 32; i++) regs[i] = tid * 32u + i + sink[tid];

  uint32_t my_tmem = tmem_base + warp_id * 32u;

  uint64_t t0;
  PRAGMA_UNROLL
  for (int rep = 0; rep < 2; rep++) {
    if (rep == 1) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    for (int it = 0; it < n_iters; it++) {
      if (is_scatter_warp) {
        regs[0] += it;
        tcgen05_st_32x32b_x32(my_tmem, regs);
        tcgen05_fence_before_thread_sync();
      }
      asm volatile("bar.sync 1, %0;" :: "n"(kMathThreads));
    }
  }
  uint64_t t1;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

  // Drain: every thread does a ld so it's hard to DCE.
  uint32_t dst[32];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      " %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : "=r"(dst[0]),"=r"(dst[1]),"=r"(dst[2]),"=r"(dst[3]),
        "=r"(dst[4]),"=r"(dst[5]),"=r"(dst[6]),"=r"(dst[7]),
        "=r"(dst[8]),"=r"(dst[9]),"=r"(dst[10]),"=r"(dst[11]),
        "=r"(dst[12]),"=r"(dst[13]),"=r"(dst[14]),"=r"(dst[15]),
        "=r"(dst[16]),"=r"(dst[17]),"=r"(dst[18]),"=r"(dst[19]),
        "=r"(dst[20]),"=r"(dst[21]),"=r"(dst[22]),"=r"(dst[23]),
        "=r"(dst[24]),"=r"(dst[25]),"=r"(dst[26]),"=r"(dst[27]),
        "=r"(dst[28]),"=r"(dst[29]),"=r"(dst[30]),"=r"(dst[31])
      : "r"(tmem_base) : "memory");
  uint32_t acc = 0;
  PRAGMA_UNROLL
  for (int s = 0; s < 32; s++) acc += dst[s];
  sink[tid] = acc;

  __syncthreads();
  if (tid < 32) {
    tcgen05_relinquish_alloc_permit();
    tcgen05_dealloc<128>(tmem_base);
  }

  if (tid == 0) out_cycles[0] = t1 - t0;
}

// ----- host driver ----------------------------------------------------------

int main() {
  const int n_iters = 10000;
  uint64_t *d_cyc;
  uint32_t *d_sink;
  cudaMalloc(&d_cyc, sizeof(uint64_t));
  cudaMalloc(&d_sink, kMathThreads * sizeof(uint32_t));
  cudaMemset(d_sink, 0, kMathThreads * sizeof(uint32_t));

  auto run = [&](const char *label, auto kernel) {
    cudaMemset(d_cyc, 0xff, sizeof(uint64_t));
    kernel<<<1, kMathThreads>>>(n_iters, d_cyc, d_sink);
    cudaError_t launch_err = cudaGetLastError();
    cudaError_t err = cudaDeviceSynchronize();
    if (launch_err != cudaSuccess) {
      printf("  %-50s LAUNCH-ERR: %s\n", label, cudaGetErrorString(launch_err));
      return;
    }
    if (err != cudaSuccess) {
      printf("  %-50s SYNC-ERR: %s\n", label, cudaGetErrorString(err));
      return;
    }
    uint64_t cyc;
    uint32_t s0;
    cudaMemcpy(&cyc, d_cyc, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&s0, d_sink, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("  %-50s %8.1f cyc/iter  (raw=%llu sink0=%u)\n",
           label, double(cyc) / n_iters,
           (unsigned long long)cyc, s0);
  };

  printf("Per K-iter cycle cost (1 CTA, 256 math threads, %d iters):\n", n_iters);
  run("Path 1  (r2s + bar.sync, 16 stores/thread):", kernel_r2s);
  run("Path 1' (r2s NO bar -- store cost alone):", kernel_r2s_nobar);
  run("Path 1p (r2s PROD addressing + bar.sync):", kernel_r2s_prod_addr);
  run("Path 1c (r2s CLOSED-FORM addr + bar.sync):", kernel_r2s_closed_form);
  run("Path 2a (r2t 8-warp + fence):", kernel_r2t_8warp);
  run("Path 2b (r2t 2-warp + fence + bar.sync):", kernel_r2t_2warp);

  cudaFree(d_cyc);
  cudaFree(d_sink);
  return 0;
}
