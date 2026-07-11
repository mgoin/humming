"""
Standalone POC: measure tcgen05.mma throughput on Blackwell (sm_100/103).

Goal: see if we can actually push past the mma.sync ceiling (~580 TF bf16 on B300)
by switching the bench kernel to the Blackwell-native tcgen05.mma instruction.
"""

import argparse
import ctypes
import dataclasses
from typing import ClassVar

import cuda.bindings.driver as cbd
import torch
import triton

from humming.jit.runtime import KernelRuntime


# Instruction descriptor layout for tcgen05.mma .kind::f16 / .kind::tf32:
#   bits 0-1   : sparsity selector   = 0
#   bit 2      : sparsity            = 0 (dense)
#   bit 3      : saturate            = 0
#   bits 4-5   : D dtype             (F16=0, F32=1)
#   bit 6      : reserved            = 0
#   bits 7-9   : A dtype             (F16=0, BF16=1, TF32=2)
#   bits 10-12 : B dtype
#   bit 13     : negate A
#   bit 14     : negate B
#   bit 15     : transpose A
#   bit 16     : transpose B
#   bits 17-22 : N >> 3
#   bit 23     : reserved
#   bits 24-28 : M >> 4
#   bit 29     : reserved
#   bits 30-31 : max-shift (.ws only)
def make_idesc(m: int, n: int, ab_type_id: int, d_type_id: int,
               transpose_a: int = 0, transpose_b: int = 0) -> int:
    assert m % 16 == 0 and (m >> 4) < 32
    assert n % 8 == 0 and (n >> 3) < 64
    v = 0
    v |= (d_type_id & 0x3) << 4
    v |= (ab_type_id & 0x7) << 7
    v |= (ab_type_id & 0x7) << 10
    v |= (transpose_a & 1) << 15
    v |= (transpose_b & 1) << 16
    v |= ((n >> 3) & 0x3F) << 17
    v |= ((m >> 4) & 0x1F) << 24
    return v & 0xFFFFFFFF


# Shared memory descriptor for tcgen05 (Blackwell).
# Different from WGMMA descriptor: bits 46-48 must be 0b001.
#   bits 0-13   : (addr >> 4) & 0x3FFF
#   bits 16-29  : (leading >> 4) & 0x3FFF
#   bits 32-45  : (stride  >> 4) & 0x3FFF
#   bits 46-48  : 0b001
#   bits 49-51  : matrix base offset (0 if 1024-aligned for 128B swizzle)
#   bit 52      : leading-dim mode (0=relative offset)
#   bits 61-63  : swizzle (0=none, 2=128B, 4=64B, 6=32B)
def make_smem_desc_init(leading: int, stride: int, swizzle: int) -> int:
    v = 0
    v |= ((leading >> 4) & 0x3FFF) << 16
    v |= ((stride >> 4) & 0x3FFF) << 32
    v |= 1 << 46
    v |= (swizzle & 0x7) << 61
    return v


CODE_TEMPLATE = r"""
#include <cstdint>
#include <cuda.h>
#include <humming/utils/base.cuh>

// 64-bit SMEM descriptor: we patch the low 14 bits at runtime with the actual SMEM addr.
__device__ __forceinline__ uint64_t make_desc(void *smem_ptr, uint64_t desc_init) {
  uint32_t addr = cast_smem_ptr_to_uint(smem_ptr);
  return desc_init | (((uint64_t)(addr) >> 4) & 0x3FFF);
}

// cta_group::1 kernel — 1 CTA per SM, M up to 128, ~half of B300 tensor-core peak
template <uint32_t kM, uint32_t kN, uint32_t kK, uint32_t kABBits, uint32_t kAccBits,
          uint32_t kIDesc, uint64_t kDescAInit, uint64_t kDescBInit,
          uint32_t kRepeatCount, uint32_t kUnrollCount>
__global__ void tcgen05_bench(uint32_t *out_ptr) {
  constexpr uint32_t kSmemABytes = kM * kK * kABBits / 8;
  constexpr uint32_t kSmemBBytes = kK * kN * kABBits / 8;
  constexpr uint32_t kTmemColsRaw = (kAccBits == 32) ? kN : (kN / 2);
  constexpr uint32_t kTmemCols =
      kTmemColsRaw <= 32 ? 32 : (kTmemColsRaw <= 64 ? 64 : (kTmemColsRaw <= 128 ? 128
                                : kTmemColsRaw <= 256 ? 256 : 512));

  __shared__ alignas(1024) char smem_a[kSmemABytes];
  __shared__ alignas(1024) char smem_b[kSmemBBytes];
  __shared__ alignas(16) uint32_t tmem_ptr_slot;
  __shared__ alignas(8) uint64_t mbar;

  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 :: "r"(cast_smem_ptr_to_uint(&mbar)) : "memory");
  }
  if (threadIdx.x < 32) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :: "r"(cast_smem_ptr_to_uint(&tmem_ptr_slot)), "n"(kTmemCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
  }
  __syncthreads();
  uint32_t d_tmem = tmem_ptr_slot;

  uint64_t desc_a = make_desc(smem_a, kDescAInit);
  uint64_t desc_b = make_desc(smem_b, kDescBInit);

  if (threadIdx.x == 0) {
    PRAGMA_UNROLL_COUNT(kUnrollCount)
    for (uint32_t i = 0; i < kRepeatCount; i++) {
      asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, 1, 0;\n"
        "  tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
        "}\n"
        :: "r"(d_tmem), "l"(desc_a), "l"(desc_b), "r"(kIDesc)
        : "memory"
      );
    }
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
                 :: "r"(cast_smem_ptr_to_uint(&mbar)) : "memory");
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    asm volatile(
      "{\n"
      "  .reg .pred P;\n"
      "LAB_WAIT:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0;\n"
      "  @P bra END_WAIT;\n"
      "  bra LAB_WAIT;\n"
      "END_WAIT:\n"
      "}\n"
      :: "r"(cast_smem_ptr_to_uint(&mbar)) : "memory"
    );
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :: "r"(d_tmem), "n"(kTmemCols) : "memory");
  }

  if (blockIdx.x == 8192) out_ptr[0] = d_tmem;
}


// cta_group::2 kernel — pair two CTAs into one tensor-core operation.
// The cluster dim is (2,1,1); the "leader" (cluster CTA 0, blockIdx.x even) issues
// the mma, the "follower" (cluster CTA 1, blockIdx.x odd) only participates in alloc/dealloc.
// idesc.M is the *total* M across the two CTAs (each holds M/2 of the accumulator in its own
// TMEM). For .kind::f16 the supported total-M is {128, 256}.
template <uint32_t kM, uint32_t kN, uint32_t kK, uint32_t kABBits, uint32_t kAccBits,
          uint32_t kIDesc, uint64_t kDescAInit, uint64_t kDescBInit,
          uint32_t kRepeatCount, uint32_t kUnrollCount>
__global__ __cluster_dims__(2, 1, 1) void tcgen05_bench_cg2(uint32_t *out_ptr) {
  constexpr uint32_t kSmemABytes = (kM / 2) * kK * kABBits / 8;  // per-CTA A: M/2 rows
  constexpr uint32_t kSmemBBytes = kK * kN * kABBits / 8;        // B is shared
  // TMEM cols per CTA: each CTA holds M/2 rows of the M*N accumulator.
  // Each lane in TMEM holds one row's worth; with 128 lanes and (M/2) ≤ 128, layout fits.
  constexpr uint32_t kTmemColsRaw = (kAccBits == 32) ? kN : (kN / 2);
  constexpr uint32_t kTmemCols =
      kTmemColsRaw <= 32 ? 32 : (kTmemColsRaw <= 64 ? 64 : (kTmemColsRaw <= 128 ? 128
                                : kTmemColsRaw <= 256 ? 256 : 512));

  __shared__ alignas(1024) char smem_a[kSmemABytes];
  __shared__ alignas(1024) char smem_b[kSmemBBytes];
  __shared__ alignas(16) uint32_t tmem_ptr_slot;
  __shared__ alignas(8) uint64_t mbar;

  const bool is_leader = (blockIdx.x & 1) == 0;

  if (is_leader && threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 :: "r"(cast_smem_ptr_to_uint(&mbar)) : "memory");
  }
  // Both CTAs must participate in cta_group::2 alloc.
  if (threadIdx.x < 32) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :: "r"(cast_smem_ptr_to_uint(&tmem_ptr_slot)), "n"(kTmemCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n" ::: "memory");
  }
  __syncthreads();
  uint32_t d_tmem = tmem_ptr_slot;

  uint64_t desc_a = make_desc(smem_a, kDescAInit);
  uint64_t desc_b = make_desc(smem_b, kDescBInit);

  // Only leader issues the mma loop.
  if (is_leader && threadIdx.x == 0) {
    PRAGMA_UNROLL_COUNT(kUnrollCount)
    for (uint32_t i = 0; i < kRepeatCount; i++) {
      // p=0 → don't accumulate, breaks the TMEM RAW chain to expose pure issue rate
      asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, 0, 0;\n"
        "  tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n"
        "}\n"
        :: "r"(d_tmem), "l"(desc_a), "l"(desc_b), "r"(kIDesc)
        : "memory"
      );
    }
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
                 :: "r"(cast_smem_ptr_to_uint(&mbar)) : "memory");
  }
  // Cluster-wide barrier so follower waits for leader before deallocating.
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");

  if (is_leader && threadIdx.x == 0) {
    asm volatile(
      "{\n"
      "  .reg .pred P;\n"
      "LAB_WAIT:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0;\n"
      "  @P bra END_WAIT;\n"
      "  bra LAB_WAIT;\n"
      "END_WAIT:\n"
      "}\n"
      :: "r"(cast_smem_ptr_to_uint(&mbar)) : "memory"
    );
  }
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");

  if (threadIdx.x < 32) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n"
                 :: "r"(d_tmem), "n"(kTmemCols) : "memory");
  }

  if (blockIdx.x == 8192) out_ptr[0] = d_tmem;
}
"""


# ab dtype id (atype/btype field in idesc), accumulator dtype id (D field)
DTYPE_INFO = {
    # name      ab_bits  ab_id  acc_bits  acc_id
    "float16":  (16,     0,     32,       1),
    "bfloat16": (16,     1,     32,       1),
}


@dataclasses.dataclass(kw_only=True)
class Tcgen05BenchKernel(KernelRuntime):
    name: ClassVar[str] = "tcgen05_bench"
    ab_dtype: str
    mma_m: int
    mma_n: int
    mma_k: int
    repeat_count: int
    unroll_count: int
    cta_group: int = 1  # 1 or 2

    def init_kernel(self):
        assert self.cta_group in (1, 2)
        ab_bits, ab_id, acc_bits, acc_id = DTYPE_INFO[self.ab_dtype]
        # For cta_group::2, idesc.M is the total M across the cluster.
        idesc = make_idesc(self.mma_m, self.mma_n, ab_id, acc_id)
        desc_a_init = make_smem_desc_init(leading=32, stride=32, swizzle=0)
        desc_b_init = make_smem_desc_init(leading=32, stride=32, swizzle=0)

        self.code = CODE_TEMPLATE
        func_name = "tcgen05_bench" if self.cta_group == 1 else "tcgen05_bench_cg2"
        self.name = func_name  # for find_kernel_name_in_cubin
        self.kernel_expr = (
            f"{func_name}<{self.mma_m}, {self.mma_n}, {self.mma_k}, "
            f"{ab_bits}, {acc_bits}, "
            f"{idesc}u, {desc_a_init}ull, {desc_b_init}ull, "
            f"{self.repeat_count}, {self.unroll_count}>"
        )
        self.arg_types = (ctypes.c_void_p,)
        self.prepare()

        self.sm_count = torch.cuda.get_device_properties().multi_processor_count
        # tcgen05.mma is per-CTA-group. cg1 → 1 CTA/SM (148 CTAs). cg2 → 2 CTAs/cluster,
        # one cluster per pair of SMs (148 CTAs total still).
        # Use num_ctas = sm_count for both. cg2 requires it to be a multiple of 2.
        self.num_ctas = self.sm_count if self.cta_group == 1 else (self.sm_count // 2) * 2
        self.block_threads = 32

        # ops per single mma instruction (idesc.M is total M for cg2)
        self.ops_per_mma = self.mma_m * self.mma_n * self.mma_k * 2
        # For cg1: kRepeatCount mmas per CTA → num_ctas independent issue queues.
        # For cg2: kRepeatCount mmas per cluster, num_clusters = num_ctas/2.
        if self.cta_group == 1:
            self.ops_per_call = self.ops_per_mma * self.num_ctas
        else:
            self.ops_per_call = self.ops_per_mma * (self.num_ctas // 2)

    def __call__(self):
        self.check_context()
        config = cbd.CUlaunchConfig()
        config.gridDimX = self.num_ctas
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = self.block_threads
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream().cuda_stream

        # cluster attribute for cta_group::2: cluster of 2 CTAs along X
        if self.cta_group == 2:
            attr = cbd.CUlaunchAttribute()
            attr.id = cbd.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            attr.value.clusterDim.x = 2
            attr.value.clusterDim.y = 1
            attr.value.clusterDim.z = 1
            config.attrs = [attr]
            config.numAttrs = 1

        tensor = torch.empty((1,), dtype=torch.uint32, device="cuda:0")
        arg_values = (tensor.data_ptr(),)

        result = cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), 0)
        assert result[0] == 0, repr(result)


def run(dtype: str, mma_m: int, mma_n: int, mma_k: int,
        repeat_count: int = 65536, unroll_count: int = 64,
        cta_group: int = 1) -> float:
    kernel = Tcgen05BenchKernel(
        ab_dtype=dtype,
        mma_m=mma_m, mma_n=mma_n, mma_k=mma_k,
        repeat_count=repeat_count, unroll_count=unroll_count,
        cta_group=cta_group,
    )
    ops_per_call = kernel.ops_per_call
    t = triton.testing.do_bench(kernel, warmup=100, rep=1000)
    return repeat_count * ops_per_call / t / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=list(DTYPE_INFO), default="bfloat16")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=65536)
    parser.add_argument("--unroll", type=int, default=64)
    parser.add_argument("--cta_group", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()

    tflops = run(args.dtype, args.m, args.n, args.k,
                 repeat_count=args.repeat, unroll_count=args.unroll,
                 cta_group=args.cta_group)
    print(f"tcgen05 cg{args.cta_group} {args.dtype} m{args.m}n{args.n}k{args.k}: {tflops:.1f} TFLOPS")


if __name__ == "__main__":
    main()
