"""Standalone unit test for the vectorized transposed TMEM drain
(humming/epilogue/tmem_ts_drain.cuh).

Fills the TMEM D region with a known [n=128, m=BlockM] f32 pattern via
tcgen05.st (the same transposed orientation the TS-mode MMA produces),
drains it into a smem.reduce-shaped buffer with either the scalar
reference (mirroring tcgen05_ts_mma.cuh) or the vectorized 8x8
transpose + 128-bit-store routine, copies SMEM to gmem, and asserts
bit-exactness against a Python model of the sectioned XOR-swizzled
layout. Guards: the register-transpose bookkeeping (3 shfl stages +
prmt) and the swizzled int4 addressing must place every bf16 exactly
where gmem_writer expects it.
"""

import ctypes
import dataclasses
from typing import ClassVar

import cuda.bindings.driver as cbd
import pytest
import torch

from humming.jit.runtime import KernelRuntime

CODE = r"""
#include <cstdint>
#include <cuda.h>
#include <humming/utils/base.cuh>
#include <humming/utils/ptx/tcgen05.cuh>
#include <humming/epilogue/tmem_ts_drain.cuh>

template <uint32_t kBlockM, uint32_t kSmemBase, bool kVectorized>
__global__ void tmem_drain_test(const float *src, int4 *out) {
  __shared__ alignas(16) uint32_t tmem_slot;
  constexpr uint32_t kReduceInt4 = 2u * kBlockM * 8u;  // 2 sections
  __shared__ alignas(1024) int4 reduce[kReduceInt4];

  if (threadIdx.x < 32) {
    tcgen05_alloc<128>(cast_smem_ptr_to_uint(&tmem_slot));
  }
  __syncthreads();
  uint32_t tmem_base = tmem_slot;

  uint32_t warp = threadIdx.x / 32u;
  uint32_t lane = threadIdx.x % 32u;
  uint32_t n = (warp % 4u) * 32u + lane;

  // Fill TMEM (lane = n, col = m) with the pattern via r2t.
  for (uint32_t col0 = 0; col0 < kBlockM; col0 += 8u) {
    uint32_t vals[8];
    PRAGMA_UNROLL
    for (uint32_t j = 0; j < 8u; j++) {
      float f = src[n * kBlockM + col0 + j];
      vals[j] = *reinterpret_cast<uint32_t *>(&f);
    }
    uint32_t addr = (tmem_base + col0) | ((warp % 4u) * 32u << 16);
    tcgen05_st_32x32b_x8(addr, vals);
  }
  tcgen05_wait_st();
  tcgen05_fence_before_thread_sync();
  __syncthreads();
  tcgen05_fence_after_thread_sync();

  if constexpr (kVectorized) {
    tmem_ts_drain_transposed<kBlockM>(tmem_base, n, reduce, kSmemBase, 0.0f);
  } else {
    // Scalar reference: mirrors tcgen05_ts_mma.cuh final_regs_c_as_ptr.
    uint16_t *red16 = reinterpret_cast<uint16_t *>(reduce);
    uint32_t section_row_base = (n / 64u) * kBlockM;
    uint32_t section_col = (n / 8u) % 8u;
    PRAGMA_UNROLL
    for (uint32_t chunk = 0; chunk < kBlockM / 32u; chunk++) {
      uint32_t tmp[32];
      tcgen05_ld_32x32b_x32(tmem_base + chunk * 32u, tmp);
      tcgen05_wait_ld();
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 32u; i++) {
        uint32_t m = chunk * 32u + i;
        float f = *reinterpret_cast<float *>(&tmp[i]);
        uint32_t smem_row = section_row_base + m;
        uint32_t col = section_col ^ ((smem_row + kSmemBase) % 8u);
        __nv_bfloat16 fb = __float2bfloat16(f);
        red16[(smem_row * 8u + col) * 8u + (n % 8u)] =
            *reinterpret_cast<uint16_t *>(&fb);
      }
    }
  }
  __syncthreads();

  for (uint32_t i = threadIdx.x; i < kReduceInt4; i += blockDim.x) {
    out[i] = reduce[i];
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05_relinquish_alloc_permit();
    tcgen05_dealloc<128>(tmem_slot);
  }
}
"""


@dataclasses.dataclass(kw_only=True)
class TmemDrainTest(KernelRuntime):
    name: ClassVar[str] = "tmem_drain_test"
    block_m: int
    smem_base: int
    vectorized: bool

    def init_kernel(self):
        self.code = CODE
        self.kernel_expr = (
            f"tmem_drain_test<{self.block_m}u, {self.smem_base}u, "
            f"{'true' if self.vectorized else 'false'}>"
        )
        self.arg_types = (ctypes.c_void_p, ctypes.c_void_p)
        self.prepare()

    def __call__(self, src, out):
        self.check_context()
        config = cbd.CUlaunchConfig()
        config.gridDimX = 1
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 128
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream().cuda_stream
        arg_values = (src.data_ptr(), out.data_ptr())
        result = cbd.cuLaunchKernelEx(
            config, self.kernel, (arg_values, self.arg_types), 0)
        assert result[0] == cbd.CUresult.CUDA_SUCCESS, repr(result)


def expected_reduce(src: torch.Tensor, block_m: int,
                    smem_base: int) -> torch.Tensor:
    """Python model of gmem_writer's sectioned XOR-swizzled layout."""
    bf = src.to(torch.bfloat16).view(torch.uint16).cpu().to(torch.int32)
    out = torch.zeros(2 * block_m * 8 * 8, dtype=torch.int32)
    n_idx = torch.arange(128).view(128, 1).expand(128, block_m)
    m_idx = torch.arange(block_m).view(1, block_m).expand(128, block_m)
    row = (n_idx // 64) * block_m + m_idx
    col = ((n_idx // 8) % 8) ^ ((row + smem_base) % 8)
    flat = (row * 8 + col) * 8 + (n_idx % 8)
    out[flat.reshape(-1)] = bf.reshape(-1)
    return out.to(torch.uint16)


@pytest.mark.parametrize("block_m", [64, 128])
@pytest.mark.parametrize("smem_base", [0, 3])
@pytest.mark.parametrize("vectorized", [False, True])
def test_tmem_ts_drain(block_m, smem_base, vectorized):
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("tcgen05 requires sm_100+")
    torch.manual_seed(7)
    src = torch.randn(128, block_m, dtype=torch.float32, device="cuda")
    out = torch.zeros(2 * block_m * 8 * 8, dtype=torch.uint16,
                      device="cuda").view(torch.uint16)
    kernel = TmemDrainTest(block_m=block_m, smem_base=smem_base,
                           vectorized=vectorized)
    # int4 buffer aliases the uint16 tensor (8 uint16 per int4).
    kernel(src, out)
    torch.cuda.synchronize()
    exp = expected_reduce(src, block_m, smem_base)
    got = out.cpu()
    if not torch.equal(got, exp):
        bad = (got != exp).nonzero().flatten()
        raise AssertionError(
            f"{bad.numel()} mismatched u16 slots; first 10: "
            f"{bad[:10].tolist()}; got {got[bad[:10]].tolist()} "
            f"exp {exp[bad[:10]].tolist()}")
