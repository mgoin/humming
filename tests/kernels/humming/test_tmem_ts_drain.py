"""Standalone unit test for the vectorized transposed TMEM drain.

Fills the TMEM D region with a known [n=128, m=BlockM] f32 pattern via
tcgen05.st -- the transposed orientation the TS-mode MMA produces -- drains it
with either the scalar reference or the vectorized 8x8 transpose, and compares
against a Python model of gmem_writer's sectioned XOR-swizzled smem.reduce
layout. Guards the register-transpose bookkeeping and the swizzled addressing
independently of the mainloop.
"""

import ctypes
import dataclasses
from typing import ClassVar

import cuda.bindings.driver as cbd
import pytest
import torch

from humming.jit.runtime import KernelRuntime
from humming.testing import skip_if_unsupported

CODE = r"""
#include <cstdint>
#include <cuda.h>
#include <humming/utils/base.cuh>
#include <humming/utils/ptx/tcgen05.cuh>
#include <humming/epilogue/tmem_ts_drain.cuh>

template <uint32_t kBlockM, uint32_t kSmemBase, bool kVectorized, bool kApplyRowScale>
__global__ void tmem_drain_test(const float *src, int4 *out, float bias_val, float scale_val) {
  __shared__ alignas(16) uint32_t tmem_slot;
  constexpr uint32_t kReduceInt4 = 2u * kBlockM * 8u;  // two 64-row sections
  __shared__ alignas(1024) int4 reduce[kReduceInt4];

  if (threadIdx.x < 32) {
    tcgen05_alloc<128>(cast_smem_ptr_to_uint(&tmem_slot));
  }
  __syncthreads();
  uint32_t tmem_base = tmem_slot;

  uint32_t warp = threadIdx.x / 32u;
  uint32_t lane = threadIdx.x % 32u;
  uint32_t n = (warp % 4u) * 32u + lane;

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
    tmem_ts_drain_transposed<kBlockM, BFloat16, kApplyRowScale>(
        tmem_base, n, reduce, kSmemBase, bias_val, scale_val);
  } else {
    // Scalar reference: the 2-byte scatter the vectorized drain replaced.
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
        if constexpr (kApplyRowScale) {
          f *= scale_val;
        }
        uint32_t smem_row = section_row_base + m;
        uint32_t col = section_col ^ ((smem_row + kSmemBase) % 8u);
        __nv_bfloat16 fb = __float2bfloat16(f + bias_val);
        red16[(smem_row * 8u + col) * 8u + (n % 8u)] = *reinterpret_cast<uint16_t *>(&fb);
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
    apply_row_scale: bool = False

    def init_kernel(self):
        self.code = CODE
        self.kernel_expr = (
            f"tmem_drain_test<{self.block_m}u, {self.smem_base}u, "
            f"{'true' if self.vectorized else 'false'}, "
            f"{'true' if self.apply_row_scale else 'false'}>"
        )
        self.arg_types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_float)
        self.prepare()

    def __call__(self, src, out, bias_val, scale_val):
        self.check_context()
        config = cbd.CUlaunchConfig()
        config.gridDimX = 1
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 128
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream().cuda_stream
        arg_values = (src.data_ptr(), out.data_ptr(), bias_val, scale_val)
        result = cbd.cuLaunchKernelEx(config, self.func, (arg_values, self.arg_types), 0)
        assert result[0] == cbd.CUresult.CUDA_SUCCESS, repr(result)


def _expected_reduce(src, block_m, smem_base, bias_val, scale_val):
    values = (src * scale_val + bias_val).to(torch.bfloat16).view(torch.uint16).cpu().to(torch.int32)
    expected = torch.zeros(2 * block_m * 8 * 8, dtype=torch.int32)
    n_index = torch.arange(128).view(128, 1).expand(128, block_m)
    m_index = torch.arange(block_m).view(1, block_m).expand(128, block_m)
    row = (n_index // 64) * block_m + m_index
    col = ((n_index // 8) % 8) ^ ((row + smem_base) % 8)
    flat = (row * 8 + col) * 8 + (n_index % 8)
    expected[flat.reshape(-1)] = values.reshape(-1)
    return expected.to(torch.uint16)


def _run_drain(block_m, smem_base, vectorized, apply_row_scale, bias_val, scale_val):
    skip_if_unsupported(mma_type="tcgen05")
    torch.manual_seed(7)
    src = torch.randn(128, block_m, dtype=torch.float32, device="cuda")
    out = torch.zeros(2 * block_m * 8 * 8, dtype=torch.uint16, device="cuda")
    kernel = TmemDrainTest(
        block_m=block_m,
        smem_base=smem_base,
        vectorized=vectorized,
        apply_row_scale=apply_row_scale,
    )
    kernel(src, out, bias_val, scale_val)
    torch.cuda.synchronize()

    expected = _expected_reduce(src, block_m, smem_base, bias_val, scale_val if apply_row_scale else 1.0)
    got = out.cpu()
    if not torch.equal(got, expected):
        bad = (got != expected).nonzero().flatten()
        raise AssertionError(
            f"{bad.numel()} mismatched u16 slots; first 10: {bad[:10].tolist()}; "
            f"got {got[bad[:10]].tolist()} expected {expected[bad[:10]].tolist()}"
        )


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("smem_base", [0, 3])
@pytest.mark.parametrize("block_m", [64, 128])
def test_tmem_ts_drain(block_m, smem_base, vectorized):
    _run_drain(block_m, smem_base, vectorized, False, 0.0, 1.0)


@pytest.mark.parametrize("vectorized", [False, True])
def test_tmem_ts_drain_bias(vectorized):
    _run_drain(128, 3, vectorized, False, -0.5, 1.0)


@pytest.mark.parametrize("vectorized", [False, True])
def test_tmem_ts_drain_row_scale(vectorized):
    _run_drain(128, 3, vectorized, True, 0.25, 0.5)
