"""First TCGEN05 path correctness test.

Constructs HummingKernel directly with `mma_type='tcgen05'` and a minimal
W4A16 config. This is the smallest viable instance:

  bf16 A x uint4 B with per-group bf16 scales (gs=128) and zero-points,
  block (M=64, N=128, K=128), 2-stage pipeline, single-CTA, no warp-spec.

Failure modes we expect to hit on first runs (recorded in workbook.md):
  * SMEM layout mismatch between r2s and tcgen05.mma's swizzle expectations.
    Math will be wrong, not segfault.
  * Missing mbarrier-based commit -- the __syncthreads in TCGEN05::run()
    should be safe but slow.
  * Instruction descriptor / dtype-code bit layout off-by-one. Will likely
    show as wrong-shaped accumulator or ptxas error.

We start the test loose (atol=0.5) to discover ordering; will tighten as
the path stabilises.
"""

from __future__ import annotations

import pytest
import torch

from humming import dtypes
from humming.kernel.humming import HummingKernel
from humming.utils.test import generate_random_inputs, generate_random_weight
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_zero_point,
)


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _is_blackwell(), reason="tcgen05 needs sm_100+")


@pytest.mark.xfail(
    reason=(
        "Phase B.4 in progress: completion path (mbarrier-based commit/wait) "
        "is wired up so the kernel runs to completion, but the r2s lane "
        "mapping in transform_b doesn't yet match what tcgen05.mma expects "
        "from the SMEM B descriptor (workbook 'B1 vs B2'). Expect wrong "
        "math until the m16n8k16 B-fragment scatter lands."
    ),
    strict=False,
)
def test_tcgen05_w4a16_smallest():
    a_dtype = dtypes.bfloat16
    b_dtype = dtypes.uint4
    c_dtype = dtypes.bfloat16
    bs_dtype = dtypes.bfloat16
    group_size = 128

    # Smallest viable shape (still a multiple of the kernel's tile).
    # Problem shape divides BlockShape evenly.
    shape_m = 128
    shape_n = 128
    shape_k = 256

    # Phase B.4 smallest viable: BlockM=64 (min for tcgen05.mma kind::f16),
    # BlockN=64 keeps the per-warp t2r footprint at uint32[64] per thread
    # using 4 calls of 32dp32b32x.
    block_shape = (64, 64, 128)
    warp_shape = (64, 64, 32)

    # Build the W4A16 problem identical to what test_shape uses.
    random_weight = generate_random_weight(
        n=shape_n, k=shape_k, group_size=group_size,
        dtype=b_dtype, scale_dtype=bs_dtype,
        has_zero_point=True,
    )
    _, weight_ref, weight, weight_scale, zero_point, _ = random_weight
    weight = prepare_humming_weight(weight, b_dtype, a_dtype, use_wgmma=False)
    weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=False)
    zero_point = prepare_humming_zero_point(zero_point, dtype=b_dtype)

    _, inputs_ref, inputs, _ = generate_random_inputs(
        m=shape_m, k=shape_k, group_size=0, dtype=a_dtype,
    )

    # Construct the TCGEN05 kernel explicitly. The novel flag is mma_type
    # + use_tcgen05 -- everything else is "boring" mainloop configuration.
    kernel = HummingKernel(
        shape_n=shape_n,
        shape_k=shape_k,
        block_shape=block_shape,
        warp_shape=warp_shape,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        weight_scale_group_size=group_size,
        has_zero_point=True,
        num_stages=2,
        use_warp_spec=False,
        use_tma=False,
        use_cp_async=True,
        has_bias=False,
        mma_type="tcgen05",
        use_tcgen05=True,
        use_stream_k=False,
    )

    # Compute the reference FIRST so a misbehaving tcgen05 kernel that
    # corrupts CUDA context doesn't take cublas down with it.
    outputs_ref = inputs_ref.matmul(weight_ref.T)
    torch.cuda.synchronize()

    from humming import ops
    outputs = torch.empty(
        (shape_m, shape_n),
        dtype=dtypes.torch_dtype_map[c_dtype],
        device=inputs.device,
    )
    outputs = ops.launch_kernel(
        configs=[kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        weight_scale=weight_scale,
        zero_point=zero_point,
    )
    torch.cuda.synchronize()

    outputs_ref = outputs_ref.to(outputs.dtype)

    # Diagnostics first -- tells us whether the kernel produced garbage,
    # all-zeros, or something with the right magnitude in the wrong layout.
    abs_err = (outputs.float() - outputs_ref.float()).abs()
    print(
        f"\n  out[:2,:4]={outputs[:2,:4].tolist()}\n"
        f"  ref[:2,:4]={outputs_ref[:2,:4].tolist()}\n"
        f"  max|err|={abs_err.max().item():.3e}  mean|err|={abs_err.mean().item():.3e}\n"
        f"  |ref|: mean={outputs_ref.float().abs().mean().item():.3e}  "
        f"max={outputs_ref.float().abs().max().item():.3e}\n"
        f"  out nonzero frac: {(outputs.float().abs() > 1e-6).float().mean().item():.3f}"
    )

    # Loose tolerance for first pass -- tighten once correct.
    torch.testing.assert_close(outputs, outputs_ref, rtol=0.5, atol=0.5)
