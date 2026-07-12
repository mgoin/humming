"""Stream-split probe for the B.37 WS+TMA race (track f round 3).

Runs the failing config (uint4 zp=T, (512,512,4096), block (128,128,128)
s4, WS+TMA) with ONE input stream at a time replaced by a constant
tensor. A stream whose corruption is "stale previous-phase data" becomes
invisible when its values are constant, so:
  * CONST_ZP clean while BASE fails  -> the zp stream carries the race
  * CONST_BS clean while BASE fails  -> the scale stream carries it
  * etc.
References are rebuilt from the modified tensors, so the correctness
check itself stays exact.

Usage:
  CUDA_VISIBLE_DEVICES=5 .venv/bin/python benchmarks/probe_stream_split.py [reps]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.utils.test import generate_random_inputs, generate_random_weight
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_zero_point,
)

M, N, K, GS = 512, 512, 4096, 128
BLOCK, WARP, STAGES = (128, 128, 128), (32, 64, 128), 4
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def run_variant(variant: str):
    a_dtype, b_dtype = dtypes.bfloat16, dtypes.uint4

    random_weight = generate_random_weight(
        n=N, k=K, group_size=GS, dtype=b_dtype,
        scale_dtype=dtypes.bfloat16, has_zero_point=True,
    )
    _, _, quanted_weight, weight_scale, zero_point, _ = random_weight
    _, _, inputs, _ = generate_random_inputs(m=M, k=K, group_size=0, dtype=a_dtype)

    if variant == "const_zp":
        zero_point.fill_(8)
    elif variant == "const_bs":
        weight_scale.fill_(1.0)
    elif variant == "const_b":
        quanted_weight.fill_(11)
    elif variant == "const_a":
        inputs.fill_(0.5)
    else:
        assert variant == "base"

    # Rebuild the reference from the (possibly modified) tensors.
    weight_ref = quanted_weight.float() - zero_point.float().repeat_interleave(GS, -1)
    weight_ref = weight_ref * weight_scale.float().repeat_interleave(GS, -1)
    inputs_ref = inputs.float()

    weight_prep = prepare_humming_weight(
        quanted_weight, b_dtype, a_dtype, zero_point=zero_point, use_wgmma=False)
    weight_scale_prep = prepare_humming_weight_scale(weight_scale, to_apply_on_c=False)
    zp_prep = prepare_humming_zero_point(zero_point, dtype=b_dtype)

    kernel = HummingKernel(
        shape_n=N, shape_k=K, block_shape=BLOCK, warp_shape=WARP,
        a_dtype=a_dtype, b_dtype=b_dtype,
        c_dtype=dtypes.bfloat16, bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=GS, has_zero_point=True,
        num_stages=STAGES,
        use_warp_spec=True, use_tma=True, use_cp_async=False,
        use_mbarrier=True, use_tma_bzp=False,
        has_bias=False, mma_type="tcgen05", use_tcgen05=True,
        use_stream_k=False,
    )

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch.bfloat16)
    outputs = torch.empty((M, N), dtype=torch.bfloat16, device=inputs.device)
    ops.launch_kernel(
        configs=[kernel.kernel_id], inputs=inputs, weight=weight_prep,
        outputs=outputs, weight_scale=weight_scale_prep, zero_point=zp_prep)
    torch.cuda.synchronize()

    err = (outputs.float() - outputs_ref.float()).abs()
    max_err = err.max().item()
    bad = (err > 8.0).nonzero()
    detail = ""
    if bad.numel():
        cols = sorted(set((bad[:, 1] % 128).tolist()))
        detail = f" bad_cells={bad.shape[0]} cols%128=[{cols[0]}..{cols[-1]}]"
    return max_err, detail


for variant in ["base", "const_zp", "const_bs", "const_b", "const_a"]:
    results = [run_variant(variant) for _ in range(REPS)]
    errs = ", ".join(f"{e:.1f}" for e, _ in results)
    details = next((d for _, d in results if d), "")
    print(f"{variant:9s}: max|err| x{REPS} = {errs}{details}")
