"""Alias-identity forensics for B.37: are corrupted C cells bit-exactly
the stage-0 A activations that alias smem.reduce?

smem.reduce (offset 0 of the SMEM union) aliases stages[0].a. Both use
the same col^row%8 128B swizzle, which cancels in the mapping, so a
corrupted C cell (m, n) [section 0: n%128 < 64] should hold
  A[m_tile*128 + m%128, k_last_block_base + (n % 64)]
where k_last_block_base = (num_k_blocks - num_stages) * BlockK is the
last k-block resident in stage 0 (32 blocks, s4 -> block 28 -> 3584).

Usage:
  CUDA_VISIBLE_DEVICES=5 .venv/bin/python benchmarks/probe_b37_alias.py
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
K_LAST_BLOCK = (K // BLOCK[2] - STAGES) * BLOCK[2]  # 3584

a_dtype, b_dtype = dtypes.bfloat16, dtypes.uint4

for rep in range(8):
    random_weight = generate_random_weight(
        n=N, k=K, group_size=GS, dtype=b_dtype,
        scale_dtype=dtypes.bfloat16, has_zero_point=True)
    _, weight_ref, quanted_weight, weight_scale, zero_point, _ = random_weight
    _, inputs_ref, inputs, _ = generate_random_inputs(m=M, k=K, group_size=0,
                                                      dtype=a_dtype)
    weight_prep = prepare_humming_weight(
        quanted_weight, b_dtype, a_dtype, zero_point=zero_point,
        use_wgmma=False)
    weight_scale_prep = prepare_humming_weight_scale(weight_scale,
                                                     to_apply_on_c=False)
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
        use_stream_k=False)

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch.bfloat16)
    outputs = torch.empty((M, N), dtype=torch.bfloat16, device=inputs.device)
    ops.launch_kernel(
        configs=[kernel.kernel_id], inputs=inputs, weight=weight_prep,
        outputs=outputs, weight_scale=weight_scale_prep, zero_point=zp_prep)
    torch.cuda.synchronize()

    err = (outputs.float() - outputs_ref.float()).abs()
    bad = (err > 8.0).nonzero()
    if not bad.numel():
        print(f"rep {rep}: clean")
        continue

    match = 0
    total = 0
    mismatches = []
    for i in range(bad.shape[0]):
        m, n = int(bad[i, 0]), int(bad[i, 1])
        n_in_tile = n % 128
        if n_in_tile >= 64:
            k_alias = K_LAST_BLOCK + 64 + (n_in_tile - 64)
        else:
            k_alias = K_LAST_BLOCK + n_in_tile
        predicted = inputs[m, k_alias]
        total += 1
        if predicted == outputs[m, n]:
            match += 1
        elif len(mismatches) < 4:
            mismatches.append((m, n, float(outputs[m, n]), float(predicted)))
    print(f"rep {rep}: bad={total} alias-identity match={match} "
          f"({100.0 * match / total:.1f}%)"
          + (f" mism={mismatches}" if mismatches else ""))
