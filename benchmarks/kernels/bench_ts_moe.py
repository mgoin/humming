"""expand/moe M4: TS-mode grouped GEMM vs mma.sync grouped on realistic
MoE shapes.

Sweeps GROUPED_CONTIGUOUS + GROUPED_MASKED over the target-model
per-expert projections at a few (num_experts, top_k, batch) points, and
measures TS (BlockM from the grouped heuristic) vs the mma.sync grouped
path. Correctness of every point is verified against the per-expert
dequant reference before timing (finite + max|err| under the bf16 tol).

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/bench_ts_moe.py
"""
import time

import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.utils.test import (
    generate_random_inputs,
    generate_random_moe_tensors,
    generate_random_weight,
)
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_zero_point,
)

A_DTYPE = dtypes.bfloat16
B_DTYPE = dtypes.uint4
C_DTYPE = dtypes.bfloat16
BS_DTYPE = dtypes.bfloat16
GROUP_SIZE = 128
TORCH_DTYPE = dtypes.torch_dtype_map[C_DTYPE]

# (label, N, K, [(num_experts, top_k), ...]) per-expert projections at
# each model's realistic expert config (plus a coarse E=8 point for the
# tile-fill comparison). num_tokens fixed below.
SHAPES = [
    ("Qwen3-MoE gate/up", 1536, 4096, [(8, 2), (128, 8)]),
    ("Qwen3-MoE down", 4096, 1536, [(8, 2), (128, 8)]),
    ("DeepSeek-V3/V4 gate/up", 2048, 7168, [(8, 2), (256, 8)]),
    ("DeepSeek-V3/V4 down", 7168, 2048, [(8, 2), (256, 8)]),
    ("Mixtral gate/up", 14336, 4096, [(8, 2)]),
    ("Mixtral down", 4096, 14336, [(8, 2)]),
]
NUM_TOKENS = 512


def time_kernel(launch_fn, warmup=10, iters=30):
    for _ in range(warmup):
        launch_fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        launch_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def _make_layout(gemm_type, num_experts, m, top_k):
    avg = (m * top_k + num_experts - 1) // num_experts
    emt = None if gemm_type == "grouped_contiguous" else max(64, 4 * avg)
    for _ in range(6):
        try:
            _, layout, *_ = generate_random_moe_tensors(
                m, num_experts=num_experts, top_k=top_k,
                gemm_type=gemm_type, expert_max_tokens=emt)
            return layout, emt
        except AssertionError:
            emt *= 2
    raise RuntimeError("emt sizing failed")


def _block_m_grouped(shape_m, num_experts):
    return 128 if shape_m // max(num_experts, 1) >= 128 else 64


def _run(gemm_type, N, K, num_experts, top_k, m, use_ts):
    torch.manual_seed(0)
    layout, emt = _make_layout(gemm_type, num_experts, m, top_k)
    _, wref, wc, ws, zp, _ = generate_random_weight(
        n=N, k=K, group_size=GROUP_SIZE, dtype=B_DTYPE, scale_dtype=BS_DTYPE,
        num_experts=num_experts, has_zero_point=True)
    m_new = m * top_k if gemm_type == "grouped_contiguous" else num_experts * emt
    _, iref, inputs, _ = generate_random_inputs(
        m=m_new, k=K, group_size=0, dtype=A_DTYPE)

    weight = prepare_humming_weight(
        wc, B_DTYPE, A_DTYPE, zero_point=zp, use_tcgen05_ts=use_ts)
    block_m = _block_m_grouped(m_new, num_experts)
    if use_ts:
        wsp = prepare_humming_weight_scale(ws, to_apply_on_c=False, use_tcgen05_ts=True)
        zpp = prepare_humming_zero_point(zp, B_DTYPE, packed=False, use_tcgen05_ts=True)
        kern = HummingKernel(
            shape_n=N, shape_k=K, block_shape=(block_m, 128, 64),
            warp_shape=(block_m, 32, 64), a_dtype=A_DTYPE, b_dtype=B_DTYPE,
            c_dtype=C_DTYPE, bs_dtype=BS_DTYPE, num_experts=num_experts,
            num_stages=4, use_warp_spec=True, has_bias=False, has_zero_point=True,
            weight_scale_group_size=GROUP_SIZE, mma_type="tcgen05",
            use_tcgen05=True, use_tcgen05_ts=True, use_tma=True,
            use_mbarrier=True, use_tma_bzp=False, use_stream_k=False,
            gemm_type=gemm_type)
    else:
        wsp = prepare_humming_weight_scale(ws, to_apply_on_c=False)
        zpp = prepare_humming_zero_point(zp, B_DTYPE, packed=False)
        bm = min(block_m, 64)
        kern = HummingKernel(
            shape_n=N, shape_k=K,
            block_shape=(bm, 256, 32), warp_shape=(bm, 64, 32),
            a_dtype=A_DTYPE, b_dtype=B_DTYPE, c_dtype=C_DTYPE, bs_dtype=BS_DTYPE,
            num_experts=num_experts, num_stages=3, use_warp_spec=False,
            has_bias=False, has_zero_point=True, weight_scale_group_size=GROUP_SIZE,
            mma_type="mma", use_tma=False, use_stream_k=False, gemm_type=gemm_type)

    out = torch.zeros((m_new, N), dtype=TORCH_DTYPE, device=inputs.device)

    def launch():
        return ops.launch_kernel(
            configs=[kern.kernel_id], inputs=inputs, weight=weight, outputs=out,
            weight_scale=wsp, zero_point=zpp, expert_layout=layout)

    res = launch().view(-1, N)
    torch.cuda.synchronize()

    ref = torch.zeros_like(res)
    wr = wref.to(torch.bfloat16).float()
    for e in range(num_experts):
        if gemm_type == "grouped_contiguous":
            o1 = int(layout[e])
            o2 = m_new if e == num_experts - 1 else int(layout[e + 1])
        else:
            o1 = emt * e
            o2 = o1 + int(layout[e])
        if o2 > o1:
            ref[o1:o2] = iref[o1:o2].matmul(wr[e].T).to(TORCH_DTYPE)
    err = (res.float() - ref.float()).abs()
    tol = 0.5 + 1e-2 * ref.float().abs().max().item()
    ok = torch.isfinite(res).all().item() and err.max().item() <= tol

    us = time_kernel(launch)
    return us, block_m, ok, err.max().item()


def main():
    print(f"device: {torch.cuda.get_device_name(0)}\n")
    hdr = f"{'shape':<26}{'mode':<12}{'E':>4}{'tk':>3}{'m':>6}{'bm':>4}" \
          f"{'TS us':>10}{'mma us':>10}{'TS/mma':>8}{'ok':>4}"
    for gemm_type in ["grouped_contiguous", "grouped_masked"]:
        print(hdr)
        print("-" * len(hdr))
        for label, N, K, points in SHAPES:
            for ne, tk in points:
                m = NUM_TOKENS
                try:
                    ts_us, bm, ts_ok, ts_err = _run(gemm_type, N, K, ne, tk, m, True)
                    mma_us, _, mma_ok, _ = _run(gemm_type, N, K, ne, tk, m, False)
                    ratio = ts_us / mma_us if mma_us else float("nan")
                    print(f"{label:<26}{gemm_type.split('_')[1]:<12}{ne:>4}{tk:>3}{m:>6}"
                          f"{bm:>4}{ts_us:>10.1f}{mma_us:>10.1f}{ratio:>8.2f}"
                          f"{'Y' if ts_ok and mma_ok else 'N':>4}")
                except Exception as e:
                    print(f"{label:<26}{gemm_type.split('_')[1]:<12}{ne:>4}"
                          f"{tk:>3}{m:>6}  ERR {str(e)[:40]}")
        print()


if __name__ == "__main__":
    main()
