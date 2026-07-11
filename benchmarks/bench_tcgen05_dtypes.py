"""Benchmark TCGEN05 across the supported B-dtype matrix at the
production WS config.

For each (B_dtype, has_zero_point) combo that humming's TCGEN05 path
accepts, bench at one representative Llama-3 weight shape per "size
class" (small-decode, mid-prefill, large-prefill) and compare to the
corresponding mma.sync reference.

The set of B-dtypes mirrors `tests/test_tcgen05_dtypes.py`:
  * unsigned ints uint{1..8} (asymmetric -- has_zero_point=True)
  * signed ints int{2..8} (symmetric -- has_zero_point=False)
  * sub-byte floats float{4e2m1, 6e2m3, 6e3m2}
  * 8-bit floats float8{e4m3, e5m2}

Run with:
    CUDA_VISIBLE_DEVICES=<idle GPU> \\
        ~/venvs/vllm-rel/bin/python benchmarks/bench_tcgen05_dtypes.py
"""
import os
import sys
import time

import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.utils.test import generate_random_inputs, generate_random_weight
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_zero_point,
)


A_DTYPE = dtypes.bfloat16
C_DTYPE = dtypes.bfloat16
BS_DTYPE = dtypes.bfloat16
GROUP_SIZE = 128


B_DTYPES = [
    # (b_name, has_zero_point)
    ("uint1", False),
    ("uint2", True),
    ("uint3", True),
    ("uint4", True),
    ("uint4", False),
    ("uint5", True),
    ("uint6", True),
    ("uint7", True),
    ("uint8", True),
    ("int2", False),
    ("int3", False),
    ("int4", False),
    ("int6", False),
    ("int8", False),
    ("float4e2m1", False),
    ("float6e2m3", False),
    ("float6e3m2", False),
    ("float8e4m3", False),
    ("float8e5m2", False),
]


# (label, N, K) -- representative Llama-3 weight shapes.
SHAPES = [
    ("Llama8B  gate", 14336, 4096),
    ("Llama70B down",  8192, 28672),
]

# Benchmark each at one "medium" prefill batch for a quick sweep; the
# uint4 reference numbers come from bench_tcgen05_vs_wmma.py.
MS = [256, 2048]


def time_kernel(launch_fn, warmup=10, iters=50):
    for _ in range(warmup):
        launch_fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        launch_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def round_up(x, m):
    return ((x + m - 1) // m) * m


# Ladder of TCGEN05 configs to try, biggest first. Wider B-dtypes
# (uint{5..8}, fp{6,8}) need smaller BlockK or fewer stages to fit
# the 232 KiB SMEM cap on cc 10.x.
TCGEN05_CONFIGS = [
    # (block_shape, warp_shape, num_stages)
    ((128, 128, 128), (32, 64, 128), 4),  # full prod config (fits uint{1..4}, fp4)
    ((128, 128, 128), (32, 64, 128), 3),
    ((128, 128, 128), (32, 64, 128), 2),
    ((128, 128,  64), (32, 64,  64), 4),
    ((128, 128,  64), (32, 64,  64), 3),
    ((128, 128,  64), (32, 64,  64), 2),
    ((64,  128,  64), (16, 64,  64), 3),
    ((64,   64,  64), (16, 64,  64), 2),
]


def build_and_verify(mma_type, b_name, has_zp, shape_m, shape_n, shape_k,
                     tcgen05_cfg_idx=0):
    """Build a launcher and verify correctness at a small probe shape.

    Returns (launch_fn, "ok") if the config builds, runs, and is
    bit-correct against a float reference; (None, "<reason>") otherwise.

    Verification uses a fresh small (128, 128, 256) problem so we
    don't pay the full bench shape just to find broken configs.
    """
    b_dtype = dtypes.DataType.from_str(b_name)
    if mma_type == "tcgen05":
        block_shape, warp_shape, num_stages = TCGEN05_CONFIGS[tcgen05_cfg_idx]
        use_warp_spec = True
        use_tma = True
        use_cp_async = False
        use_mbarrier = True
    else:
        # mma.sync baseline at the same M,N,K granularity.
        block_shape = (64, 128, 64)
        warp_shape = (16, 64, 64)
        num_stages = 3
        use_warp_spec = False
        use_tma = False
        use_cp_async = True
        use_mbarrier = False

    def _kernel_factory(n, k, m_probe=None):
        """Build the kernel + weights + inputs for either the verify
        probe (small) or the bench shape (large)."""
        m_to_use = m_probe if m_probe is not None else shape_m
        torch.manual_seed(123)
        try:
            w = generate_random_weight(
                n=n, k=k, group_size=GROUP_SIZE,
                dtype=b_dtype, scale_dtype=BS_DTYPE, has_zero_point=has_zp,
            )
        except (AssertionError, RuntimeError):
            return None
        _, w_ref, weight, weight_scale, zero_point, _ = w
        try:
            weight_p = prepare_humming_weight(
                weight, b_dtype, A_DTYPE,
                zero_point=zero_point if has_zp else None,
                use_wgmma=False,
            )
            weight_scale_p = prepare_humming_weight_scale(
                weight_scale, to_apply_on_c=False,
            )
            zp_p = (prepare_humming_zero_point(zero_point, dtype=b_dtype)
                    if has_zp else None)
        except (AssertionError, RuntimeError):
            return None
        _, in_ref, inputs, _ = generate_random_inputs(
            m=m_to_use, k=k, group_size=0, dtype=A_DTYPE,
        )
        outputs = torch.empty(
            (m_to_use, n), dtype=torch.bfloat16, device="cuda",
        )
        try:
            kernel = HummingKernel(
                shape_n=n, shape_k=k,
                block_shape=block_shape, warp_shape=warp_shape,
                a_dtype=A_DTYPE, b_dtype=b_dtype, c_dtype=C_DTYPE,
                bs_dtype=BS_DTYPE,
                weight_scale_group_size=GROUP_SIZE,
                has_zero_point=has_zp,
                num_stages=num_stages,
                use_warp_spec=use_warp_spec,
                use_tma=use_tma,
                use_cp_async=use_cp_async,
                use_mbarrier=use_mbarrier,
                use_tma_bzp=False,
                has_bias=False,
                mma_type=mma_type,
                use_tcgen05=(mma_type == "tcgen05"),
                use_stream_k=False,
            )
        except (AssertionError, RuntimeError):
            return None
        launch_kwargs = dict(
            configs=[kernel.kernel_id], inputs=inputs, weight=weight_p,
            outputs=outputs, weight_scale=weight_scale_p,
        )
        if zp_p is not None:
            launch_kwargs["zero_point"] = zp_p
        def launch():
            ops.launch_kernel(**launch_kwargs)
        return launch, in_ref, w_ref, outputs

    # Step 1: build at the verify-probe shape and check correctness.
    verify = _kernel_factory(128, 256, m_probe=128)
    if verify is None:
        return (None, "build")
    v_launch, v_in_ref, v_w_ref, v_out = verify
    try:
        v_launch()
        torch.cuda.synchronize()
    except RuntimeError:
        return (None, "launch")
    v_ref = v_in_ref.matmul(v_w_ref.T).to(torch.bfloat16)
    # Match the test's tolerance (`torch.testing.assert_close
    # rtol=1e-2, atol=0.5`): |x - y| <= atol + rtol * |y|. Plain
    # `> atol` is too strict and rejects bf16-precision drift.
    abs_err = (v_out.float() - v_ref.float()).abs()
    threshold = 0.5 + 1e-2 * v_ref.float().abs()
    if (abs_err > threshold).any():
        rel = (abs_err.max() / max(v_ref.float().abs().max().item(), 1e-9))
        return (None, f"WRONG-OUTPUT (rel {rel*100:.0f}%)")

    # Step 2: build at the actual bench shape.
    bench = _kernel_factory(shape_n, shape_k, m_probe=shape_m)
    if bench is None:
        return (None, "bench-build")
    b_launch, _, _, _ = bench
    return (b_launch, "ok")


def bench_one(b_name, has_zp, shape_m, shape_n, shape_k, mma_type):
    """Bench one (mma_type, dtype, shape).

    For mma_type=="tcgen05" walks the config ladder, **verifies
    correctness at a small probe shape**, and returns the fastest
    config that is bit-correct. For "mma" uses a single config.
    """
    sn = round_up(shape_n, 128)
    sk = round_up(shape_k, 128)
    sm = round_up(max(shape_m, 128), 128) if mma_type == "tcgen05" else max(shape_m, 1)
    if mma_type == "tcgen05":
        best = (None, None)
        for cfg_idx in range(len(TCGEN05_CONFIGS)):
            launch, why = build_and_verify(
                mma_type, b_name, has_zp, sm, sn, sk, cfg_idx,
            )
            if launch is None:
                continue
            try:
                us = time_kernel(launch)
            except RuntimeError:
                continue
            if best[0] is None or us < best[0]:
                best = (us, cfg_idx)
        return best  # (us_or_None, cfg_idx_or_None)
    else:
        launch, _ = build_and_verify(mma_type, b_name, has_zp, sm, sn, sk)
        if launch is None:
            return (None, None)
        try:
            return (time_kernel(launch), 0)
        except RuntimeError:
            return (None, None)


def cfg_label(idx):
    if idx is None:
        return "  --  "
    bs, _, ns = TCGEN05_CONFIGS[idx]
    return f"M{bs[0]}K{bs[2]}s{ns}"


def fmt(us):
    return f"{us:>9.1f}" if isinstance(us, float) else f"{'  --  ':>9s}"


def main():
    print(
        f"{'shape':<15s} {'M':>5s} {'B dtype':<14s} {'zp':>4s}  "
        f"{'tcg us':>9s} {'tcg cfg':>11s} {'mma us':>9s} {'tcg/mma':>10s}"
    )
    for label, n, k in SHAPES:
        for m in MS:
            for b_name, has_zp in B_DTYPES:
                t_tcg, cfg_idx = bench_one(b_name, has_zp, m, n, k, "tcgen05")
                t_mma, _ = bench_one(b_name, has_zp, m, n, k, "mma")
                if isinstance(t_tcg, float) and isinstance(t_mma, float):
                    ratio = f"{t_mma / t_tcg:>9.2f}x"
                else:
                    ratio = "   --   "
                if t_tcg is None and t_mma is None:
                    print(
                        f"  {label:<13s} {m:>5d} {b_name:<14s} {str(has_zp):>4s}  "
                        f"  (humming rejects -- no valid tcg or mma config)"
                    )
                    continue
                print(
                    f"  {label:<13s} {m:>5d} {b_name:<14s} {str(has_zp):>4s}  "
                    f"{fmt(t_tcg)} {cfg_label(cfg_idx):>11s} {fmt(t_mma)} {ratio:>10s}"
                )
            print()


if __name__ == "__main__":
    main()
